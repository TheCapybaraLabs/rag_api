"""Card #278 — integration: unique chunk identity + namespace-scoped batch ops.

Requires Docker (testcontainers). Run explicitly:
    pytest tests/integration/test_unique_chunk_identity.py -m integration -v

The migration under migrations/ must apply cleanly on the schema mirrored by
conftest (langchain_pg_embedding), enforce (collection_id, custom_id, user_id,
chunk_index) uniqueness for NEW-format rows only, and leave legacy rows alone.
"""

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.services.vector_store.extended_pg_vector import ExtendedPgVector
from langchain_community.vectorstores.pgvector import _get_embedding_collection_store

pytestmark = pytest.mark.integration

MIGRATION = Path(__file__).resolve().parents[2] / "migrations" / (
    "2026-10-22_card278_unique_chunk_identity.sql"
)


def _executable_sql(raw: str) -> str:
    """Strip `--` comments so only real statements remain (keeps the audit query
    and the CONCURRENTLY variant out of the executed script)."""
    lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)
    return "\n".join(lines)


def _insert_embedding(
    conn,
    collection_id,
    *,
    file_id,
    user_id,
    chunk_index,
    document,
    custom_id=None,
    with_new_metadata=True,
    digest=None,
):
    meta = {
        "file_id": file_id,
        "user_id": user_id,
        "digest": digest if digest is not None else f"d{chunk_index}",
    }
    if with_new_metadata:
        meta["chunk_index"] = chunk_index
    conn.execute(
        text(
            "INSERT INTO langchain_pg_embedding "
            "(collection_id, embedding, document, cmetadata, custom_id) "
            "VALUES (:cid, '[0.1,0.2,0.3]', :doc, CAST(:meta AS jsonb), :cust)"
        ),
        {
            "cid": collection_id,
            "doc": document,
            "meta": json.dumps(meta),
            "cust": custom_id or file_id,
        },
    )


def _store(engine, collection_id):
    """ExtendedPgVector bound to the test engine with a stubbed collection lookup."""

    class TestableStore(ExtendedPgVector):
        def __init__(self):
            self._bind = engine
            EmbeddingStore, _ = _get_embedding_collection_store(
                vector_dimension=3, use_jsonb=True
            )
            self.EmbeddingStore = EmbeddingStore

        def get_collection(self, session):
            return SimpleNamespace(uuid=collection_id)

    return TestableStore()


def test_migration_applies_and_enforces_chunk_identity(engine, collection_id):
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.exec_driver_sql(_executable_sql(MIGRATION.read_text()))

        _insert_embedding(
            conn, collection_id, file_id="f1", user_id="u1", chunk_index=0, document="a"
        )
        # same identity → blocked
        with pytest.raises(IntegrityError):
            _insert_embedding(
                conn, collection_id, file_id="f1", user_id="u1", chunk_index=0, document="a"
            )
        trans.rollback()
        trans = conn.begin()

        # different namespace → allowed (agents sharing a record each hold a copy)
        _insert_embedding(
            conn, collection_id, file_id="f1", user_id="u1", chunk_index=0, document="a"
        )
        _insert_embedding(
            conn, collection_id, file_id="f1", user_id="agent-2", chunk_index=0, document="a"
        )
        # different chunk position → allowed
        _insert_embedding(
            conn, collection_id, file_id="f1", user_id="u1", chunk_index=1, document="b"
        )
        trans.rollback()
        trans = conn.begin()

        # legacy rows (no chunk_index/user_id in cmetadata) stay outside the index
        _insert_embedding(
            conn,
            collection_id,
            file_id="legacy",
            user_id="u1",
            chunk_index=0,
            document="old",
            with_new_metadata=False,
        )
        _insert_embedding(
            conn,
            collection_id,
            file_id="legacy",
            user_id="u1",
            chunk_index=0,
            document="old",
            with_new_metadata=False,
        )
    finally:
        trans.rollback()
        conn.close()


def test_get_and_delete_file_batch_are_namespace_scoped(engine, collection_id):
    store = _store(engine, collection_id)
    try:
        with engine.begin() as conn:
            _insert_embedding(conn, collection_id, file_id="f9-scope", user_id="u1", chunk_index=0, document="a")
            _insert_embedding(conn, collection_id, file_id="f9-scope", user_id="u1", chunk_index=1, document="b")
            _insert_embedding(conn, collection_id, file_id="f9-scope", user_id="agent-2", chunk_index=0, document="a")

        digests_u1 = store.get_file_batch_digests("f9-scope", "u1")
        assert sorted(digests_u1) == ["d0", "d1"]
        assert store.get_file_batch_digests("f9-scope", "agent-2") == ["d0"]
        assert store.get_file_batch_digests("missing", "u1") is None

        removed = store.delete_file_batch("f9-scope", "u1")
        assert removed == 2
        # other namespace untouched
        assert store.get_file_batch_digests("f9-scope", "agent-2") == ["d0"]
        assert store.delete_file_batch("f9-scope", "u1") == 0
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'file_id' = 'f9-scope'")
            )


# ---------------------------------------------------------------------------
# Card #278 — replace_file_batch atômico (achado 3)
# ---------------------------------------------------------------------------

def _replace_store(engine, collection_id, embed_documents=None):
    store = _store(engine, collection_id)
    store.embedding_function = SimpleNamespace(
        embed_documents=embed_documents
        or (lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    )
    return store


def _replace_docs(content: str, n: int, file_id: str, user_id: str):
    return [
        Document(
            page_content=f"{content}-{i}",
            metadata={
                "file_id": file_id,
                "user_id": user_id,
                "chunk_index": i,
                "digest": hashlib.md5(f"{content}-{i}".encode()).hexdigest(),
            },
        )
        for i in range(n)
    ]


def test_replace_file_batch_swaps_rows_atomically(engine, collection_id):
    """Achado 3: o swap substitui o lote do namespace e preserva os outros."""
    store = _replace_store(engine, collection_id)
    try:
        with engine.begin() as conn:
            _insert_embedding(conn, collection_id, file_id="f-replace", user_id="u1", chunk_index=0, document="old")
            _insert_embedding(conn, collection_id, file_id="f-replace", user_id="agent-2", chunk_index=0, document="old")

        ids = store.replace_file_batch(
            "f-replace", "u1", _replace_docs("new", 3, "f-replace", "u1")
        )
        assert ids == ["f-replace"] * 3
        esperados = sorted(
            hashlib.md5(f"new-{i}".encode()).hexdigest() for i in range(3)
        )
        assert sorted(store.get_file_batch_digests("f-replace", "u1")) == esperados
        # outro namespace intocado
        assert store.get_file_batch_digests("f-replace", "agent-2") == ["d0"]
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'file_id' = 'f-replace'"))


def test_replace_file_batch_provider_failure_keeps_old_batch(engine, collection_id):
    """Falha do provedor de embedding ANTES do delete: vetores bons sobrevivem."""
    def broken_embed(texts):
        raise RuntimeError("provider down")

    store = _replace_store(engine, collection_id, embed_documents=broken_embed)
    try:
        with engine.begin() as conn:
            _insert_embedding(conn, collection_id, file_id="f-provider", user_id="u1", chunk_index=0, document="good")

        with pytest.raises(RuntimeError, match="provider down"):
            store.replace_file_batch(
                "f-provider", "u1", _replace_docs("new", 2, "f-provider", "u1")
            )

        # o lote antigo continua lá, intacto
        assert store.get_file_batch_digests("f-provider", "u1") == ["d0"]
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'file_id' = 'f-provider'"))


def test_replace_file_batch_insert_failure_keeps_old_batch(engine, collection_id):
    """Falha de INSERT no meio do swap (violação de identidade no lote novo):
    a transação única reverte delete+insert e o lote antigo permanece."""
    store = _replace_store(engine, collection_id)
    conn = engine.connect()
    trans = conn.begin()
    try:
        with engine.begin() as seed:
            _insert_embedding(seed, collection_id, file_id="f-insert", user_id="u1", chunk_index=0, document="good")

        # índice único ativo para o lote novo colidir consigo mesmo
        conn.exec_driver_sql(_executable_sql(MIGRATION.read_text()))
        trans.commit()
        trans = conn.begin()

        # dois chunks com o MESMO chunk_index violam o índice único → commit falha
        bad_docs = [
            Document(
                page_content="dup",
                metadata={"file_id": "f-insert", "user_id": "u1", "chunk_index": 0},
            ),
            Document(
                page_content="dup",
                metadata={"file_id": "f-insert", "user_id": "u1", "chunk_index": 0},
            ),
        ]
        with pytest.raises(IntegrityError):
            store.replace_file_batch("f-insert", "u1", bad_docs)

        assert store.get_file_batch_digests("f-insert", "u1") == ["d0"]
    finally:
        if trans.is_active:
            trans.rollback()
        conn.close()
        with engine.begin() as cleanup:
            cleanup.execute(text("DROP INDEX IF EXISTS uq_langchain_pg_embedding_chunk_identity"))
            cleanup.execute(text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'file_id' = 'f-insert'"))


def test_unique_violation_race_preserves_winning_rows(engine, collection_id):
    """Achado 4: com o índice único ativo, o insert do perdedor da corrida levanta
    IntegrityError e as linhas do VENCEDOR permanecem (nenhum rollback cru)."""
    conn = engine.connect()
    trans = conn.begin()
    try:
        # índice único + lote do vencedor no MESMO commit: a corrida é real
        conn.exec_driver_sql(_executable_sql(MIGRATION.read_text()))
        _insert_embedding(conn, collection_id, file_id="f-race", user_id="u1", chunk_index=0, document="a")
        _insert_embedding(conn, collection_id, file_id="f-race", user_id="u1", chunk_index=1, document="b")
        trans.commit()
        trans = conn.begin()

        store = _replace_store(engine, collection_id)
        loser_docs = _replace_docs("a", 2, "f-race", "u1")  # mesmas identidades (0,1)
        with pytest.raises(IntegrityError):
            store.add_documents(loser_docs, ids=["f-race"] * 2)

        sobreviventes = conn.execute(
            text(
                "SELECT cmetadata->>'chunk_index' FROM langchain_pg_embedding "
                "WHERE cmetadata->>'file_id' = 'f-race' AND cmetadata->>'user_id' = 'u1' "
                "ORDER BY 1"
            )
        ).scalars().all()
        assert sobreviventes == ["0", "1"], "linhas do vencedor devem permanecer"
    finally:
        if trans.is_active:
            trans.rollback()
        conn.close()
        # limpeza global: o índice e as linhas não podem vazar para outros testes
        with engine.begin() as cleanup:
            cleanup.execute(text("DROP INDEX IF EXISTS uq_langchain_pg_embedding_chunk_identity"))
            cleanup.execute(text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'file_id' = 'f-race'"))


class _RaceLoserStore(AsyncPgVector):
    """Store do PERDEDOR da corrida sobre o banco real.

    A primeira consulta de digests devolve None (o estado que o perdedor enxerga
    antes do commit do vencedor); as seguintes delegam ao store real. O insert vai
    ao banco de verdade e bate no índice único — a corrida não é simulada com mock.
    """

    def __init__(self, real_store):
        self._real = real_store
        self.consultas = 0
        self.deletes = 0

    async def get_file_batch_digests(self, file_id, user_id, executor=None):
        self.consultas += 1
        if self.consultas == 1:
            return None
        return self._real.get_file_batch_digests(file_id, user_id)

    async def aadd_documents(self, documents, ids=None, executor=None, **kwargs):
        return self._real.add_documents(documents, ids=ids)

    async def delete_file_batch(self, file_id, user_id, executor=None):
        self.deletes += 1
        return self._real.delete_file_batch(file_id, user_id)


def test_store_data_in_vector_db_race_reports_deduplicated_and_skips_rollback(
    engine, collection_id, monkeypatch
):
    """Rodada 3 (sugestão): o caminho app INTEIRO — store_data_in_vector_db — sob
    corrida real (índice único ativo, linhas do vencedor no banco): o perdedor
    reconhece o lote do vencedor, responde deduplicated e NÃO apaga nada."""
    import asyncio

    from app.routes import document_routes

    store = _replace_store(engine, collection_id)
    race_store = _RaceLoserStore(store)

    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.exec_driver_sql(_executable_sql(MIGRATION.read_text()))
        textos = ["conteudo-raca-a", "conteudo-raca-b"]
        for i, texto in enumerate(textos):
            _insert_embedding(
                conn,
                collection_id,
                file_id="f-app-race",
                user_id="u1",
                chunk_index=i,
                document=texto,
                digest=hashlib.md5(texto.encode()).hexdigest(),
            )
        trans.commit()
        trans = conn.begin()

        print("DEBUG vector_store global:", type(document_routes.vector_store)); print("DEBUG race_store:", type(race_store)); print("DEBUG _real:", type(getattr(race_store, "_real", None)), "| get_file_batch_digests de _real:", getattr(type(getattr(race_store, "_real", None)), "get_file_batch_digests", None)); print("DEBUG CollectionStore em _real:", hasattr(getattr(race_store, "_real", None), "CollectionStore"));
        monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0)
        # store_data_in_vector_db usa o vector_store GLOBAL do módulo de rotas —
        # o perdedor precisa estar lá para a corrida exercitar o caminho app real.
        monkeypatch.setattr(document_routes, "vector_store", race_store)
        entrada = [Document(page_content=t) for t in textos]
        resultado = asyncio.run(
            document_routes.store_data_in_vector_db(
                entrada, "f-app-race", "u1", executor=None
            )
        )

        assert resultado["deduplicated"] is True
        assert race_store.consultas == 2, "pré-checagem + reconferência pós-corrida"
        assert race_store.deletes == 0, "rollback pulado: nada é apagado na corrida"

        sobreviventes = conn.execute(
            text(
                "SELECT cmetadata->>'user_id', cmetadata->>'digest' "
                "FROM langchain_pg_embedding "
                "WHERE cmetadata->>'file_id' = 'f-app-race' "
                "ORDER BY cmetadata->>'chunk_index'"
            )
        ).all()
        assert sobreviventes == [
            ("u1", hashlib.md5(b"conteudo-raca-a").hexdigest()),
            ("u1", hashlib.md5(b"conteudo-raca-b").hexdigest()),
        ], "linhas do VENCEDOR permanecem, intactas"
    finally:
        if trans.is_active:
            trans.rollback()
        conn.close()
        with engine.begin() as cleanup:
            cleanup.execute(
                text("DROP INDEX IF EXISTS uq_langchain_pg_embedding_chunk_identity")
            )
            cleanup.execute(
                text(
                    "DELETE FROM langchain_pg_embedding "
                    "WHERE cmetadata->>'file_id' = 'f-app-race'"
                )
            )


def test_filter_eq_none_matches_only_legacy_rows(engine, collection_id):
    """/query fallback: user_id $eq None → astext IS NULL (linha sem a chave ou
    com null); linhas com namespace — inclusive '' — ficam fora."""
    store = _store(engine, collection_id)
    try:
        with engine.begin() as conn:
            _insert_embedding(conn, collection_id, file_id="f-null", user_id="u1", chunk_index=0, document="a")
            conn.execute(
                text(
                    "INSERT INTO langchain_pg_embedding "
                    "(collection_id, embedding, document, cmetadata, custom_id) "
                    "VALUES (:cid, '[0.1,0.2,0.3]', :doc, CAST(:meta AS jsonb), :cust)"
                ),
                [
                    {"cid": collection_id, "doc": "legacy", "meta": json.dumps({"file_id": "f-null"}), "cust": "f-null"},
                    {"cid": collection_id, "doc": "null-key", "meta": json.dumps({"file_id": "f-null", "user_id": None}), "cust": "f-null"},
                    {"cid": collection_id, "doc": "empty-ns", "meta": json.dumps({"file_id": "f-null", "user_id": ""}), "cust": "f-null"},
                ],
            )

        clause = store._handle_field_filter("user_id", {"$eq": None})
        with Session(store._bind) as session:
            rows = (
                session.query(store.EmbeddingStore.document)
                .filter(store.EmbeddingStore.collection_id == collection_id)
                .filter(store.EmbeddingStore.cmetadata["file_id"].astext == "f-null")
                .filter(clause)
                .all()
            )
        documentos = sorted(r[0] for r in rows)
        assert documentos == ["legacy", "null-key"], "'' NÃO é legado e fica de fora"
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'file_id' = 'f-null'"))
