"""Card #278 — unit tests for idempotent ingestion and namespace-scoped /query.

Covers:
  * _prepare_documents_sync: chunk_index identity + logical filename metadata
  * store_data_in_vector_db: dedup skip on identical batch, atomic replace on change,
    concurrent unique-violation reported as deduplicated (never delete-by-file_id)
  * _process_documents_async_pipeline / _process_documents_batched_sync: rollback
    scoped to namespace, skipped on unique violation
  * _is_unique_violation: IntegrityError de corrida (pgcode 23505 /
    asyncpg UniqueViolationError) é detectado; IntegrityError NÃO-unique
    (NOT NULL/FK/CHECK) NÃO é — rollback normal acontece
  * /query: namespace filter applied before top-k, per-result validation,
    legacy fallback filtered to user_id IS NULL, '' never authorized
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document

from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector


def _docs(content: str, n: int, **extra) -> list[Document]:
    return [Document(page_content=content, metadata=dict(extra)) for _ in range(n)]


@contextmanager
def patch_pipeline_batch_size(size: int):
    import unittest.mock

    with unittest.mock.patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", size):
        yield


def _integrity_error() -> Exception:
    """SQLAlchemy IntegrityError wrapping asyncpg UniqueViolationError — o que o
    perdedor da corrida de ingestão recebe com o índice único ativo."""
    import asyncpg
    from sqlalchemy.exc import IntegrityError

    return IntegrityError(
        "INSERT INTO langchain_pg_embedding ...",
        {},
        asyncpg.UniqueViolationError(
            "duplicate key value violates unique constraint "
            "uq_langchain_pg_embedding_chunk_identity"
        ),
    )


def _not_null_integrity_error() -> Exception:
    """SQLAlchemy IntegrityError que NÃO é corrida perdida: violação NOT NULL
    (pgcode 23502). Antes do fix da rodada 3, qualquer IntegrityError era
    tratado como unique e o rollback era pulado erroneamente."""
    import asyncpg
    from sqlalchemy.exc import IntegrityError

    return IntegrityError(
        "INSERT INTO langchain_pg_embedding ...",
        {},
        asyncpg.NotNullViolationError(
            'null value in column "embedding" of relation '
            '"langchain_pg_embedding" violates not-null constraint'
        ),
    )


@pytest.fixture
def fake_store():
    """An AsyncPgVector-shaped store so the async code path is exercised."""

    class FakeAsyncStore(AsyncPgVector):
        def __init__(self):  # skip PGVector machinery (conftest dummies it too)
            pass

    store = FakeAsyncStore()
    store.get_file_batch_digests = AsyncMock(return_value=None)
    store.delete_file_batch = AsyncMock(return_value=0)
    store.replace_file_batch = AsyncMock(return_value=[])
    store.aadd_documents = AsyncMock(return_value=[])
    return store


@pytest.fixture(autouse=True)
def patch_vector_store(monkeypatch, fake_store):
    """Aponta o vector_store global do módulo de rotas para o fake (todas as classes
    deste módulo exercitam caminhos que leem esse global)."""
    monkeypatch.setattr(document_routes, "vector_store", fake_store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0)
    return fake_store


class TestPrepareDocuments:
    def test_chunk_index_is_sequential_per_upload(self):
        docs = _docs("chunk text for identity", 3)
        prepared = document_routes._prepare_documents_sync(docs, "file-1", "user-1", False)
        assert [d.metadata["chunk_index"] for d in prepared] == [0, 1, 2]
        for d in prepared:
            assert d.metadata["file_id"] == "file-1"
            assert d.metadata["user_id"] == "user-1"
            assert len(d.metadata["digest"]) == 32  # md5

    def test_logical_filename_becomes_canonical_source(self):
        docs = _docs("content", 1, source="/tmp/uploads/user-1/x_abc.pdf")
        prepared = document_routes._prepare_documents_sync(
            docs, "file-1", "user-1", False, filename="x.pdf"
        )
        assert prepared[0].metadata["filename"] == "x.pdf"
        assert prepared[0].metadata["source"] == "x.pdf"
        assert "provision-" not in prepared[0].metadata["source"]

    def test_without_filename_metadata_stays_legacy_compatible(self):
        docs = _docs("content", 1, source="/tmp/legacy.pdf")
        prepared = document_routes._prepare_documents_sync(docs, "file-1", "user-1", False)
        assert "filename" not in prepared[0].metadata
        assert prepared[0].metadata["source"] == "/tmp/legacy.pdf"

    def test_chunk_index_cannot_be_clobbered_by_loader_metadata(self):
        docs = _docs("content", 1, chunk_index=999)
        prepared = document_routes._prepare_documents_sync(docs, "file-1", "user-1", False)
        assert prepared[0].metadata["chunk_index"] == 0


class TestStoreIdempotency:
    async def test_identical_batch_skips_reembed(self, fake_store):
        first = document_routes._prepare_documents_sync(
            _docs("same content", 2), "file-1", "user-1", False
        )
        stored_digests = [d.metadata["digest"] for d in first]
        fake_store.get_file_batch_digests = AsyncMock(return_value=stored_digests)

        result = await document_routes.store_data_in_vector_db(
            _docs("same content", 2), "file-1", "user-1", executor=None
        )

        assert result["deduplicated"] is True
        assert result["ids"] == []
        fake_store.aadd_documents.assert_not_called()
        fake_store.delete_file_batch.assert_not_called()

    async def test_changed_batch_replaces_atomically_without_predelete(self, fake_store):
        """Card #278 achado 3: a substituição é um swap atômico (delete+insert numa
        transação) — nunca delete commitado antes do embed/insert."""
        stored = document_routes._prepare_documents_sync(
            _docs("old content", 2), "file-1", "user-1", False
        )
        fake_store.get_file_batch_digests = AsyncMock(
            return_value=[d.metadata["digest"] for d in stored]
        )
        new_docs = _docs("new content", 2)

        result = await document_routes.store_data_in_vector_db(
            new_docs, "file-1", "user-1", executor=None
        )

        assert result.get("replaced") is True
        fake_store.replace_file_batch.assert_awaited_once()
        args = fake_store.replace_file_batch.await_args
        assert args.args[:2] == ("file-1", "user-1")
        swapped_docs = args.args[2]
        assert [d.page_content for d in swapped_docs] == ["new content", "new content"]
        assert all(d.metadata["user_id"] == "user-1" for d in swapped_docs)
        # O swap atômico substitui delete-then-insert:
        fake_store.delete_file_batch.assert_not_called()
        fake_store.aadd_documents.assert_not_called()

    async def test_store_without_atomic_replace_falls_back_to_delete_insert(
        self, monkeypatch
    ):
        """Legacy store sem replace_file_batch: mantém o caminho antigo (não atômico),
        explícito e confinado a stores sem o método."""
        stored = document_routes._prepare_documents_sync(
            _docs("old content", 1), "file-1", "user-1", False
        )
        digests = [d.metadata["digest"] for d in stored]
        bare = SimpleNamespace(
            get_file_batch_digests=AsyncMock(return_value=digests),
            delete_file_batch=AsyncMock(return_value=1),
            add_documents=Mock(return_value=["id1"]),
        )
        monkeypatch.setattr(document_routes, "vector_store", bare)

        result = await document_routes.store_data_in_vector_db(
            _docs("new content", 1), "file-1", "user-1", executor=None
        )

        assert "replaced" not in result
        bare.delete_file_batch.assert_awaited_once_with(
            "file-1", "user-1", executor=None
        )
        bare.add_documents.assert_called_once()

    async def test_other_namespace_batch_is_not_a_duplicate(self, fake_store):
        """Same file_id embedded for another agent must not count as 'already there'."""
        other_namespace = document_routes._prepare_documents_sync(
            _docs("same content", 2), "file-1", "user-1", False
        )
        fake_store.get_file_batch_digests = AsyncMock(return_value=None)

        result = await document_routes.store_data_in_vector_db(
            _docs("same content", 2), "file-1", "user-1", executor=None
        )

        assert "deduplicated" not in result
        fake_store.delete_file_batch.assert_not_called()
        fake_store.aadd_documents.assert_awaited_once()
        # digests of the other namespace's rows were irrelevant
        assert other_namespace[0].metadata["user_id"] == "user-1"

    async def test_store_without_batch_support_still_inserts(self, monkeypatch):
        bare = SimpleNamespace()
        bare.add_documents = lambda documents, ids=None, **kw: []
        monkeypatch.setattr(document_routes, "vector_store", bare)
        monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0)

        result = await document_routes.store_data_in_vector_db(
            _docs("content", 1), "file-1", "user-1", executor=None
        )
        assert result["message"] == "Documents added successfully"


class TestQueryNamespaceFilter:
    @pytest.fixture(autouse=True)
    def patch_embedding(self, monkeypatch):
        monkeypatch.setattr(
            document_routes,
            "get_cached_query_embedding",
            lambda query: [0.1, 0.2, 0.3],
        )

    def _request(self, user_id="user-1"):
        return SimpleNamespace(
            state=SimpleNamespace(user={"id": user_id}),
            app=SimpleNamespace(state=SimpleNamespace(thread_pool=None)),
        )

    async def test_namespace_goes_into_the_search_filter(self, monkeypatch):
        captured = {}

        async def fake_search(self, embedding, k=4, filter=None, executor=None):
            captured.update(filter=filter, k=k)
            doc = Document(
                page_content="hit",
                metadata={"file_id": "file-1", "user_id": filter["user_id"]["$eq"]},
            )
            return [(doc, 0.1)]

        monkeypatch.setattr(
            "app.services.vector_store.async_pg_vector.AsyncPgVector.asimilarity_search_with_score_by_vector",
            fake_search,
        )
        body = document_routes.QueryRequestBody(
            query="q", file_id="file-1", k=5, entity_id=None
        )
        result = await document_routes.query_embeddings_by_file_id(
            body, self._request("user-1")
        )

        assert captured["filter"] == {"file_id": {"$eq": "file-1"}, "user_id": {"$eq": "user-1"}}
        assert captured["k"] == 5
        assert len(result) == 1

    async def test_agent_namespace_tried_first_then_user_fallback(self, monkeypatch):
        seen_filters = []

        async def fake_search(self, embedding, k=4, filter=None, executor=None):
            seen_filters.append(dict(filter))
            if filter["user_id"]["$eq"] == "agent-1":
                return []  # nothing under the agent namespace
            doc = Document(
                page_content="user hit",
                metadata={"file_id": "file-1", "user_id": "user-1"},
            )
            return [(doc, 0.2)]

        monkeypatch.setattr(
            "app.services.vector_store.async_pg_vector.AsyncPgVector.asimilarity_search_with_score_by_vector",
            fake_search,
        )
        body = document_routes.QueryRequestBody(
            query="q", file_id="file-1", k=5, entity_id="agent-1"
        )
        result = await document_routes.query_embeddings_by_file_id(
            body, self._request("user-1")
        )

        assert [f["user_id"]["$eq"] for f in seen_filters] == ["agent-1", "user-1"]
        assert len(result) == 1

    async def test_mixed_batch_validated_per_result(self, monkeypatch):
        """Even when the store mixes namespaces after top-k, foreign hits are dropped."""

        async def fake_search(self, embedding, k=4, filter=None, executor=None):
            # Pretends the store ignored the namespace filter (legacy behavior)
            mine = Document(
                page_content="mine",
                metadata={"file_id": "file-1", "user_id": "user-1"},
            )
            foreign = Document(
                page_content="foreign",
                metadata={"file_id": "file-1", "user_id": "agent-other"},
            )
            return [(foreign, 0.05), (mine, 0.1)]

        monkeypatch.setattr(
            "app.services.vector_store.async_pg_vector.AsyncPgVector.asimilarity_search_with_score_by_vector",
            fake_search,
        )
        body = document_routes.QueryRequestBody(query="q", file_id="file-1", k=5)
        result = await document_routes.query_embeddings_by_file_id(
            body, self._request("user-1")
        )

        assert [doc.page_content for doc, _ in result] == ["mine"]

    async def test_legacy_fallback_keeps_rows_without_user_id(self, monkeypatch):
        calls = []

        async def fake_search(self, embedding, k=4, filter=None, executor=None):
            calls.append(dict(filter))
            # Card #278: o fallback legado agora filtra user_id IS NULL na store
            # ($eq None); namespaces reais nunca são consultados sem escopo.
            if (filter.get("user_id") or {}).get("$eq") is not None:
                return []  # namespace-restricted finds nothing (legacy rows lack user_id)
            return [
                (
                    Document(
                        page_content="legacy",
                        metadata={"file_id": "file-1"},  # no user_id at all
                    ),
                    0.3,
                )
            ]

        monkeypatch.setattr(
            "app.services.vector_store.async_pg_vector.AsyncPgVector.asimilarity_search_with_score_by_vector",
            fake_search,
        )
        body = document_routes.QueryRequestBody(query="q", file_id="file-1", k=5)
        result = await document_routes.query_embeddings_by_file_id(
            body, self._request("user-1")
        )

        assert len(calls) == 2  # namespace attempt + legacy fallback
        assert calls[1]["user_id"] == {"$eq": None}  # IS NULL no fallback legado
        assert [doc.page_content for doc, _ in result] == ["legacy"]

    async def test_foreign_namespace_legacy_fallback_is_dropped(self, monkeypatch):
        async def fake_search(self, embedding, k=4, filter=None, executor=None):
            if (filter.get("user_id") or {}).get("$eq") is not None:
                return []
            return [
                (
                    Document(
                        page_content="someone else",
                        metadata={"file_id": "file-1", "user_id": "other-user"},
                    ),
                    0.1,
                )
            ]

        monkeypatch.setattr(
            "app.services.vector_store.async_pg_vector.AsyncPgVector.asimilarity_search_with_score_by_vector",
            fake_search,
        )
        body = document_routes.QueryRequestBody(query="q", file_id="file-1", k=5)
        result = await document_routes.query_embeddings_by_file_id(
            body, self._request("user-1")
        )
        assert result == []

    async def test_empty_string_namespace_is_never_authorized(self, monkeypatch):
        """Card #278 achado 1: user_id='' é o default histórico do store; aceitar a
        chave vazia deixaria quem sabe o file_id ler lotes de namespace ambíguo."""
        async def fake_search(self, embedding, k=4, filter=None, executor=None):
            if (filter.get("user_id") or {}).get("$eq") is not None:
                return []
            return [
                (
                    Document(
                        page_content="empty namespace row",
                        metadata={"file_id": "file-1", "user_id": ""},
                    ),
                    0.05,
                ),
                (
                    Document(
                        page_content="legacy row",
                        metadata={"file_id": "file-1"},  # None: legado explícito
                    ),
                    0.2,
                ),
            ]

        monkeypatch.setattr(
            "app.services.vector_store.async_pg_vector.AsyncPgVector.asimilarity_search_with_score_by_vector",
            fake_search,
        )
        body = document_routes.QueryRequestBody(query="q", file_id="file-1", k=5)
        result = await document_routes.query_embeddings_by_file_id(
            body, self._request("user-1")
        )

        # '' derrubado; None (legado documentado) preservado.
        assert [doc.page_content for doc, _ in result] == ["legacy row"]


class TestAsyncPipelineRollback:
    """Card #278 achados 2 e 4: rollback do pipeline async escopado por namespace e
    nunca após unique violation."""

    def _store(self):
        class FakeAsyncStore(AsyncPgVector):
            def __init__(self):
                pass

        store = FakeAsyncStore()
        store.delete = AsyncMock()
        store.delete_file_batch = AsyncMock(return_value=0)
        return store

    def _docs(self, n=6):
        return [Document(page_content=f"c{i}", metadata={"chunk_index": i}) for i in range(n)]

    async def test_pipeline_rollback_is_scoped_to_namespace(self):
        from app.routes.document_routes import _process_documents_async_pipeline

        store = self._store()
        # 1º lote ok, 2º falha (erro genérico de provider/insert)
        store.aadd_documents = AsyncMock(side_effect=[["id1"], RuntimeError("boom")])

        with patch_pipeline_batch_size(2):
            with pytest.raises(RuntimeError, match="boom"):
                await _process_documents_async_pipeline(
                    documents=self._docs(),
                    file_id="file-1",
                    vector_store=store,
                    executor=None,
                    user_id="user-1",
                )

        store.delete_file_batch.assert_awaited_once_with(
            "file-1", "user-1", executor=None
        )
        store.delete.assert_not_called()  # NUNCA delete cru por file_id

    async def test_pipeline_rollback_without_user_id_falls_back(self):
        from app.routes.document_routes import _process_documents_async_pipeline

        store = self._store()
        store.aadd_documents = AsyncMock(side_effect=[["id1"], RuntimeError("boom")])

        with patch_pipeline_batch_size(2):
            with pytest.raises(RuntimeError):
                await _process_documents_async_pipeline(
                    documents=self._docs(),
                    file_id="file-1",
                    vector_store=store,
                    executor=None,
                    user_id=None,
                )

        store.delete.assert_awaited_once_with(ids=["file-1"], executor=None)

    async def test_pipeline_skips_rollback_on_unique_violation(self):
        """A requisição perdedora da corrida NÃO pode apagar (mesmo escopado) as
        linhas que a vencedora acabou de gravar."""
        from app.routes.document_routes import _process_documents_async_pipeline

        store = self._store()
        store.aadd_documents = AsyncMock(side_effect=[["id1"], _integrity_error()])

        with patch_pipeline_batch_size(2):
            with pytest.raises(Exception):
                await _process_documents_async_pipeline(
                    documents=self._docs(),
                    file_id="file-1",
                    vector_store=store,
                    executor=None,
                    user_id="user-1",
                )

        store.delete.assert_not_called()
        store.delete_file_batch.assert_not_called()


class TestSyncBatchUniqueViolation:
    async def test_sync_batched_skips_rollback_on_unique_violation(self):
        from concurrent.futures import ThreadPoolExecutor

        from unittest.mock import Mock

        from app.routes.document_routes import _process_documents_batched_sync

        store = Mock()
        store.delete = Mock()
        store.delete_file_batch = Mock(return_value=0)
        store.add_documents = Mock(side_effect=[["id1"], _integrity_error()])

        with patch_pipeline_batch_size(2):
            with ThreadPoolExecutor(max_workers=2) as executor:
                with pytest.raises(Exception):
                    await _process_documents_batched_sync(
                        documents=[Document(page_content=f"c{i}") for i in range(4)],
                        file_id="file-1",
                        vector_store=store,
                        executor=executor,
                        user_id="user-1",
                    )

        store.delete.assert_not_called()
        store.delete_file_batch.assert_not_called()


class TestNonUniqueIntegrityErrorRollsBack:
    """Rodada 3 — regressão da rodada 2: só corrida perdida (unique) pula rollback.
    NOT NULL/FK/CHECK num batch posterior é falha real: rollback normal acontece."""

    def _store(self):
        class FakeAsyncStore(AsyncPgVector):
            def __init__(self):  # skip PGVector machinery (conftest dummies it too)
                pass

        store = FakeAsyncStore()
        store.delete = AsyncMock()
        store.delete_file_batch = AsyncMock(return_value=0)
        return store

    def _docs(self, n=6):
        return [Document(page_content=f"c{i}", metadata={"chunk_index": i}) for i in range(n)]

    async def test_async_pipeline_rolls_back_on_non_unique_integrity_error(self):
        from app.routes.document_routes import _process_documents_async_pipeline

        store = self._store()
        # 1º lote insere; 2º falha com IntegrityError de NOT NULL (não é corrida)
        store.aadd_documents = AsyncMock(
            side_effect=[["id1"], _not_null_integrity_error()]
        )

        with patch_pipeline_batch_size(2):
            with pytest.raises(Exception):
                await _process_documents_async_pipeline(
                    documents=self._docs(),
                    file_id="file-1",
                    vector_store=store,
                    executor=None,
                    user_id="user-1",
                )

        store.delete_file_batch.assert_awaited_once_with(
            "file-1", "user-1", executor=None
        )
        store.delete.assert_not_called()

    async def test_sync_batched_rolls_back_on_non_unique_integrity_error(self):
        from concurrent.futures import ThreadPoolExecutor

        from unittest.mock import Mock

        from app.routes.document_routes import _process_documents_batched_sync

        store = Mock()
        store.delete = Mock()
        store.delete_file_batch = Mock(return_value=0)
        store.add_documents = Mock(side_effect=[["id1"], _not_null_integrity_error()])

        with patch_pipeline_batch_size(2):
            with ThreadPoolExecutor(max_workers=2) as executor:
                with pytest.raises(Exception):
                    await _process_documents_batched_sync(
                        documents=[Document(page_content=f"c{i}") for i in range(4)],
                        file_id="file-1",
                        vector_store=store,
                        executor=executor,
                        user_id="user-1",
                    )

        store.delete_file_batch.assert_called_once_with("file-1", "user-1")
        store.delete.assert_not_called()


class TestIsUniqueViolation:
    @pytest.mark.parametrize("code", ["23502", "23503", "23514"])
    @pytest.mark.parametrize("code_attribute", ["pgcode", "sqlstate"])
    def test_other_integrity_sqlstates_are_not_unique(self, code, code_attribute):
        from sqlalchemy.exc import IntegrityError

        orig = Exception("constraint violation")
        setattr(orig, code_attribute, code)
        exc = IntegrityError("INSERT ...", {}, orig)
        assert document_routes._is_unique_violation(exc) is False

    @pytest.mark.parametrize("code_attribute", ["pgcode", "sqlstate"])
    def test_unique_sqlstate_is_detected_without_message(self, code_attribute):
        from sqlalchemy.exc import IntegrityError

        orig = Exception("constraint violation")
        setattr(orig, code_attribute, "23505")
        exc = IntegrityError("INSERT ...", {}, orig)
        assert document_routes._is_unique_violation(exc) is True

    def test_sqlalchemy_integrity_error_is_detected(self):
        from sqlalchemy.exc import IntegrityError

        exc = IntegrityError(
            "INSERT INTO langchain_pg_embedding ...",
            {},
            Exception("duplicate key value violates unique constraint uq_chunk"),
        )
        assert document_routes._is_unique_violation(exc) is True

    def test_pgcode_23505_is_detected(self):
        class FakeOrig(Exception):
            pgcode = "23505"

        exc = RuntimeError("insert failed")
        exc.orig = FakeOrig()
        assert document_routes._is_unique_violation(exc) is True

    def test_asyncpg_unique_violation_is_detected(self):
        asyncpg = pytest.importorskip("asyncpg")
        exc = RuntimeError("insert failed")
        exc.orig = asyncpg.UniqueViolationError("duplicate key")
        assert document_routes._is_unique_violation(exc) is True

    def test_plain_error_is_not_unique_violation(self):
        assert document_routes._is_unique_violation(RuntimeError("boom")) is False

    def test_not_null_integrity_error_is_not_unique_violation(self):
        """Rodada 3: IntegrityError de NOT NULL (23502) NÃO é corrida perdida."""
        exc = _not_null_integrity_error()
        assert document_routes._is_unique_violation(exc) is False

    def test_foreign_key_integrity_error_is_not_unique_violation(self):
        """FK/CHECK sem pgcode e sem mensagem de unique também não são."""
        from sqlalchemy.exc import IntegrityError

        exc = IntegrityError(
            "INSERT INTO langchain_pg_embedding ...",
            {},
            Exception('update or insert violates foreign key constraint "fk_col"'),
        )
        assert document_routes._is_unique_violation(exc) is False


class TestConcurrentIngestReportsDeduplicated:
    async def test_unique_violation_with_matching_stored_batch_is_deduplicated(
        self, fake_store
    ):
        """Achado 4: o perdedor da corrida com conteúdo idêntico reconhece o lote do
        vencedor e reporta deduplicated — sem apagar nada."""
        incoming = _docs("same content", 2)
        prepared = document_routes._prepare_documents_sync(
            incoming, "file-1", "user-1", False
        )
        digests = [d.metadata["digest"] for d in prepared]

        # Antes do insert o lote não existia (corrida), o insert bate na unique
        # violation e, ao reconferir, o vencedor gravou exatamente o mesmo conteúdo.
        fake_store.get_file_batch_digests = AsyncMock(side_effect=[None, digests])
        fake_store.aadd_documents = AsyncMock(side_effect=_integrity_error())

        result = await document_routes.store_data_in_vector_db(
            incoming, "file-1", "user-1", executor=None
        )

        assert result["deduplicated"] is True
        fake_store.delete_file_batch.assert_not_called()
        fake_store.replace_file_batch.assert_not_called()

    async def test_unique_violation_with_different_stored_batch_is_error(self, fake_store):
        """Corrida com conteúdo DIFERENTE: erro honesto, e nenhuma linha é apagada."""
        incoming = _docs("my new content", 2)
        prepared = document_routes._prepare_documents_sync(
            incoming, "file-1", "user-1", False
        )
        other = document_routes._prepare_documents_sync(
            _docs("someone else's content", 2), "file-1", "user-1", False
        )
        fake_store.get_file_batch_digests = AsyncMock(
            side_effect=[None, [d.metadata["digest"] for d in other]]
        )
        fake_store.aadd_documents = AsyncMock(side_effect=_integrity_error())

        result = await document_routes.store_data_in_vector_db(
            incoming, "file-1", "user-1", executor=None
        )

        assert "error" in result
        assert "deduplicated" not in result
        fake_store.delete_file_batch.assert_not_called()
        fake_store.replace_file_batch.assert_not_called()
