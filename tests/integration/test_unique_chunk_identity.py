"""Card #278 — integration: unique chunk identity + namespace-scoped batch ops.

Requires Docker (testcontainers). Run explicitly:
    pytest tests/integration/test_unique_chunk_identity.py -m integration -v

The migration under migrations/ must apply cleanly on the schema mirrored by
conftest (langchain_pg_embedding), enforce (collection_id, custom_id, user_id,
chunk_index) uniqueness for NEW-format rows only, and leave legacy rows alone.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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
):
    meta = {"file_id": file_id, "user_id": user_id, "digest": f"d{chunk_index}"}
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
