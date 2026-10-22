"""Card #278 — unit tests for idempotent ingestion and namespace-scoped /query.

Covers:
  * _prepare_documents_sync: chunk_index identity + logical filename metadata
  * store_data_in_vector_db: dedup skip on identical batch, scoped replace on change
  * /query: namespace filter applied before top-k, per-result validation,
    legacy fallback for rows without user_id
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector


def _docs(content: str, n: int, **extra) -> list[Document]:
    return [Document(page_content=content, metadata=dict(extra)) for _ in range(n)]


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
    @pytest.fixture
    def fake_store(self):
        """An AsyncPgVector-shaped store so the async code path is exercised."""

        class FakeAsyncStore(AsyncPgVector):
            def __init__(self):  # skip PGVector machinery (conftest dummies it too)
                pass

        store = FakeAsyncStore()
        store.get_file_batch_digests = AsyncMock(return_value=None)
        store.delete_file_batch = AsyncMock(return_value=0)
        store.aadd_documents = AsyncMock(return_value=[])
        return store

    @pytest.fixture(autouse=True)
    def patch_vector_store(self, monkeypatch, fake_store):
        monkeypatch.setattr(document_routes, "vector_store", fake_store)
        monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0)
        return fake_store

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

    async def test_changed_batch_replaces_only_the_namespace(self, fake_store):
        stored = document_routes._prepare_documents_sync(
            _docs("old content", 2), "file-1", "user-1", False
        )
        fake_store.get_file_batch_digests = AsyncMock(
            return_value=[d.metadata["digest"] for d in stored]
        )
        fake_store.delete_file_batch = AsyncMock(return_value=2)

        result = await document_routes.store_data_in_vector_db(
            _docs("new content", 2), "file-1", "user-1", executor=None
        )

        assert "deduplicated" not in result
        fake_store.delete_file_batch.assert_awaited_once_with(
            "file-1", "user-1", executor=None
        )
        fake_store.aadd_documents.assert_awaited_once()

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
            if "user_id" in filter:
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
        assert [doc.page_content for doc, _ in result] == ["legacy"]

    async def test_foreign_namespace_legacy_fallback_is_dropped(self, monkeypatch):
        async def fake_search(self, embedding, k=4, filter=None, executor=None):
            if "user_id" in filter:
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
