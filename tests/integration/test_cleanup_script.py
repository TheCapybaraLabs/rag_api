"""Card #278 — testes de integração do script de limpeza contra pgvector real.

Exercita o CLI de ponta a ponta (subprocess) sobre um container efêmero:
  * dry-run: agrupamento SEMPRE por namespace+chunk_index (achados 5 e 6),
    nada é alterado, manifesto com manter/remover;
  * --execute aplica EXATAMENTE o manifesto do dry-run (achado 7): uuids
    verificados, snapshot+delete numa transação, manifesto intocado, relatório
    de execução com rollback de lista concreta;
  * sobrevivente é a linha mais antiga por ctid (achado 8);
  * recusas do CLI (--execute sem manifesto, sem namespace, com escopo).
"""

import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path

import asyncpg
import pytest

from scripts.cleanup_duplicate_vectors import GROUPING_DESCRIPTION

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "cleanup_duplicate_vectors.py"
COLLECTION = "card278_cleanup_test"
PG_DSN: dict = {}


@pytest.fixture(scope="module")
def dsn(pg_container):
    """DSN cru (asyncpg) a partir do container da sessão."""
    url = pg_container.get_connection_url()  # postgresql+psycopg2://...
    return url.replace("postgresql+psycopg2://", "postgresql://")


async def _connect():
    return await asyncpg.connect(dsn=PG_DSN["dsn"])


@pytest.fixture(scope="module")
def _schema(dsn):
    """Schema espelhado + coleção do teste (uma vez por módulo)."""

    async def _setup():
        conn = await asyncpg.connect(dsn=dsn)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR NOT NULL UNIQUE,
                    cmetadata JSONB
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    collection_id UUID REFERENCES langchain_pg_collection(uuid)
                        ON DELETE CASCADE,
                    embedding vector(3),
                    document VARCHAR,
                    cmetadata JSONB,
                    custom_id VARCHAR
                )
                """
            )
            await conn.execute(
                "INSERT INTO langchain_pg_collection (name, cmetadata) "
                "VALUES ($1, '{}') ON CONFLICT (name) DO NOTHING",
                COLLECTION,
            )
        finally:
            await conn.close()

    asyncio.run(_setup())
    yield


async def _insert_rows(rows):
    """rows: (file_id, user_id, page, chunk_index, texto, source)."""
    conn = await _connect()
    try:
        for file_id, user_id, page, chunk_index, body, source in rows:
            meta = {
                "file_id": file_id,
                "user_id": user_id,
                "source": source,
                "digest": uuid.uuid4().hex[:8],
            }
            if page is not None:
                meta["page"] = str(page)
            if chunk_index != "":
                meta["chunk_index"] = str(chunk_index)
            await conn.execute(
                """
                INSERT INTO langchain_pg_embedding
                    (collection_id, embedding, document, cmetadata, custom_id)
                SELECT uuid, '[0.1,0.2,0.3]', $2, CAST($3 AS jsonb), $4
                FROM langchain_pg_collection WHERE name = $1
                """,
                COLLECTION,
                body,
                json.dumps(meta),
                file_id,
            )
    finally:
        await conn.close()


async def _count(where: str = "TRUE") -> int:
    conn = await _connect()
    try:
        return await conn.fetchval(
            f"SELECT count(*) FROM langchain_pg_embedding WHERE {where}"
        )
    finally:
        await conn.close()


async def _drop_backups():
    conn = await _connect()
    try:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE tablename LIKE 'card278_backup_%'"
        )
        for t in tables:
            await conn.execute(f'DROP TABLE IF EXISTS "{t["tablename"]}"')
    finally:
        await conn.close()


def _run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--dsn", PG_DSN["dsn"], *args],
        capture_output=True,
        text=True,
        env={"OPENAI_API_KEY": "test_key", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        cwd=str(REPO),
        timeout=120,
    )


@pytest.fixture(autouse=True)
def _bind_dsn_and_clean(dsn, _schema):
    """Liga o DSN aos helpers e isola cada teste (tabela zerada + backups fora)."""
    PG_DSN["dsn"] = dsn

    async def _clean():
        conn = await _connect()
        try:
            await conn.execute("DELETE FROM langchain_pg_embedding")
        finally:
            await conn.close()

    asyncio.run(_clean())
    asyncio.run(_drop_backups())
    yield dsn
    asyncio.run(_drop_backups())


class TestDryRun:
    def test_by_namespace_is_default_and_preserves_agent_copies(self, tmp_path):
        """Achado 5: agrupamento por namespace é o default; cópia legítima de outro
        agente NÃO é candidata, repetição dentro do namespace é."""
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("dup-doc", "user-1", 0, 0, "TEXTO IGUAL", f"/tmp/provision-x-{i}.pdf")
                    for i in range(3)
                ]
                + [("dup-doc", "agent-2", 0, 0, "TEXTO IGUAL", "/tmp/original.pdf")]
                + [("clean-doc", "user-1", 0, 0, "UNICO", "/tmp/ok.pdf")]
            )
        )
        assert asyncio.run(_count()) == 5

        run = _run_script(
            "--collection", COLLECTION,
            "--file-id", "dup-doc", "--file-id", "clean-doc",
            "--manifesto", str(manifesto),
        )
        assert run.returncode == 0, run.stderr
        assert "DRY-RUN" in run.stdout
        assert asyncio.run(_count()) == 5, "dry-run não pode alterar nada"

        m = json.loads(manifesto.read_text())
        assert m["candidatos_total"] == 2
        assert "user_id" in m["agrupamento"] and "chunk_index" in m["agrupamento"]
        grupos_user1 = [g for g in m["grupos"] if g.get("namespace") == "user-1"]
        assert len(grupos_user1) == 1
        assert grupos_user1[0]["quantidade"] == 3
        assert len(grupos_user1[0]["remover_uuids"]) == 2
        # namespaces distintos → grupos separados; agent-2 não tem duplicata
        assert not any(g.get("namespace") == "agent-2" for g in m["grupos"])

    def test_chunk_index_prevents_merging_repeated_text(self, tmp_path):
        """Achado 6: texto repetido na MESMA página em posições distintas não é
        duplicata; linhas legadas idênticas (sem posição) continuam duplicatas."""
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("pos-doc", "user-1", 0, 0, "REPETIDO", "/tmp/a.pdf"),
                    ("pos-doc", "user-1", 0, 1, "REPETIDO", "/tmp/a.pdf"),
                    ("legacy-doc", "user-1", 0, "", "IGUAL", "/tmp/b.pdf"),
                    ("legacy-doc", "user-1", 0, "", "IGUAL", "/tmp/b.pdf"),
                ]
            )
        )

        run = _run_script(
            "--collection", COLLECTION,
            "--file-id", "pos-doc", "--file-id", "legacy-doc",
            "--manifesto", str(manifesto),
        )
        assert run.returncode == 0, run.stderr
        assert asyncio.run(_count()) == 4

        m = json.loads(manifesto.read_text())
        arquivos = {g["file_id"] for g in m["grupos"]}
        assert "pos-doc" not in arquivos, "chunks de posições distintas não se fundem"
        assert "legacy-doc" in arquivos, "legadas idênticas (sem posição) seguem duplicatas"

    def test_keep_rule_oldest_by_ctid_not_by_uuid(self, tmp_path):
        """Achado 8: sobrevive a linha mais antiga (menor ctid), não a de menor uuid."""
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("keep-doc", "user-1", 0, 0, "MESMO TEXTO", "/tmp/legit.pdf"),
                    ("keep-doc", "user-1", 0, 0, "MESMO TEXTO", "/tmp/provision-lixo.pdf"),
                    ("keep-doc", "user-1", 0, 0, "MESMO TEXTO", "/tmp/provision-lixo2.pdf"),
                ]
            )
        )

        async def _oldest_uuid():
            conn = await _connect()
            try:
                row = await conn.fetchrow(
                    """
                    SELECT uuid FROM langchain_pg_embedding
                    WHERE cmetadata->>'file_id' = 'keep-doc'
                    ORDER BY ctid LIMIT 1
                    """
                )
                return str(row["uuid"])
            finally:
                await conn.close()

        esperado_manter = asyncio.run(_oldest_uuid())

        run = _run_script(
            "--collection", COLLECTION,
            "--file-id", "keep-doc",
            "--manifesto", str(manifesto),
        )
        assert run.returncode == 0, run.stderr
        m = json.loads(manifesto.read_text())
        assert m["criterio_manter"], "manifesto documenta o critério de manutenção"
        assert m["grupos"][0]["manter_uuid"] == esperado_manter


class TestExecuteAppliesManifest:
    def test_execute_deletes_exactly_the_manifest_uuids(self, tmp_path):
        """Achado 7: --execute aplica o manifesto do dry-run; duplicata surgida
        DEPOIS do dry-run não é tocada; o manifesto aprovado não é sobrescrito."""
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("ex-doc", "user-1", 0, 0, "MESMO", "/tmp/1.pdf"),
                    ("ex-doc", "user-1", 0, 0, "MESMO", "/tmp/2.pdf"),
                    ("ex-doc", "user-1", 0, 0, "MESMO", "/tmp/3.pdf"),
                ]
            )
        )

        run = _run_script(
            "--collection", COLLECTION,
            "--file-id", "ex-doc",
            "--manifesto", str(manifesto),
        )
        assert run.returncode == 0, run.stderr
        aprovado_bytes = manifesto.read_bytes()
        alvo_uuids = json.loads(manifesto.read_text())["grupos"][0]["remover_uuids"]
        assert len(alvo_uuids) == 2

        # duplicata NOVA depois do dry-run (não está no manifesto)
        asyncio.run(_insert_rows([("ex-doc", "user-1", 0, 0, "MESMO", "/tmp/4.pdf")]))

        run = _run_script("--manifesto", str(manifesto), "--execute", "--yes")
        assert run.returncode == 0, run.stderr

        assert asyncio.run(_count("cmetadata->>'file_id' = 'ex-doc'")) == 2, (
            "remove só as 2 aprovadas; a duplicata pós-dry-run sobrevive"
        )
        assert manifesto.read_bytes() == aprovado_bytes, "manifesto aprovado intocado"

        execucao_path = manifesto.with_suffix(manifesto.suffix + ".execucao.json")
        execucao = json.loads(execucao_path.read_text())
        assert execucao["removidos"] == 2
        assert execucao["uuids_ausentes"] == []
        assert "ANY('" in execucao["rollback"], "rollback com lista concreta"
        for u in alvo_uuids:
            assert u in execucao["rollback"]

    def test_rollback_restores_snapshot(self, tmp_path):
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("rb-doc", "user-1", 0, 0, "MESMO", "/tmp/1.pdf"),
                    ("rb-doc", "user-1", 0, 0, "MESMO", "/tmp/2.pdf"),
                ]
            )
        )
        assert _run_script(
            "--collection", COLLECTION, "--file-id", "rb-doc",
            "--manifesto", str(manifesto),
        ).returncode == 0
        assert _run_script(
            "--manifesto", str(manifesto), "--execute", "--yes"
        ).returncode == 0
        assert asyncio.run(_count("cmetadata->>'file_id' = 'rb-doc'")) == 1

        execucao = json.loads(
            manifesto.with_suffix(manifesto.suffix + ".execucao.json").read_text()
        )

        async def _rollback():
            conn = await _connect()
            try:
                await conn.execute(execucao["rollback"])
            finally:
                await conn.close()

        asyncio.run(_rollback())
        assert asyncio.run(_count("cmetadata->>'file_id' = 'rb-doc'")) == 2, (
            "rollback pela âncora restaura as linhas removidas"
        )

    def test_missing_uuids_are_reported_not_refetched(self, tmp_path):
        """Uuid removido entre o dry-run e o execute: vira 'ausente'; o restante é
        apagado exatamente como aprovado (nunca re-selecionado do banco vivo)."""
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("miss-doc", "user-1", 0, 0, "MESMO", "/tmp/1.pdf"),
                    ("miss-doc", "user-1", 0, 0, "MESMO", "/tmp/2.pdf"),
                    ("miss-doc", "user-1", 0, 0, "MESMO", "/tmp/3.pdf"),
                ]
            )
        )
        assert _run_script(
            "--collection", COLLECTION, "--file-id", "miss-doc",
            "--manifesto", str(manifesto),
        ).returncode == 0
        alvo = json.loads(manifesto.read_text())["grupos"][0]["remover_uuids"]

        async def _delete_one():
            conn = await _connect()
            try:
                await conn.execute(
                    "DELETE FROM langchain_pg_embedding WHERE uuid = $1::uuid", alvo[0]
                )
            finally:
                await conn.close()

        asyncio.run(_delete_one())

        run = _run_script("--manifesto", str(manifesto), "--execute", "--yes")
        assert run.returncode == 0, run.stderr
        execucao = json.loads(
            manifesto.with_suffix(manifesto.suffix + ".execucao.json").read_text()
        )
        assert execucao["uuids_ausentes"] == [alvo[0]]
        assert execucao["removidos"] == 1
        assert asyncio.run(_count("cmetadata->>'file_id' = 'miss-doc'")) == 1, (
            "3 linhas - 1 removida fora - 1 executada = só o 'manter' sobrevive"
        )


class TestExecuteRefusals:
    @pytest.mark.parametrize(
        "invalid_part",
        ["reviewer_manifest", "agrupamento", "namespace", "namespaces_presentes"],
    )
    def test_execute_refuses_cross_namespace_manifest_without_cli_flag(
        self, tmp_path, invalid_part
    ):
        """Reprodução do revisor: uuids reais de agentA/agentB no mesmo grupo.

        by_namespace=True não basta: cada outra evidência insegura deve recusar
        o arquivo por si só, mesmo sem --no-by-namespace na CLI.
        """
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("tenant-doc", "agentA", 0, 0, "MESMO", "/tmp/a.pdf"),
                    ("tenant-doc", "agentB", 0, 0, "MESMO", "/tmp/b.pdf"),
                ]
            )
        )

        async def _ids_by_namespace():
            conn = await _connect()
            try:
                rows = await conn.fetch(
                    "SELECT uuid, cmetadata->>'user_id' AS namespace "
                    "FROM langchain_pg_embedding"
                )
                return {row["namespace"]: str(row["uuid"]) for row in rows}
            finally:
                await conn.close()

        ids = asyncio.run(_ids_by_namespace())
        group = {
            "file_id": "tenant-doc",
            "namespace": "agentA",
            "namespaces_presentes": ["agentA"],
            "manter_uuid": ids["agentA"],
            "remover_uuids": [ids["agentB"]],
        }
        grouping = GROUPING_DESCRIPTION
        if invalid_part in ("reviewer_manifest", "agrupamento"):
            grouping = "file_id+page+chunk_index+md5(document) (SEM namespace — só inventário!)"
        if invalid_part in ("reviewer_manifest", "namespace"):
            del group["namespace"]
        if invalid_part in ("reviewer_manifest", "namespaces_presentes"):
            group["namespaces_presentes"] = ["agentA", "agentB"]
        manifesto.write_text(
            json.dumps(
                {
                    "card": "278-citacoes-duplicadas",
                    "by_namespace": True,
                    "agrupamento": grouping,
                    "grupos": [group],
                }
            )
        )
        approved = manifesto.read_bytes()

        run = _run_script("--manifesto", str(manifesto), "--execute", "--yes")

        assert run.returncode != 0
        assert "Recusado" in run.stderr and "namespace" in run.stderr
        assert asyncio.run(_ids_by_namespace()) == ids, "ambas as cópias devem sobreviver"
        assert manifesto.read_bytes() == approved
        assert not manifesto.with_suffix(".json.execucao.json").exists()

    def test_execute_refuses_missing_keeper_without_deleting_any_group(self, tmp_path):
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("safe-doc", "agentA", 0, 0, "MESMO", f"/tmp/a{i}.pdf")
                    for i in range(3)
                ]
                + [
                    ("stale-doc", "agentB", 0, 0, "OUTRO", f"/tmp/b{i}.pdf")
                    for i in range(2)
                ]
            )
        )
        dry = _run_script(
            "--collection", COLLECTION,
            "--file-id", "safe-doc", "--file-id", "stale-doc",
            "--manifesto", str(manifesto),
        )
        assert dry.returncode == 0, dry.stderr
        approved = manifesto.read_bytes()
        groups = json.loads(approved)["grupos"]
        keeper = next(g["manter_uuid"] for g in groups if g["file_id"] == "stale-doc")

        async def _delete_keeper():
            conn = await _connect()
            try:
                await conn.execute(
                    "DELETE FROM langchain_pg_embedding WHERE uuid = $1::uuid", keeper
                )
            finally:
                await conn.close()

        asyncio.run(_delete_keeper())
        assert asyncio.run(_count()) == 4

        run = _run_script("--manifesto", str(manifesto), "--execute", "--yes")

        assert run.returncode != 0
        assert "manter_uuid" in run.stderr and "ausente" in run.stderr
        assert keeper in run.stderr
        assert asyncio.run(_count("cmetadata->>'file_id' = 'safe-doc'")) == 3
        assert asyncio.run(_count("cmetadata->>'file_id' = 'stale-doc'")) == 1
        assert manifesto.read_bytes() == approved
        assert not manifesto.with_suffix(".json.execucao.json").exists()

    def test_execute_without_manifest_refused(self):
        run = _run_script("--execute", "--yes")
        assert run.returncode != 0
        assert "manifesto" in (run.stderr + run.stdout)

    def test_execute_with_no_by_namespace_refused(self, tmp_path):
        run = _run_script(
            "--execute", "--yes", "--no-by-namespace",
            "--manifesto", str(tmp_path / "m.json"),
        )
        assert run.returncode != 0
        assert "namespace" in (run.stderr + run.stdout)

    def test_execute_with_scope_flags_refused(self, tmp_path):
        run = _run_script(
            "--execute", "--yes",
            "--manifesto", str(tmp_path / "m.json"),
            "--file-id", "qualquer",
        )
        assert run.returncode != 0
        assert "manifesto" in (run.stderr + run.stdout)

    def test_execute_refuses_manifest_generated_without_namespace(self, tmp_path):
        """Rodada 3 (bloqueante): dry-run com --no-by-namespace seguido de
        --execute SEM a flag é recusado — o manifesto prova COMO foi gerado
        (by_namespace=false) e a fusão de cópias legítimas não vira delete."""
        manifesto = tmp_path / "manifesto.json"
        asyncio.run(
            _insert_rows(
                [
                    ("ns-doc", "user-1", 0, 0, "MESMO", "/tmp/1.pdf"),
                    ("ns-doc", "user-1", 0, 0, "MESMO", "/tmp/1b.pdf"),
                    ("ns-doc", "agent-2", 0, 0, "MESMO", "/tmp/2.pdf"),
                ]
            )
        )

        dry = _run_script(
            "--collection", COLLECTION,
            "--file-id", "ns-doc",
            "--no-by-namespace",
            "--manifesto", str(manifesto),
        )
        assert dry.returncode == 0, dry.stderr
        assert asyncio.run(_count()) == 3
        m = json.loads(manifesto.read_text())
        assert m["by_namespace"] is False
        # O inventário sem namespace tem candidatos, mas não autoriza exclusão.
        assert m["candidatos_total"] >= 1

        run = _run_script("--manifesto", str(manifesto), "--execute", "--yes")
        assert run.returncode != 0
        assert "namespace" in (run.stderr + run.stdout)
        # nada foi apagado: cópias de ambos os namespaces sobrevivem
        assert asyncio.run(_count()) == 3

    def test_execute_warns_when_schema_diverges(self, tmp_path):
        """Rodada 3 (sugestão): --schema diferente do schema do manifesto avisa
        antes da confirmação (a execução é abortada pelo keeper ausente)."""
        manifesto = tmp_path / "manifesto.json"
        manifesto.write_text(
            json.dumps(
                {
                    "card": "278-citacoes-duplicadas",
                    "gerado_em": "2026-10-22T00:00:00Z",
                    "schema": "schema_do_dry_run",
                    "by_namespace": True,
                    "agrupamento": GROUPING_DESCRIPTION,
                    "grupos": [{
                        "namespace": "user-1",
                        "namespaces_presentes": ["user-1"],
                        "manter_uuid": str(uuid.uuid4()),
                        "remover_uuids": [str(uuid.uuid4())],
                    }],
                }
            )
        )
        run = _run_script("--manifesto", str(manifesto), "--execute", "--yes")
        assert run.returncode != 0  # keeper ausente, para depois do aviso
        combined = run.stdout + run.stderr
        assert "diverge" in combined
        assert "schema_do_dry_run" in combined
