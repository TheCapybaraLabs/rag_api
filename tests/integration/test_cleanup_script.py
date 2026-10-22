"""Smoke test do script de limpeza (card #278) contra um Postgres/pgvector descartável.

Sobe um container, cria o schema espelhado, insere duplicatas de dois tipos
(mesmo namespace / namespaces distintos), roda o script em dry-run e depois
com --execute, e valida snapshot/manifesto/rollback documentado.

Uso: OPENAI_API_KEY=test_key python tests/integration/test_cleanup_script.py
(não é pytest — script direto para exercitar o CLI de ponta a ponta)
"""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import testcontainers.postgres as tp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "cleanup_duplicate_vectors.py"

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR NOT NULL UNIQUE,
    cmetadata JSONB
);
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
    embedding vector(3),
    document VARCHAR,
    cmetadata JSONB,
    custom_id VARCHAR
);
INSERT INTO langchain_pg_collection (name, cmetadata) VALUES ('testcollection', '{}');
"""


def insert_row(conn, file_id, user_id, page, text, source):
    conn.execute(
        "INSERT INTO langchain_pg_embedding (collection_id, embedding, document, cmetadata, custom_id) "
        "SELECT uuid, '[0.1,0.2,0.3]', %s, %s, %s FROM langchain_pg_collection WHERE name = 'testcollection'",
        (
            text,
            json.dumps(
                {
                    "file_id": file_id,
                    "user_id": user_id,
                    "page": page,
                    "source": source,
                    "digest": uuid.uuid4().hex[:8],
                }
            ),
            file_id,
        ),
    )


def main():
    with tp.PostgresContainer("pgvector/pgvector:pg16", driver="psycopg2") as pg:
        dsn_url = pg.get_connection_url()  # sqlalchemy style: postgresql+psycopg2://...
        dsn = dsn_url.replace("postgresql+psycopg2://", "postgresql://")
        import psycopg2

        raw = psycopg2.connect(dsn)
        raw.autocommit = True
        cur = raw.cursor()
        cur.execute(DDL)

        collection = "testcollection"
        # 3x o mesmo chunk no MESMO namespace (duplicata efetiva) + 1x em OUTRO
        # namespace (cópia legítima de agente) + 1 arquivo limpo.
        for i in range(3):
            insert_row(cur, "dup-doc", "user-1", 0, "TEXTO IGUAL", f"/tmp/provision-x-{i}.pdf")
        insert_row(cur, "dup-doc", "agent-2", 0, "TEXTO IGUAL", "/tmp/original.pdf")
        insert_row(cur, "clean-doc", "user-1", 0, "UNICO", "/tmp/ok.pdf")

        env = dict(os.environ, OPENAI_API_KEY="test_key")

        # ---- dry-run: nada muda
        manifest = "/tmp/card278_manifesto_smoke.json"
        run = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dsn", dsn, "--collection", collection,
                "--file-id", "dup-doc", "--file-id", "clean-doc",
                "--by-namespace", "--manifesto", manifest,
            ],
            capture_output=True, text=True, env=env,
        )
        print(run.stdout[-2000:])
        assert run.returncode == 0, run.stderr
        assert "DRY-RUN" in run.stdout
        cur.execute("SELECT count(*) FROM langchain_pg_embedding")
        antes = cur.fetchone()[0]
        assert antes == 5, antes

        m = json.loads(Path(manifest).read_text())
        assert m["candidatos_total"] == 2, m["candidatos_total"]
        grupo = [g for g in m["grupos"] if g.get("namespace") == "user-1"][0]
        assert grupo["quantidade"] == 3 and len(grupo["remover_uuids"]) == 2
        assert not any("provision" in str(g.get("manter_uuid")) for g in m["grupos"])

        # ---- execute: remove exatamente os candidatos do manifesto
        run = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dsn", dsn, "--collection", collection,
                "--file-id", "dup-doc", "--file-id", "clean-doc",
                "--by-namespace", "--manifesto", manifest,
                "--execute", "--yes",
            ],
            capture_output=True, text=True, env=env,
        )
        print(run.stdout[-2000:])
        assert run.returncode == 0, run.stderr

        m = json.loads(Path(manifest).read_text())
        assert m["removidos"] == 2 and m["total_linhas_depois"] == 3
        assert m["snapshot_linhas"] == 2 and "card278_backup_" in m["snapshot_table"]

        cur.execute("SELECT count(*) FROM langchain_pg_embedding")
        assert cur.fetchone()[0] == 3
        cur.execute(
            "SELECT count(*) FROM langchain_pg_embedding "
            "WHERE cmetadata->>'file_id'='dup-doc' AND cmetadata->>'user_id'='agent-2'"
        )
        assert cur.fetchone()[0] == 1, "cópia legítima do agente deve sobreviver"

        # ---- rollback pela âncora (como documentado): restaura o snapshot inteiro
        cur.execute(f"""
            INSERT INTO langchain_pg_embedding (uuid, collection_id, embedding, document, cmetadata, custom_id)
            SELECT uuid, collection_id, embedding, document, cmetadata, custom_id
            FROM {m['snapshot_table']}
        """)
        cur.execute("SELECT count(*) FROM langchain_pg_embedding")
        assert cur.fetchone()[0] == 5, "rollback deve restaurar as 5 linhas"

        print("SMOKE_OK: dry-run, execute, manifesto, snapshot e rollback verificados")


if __name__ == "__main__":
    main()
