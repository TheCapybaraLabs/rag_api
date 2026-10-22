#!/usr/bin/env python3
"""Card #278 — inventário e limpeza de vetores duplicados no pgvector.

⚠⚠⚠  NÃO EXECUTAR CONTRA PRODUÇÃO SEM O OK EXPLÍCITO DO DONO DOS TENANTS.  ⚠⚠⚠

Este script NÃO faz parte da inicialização do serviço. Ele existe para preparar,
AUDITAR e — só com autorização — corrigir a duplicação de ingestão descrita no
card #278 (reingestão sem idempotência antes do fix de dedup/UNIQUE).

Regras da casa (não negociáveis):
  * Nunca deletar por prefixo de nome (nada de `provision-%`): o agrupamento é
    por (file_id, página, md5 do texto) + namespace, e a exclusão é SOMENTE por
    uuid físico listado no manifesto gerado na mesma execução.
  * Antes de qualquer exclusão: snapshot das linhas afetadas, contagens e
    manifesto com os ids candidatos. O snapshot é a âncora de rollback.
  * Modo padrão é --dry-run: nenhuma linha é alterada sem --execute.

Agrupamento (por coleção):
    file_id = cmetadata->>'file_id'
    página  = cmetadata->>'page'
    texto   = md5(document)
    namespace (opcional via --by-namespace, recomendado) = cmetadata->>'user_id'
Duplicata = grupo com count(*) > 1. Mantém-se a linha mais antiga (menor uuid
para desempate estável); as demais entram no manifesto como candidatas.

Rollback (âncora — ver também README_CARD_278_LIMPEZA.md):
    INSERT INTO langchain_pg_embedding (uuid, collection_id, embedding, document,
                                        cmetadata, custom_id)
    SELECT uuid, collection_id, embedding, document, cmetadata, custom_id
    FROM <schema>.<snapshot_table>
    WHERE uuid = ANY(:removidos_do_manifesto);

Uso:
    # 1. inventário (não altera nada)
    python scripts/cleanup_duplicate_vectors.py --file-id <id> [--file-id <id2> ...] \
        [--by-namespace] [--manifesto /tmp/manifesto.json]

    # 2. mesmo comando + --execute --yes SÓ COM OK DO DONO
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

# Reaproveita a configuração do serviço (POSTGRES_* / POSTGRES_SCHEMA) — mesma
# fonte de verdade do app, sem segredos em linha de comando.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from app.config import DSN, POSTGRES_SCHEMA  # noqa: E402
except Exception:  # pragma: no cover - uso fora do serviço
    DSN = None
    POSTGRES_SCHEMA = None

EMBEDDING_TABLE = "langchain_pg_embedding"

GROUP_BY_NAMESPACE = """
    cmetadata->>'file_id', cmetadata->>'user_id', cmetadata->>'page', md5(document)
"""
GROUP_WITHOUT_NAMESPACE = """
    cmetadata->>'file_id', cmetadata->>'page', md5(document)
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventário/limpeza de vetores duplicados (card #278).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(__doc__ or "").split("Uso:")[0],
    )
    parser.add_argument(
        "--file-id",
        action="append",
        default=[],
        help="file_id alvo (repetível). Sem --all, exige pelo menos um.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Processa TODA a coleção (alto custo — use só com inventário).",
    )
    parser.add_argument(
        "--by-namespace",
        action="store_true",
        help="Agrupa também por user_id: separa repetição dentro do namespace de "
        "cópias legítimas entre agentes (recomendado; ver card #278).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Nome da coleção (COLLECTION_NAME do serviço). Default: todas as coleções.",
    )
    parser.add_argument(
        "--manifesto",
        default=None,
        help="Caminho do JSON de manifesto. Default: card278_manifesto_<ts>.json no cwd.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Executa a exclusão dos uuids candidatos. SEM ISTO O SCRIPT SÓ AUDITA.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Não pede confirmação interativa (para uso automatizado autorizado).",
    )
    parser.add_argument("--dsn", default=DSN, help="DSN Postgres (default: env do app).")
    parser.add_argument(
        "--schema", default=POSTGRES_SCHEMA, help="Schema (default: POSTGRES_SCHEMA do app)."
    )
    return parser


async def connect(dsn: str, schema: str | None) -> asyncpg.Connection:
    if not dsn:
        raise SystemExit("DSN não definido: passe --dsn ou configure POSTGRES_* no ambiente.")
    conn = await asyncpg.connect(dsn=dsn)
    if schema:
        await conn.execute(f'SET search_path TO "{schema}"')
    return conn


def qualify(table: str, schema: str | None) -> str:
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


async def resolve_collection_ids(conn, collection_name: str | None) -> list[str] | None:
    if not collection_name:
        return None
    rows = await conn.fetch(
        f'SELECT uuid FROM {qualify("langchain_pg_collection", None)} WHERE name = $1',
        collection_name,
    )
    if not rows:
        raise SystemExit(f"Coleção '{collection_name}' não encontrada.")
    return [str(r["uuid"]) for r in rows]


async def count_rows(conn, collection_ids: list[str] | None, file_ids: list[str]) -> int:
    query = f"SELECT count(*) FROM {qualify(EMBEDDING_TABLE, None)} WHERE TRUE"
    args: list = []
    if collection_ids is not None:
        query += f" AND collection_id = ANY(${len(args) + 1}::uuid[])"
        args.append(collection_ids)
    if file_ids:
        query += f" AND cmetadata->>'file_id' = ANY(${len(args) + 1}::text[])"
        args.append(file_ids)
    return await conn.fetchval(query, *args)


async def fetch_duplicate_groups(
    conn,
    collection_ids: list[str] | None,
    file_ids: list[str],
    by_namespace: bool,
) -> list[asyncpg.Record]:
    group_by = GROUP_BY_NAMESPACE if by_namespace else GROUP_WITHOUT_NAMESPACE
    query = f"""
        SELECT (cmetadata->>'file_id')                          AS file_id,
               {'(cmetadata->>' + "'user_id'" + ') AS namespace,' if by_namespace else ''}
               (cmetadata->>'page')                             AS page,
               md5(document)                                    AS text_hash,
               count(*)                                         AS quantidade,
               array_agg(uuid ORDER BY uuid)                    AS registros,
               array_agg(DISTINCT cmetadata->>'user_id')        AS namespaces,
               array_agg(DISTINCT cmetadata->>'source')         AS origens
        FROM {qualify(EMBEDDING_TABLE, None)}
        WHERE TRUE
    """
    args: list = []
    if collection_ids is not None:
        query += f" AND collection_id = ANY(${len(args) + 1}::uuid[])"
        args.append(collection_ids)
    if file_ids:
        query += f" AND cmetadata->>'file_id' = ANY(${len(args) + 1}::text[])"
        args.append(file_ids)
    query += f" GROUP BY {group_by} HAVING count(*) > 1 ORDER BY quantidade DESC"
    return await conn.fetch(query, *args)


async def snapshot_affected_rows(
    conn, snapshot_table: str, candidate_ids: list[str]
) -> int:
    """Âncora de rollback: cópia integral (embedding + metadados) antes de excluir."""
    await conn.execute(f"DROP TABLE IF EXISTS {snapshot_table}")
    result = await conn.execute(
        f"""
        CREATE TABLE {snapshot_table} AS
        SELECT uuid, collection_id, embedding, document, cmetadata, custom_id
        FROM {qualify(EMBEDDING_TABLE, None)}
        WHERE uuid = ANY($1::uuid[])
        """,
        candidate_ids,
    )
    return int(result.split()[-1])


async def delete_candidates(conn, candidate_ids: list[str]) -> int:
    result = await conn.execute(
        f"DELETE FROM {qualify(EMBEDDING_TABLE, None)} WHERE uuid = ANY($1::uuid[])",
        candidate_ids,
    )
    return int(result.split()[-1])


async def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.all and not args.file_id:
        raise SystemExit("Informe --file-id (repetível) ou --all.")
    if args.execute and args.all:
        raise SystemExit(
            "Recusado: --execute com --all atinge a coleção inteira. Liste os file_id "
            "afetados (audite antes com --dry-run) ou consulte o dono dos tenants."
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = await connect(args.dsn, args.schema)
    try:
        collection_ids = await resolve_collection_ids(conn, args.collection)
        total_antes = await count_rows(conn, collection_ids, args.file_id)
        grupos = await fetch_duplicate_groups(
            conn, collection_ids, args.file_id, args.by_namespace
        )

        # Mantém o primeiro uuid (ordem estável); os demais são candidatos.
        candidatos: list[str] = []
        manifest_groups = []
        for g in grupos:
            registros = [str(u) for u in g["registros"]]
            manter, remover = registros[0], registros[1:]
            candidatos.extend(remover)
            manifest_groups.append(
                {
                    "file_id": g["file_id"],
                    **({"namespace": g["namespace"]} if args.by_namespace else {}),
                    "page": g["page"],
                    "text_hash_md5": g["text_hash"],
                    "quantidade": g["quantidade"],
                    "manter_uuid": manter,
                    "remover_uuids": remover,
                    "namespaces_presentes": g["namespaces"],
                    "origens": g["origens"],
                }
            )

        print(f"[{ts}] Coleção: {args.collection or 'TODAS'}")
        print(f"Linhas no escopo antes: {total_antes}")
        print(f"Grupos duplicados: {len(grupos)} | candidatos a remover: {len(candidatos)}")
        for g in manifest_groups[:20]:
            ns = f" ns={g.get('namespace')}" if args.by_namespace else ""
            print(
                f"  - file={g['file_id']}{ns} page={g['page']} md5={g['text_hash_md5'][:12]} "
                f"qtd={g['quantidade']} remover={len(g['remover_uuids'])}"
            )
        if len(manifest_groups) > 20:
            print(f"  ... e mais {len(manifest_groups) - 20} grupos (ver manifesto).")

        manifesto = {
            "card": "278-citacoes-duplicadas",
            "gerado_em": ts,
            "schema": args.schema,
            "colecao": args.collection,
            "escopo_file_ids": args.file_id or "ALL",
            "agrupamento": "file_id+user_id+page+md5(document)" if args.by_namespace
            else "file_id+page+md5(document)",
            "total_linhas_antes": total_antes,
            "grupos": manifest_groups,
            "candidatos_total": len(candidatos),
        }
        manifesto_path = Path(args.manifesto or f"card278_manifesto_{ts}.json")
        manifesto_path.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False))
        print(f"Manifesto: {manifesto_path.resolve()}")

        if not args.execute:
            print("\nDRY-RUN: nada foi alterado. Execute novamente com --execute")
            print("(e ok do dono dos tenants) para aplicar o manifesto acima.")
            return 0

        if not candidatos:
            print("Nada a remover.")
            return 0

        if not args.yes:
            resposta = input(
                f"Confirma a exclusão de {len(candidatos)} linhas? "
                "Digite o número exato para confirmar: "
            )
            if resposta.strip() != str(len(candidatos)):
                print("Confirmação divergente — abortado. Nada foi alterado.")
                return 1

        snapshot_table = f"card278_backup_{ts.replace(':', '')}"
        snap_count = await snapshot_affected_rows(conn, snapshot_table, candidatos)
        removidos = await delete_candidates(conn, candidatos)
        total_depois = await count_rows(conn, collection_ids, args.file_id)

        manifesto.update(
            {
                "executado_em": datetime.now(timezone.utc).isoformat(),
                "snapshot_table": snapshot_table,
                "snapshot_linhas": snap_count,
                "removidos": removidos,
                "total_linhas_depois": total_depois,
                "rollback": (
                    f"INSERT INTO {qualify(EMBEDDING_TABLE, None)} "
                    "(uuid, collection_id, embedding, document, cmetadata, custom_id) "
                    f"SELECT uuid, collection_id, embedding, document, cmetadata, custom_id "
                    f"FROM {snapshot_table} WHERE uuid = ANY(removidos_do_manifesto);"
                ),
            }
        )
        manifesto_path.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False))
        print(f"\nSnapshot (âncora de rollback): {snapshot_table} ({snap_count} linhas)")
        print(f"Removidos: {removidos} | Linhas no escopo depois: {total_depois}")
        print(f"Manifesto atualizado: {manifesto_path.resolve()}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
