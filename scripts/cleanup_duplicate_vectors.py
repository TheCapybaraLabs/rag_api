#!/usr/bin/env python3
"""Card #278 — inventário e limpeza de vetores duplicados no pgvector.

⚠⚠⚠  NÃO EXECUTAR CONTRA PRODUÇÃO SEM O OK EXPLÍCITO DO DONO DOS TENANTS.  ⚠⚠⚠

Este script NÃO faz parte da inicialização do serviço. Ele existe para preparar,
AUDITAR e — só com autorização — corrigir a duplicação de ingestão descrita no
card #278 (reingestão sem idempotência antes do fix de dedup/UNIQUE).

Regras da casa (não negociáveis):
  * Nunca deletar por prefixo de nome (nada de `provision-%`): a exclusão é
    SOMENTE por uuid físico, e no modo --execute EXATAMENTE os uuids aprovados
    no manifesto do dry-run — nunca um reagrupamento refeito na hora.
  * O agrupamento é SEMPRE por namespace (user_id): cópias legítimas do mesmo
    chunk entre agentes são preservadas. `--no-by-namespace` existe só para
    inventário e é RECUSADO com --execute — e o manifesto grava `by_namespace`:
    --execute também RECUSA manifesto gerado sem namespace (ou sem o campo,
    de versão anterior do script).
  * A posição do chunk (chunk_index) entra no agrupamento: textos repetidos na
    mesma página em posições distintas são chunks distintos, não duplicatas.
  * Mantém-se a linha mais antiga (menor ctid ≈ ordem física de inserção), não
    a de menor uuid — uuid v4 é aleatório e pode preservar lixo de provisionamento.
  * Modo padrão é --dry-run: nenhuma linha é alterada sem --execute, e --execute
    EXIGE o manifesto gerado por um dry-run prévio.
  * Snapshot + delete rodam na MESMA transação; o snapshot é a âncora de rollback.

Agrupamento (por coleção):
    file_id      = cmetadata->>'file_id'
    namespace    = cmetadata->>'user_id'        (sempre; ver --no-by-namespace)
    página       = cmetadata->>'page'
    posição      = cmetadata->>'chunk_index'
    texto        = md5(document)
Duplicata = grupo com count(*) > 1. Mantém-se a linha de menor ctid; as demais
entram no manifesto como candidatas.

Uso:
    # 1. inventário / dry-run (não altera nada; grava o manifesto)
    python scripts/cleanup_duplicate_vectors.py --file-id <id> [--file-id <id2> ...] \
        [--manifesto /tmp/manifesto.json]

    # 2. executar EXATAMENTE o manifesto revisado (SÓ COM OK DO DONO DOS TENANTS)
    python scripts/cleanup_duplicate_vectors.py --manifesto /tmp/manifesto.json \
        --execute --yes

Rollback (âncora — emitido com a lista concreta de uuids no relatório de execução;
ver também README_CARD_278_LIMPEZA.md):
    INSERT INTO langchain_pg_embedding (uuid, collection_id, embedding, document,
                                        cmetadata, custom_id)
    SELECT uuid, collection_id, embedding, document, cmetadata, custom_id
    FROM <schema>.<snapshot_table>
    WHERE uuid = ANY('{<uuid1>,<uuid2>,...}'::uuid[]);
"""

from __future__ import annotations

import argparse
import asyncio
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

# Agrupamento SEMPRE por namespace + chunk_index: preserva cópias legítimas entre
# agentes e não funde chunks distintos de texto repetido na mesma página.
GROUP_KEYS = """
    cmetadata->>'file_id', cmetadata->>'user_id', cmetadata->>'page',
    cmetadata->>'chunk_index', md5(document)
"""
GROUPING_DESCRIPTION = "file_id+user_id+page+chunk_index+md5(document)"
# Sinal de inserção: ctid é a posição física da tupla — a linha de menor ctid é,
# em geral, a mais antiga (uuid v4 NÃO é sinal de idade).
KEEP_RULE_DESCRIPTION = (
    "menor ctid (ordem física de inserção; sobrevive a linha mais antiga)"
)


def quote_ident(identifier: str) -> str:
    """Quoting seguro de identificador (schema/tabela) contra interpolação hostil."""
    return '"' + identifier.replace('"', '""') + '"'


def qualify(table: str, schema: str | None) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}" if schema else quote_ident(table)


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
        help="[dry-run] file_id alvo (repetível). Obrigatório (sem --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="[dry-run] Processa TODA a coleção (alto custo — só para inventário; "
        "--execute nunca usa escopo, só o manifesto).",
    )
    parser.add_argument(
        "--by-namespace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Agrupa também por user_id (DEFAULT: preserva cópias legítimas entre "
        "agentes). --no-by-namespace só serve para inventário; --execute recusa.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Nome da coleção (COLLECTION_NAME do serviço). Default: todas as coleções.",
    )
    parser.add_argument(
        "--manifesto",
        default=None,
        help="Caminho do JSON de manifesto. No dry-run é ESCRITO aqui; no --execute é "
        "LIDO daqui (os uuids aprovados) e nunca sobrescrito.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Executa a exclusão dos uuids do manifesto do dry-run. SEM ISTO O SCRIPT "
        "SÓ AUDITA.",
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
        await conn.execute(f"SET search_path TO {quote_ident(schema)}")
    return conn


async def resolve_collection_ids(conn, collection_name: str | None) -> list[str] | None:
    if not collection_name:
        return None
    rows = await conn.fetch(
        f"SELECT uuid FROM {qualify('langchain_pg_collection', None)} WHERE name = $1",
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
    namespace_select = "(cmetadata->>'user_id') AS namespace," if by_namespace else ""
    namespace_group = "cmetadata->>'user_id'," if by_namespace else ""
    query = f"""
        SELECT (cmetadata->>'file_id')                          AS file_id,
               {namespace_select}
               (cmetadata->>'page')                             AS page,
               (cmetadata->>'chunk_index')                      AS chunk_index,
               md5(document)                                    AS text_hash,
               count(*)                                         AS quantidade,
               array_agg(uuid ORDER BY ctid)                    AS registros,
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
    group_by = f"{namespace_group} {GROUP_KEYS}" if by_namespace else GROUP_KEYS
    query += f" GROUP BY {group_by} HAVING count(*) > 1 ORDER BY quantidade DESC"
    return await conn.fetch(query, *args)


def build_manifest_groups(
    grupos, by_namespace: bool
) -> tuple[list[str], list[dict]]:
    """Mantém o primeiro uuid (menor ctid ≈ inserção mais antiga); os demais são
    candidatos à remoção."""
    candidatos: list[str] = []
    manifest_groups = []
    for g in grupos:
        registros = [str(u) for u in g["registros"]]
        manter, remover = registros[0], registros[1:]
        candidatos.extend(remover)
        manifest_groups.append(
            {
                "file_id": g["file_id"],
                **({"namespace": g["namespace"]} if by_namespace else {}),
                "page": g["page"],
                "chunk_index": g["chunk_index"],
                "text_hash_md5": g["text_hash"],
                "quantidade": g["quantidade"],
                "manter_uuid": manter,
                "remover_uuids": remover,
                "namespaces_presentes": g["namespaces"],
                "origens": g["origens"],
            }
        )
    return candidatos, manifest_groups


def write_manifest(
    path: Path,
    *,
    schema: str | None,
    collection: str | None,
    file_ids: list[str] | None,
    by_namespace: bool,
    total_antes: int,
    manifest_groups: list[dict],
    candidatos_total: int,
    ts: str,
) -> None:
    agrupamento = (
        GROUPING_DESCRIPTION
        if by_namespace
        else "file_id+page+chunk_index+md5(document) (SEM namespace — só inventário!)"
    )
    manifesto = {
        "card": "278-citacoes-duplicadas",
        "gerado_em": ts,
        "schema": schema,
        "colecao": collection,
        "escopo_file_ids": file_ids if file_ids else "ALL",
        "by_namespace": bool(by_namespace),
        "agrupamento": agrupamento,
        "criterio_manter": KEEP_RULE_DESCRIPTION,
        "total_linhas_antes": total_antes,
        "grupos": manifest_groups,
        "candidatos_total": candidatos_total,
    }
    path.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False))


def load_manifest_remover_uuids(manifest_path: Path) -> tuple[list[str], dict]:
    """Lê os remover_uuids APROVADOS no manifesto do dry-run.

    O manifesto é a ÚNICA fonte de exclusão — e ele tem de provar COMO foi
    gerado: exige `by_namespace: true`, a descrição exata do agrupamento padrão
    e um único namespace consistente por grupo. Manifestos entre agentes ou de
    versões anteriores são recusados antes de qualquer conexão ao banco.
    """
    if not manifest_path.exists():
        raise SystemExit(
            f"Manifesto não encontrado: {manifest_path}. Rode o dry-run antes."
        )
    try:
        manifesto = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Manifesto inválido ({manifest_path}): {exc}")
    if manifesto.get("card") != "278-citacoes-duplicadas":
        raise SystemExit(f"Manifesto não é do card #278: {manifest_path}")
    if manifesto.get("by_namespace") is not True:
        raise SystemExit(
            "Recusado: o manifesto não registra agrupamento por namespace "
            f"(by_namespace={manifesto.get('by_namespace')!r}). Ele foi gerado com "
            "--no-by-namespace (só inventário) ou por versão anterior do script; "
            "aplicá-lo fundiria cópias legítimas do mesmo chunk entre agentes. "
            "Rode o dry-run de novo com o agrupamento padrão (por namespace)."
        )
    if manifesto.get("agrupamento") != GROUPING_DESCRIPTION:
        raise SystemExit(
            "Recusado: agrupamento do manifesto não corresponde ao padrão por "
            "namespace. Rode o dry-run de novo com o agrupamento padrão."
        )
    removidos: list[str] = []
    for g in manifesto.get("grupos", []):
        if "namespace" not in g or g.get("namespaces_presentes") != [g["namespace"]]:
            raise SystemExit(
                "Recusado: grupo do manifesto sem namespace único e consistente "
                f"(file_id={g.get('file_id')!r}). Nenhum uuid será apagado."
            )
        removidos.extend(g.get("remover_uuids", []))
    if not removidos:
        raise SystemExit("Manifesto não contém candidatos (remover_uuids vazio).")
    if len(set(removidos)) != len(removidos):
        raise SystemExit("Manifesto contém uuids duplicados — recusado.")
    return removidos, manifesto


def rollback_sql(snapshot_table: str, schema: str | None, uuids: list[str]) -> str:
    """Rollback com a lista CONCRETA de uuids (sem placeholder solto)."""
    uuid_list = ",".join(sorted(set(uuids)))
    return (
        f"INSERT INTO {qualify(EMBEDDING_TABLE, schema)} "
        "(uuid, collection_id, embedding, document, cmetadata, custom_id) "
        "SELECT uuid, collection_id, embedding, document, cmetadata, custom_id "
        f"FROM {qualify(snapshot_table, schema)} "
        f"WHERE uuid = ANY('{{{uuid_list}}}'::uuid[]);"
    )


async def execute_manifest(
    conn,
    schema: str | None,
    remover_uuids: list[str],
    manifest_groups: list[dict],
) -> tuple[str, int, int, list[str]]:
    """Snapshot + delete dos uuids do manifesto, numa ÚNICA transação.

    Verifica e bloqueia os manter_uuid antes de qualquer exclusão; keeper ausente
    recusa a execução inteira. Apaga EXATAMENTE os candidatos ainda presentes
    (nunca um reagrupamento vivo). Retorna (snapshot_table, snap_count, removidos,
    ausentes).
    """
    async with conn.transaction():
        # O lock preserva os keepers até o commit do snapshot/delete.
        manter_uuids = [g.get("manter_uuid") for g in manifest_groups]
        keepers = await conn.fetch(
            f"SELECT uuid FROM {qualify(EMBEDDING_TABLE, schema)} "
            "WHERE uuid = ANY($1::uuid[]) FOR KEY SHARE",
            manter_uuids,
        )
        keepers_ids = {str(r["uuid"]) for r in keepers}
        for g in manifest_groups:
            if g.get("manter_uuid") not in keepers_ids:
                raise SystemExit(
                    f"Recusado: manter_uuid {g.get('manter_uuid')!r} ausente "
                    f"(file_id={g.get('file_id')!r}, namespace={g.get('namespace')!r}). "
                    "Nada foi apagado. Rode o dry-run de novo."
                )

        # Quais uuids aprovados ainda existem?
        presentes = await conn.fetch(
            f"SELECT uuid FROM {qualify(EMBEDDING_TABLE, None)} "
            "WHERE uuid = ANY($1::uuid[])",
            remover_uuids,
        )
        presentes_ids = [str(r["uuid"]) for r in presentes]
        ausentes = sorted(set(remover_uuids) - set(presentes_ids))
        if ausentes:
            print(
                f"Aviso: {len(ausentes)} uuid(s) do manifesto não existem mais "
                "(já removidos?):\n  " + "\n  ".join(ausentes[:10])
            )
        if not presentes_ids:
            return "", 0, 0, ausentes

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        snapshot_table = f"card278_backup_{ts}"

        # Snapshot + delete: ou os dois acontecem, ou nenhum.
        result = await conn.execute(
            f"""
            CREATE TABLE {qualify(snapshot_table, schema)} AS
            SELECT uuid, collection_id, embedding, document, cmetadata, custom_id
            FROM {qualify(EMBEDDING_TABLE, schema)}
            WHERE uuid = ANY($1::uuid[])
            """,
            presentes_ids,
        )
        snap_count = int(result.split()[-1])
        result = await conn.execute(
            f"DELETE FROM {qualify(EMBEDDING_TABLE, schema)} WHERE uuid = ANY($1::uuid[])",
            presentes_ids,
        )
        removed = int(result.split()[-1])
        if removed != snap_count:
            raise RuntimeError(
                f"Delete ({removed}) divergiu do snapshot ({snap_count}); "
                "transação revertida."
            )
    return snapshot_table, snap_count, removed, ausentes


async def run_dry_run(args: argparse.Namespace) -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = await connect(args.dsn, args.schema)
    try:
        collection_ids = await resolve_collection_ids(conn, args.collection)
        total_antes = await count_rows(conn, collection_ids, args.file_id)
        grupos = await fetch_duplicate_groups(
            conn, collection_ids, args.file_id, args.by_namespace
        )
        candidatos, manifest_groups = build_manifest_groups(grupos, args.by_namespace)

        print(f"[{ts}] Coleção: {args.collection or 'TODAS'}")
        print(f"Agrupamento: {GROUPING_DESCRIPTION}")
        print(f"Critério de manutenção: {KEEP_RULE_DESCRIPTION}")
        print(f"Linhas no escopo antes: {total_antes}")
        print(
            f"Grupos duplicados: {len(grupos)} | candidatos a remover: {len(candidatos)}"
        )
        for g in manifest_groups[:20]:
            ns = f" ns={g.get('namespace')}" if args.by_namespace else ""
            print(
                f"  - file={g['file_id']}{ns} page={g['page']} "
                f"chunk_index={g['chunk_index']} md5={g['text_hash_md5'][:12]} "
                f"qtd={g['quantidade']} remover={len(g['remover_uuids'])}"
            )
        if len(manifest_groups) > 20:
            print(f"  ... e mais {len(manifest_groups) - 20} grupos (ver manifesto).")

        manifesto_path = Path(args.manifesto or f"card278_manifesto_{ts}.json")
        write_manifest(
            manifesto_path,
            schema=args.schema,
            collection=args.collection,
            file_ids=args.file_id or None,
            by_namespace=args.by_namespace,
            total_antes=total_antes,
            manifest_groups=manifest_groups,
            candidatos_total=len(candidatos),
            ts=ts,
        )
        print(f"Manifesto: {manifesto_path.resolve()}")

        print("\nDRY-RUN: nada foi alterado. Revise o manifesto e execute novamente com:")
        print(
            f"  {Path(sys.argv[0]).name} --manifesto {manifesto_path.resolve()} --execute"
        )
        print("(e ok do dono dos tenants) para aplicar EXATAMENTE o manifesto acima.")
        return 0
    finally:
        await conn.close()


async def run_execute(args: argparse.Namespace) -> int:
    """--execute: aplica o MANIFESTO revisado, não um reagrupamento vivo."""
    if not args.manifesto:
        raise SystemExit(
            "Recusado: --execute exige --manifesto <arquivo do dry-run>. O script NUNCA "
            "reagrupa por conta própria na hora de apagar — os uuids removidos devem ser "
            "exatamente os aprovados na revisão do dry-run."
        )
    manifesto_path = Path(args.manifesto)
    remover_uuids, manifesto = load_manifest_remover_uuids(manifesto_path)
    print(f"Manifesto revisado: {manifesto_path.resolve()}")
    print(f"Agrupamento: {manifesto.get('agrupamento')}")
    print(f"Critério de manutenção: {manifesto.get('criterio_manter')}")
    print(f"Candidatos aprovados no manifesto: {len(remover_uuids)}")
    if manifesto.get("schema") != args.schema:
        print(
            f"Aviso: schema do manifesto ({manifesto.get('schema')!r}) diverge do "
            f"--schema alvo ({args.schema!r}). O snapshot/delete rodam no schema "
            "ALVO — confira antes de confirmar."
        )

    if not args.yes:
        resposta = input(
            f"Confirma a exclusão de EXATAMENTE esses {len(remover_uuids)} uuids? "
            "Digite o número exato para confirmar: "
        )
        if resposta.strip() != str(len(remover_uuids)):
            print("Confirmação divergente — abortado. Nada foi alterado.")
            return 1

    conn = await connect(args.dsn, args.schema)
    try:
        snapshot_table, snap_count, removidos, ausentes = await execute_manifest(
            conn, args.schema, remover_uuids, manifesto["grupos"]
        )
        if not snapshot_table:
            print("Nenhum uuid do manifesto existe mais. Nada a fazer.")
            return 0
        escopo = manifesto.get("escopo_file_ids")
        total_depois = await count_rows(
            conn,
            await resolve_collection_ids(conn, manifesto.get("colecao")),
            escopo if isinstance(escopo, list) else [],
        )

        # Relatório de execução em arquivo PRÓPRIO — o manifesto aprovado jamais é
        # sobrescrito.
        execucao = {
            "card": "278-citacoes-duplicadas",
            "manifesto_origem": str(manifesto_path.resolve()),
            "manifesto_gerado_em": manifesto.get("gerado_em"),
            "executado_em": datetime.now(timezone.utc).isoformat(),
            "schema": args.schema,
            "snapshot_table": snapshot_table,
            "snapshot_linhas": snap_count,
            "removidos": removidos,
            "uuids_ausentes": ausentes,
            "total_linhas_depois": total_depois,
            "rollback": rollback_sql(snapshot_table, args.schema, remover_uuids),
        }
        execucao_path = manifesto_path.with_suffix(
            manifesto_path.suffix + ".execucao.json"
        )
        execucao_path.write_text(json.dumps(execucao, indent=2, ensure_ascii=False))

        print(f"\nSnapshot (âncora de rollback): {snapshot_table} ({snap_count} linhas)")
        print(f"Removidos: {removidos} | ausentes do manifesto: {len(ausentes)}")
        print(f"Linhas no escopo depois: {total_depois}")
        print(f"Relatório de execução (rollback incluso): {execucao_path.resolve()}")
        print("Manifesto original preservado sem alteração.")
        return 0
    finally:
        await conn.close()


async def main() -> int:
    args = build_arg_parser().parse_args()

    if args.execute:
        if args.file_id or args.all:
            raise SystemExit(
                "Recusado: com --execute a seleção vem SEMPRE do manifesto "
                "(--manifesto). Passe apenas --manifesto (e --yes se automatizado)."
            )
        if not args.by_namespace:
            raise SystemExit(
                "Recusado: --execute exige o agrupamento por namespace (padrão). O modo "
                "--no-by-namespace funde chunks idênticos entre agentes/namespaces "
                "mantendo só uma cópia e viola a preservação de cópias legítimas."
            )
        return await run_execute(args)

    # Dry-run / inventário
    if not args.all and not args.file_id:
        raise SystemExit("Informe --file-id (repetível) ou --all.")
    return await run_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
