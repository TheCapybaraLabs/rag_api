"""Card #278 — testes unitários do CLI do script de limpeza (sem banco).

Cobre os achados 5 e 7 no nível que não exige Postgres:
  * --by-namespace é o DEFAULT do agrupamento; --execute recusa --no-by-namespace
  * --execute exige --manifesto (nunca reagrupa o banco vivo na hora de apagar)
  * --execute recusa --file-id/--all (a seleção vem do manifesto)
  * load_manifest_remover_uuids valida o manifesto antes de apagar qualquer coisa
  * rollback_sql emite a lista CONCRETA de uuids
  * quote_ident/qualify escapan identificadores
"""

import json
from pathlib import Path

import pytest

from scripts.cleanup_duplicate_vectors import (
    GROUPING_DESCRIPTION,
    GROUP_KEYS,
    build_arg_parser,
    load_manifest_remover_uuids,
    qualify,
    quote_ident,
    rollback_sql,
)


def parse(*argv):
    return build_arg_parser().parse_args(argv)


class TestNamespaceGroupingDefault:
    def test_by_namespace_is_the_default(self):
        """Achado 5: fundir cópias entre agentes NUNCA é o comportamento padrão."""
        assert parse("--file-id", "f1").by_namespace is True

    def test_explicit_opt_out_is_possible_for_inventory(self):
        assert parse("--file-id", "f1", "--no-by-namespace").by_namespace is False

    def test_group_keys_always_include_namespace_and_chunk_index(self):
        """Achado 6: namespace e chunk_index fazem parte da chave de agrupamento."""
        assert "user_id" in GROUP_KEYS
        assert "chunk_index" in GROUP_KEYS
        assert "user_id" in GROUPING_DESCRIPTION


class TestExecuteRefusals:
    @staticmethod
    def _run_main(argv):
        """main() lê sys.argv; as recusas acontecem antes de qualquer conexão."""
        import sys

        import asyncio

        from scripts.cleanup_duplicate_vectors import main

        old = sys.argv
        sys.argv = ["cleanup_duplicate_vectors.py", *argv]
        try:
            asyncio.run(main())
        finally:
            sys.argv = old

    def test_execute_requires_manifest(self):
        """Achado 7: sem manifesto não há exclusão — reagrupar na hora é proibido."""
        with pytest.raises(SystemExit, match="manifesto"):
            self._run_main(["--execute", "--yes", "--dsn", "postgresql://x"])

    def test_execute_refuses_scope_flags(self):
        """A seleção da exclusão vem SEMPRE do manifesto revisado."""
        with pytest.raises(SystemExit, match="manifesto"):
            self._run_main(
                ["--execute", "--yes", "--file-id", "f1", "--manifesto", "/tmp/m.json"]
            )

    def test_execute_refuses_no_by_namespace(self):
        """Achado 5: --execute sem agrupamento por namespace é recusado."""
        with pytest.raises(SystemExit, match="namespace"):
            self._run_main(
                [
                    "--execute",
                    "--yes",
                    "--manifesto",
                    "/tmp/m.json",
                    "--no-by-namespace",
                ]
            )


class TestManifestValidation:
    def _manifest(self, tmp_path, grupos):
        path = tmp_path / "manifesto.json"
        path.write_text(
            json.dumps(
                {
                    "card": "278-citacoes-duplicadas",
                    "gerado_em": "2026-10-22T00:00:00Z",
                    "grupos": grupos,
                }
            )
        )
        return path

    def test_reads_all_remover_uuids(self, tmp_path):
        path = self._manifest(
            tmp_path,
            [
                {"remover_uuids": ["u1", "u2"]},
                {"remover_uuids": ["u3"]},
            ],
        )
        uuids, manifesto = load_manifest_remover_uuids(path)
        assert uuids == ["u1", "u2", "u3"]
        assert manifesto["card"] == "278-citacoes-duplicadas"

    def test_missing_manifest_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="dry-run"):
            load_manifest_remover_uuids(tmp_path / "nao-existe.json")

    def test_wrong_card_is_refused(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"card": "outro-card", "grupos": []}))
        with pytest.raises(SystemExit, match="card #278"):
            load_manifest_remover_uuids(path)

    def test_empty_manifest_is_refused(self, tmp_path):
        path = self._manifest(tmp_path, [])
        with pytest.raises(SystemExit, match="vazio"):
            load_manifest_remover_uuids(path)

    def test_duplicated_uuids_are_refused(self, tmp_path):
        path = self._manifest(
            tmp_path,
            [{"remover_uuids": ["u1"]}, {"remover_uuids": ["u1"]}],
        )
        with pytest.raises(SystemExit, match="duplicad"):
            load_manifest_remover_uuids(path)


class TestRollbackSql:
    def test_rollback_uses_concrete_uuid_list(self):
        sql = rollback_sql("card278_backup_1", None, ["u2", "u1"])
        assert "ANY('{u1,u2}'::uuid[])" in sql  # array literal com chaves
        assert "removidos_do_manifesto" not in sql  # sem placeholder solto

    def test_rollback_qualifies_schema(self):
        sql = rollback_sql("snap", "meu_schema", ["u1"])
        assert '"meu_schema"."snap"' in sql
        assert '"meu_schema"."langchain_pg_embedding"' in sql


class TestIdentifiers:
    def test_quote_ident_escapes_embedded_quotes(self):
        assert quote_ident('sch"ema') == '"sch""ema"'

    def test_qualify_quotes_schema_and_table(self):
        assert qualify("t", "s") == '"s"."t"'
        assert qualify("t", None) == '"t"'
