"""A execução deve preservar as cópias restantes se qualquer keeper desaparecer."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.cleanup_duplicate_vectors import execute_manifest


@pytest.mark.parametrize("missing_keeper", ["keeper-a", "keeper-b"])
async def test_missing_keeper_refuses_every_group_before_snapshot_or_delete(missing_keeper):
    groups = [
        {
            "file_id": "doc",
            "namespace": namespace,
            "manter_uuid": keeper,
            "remover_uuids": [candidate],
        }
        for namespace, keeper, candidate in (
            ("agentA", "keeper-a", "candidate-a"),
            ("agentB", "keeper-b", "candidate-b"),
        )
    ]
    conn = SimpleNamespace(
        transaction=MagicMock(),
        fetch=AsyncMock(
            return_value=[
                {"uuid": group["manter_uuid"]}
                for group in groups
                if group["manter_uuid"] != missing_keeper
            ]
        ),
        execute=AsyncMock(),
    )

    with pytest.raises(SystemExit, match=f"manter_uuid '{missing_keeper}' ausente"):
        await execute_manifest(conn, None, ["candidate-a", "candidate-b"], groups)

    conn.execute.assert_not_awaited()
    conn.transaction.return_value.__aexit__.assert_awaited_once()
