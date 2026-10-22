-- ============================================================================
-- Card #278 — citações duplicadas: identidade única de chunk no pgvector
-- Repo: TheCapybaraLabs/rag_api
--
-- ⚠️  NÃO APLICAR EM PRODUÇÃO POR CONTA PRÓPRIA.
--     Executar SOMENTE após: (1) ok do dono dos tenants, (2) rodada de auditoria
--     do scripts/cleanup_duplicate_vectors.py --dry-run no ambiente alvo, (3)
--     limpeza (ou pelo menos inventário) das duplicatas de linhas novas.
--
-- Versão de produção confirmada (labschat-ansible, roles/labschat/templates/
-- docker-compose.yml.j2): imagem pgvector/pgvector:0.8.0-pg15-trixie
-- → PostgreSQL 15 + pgvector 0.8.0. Todo DDL abaixo usa recursos do PG ≥ 9.0
-- (índices parciais e de expressão) — sem dependência da extensão pgvector.
--
-- O que este índice garante:
--   Para linhas NOVAS (que carregam chunk_index + user_id no cmetadata), o par
--   (coleção, arquivo, namespace, posição do chunk) é único. Uma segunda
--   ingestão idêntica vira violação de índice em vez de linha duplicada — a
--   segunda barreira de defesa atrás da dedup aplicada no /embed.
--   Linhas legadas (sem chunk_index/user_id) ficam FORA do índice de propósito:
--   não bloqueiam a criação do índice e continuam acessíveis até a limpeza.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Passo 0 (pré-voo, somente leitura): existe duplicata entre linhas NOVAS?
-- Se retornar linhas, rodar o cleanup (card #278) antes do passo 2.
-- ---------------------------------------------------------------------------
--
-- SELECT cmetadata->>'file_id'            AS file_id,
--        cmetadata->>'user_id'            AS namespace,
--        cmetadata->>'chunk_index'        AS chunk_index,
--        count(*)                         AS quantidade,
--        array_agg(uuid)                  AS registros
-- FROM langchain_pg_embedding
-- WHERE cmetadata ? 'chunk_index' AND cmetadata ? 'user_id'
-- GROUP BY 1, 2, 3
-- HAVING count(*) > 1;

-- ---------------------------------------------------------------------------
-- Passo 1: criar o índice único parcial (linhas novas apenas).
-- Fora de transação é possível usar CONCURRENTLY (recomendado em produção;
-- descomente a variante abaixo e comente a variante simples).
-- ---------------------------------------------------------------------------
--
-- Variante CONCURRENTLY (produção — rodar fora de BEGIN/COMMIT):
-- CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_langchain_pg_embedding_chunk_identity
--     ON langchain_pg_embedding (collection_id, custom_id,
--                                (cmetadata->>'user_id'),
--                                (cmetadata->>'chunk_index'))
--     WHERE cmetadata ? 'chunk_index' AND cmetadata ? 'user_id';

CREATE UNIQUE INDEX IF NOT EXISTS uq_langchain_pg_embedding_chunk_identity
    ON langchain_pg_embedding (collection_id, custom_id,
                               (cmetadata->>'user_id'),
                               (cmetadata->>'chunk_index'))
    WHERE cmetadata ? 'chunk_index' AND cmetadata ? 'user_id';

-- ---------------------------------------------------------------------------
-- Verificação pós-criação:
--   SELECT indexname FROM pg_indexes
--    WHERE indexname = 'uq_langchain_pg_embedding_chunk_identity';
--
-- ROLLBACK (âncora):
--   DROP INDEX IF EXISTS uq_langchain_pg_embedding_chunk_identity;
--
-- Nota: com o índice ativo, o /embed idempotente (dedup por digest no
-- document_routes.py) evita a violação na reingestão idêntica; conteúdo
-- ALTERADO passa pelo swap atômico (replace_file_batch: delete escopado por
-- namespace + reinserção numa ÚNICA transação — falha de provider/insert
-- reverte tudo e o lote anterior permanece). Uma corrida de ingestão
-- concorrente vira violação DESTE índice, tratada sem delete por file_id.
-- ============================================================================
