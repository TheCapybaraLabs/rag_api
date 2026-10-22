# Card #278 — Limpeza de vetores duplicados (pgvector)

> ⚠️ **A LIMPEZA ESPERA O OK DO DONO DOS TENANTS. Nada aqui roda sozinho: o
> script é ferramenta de auditoria/preparação e exige comando explícito com
> credenciais do ambiente.** Nenhum passo deste documento é executado por
> deploy, CI ou inicialização do serviço.

## Contexto (causa raiz)

Antes do fix de dedup/UNIQUE desta PR, a reingestão de um arquivo (reindexação,
retry, callback de provision) reenviava o conteúdo ao `/embed`, que inseria um
lote novo sem verificar existência — linhas duplicadas em
`langchain_pg_embedding` para o mesmo `file_id`,namespace,chunk. O nome
`provision-<uuid>` nos metadados é sintoma de um dos lotes (veio do caminho
temporário do provisionador), mas **não é critério de limpeza**.

## Regras (card #278)

1. **Nunca deletar por prefixo** `provision-` ou por nome. A identificação é por
   agrupamento de conteúdo e a exclusão é **somente por uuid físico** listado no
   manifesto gerado na mesma execução.
2. **Âncora de rollback antes de excluir**: snapshot das linhas candidatas
   (`CREATE TABLE ... AS SELECT` com embedding completo + metadados), contagens
   antes/depois e manifesto JSON.
3. **Agrupamento**: `file_id + page + md5(document)`; com `--by-namespace`,
   `file_id + user_id + page + md5(document)` — separa repetição dentro do mesmo
   namespace (duplicata efetiva) de cópias legítimas entre agentes (preservar).
4. **Não usar `DELETE /documents`** para este fim: apaga por arquivo e atinge
   todos os namespaces.

## Confirmação de produção (backend/versão)

* Backend: **pgvector** (`VECTOR_DB_TYPE` default; compose do labschat não define
  Atlas). Confirme por tenant antes de agir: `VECTOR_DB_TYPE`, `POSTGRES_SCHEMA`,
  `COLLECTION_NAME`.
* Versão da imagem (labschat-ansible, roles/labschat): **pgvector/pgvector:0.8.0-pg15-trixie**
  → PostgreSQL 15, pgvector 0.8.0. Se algum tenant estiver em Atlas Mongo, este
  procedimento **não se aplica** (identidade lá é `file_id_digest` — comportamento
  diferente; auditar à parte).

## Procedimento

```bash
# 0. Inventário somente-leitura em TODA coleção alvo (dry-run é o default)
python scripts/cleanup_duplicate_vectors.py \
    --collection <COLLECTION_NAME> \
    --file-id <file_id_1> --file-id <file_id_2> \
    --by-namespace \
    --manifesto /tmp/card278_manifesto.json

# → imprime grupos duplicados, grava manifesto com uuids "manter"/"remover".

# 1. Com OK DO DONO dos tenants, aplicar o manifesto:
python scripts/cleanup_duplicate_vectors.py \
    --collection <COLLECTION_NAME> \
    --file-id <file_id_1> --file-id <file_id_2> \
    --by-namespace \
    --manifesto /tmp/card278_manifesto.json \
    --execute            # pede confirmação interativa (digite o total exato)
```

O que o modo `--execute` faz, em ordem:

1. Recalcula os grupos no estado atual do banco.
2. Cria snapshot `card278_backup_<ts>` com as linhas candidatas completas
   (embedding + `cmetadata` + `document`).
3. Exclui **exclusivamente** os uuids `remover_uuids` do manifesto.
4. Reconta e grava `removidos`, `total_linhas_depois` e o SQL de rollback no
   próprio manifesto.

## Rollback (âncora)

Todo modo `--execute` grava no manifesto o SQL exato; genericamente:

```sql
INSERT INTO langchain_pg_embedding (uuid, collection_id, embedding, document, cmetadata, custom_id)
SELECT uuid, collection_id, embedding, document, cmetadata, custom_id
FROM card278_backup_<ts>
WHERE uuid = ANY(<removidos_do_manifesto>);
```

Validar a restauração em ambiente isolado antes de qualquer execução real
(duplicar o snapshot em um banco de teste e conferir contagens por
`file_id+namespace`).

## Depois da limpeza

* Aplicar `migrations/2026-10-22_card278_unique_chunk_identity.sql` (índice
  único parcial) — só depois do ok do dono e com o pré-voo zerado (passo 0 do
  SQL).
* Reindexar um arquivo de teste e conferir que a contagem **não muda**
  (idempotência do `/embed` corrigido nesta PR).
* Se lotes antigos tiverem `source` = caminho temporário, o nome exibido em
  citações novas vem de `metadata.filename`/cadastro Mongo — limpar não reescreve
  metadados de linhas sobreviventes (o LabsChat atual normaliza pelo cadastro;
  ver PR do fork).
