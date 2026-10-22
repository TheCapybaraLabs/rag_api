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
   agrupamento de conteúdo e a exclusão é **somente por uuid físico** — no modo
   `--execute`, **exatamente** os uuids aprovados no manifesto do dry-run
   (o script nunca reagrupa o banco vivo na hora de apagar).
2. **Âncora de rollback antes de excluir**: snapshot das linhas candidatas
   (`CREATE TABLE ... AS SELECT` com embedding completo + metadados) na **mesma
   transação** do delete — ou os dois acontecem, ou nenhum.
3. **Agrupamento (sempre por namespace e posição)**:
   `file_id + user_id + page + chunk_index + md5(document)`.
   * `user_id` no grupo preserva cópias legítimas do mesmo chunk entre
     agentes/namespaces (`--no-by-namespace` existe só para inventário e é
     **recusado com `--execute`**).
   * `chunk_index` no grupo não funde chunks distintos de texto repetido na
     mesma página; linhas legadas sem `chunk_index` agrupam entre si (NULL).
4. **Sobrevivente = linha mais antiga**: menor `ctid` (ordem física de inserção).
   **Não** se usa o menor uuid — uuid v4 é aleatório e preservaria o lote por
   acaso (possivelmente o lixo `provision-/temp`).
5. **Não usar `DELETE /documents`** para este fim: apaga por arquivo e atinge
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
# 0. Inventário / dry-run (default; não altera nada; grava o manifesto)
python scripts/cleanup_duplicate_vectors.py \
    --collection <COLLECTION_NAME> \
    --file-id <file_id_1> --file-id <file_id_2> \
    --manifesto /tmp/card278_manifesto.json

# → imprime grupos duplicados, grava manifesto com uuid "manter"/"remover".
#   (--by-namespace é o DEFAULT; --no-by-namespace só para inventário. --all
#    também é só inventário.)

# 1. REVISE o manifesto (uuids a manter/remover por grupo) e obtenha o OK do
#    dono dos tenants.

# 2. Com OK, aplicar EXATAMENTE o manifesto revisado:
python scripts/cleanup_duplicate_vectors.py \
    --manifesto /tmp/card278_manifesto.json \
    --execute --yes        # sem --yes pede confirmação interativa (digite o total)
```

O que o modo `--execute` faz, em ordem:

1. **Lê os `remover_uuids` do manifesto do dry-run** (exige `--manifesto`;
   recusa `--file-id`/`--all`, que não participam da seleção).
2. Verifica quais desses uuids **ainda existem** (avisa os sumidos; só eles
   diferem da lista aprovada).
3. Numa **única transação**: cria o snapshot `card278_backup_<ts>` com as linhas
   completas (embedding + `cmetadata` + `document`) e exclui **exatamente** os
   uuids verificados. Se o count do delete divergir do snapshot, a transação é
   revertida.
4. Grava um **relatório de execução separado**
   (`<manifesto>.execucao.json`) com `removidos`, `uuids_ausentes`,
   `total_linhas_depois` e o SQL de rollback com a **lista concreta de uuids**.
   O manifesto aprovado **não é sobrescrito**.

## Rollback (âncora)

Todo modo `--execute` grava o SQL exato no relatório de execução; genericamente:

```sql
INSERT INTO langchain_pg_embedding (uuid, collection_id, embedding, document, cmetadata, custom_id)
SELECT uuid, collection_id, embedding, document, cmetadata, custom_id
FROM card278_backup_<ts>
WHERE uuid = ANY('{<uuid1>,<uuid2>,...}'::uuid[]);  -- lista concreta, do relatório
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
