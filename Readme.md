# Lambot
A chatbot written using [rust-bert](https://github.com/guillaume-be/rust-bert) and openai models. 

## TODO
- [x] Knowledge graph from plain text using [pg_embeddings](https://neon.tech/blog/pg-embedding-extension-for-vector-search)
- [x] Pdf to text to Knowledge graph
- [x] Audio to text to Knowledge graph
- [x] Chat history
- [x] CLI
- [x] Websocket interface 
- [ ] Refactor utils 

### How to setup
#### Required env variables
```bash
export INDEX_PATH= # path to keep track of already indexed files
export DATA_DIR= # path to a directory containing files to index
export PG_URI= # postgres connection string
export OPEN_AI_TOKEN= # openai api token
```
#### Setup Table
Using [Neon](https://neon.tech/ai)

```sql
CREATE TABLE documents (id BIGSERIAL PRIMARY KEY, embedding real[], raw TEXT, doc_ref TEXT, segment bigint);
CREATE INDEX ON documents USING hnsw(embedding) WITH (dims=384);
SET enable_seqscan = off;
```

#### Setup libtorch and rustbert
rust-bert [getting started](https://github.com/guillaume-be/rust-bert#getting-started)\
Model for embedding [AllMiniLmL6V2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
