# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vector database benchmarking toolkit comparing **Milvus**, **Qdrant**, and **pgvector** performance using hybrid search (dense + sparse vectors) with the BGE-M3 embedding model. Load testing is done via Locust. This is a proof-of-concept, not production-ready.

## Commands

### Task Runner (primary automation)
```bash
task bootstrap          # Install pipenv deps + requirements
task start-milvus       # Start Milvus Docker stack (etcd, minio, milvus, attu)
task start-qdrant       # Start Qdrant Docker stack
task start-pgvector     # Start pgvector Docker stack (PostgreSQL 17)
task start-openlit      # Start OpenLit observability stack
task stop-milvus        # Stop Milvus stack
task stop-qdrant        # Stop Qdrant stack
task stop-pgvector      # Stop pgvector stack
task init-db-milvus     # Load embeddings into Milvus
task init-db-qdrant     # Load embeddings into Qdrant
task init-db-pgvector   # Load embeddings into pgvector
task extract-gz-data    # Extract compressed videogame data
task download-hf-ds     # Download HuggingFace dataset
```

### Running Benchmarks
```bash
locust -f benchmark_locust_milvus.py --host http://localhost:19530
locust -f benchmark_locust_qdrant.py --host http://localhost:6333
locust -f benchmark_locust_pgvector.py --host http://localhost:5432
# Locust UI at http://localhost:8089
```

### Linting
```bash
ruff check .            # Lint
ruff format .           # Format
```

Ruff config (`ruff.toml`): ignores F722/F821, enables import sorting (I).

### Python Environment
- Python 3.11+ required (Pipfile specifies 3.12.8)
- Uses pipenv: `task shell` to activate

## Architecture

### Pipeline Flow
```
1. Prepare dataset    →  prepare_dataset_videogame.py / prepare_dataset_hf.py
2. Load into DB       →  prepare_db_milvus.py / prepare_db_qdrant.py / prepare_db_pgvector.py
3. Generate queries   →  prepare_custom_query-milvus.py / prepare_custom_query-qdrant.py (testset.json is DB-agnostic)
4. Run benchmarks     →  benchmark_locust_milvus.py / benchmark_locust_qdrant.py / benchmark_locust_pgvector.py
```

### Key Constants (`common.py`)
All scripts import shared config from `common.py`:
- `DIMENSION = 1024` (BGE-M3 dense vector size)
- `VOCAB_SIZE = 250002` (BGE-M3 vocabulary)
- `COLLECTION_NAME = "games"`, `DATABASE_NAME = "gamedb"`
- Data paths: `data/videogame/` for datasets, `data/testset.json` for queries

### Hybrid Search Weights
Sparse: 0.45, Dense: 0.55 — hardcoded in benchmark scripts.

### Embedding Model
BGE-M3 (`BAAI/bge-m3`) via FlagEmbedding. Auto-detects device (CUDA > MPS > CPU). Produces both dense (1024-dim) and sparse (CSR format) vectors.

### Database Differences
- **Milvus**: Batch size 10,000. Uses SPARSE_INVERTED_INDEX (IP) + FLAT (COSINE). Schema with typed fields.
- **Qdrant**: Batch size 10. Uses VectorParams with COSINE distance. Sparse vectors stored as payload. Has retry logic with exponential backoff.
- **pgvector**: Batch size 1,000. PostgreSQL + pgvector extension. HNSW index with cosine ops. Sparse vectors in JSONB column. Dense search via SQL + sparse reranking in Python (like Qdrant). Uses asyncpg client.

### Docker Stacks
- `milvus/docker-compose.yaml` — etcd + minio + milvus standalone + attu UI (ports 19530, 3002)
- `qdrant/docker-compose.yaml` — qdrant (ports 6333, 6334)
- `pgvector/docker-compose.yaml` — PostgreSQL 17 + pgvector (port 5432)
- `openlit/docker-compose.yml` — clickhouse + openlit + otel-collector (port 3001)

### Environment Variables
```bash
MILVUS_HOST=localhost  MILVUS_PORT=19530
QDRANT_HOST=localhost  QDRANT_PORT=6333
PG_HOST=localhost  PG_PORT=5432  PG_USER=benchmark  PG_PASSWORD=benchmark  PG_DATABASE=benchmark
```

## Data Files (Git LFS)
Large files in `data/` are tracked by Git LFS (`.gitattributes`). Run `git lfs pull` after cloning to fetch them.
