Je vais réorganiser le README pour le rendre plus cohérent et complet. Voici la structure proposée :

# Milvus-Qdrant Benchmark 🔍

A comprehensive toolkit for benchmarking Milvus and Qdrant vector databases using Locust as a load testing tool. The project focuses on comparing performance with hybrid search (dense and sparse vectors) using the BGE-M3 model.

## Features 🌟

- Hybrid search benchmarking (dense + sparse vectors)
- BGE-M3 embeddings generation
- Automated test scenarios
- Detailed performance metrics
- Docker-based deployments

## Quick Start 🚀

### Prerequisites

- Python 3.11
- pip
- pipenv
- Docker and Docker Compose

### Installation

```bash
 

# Install dependencies
task bootstrap
```

### Environment Setup

```bash
# For Milvus
export MILVUS_HOST=localhost
export MILVUS_PORT=19530

# Start databases
task start-milvus  # For Milvus
task start-qdrant  # For Qdrant
```

## Project Structure 📂

```
.
├── data/                      # Data files
│   ├── dataset.csv           # Source dataset
│   ├── questions.txt         # Test queries
│   └── testset.json         # Processed queries
├── tools/                    # Data preparation tools
│   ├── prepare_dataset.py    # Dataset processing
│   └── prepare_custom_query*.py # Query preparation
├── milvus/                   # Milvus configuration
└── qdrant/                   # Qdrant configuration
```

## Data Preparation 🛠️

1. Extract dataset:
```bash
task prepare-data
```

2. Process dataset:
```bash
python tools/prepare_dataset.py
```

3. Prepare queries:
```bash
# For Milvus
python tools/prepare_custom_query-milvus.py generate_testset

# For Qdrant
python tools/prepare_custom_query-qdrant.py generate_testset
```

## Running Benchmarks 📊

1. Start Locust:
```bash
locust -f benchmark_locust_milvus.py --host http://localhost:19530
```

2. Access UI at `http://localhost:8089` and configure:
   - Number of users
   - Spawn rate
   - Host URL

3. Available tests:
   - Dense vector search
   - Sparse vector search
   - Hybrid search
   - Bulk insertions
   - Metadata queries

### Test Execution 🏃

1. Start Locust for Milvus:
```bash
locust -f benchmark_locust_milvus.py --host http://localhost:19530
```

2. Access Locust Web Interface:
   - Navigate to `http://localhost:8089`
   - Configure test parameters:
     - Number of users: simulated user count
     - Spawn rate: user creation rate
     - Host: Milvus instance URL

3. Available Metrics:
   - Response times (min, max, average)
   - Requests per second
   - Error rate
   - Response time distribution

### Test Customization 🎛️

To modify test scenarios:

1. Edit queries in `questions.txt`
2. Adjust parameters in `common.py`:
```python
# common.py
DIMENSION = 1024      # BGE-M3's dense vector dimension
VOCAB_SIZE = 250002   # BGE-M3's vocabulary size
COLLECTION_NAME = "games"
DATABASE_NAME = "gamedb"
```

### Tools Features

- Dataset processing:
  - Chunk size: 500 entries
  - BGE-M3 embeddings
  - Dense and sparse vectors

- Hybrid search weights:
  - Sparse: 0.45
  - Dense: 0.55

## Task Automation 🤖

```bash
task: Available tasks:
* bootstrap:     Setup environment
* start-milvus:  Start Milvus stack
* start-qdrant:  Start Qdrant stack
* prepare-data:  Prepare dataset
* init-db:       Initialize database
```

## Contributing 🤝

Contributions are welcome! Feel free to:
- Open issues
- Submit pull requests
- Improve documentation
- Add new features

## License ⚖️

[License to be defined]