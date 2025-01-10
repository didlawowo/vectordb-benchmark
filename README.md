# Milvus-Qdrant Benchmark 🔍

A comprehensive toolkit and collection of scripts for performing comparative benchmarks between Milvus and Qdrant vector databases using Locust as a load testing tool.

## Project Structure 📂

### Data Files (`/data`)

- `dataset.csv`: Source dataset
- `questions.txt`: Test query list
- `testset.json`: Formatted test dataset

### Generate Dataset from Archive 📦

```shell
# 💾 Decompress the archive file
gzcat data/output_dataset.jsonl.gz > data/output_dataset.jsonl
```

### Milvus Configuration (`/milvus-local`) 🛠️

- `docker-compose.yaml`: Docker configuration for local Milvus deployment

### Preparation Scripts 📝

- `prepare_custom_query.py`: Custom query generation (converts questions.txt to testset.json)
- `prepare_dataset.py`: Dataset preparation and transformation (converts csv to jsonl)
- `prepare_milvus_db.py`: Milvus database initialization and setup
- `benchmark_locust_milvus.py`: Locust benchmark script for Milvus

### Project Configuration 🔧

- `Pipfile` & `Pipfile.lock`: Python dependency management using Pipenv
- `pyproject.toml`: Python project configuration
- `common.py`: Shared constants and utility functions

## Installation Guide 🚀

### Prerequisites

- Python 3.11
- pip
- pipenv

---

1. Install Dependencies:
```bash
pipenv install
```

## Milvus Setup ⚙️

### Starting Milvus with Docker Compose

1. Launch Milvus Services:
```bash
cd milvus-local
docker-compose up -d
```

2. Verify Service Status:
```bash
docker-compose ps
```

3. Initialize Database:

### Environment Variables Configuration 🌍

```bash
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
```

```bash
python prepare_milvus_db.py init
```

## Performance Testing with Locust 📊

### Locust Configuration

1. Test Script Structure (`benchmark_locust_milvus.py`):
```python
from locust import HttpUser, task, between

class MilvusUser(HttpUser):
    wait_time = between(1, 2)
    
    @task
    def search_vectors(self):
        # Test task definitions
```

2. Implemented Test Types:
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
DIMENSION = 768  # Vector dimensions
COLLECTION_NAME = "benchmark_collection"
DATABASE_NAME = "benchmark_db"
```

3. Modify load patterns in `benchmark_locust_milvus.py`:
```python
@task(3)  # Relative task weight
def search_vectors(self):
    # Search configuration
```

### Results Analysis 📈

Locust generates detailed reports in HTML and CSV formats, including:
- Real-time performance graphs
- Per-endpoint statistics
- Response time distribution
- Error logs

Reports are saved in the `locust-reports/` directory after each test session.

## Contributing 🤝

Contributions are welcome! Feel free to open an issue or submit a pull request to improve benchmarks or add new tests.

## License ⚖️

[License to be defined]