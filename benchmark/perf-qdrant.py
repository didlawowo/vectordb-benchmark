import time
import statistics
from qdrant_client import QdrantClient
import psutil

from typing import List, Dict
from loguru import logger


class QdrantBenchmark:
    def __init__(self, host: str, port: int = 6333):
        """Initialize benchmark class with Qdrant connection"""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "ultrachat200k"

    def measure_search_latency(
        self, queries: List[str], k: int = 10, num_iterations: int = 100
    ) -> Dict:
        """Measure search latency statistics"""
        latencies = []

        for query in queries:
            for _ in range(num_iterations):
                start_time = time.time()

                # Perform search
                self.client.search(
                    collection_name=self.collection_name, query_vector=query, limit=k
                )

                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)

        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[
                18
            ],  # 95th percentile
            "p99_latency_ms": statistics.quantiles(latencies, n=100)[
                98
            ],  # 99th percentile
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        }

    def measure_throughput(self, queries: List[str], duration: int = 60) -> Dict:
        """Measure queries per second over a fixed duration"""
        start_time = time.time()
        query_count = 0

        while (time.time() - start_time) < duration:
            for query in queries:
                self.client.search(
                    collection_name=self.collection_name, query_vector=query, limit=10
                )
                query_count += 1

        total_time = time.time() - start_time
        qps = query_count / total_time

        return {
            "queries_per_second": qps,
            "total_queries": query_count,
            "duration_seconds": total_time,
        }

    def measure_resource_usage(self, queries: List[str], duration: int = 60) -> Dict:
        """Monitor CPU and memory usage during search operations"""
        cpu_percent = []
        memory_percent = []

        start_time = time.time()

        while (time.time() - start_time) < duration:
            for query in queries:
                self.client.search(
                    collection_name=self.collection_name, query_vector=query, limit=10
                )

                cpu_percent.append(psutil.cpu_percent())
                memory_percent.append(psutil.Process().memory_percent())

        return {
            "avg_cpu_percent": statistics.mean(cpu_percent),
            "max_cpu_percent": max(cpu_percent),
            "avg_memory_percent": statistics.mean(memory_percent),
            "max_memory_percent": max(memory_percent),
        }

    def run_comprehensive_benchmark(self, test_queries: List[str]) -> Dict:
        """Run all benchmark tests and return comprehensive results"""
        results = {}

        logger.info("Starting latency measurements...")
        results["latency"] = self.measure_search_latency(test_queries)

        logger.info("Starting throughput measurements...")
        results["throughput"] = self.measure_throughput(test_queries)

        logger.info("Starting resource usage measurements...")
        results["resource_usage"] = self.measure_resource_usage(test_queries)

        return results


def main():
    # Sample test queries - replace with actual queries from your dataset
    test_queries = [
        "How does science education impact students?",
        "What are the benefits of tracking system in education?",
        "Explain the concept of science alienation",
    ]

    benchmark = QdrantBenchmark(host="localhost")
    results = benchmark.run_comprehensive_benchmark(test_queries)

    # Print results
    print("\n=== Benchmark Results ===")
    print("\nLatency Metrics:")
    for key, value in results["latency"].items():
        print(f"{key}: {value:.2f}ms")

    print("\nThroughput Metrics:")
    for key, value in results["throughput"].items():
        print(f"{key}: {value:.2f}")

    print("\nResource Usage Metrics:")
    for key, value in results["resource_usage"].items():
        print(f"{key}: {value:.2f}%")


if __name__ == "__main__":
    main()
