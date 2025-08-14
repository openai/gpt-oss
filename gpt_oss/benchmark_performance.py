#!/usr/bin/env python3
"""
Performance benchmarking script for GPT-OSS optimizations.

This script helps measure the impact of various optimizations on inference performance.
"""

import argparse
import time
import torch
import statistics
from typing import Dict, List, Tuple
import json
import os

from gpt_oss.tokenizer import get_tokenizer


class PerformanceBenchmark:
    """Benchmark class for measuring GPT-OSS performance optimizations."""
    
    def __init__(self, model_path: str, backend: str = "triton"):
        self.model_path = model_path
        self.backend = backend
        self.tokenizer = get_tokenizer()
        self.results = {}
        
    def setup_model(self):
        """Initialize the model based on the selected backend."""
        print(f"Setting up {self.backend} backend...")
        
        if self.backend == "triton":
            from gpt_oss.triton.model import TokenGenerator
            self.model = TokenGenerator(self.model_path, context=4096)
        elif self.backend == "torch":
            from gpt_oss.torch.model import TokenGenerator
            self.model = TokenGenerator(self.model_path)
        elif self.backend == "metal":
            from gpt_oss.metal import Context, Model
            model = Model(self.model_path)
            self.model = Context(model)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
            
    def benchmark_generation(self, prompt: str, max_tokens: int = 100, num_runs: int = 5) -> Dict:
        """Benchmark text generation performance."""
        print(f"Benchmarking generation with {num_runs} runs...")
        
        tokens = self.tokenizer.encode(prompt)
        latencies = []
        throughputs = []
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            
            if self.backend == "metal":
                # Metal backend has different API
                self.model.reset()
                for token in tokens:
                    self.model.append(token)
                self.model.process()
                
                generated_tokens = []
                for _ in range(max_tokens):
                    token = int(self.model.sample(temperature=0.0))
                    generated_tokens.append(token)
                    self.model.append(token)
                    self.model.process()
                    
            else:
                # PyTorch/Triton backends
                generated_tokens = []
                for token, _ in self.model.generate(tokens, max_tokens=max_tokens, temperature=0.0):
                    generated_tokens.append(token)
                    if len(generated_tokens) >= max_tokens:
                        break
                        
            end_time = time.perf_counter()
            latency = end_time - start_time
            throughput = len(generated_tokens) / latency
            
            latencies.append(latency)
            throughputs.append(throughput)
            
            print(f"Run {i+1}: {latency:.3f}s, {throughput:.1f} tokens/s")
            
        return {
            "latency": {
                "mean": statistics.mean(latencies),
                "std": statistics.stdev(latencies),
                "min": min(latencies),
                "max": max(latencies)
            },
            "throughput": {
                "mean": statistics.mean(throughputs),
                "std": statistics.stdev(throughputs),
                "min": min(throughputs),
                "max": max(throughputs)
            },
            "total_tokens": len(generated_tokens)
        }
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage."""
        print("Benchmarking memory usage...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run a small generation to measure memory
            tokens = self.tokenizer.encode("Hello, world!")
            if self.backend != "metal":
                for token, _ in self.model.generate(tokens, max_tokens=10, temperature=0.0):
                    pass
                    
            max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            return {"gpu_memory_gb": max_memory}
        else:
            return {"gpu_memory_gb": 0}
    
    def benchmark_batch_processing(self, batch_sizes: List[int] = [1, 2, 4, 8]) -> Dict:
        """Benchmark batch processing performance."""
        print("Benchmarking batch processing...")
        
        results = {}
        prompt = "The quick brown fox jumps over the lazy dog."
        tokens = self.tokenizer.encode(prompt)
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            latencies = []
            
            for _ in range(3):  # 3 runs per batch size
                start_time = time.perf_counter()
                
                if self.backend != "metal":
                    # Process multiple sequences
                    for _ in range(batch_size):
                        for token, _ in self.model.generate(tokens, max_tokens=50, temperature=0.0):
                            pass
                            
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)
                
            avg_latency = statistics.mean(latencies)
            throughput = (batch_size * 50) / avg_latency  # 50 tokens per sequence
            
            results[f"batch_{batch_size}"] = {
                "latency": avg_latency,
                "throughput": throughput
            }
            
        return results
    
    def run_full_benchmark(self, output_file: str = None) -> Dict:
        """Run a comprehensive performance benchmark."""
        print("Starting comprehensive performance benchmark...")
        
        self.setup_model()
        
        # Test prompts of different lengths
        test_prompts = [
            "Hello",
            "The quick brown fox jumps over the lazy dog.",
            "In a world where artificial intelligence continues to advance rapidly, we find ourselves at the intersection of human creativity and machine learning capabilities.",
        ]
        
        self.results = {
            "backend": self.backend,
            "model_path": self.model_path,
            "generation_benchmarks": {},
            "memory_usage": self.benchmark_memory_usage(),
            "batch_processing": self.benchmark_batch_processing()
        }
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nBenchmarking prompt {i+1}: {prompt[:50]}...")
            self.results["generation_benchmarks"][f"prompt_{i+1}"] = self.benchmark_generation(
                prompt, max_tokens=100, num_runs=3
            )
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        
        return self.results
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            print("No benchmark results available. Run benchmark first.")
            return
            
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        print(f"Backend: {self.results['backend']}")
        print(f"Model: {self.results['model_path']}")
        
        print("\nGeneration Performance:")
        for prompt_name, results in self.results["generation_benchmarks"].items():
            print(f"  {prompt_name}:")
            print(f"    Latency: {results['latency']['mean']:.3f}s ± {results['latency']['std']:.3f}s")
            print(f"    Throughput: {results['throughput']['mean']:.1f} tokens/s")
        
        print(f"\nMemory Usage: {self.results['memory_usage']['gpu_memory_gb']:.2f} GB")
        
        print("\nBatch Processing:")
        for batch_name, results in self.results["batch_processing"].items():
            print(f"  {batch_name}: {results['throughput']:.1f} tokens/s")


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Performance Benchmark")
    parser.add_argument("model_path", help="Path to the model checkpoint")
    parser.add_argument("--backend", choices=["triton", "torch", "metal"], default="triton",
                       help="Inference backend to benchmark")
    parser.add_argument("--output", help="Output file for benchmark results")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark(args.model_path, args.backend)
    
    # Run benchmark
    results = benchmark.run_full_benchmark(args.output)
    
    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
