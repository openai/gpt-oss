#!/usr/bin/env python3
"""
Run AIME evaluation using Fireworks AI API.
"""
import argparse
import json
import os
from datetime import datetime

from fireworks_sampler import FireworksSampler
from gpt_oss.evals.aime_eval import AIME25Eval


def main():
    parser = argparse.ArgumentParser(
        description="Run AIME evaluation with Fireworks AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/gpt-oss-120b",
        help="Fireworks model to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Number of threads to run.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with fewer examples"
    )
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--output", type=str, help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--reasoning-effort", 
        type=str, 
        choices=["low", "medium", "high"],
        help="Reasoning effort level for gpt-oss models"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY environment variable is required")
        return 1

    print(f"Running AIME evaluation with args: {args}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")

    # Create the sampler
    sampler = FireworksSampler(
        model=args.model,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        max_tokens=65536,
    )

    # Create the evaluation
    num_examples = None
    if args.debug:
        num_examples = 5
    elif args.examples:
        num_examples = args.examples

    eval_instance = AIME25Eval(
        n_repeats=1 if num_examples else 4,  # Use 1 repeat when limiting examples
        num_examples=num_examples,
        n_threads=args.n_threads,
    )

    print(f"Running evaluation with {len(eval_instance.examples)} examples...")
    
    # Run the evaluation
    start_time = datetime.now()
    result = eval_instance(sampler)
    end_time = datetime.now()
    
    duration = end_time - start_time
    print(f"\nEvaluation completed in {duration}")
    print(f"Score: {result.score}")
    if result.metrics:
        print(f"Metrics: {result.metrics}")

    # Save results if output file specified
    if args.output:
        output_data = {
            "model": args.model,
            "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort,
            "score": result.score,
            "metrics": result.metrics,
            "num_examples": len(eval_instance.examples),
            "duration_seconds": duration.total_seconds(),
            "timestamp": start_time.isoformat(),
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())