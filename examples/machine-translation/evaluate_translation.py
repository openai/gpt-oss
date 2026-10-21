# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Machine Translation Evaluation Script for GPT-OSS models.

This script evaluates machine translation models using standard metrics like BLEU, COMET, and chrF.
It supports evaluation on various datasets and provides detailed analysis.

Usage:
python evaluate_translation.py \
    --model_path ./gpt-oss-20b-translator \
    --source_lang en \
    --target_lang es \
    --test_file test_data.csv \
    --source_column source \
    --target_column target \
    --output_file evaluation_results.json

# Evaluate on standard datasets
python evaluate_translation.py \
    --model_path ./gpt-oss-20b-translator \
    --source_lang en \
    --target_lang de \
    --dataset_name wmt14 \
    --dataset_config de-en \
    --output_file wmt14_evaluation.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
from generate_translation import TranslationGenerator, create_translation_prompt

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. Install with: pip install sacrebleu")

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    print("Warning: evaluate library not available. Install with: pip install evaluate")


def compute_bleu_score(predictions: List[str], references: List[str]) -> Dict:
    """Compute BLEU score using sacrebleu."""
    if not SACREBLEU_AVAILABLE:
        return {"error": "sacrebleu not available"}
    
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return {
        "bleu": bleu.score,
        "bleu_signature": bleu.signature,
        "bleu_1": bleu.precisions[0],
        "bleu_2": bleu.precisions[1],
        "bleu_3": bleu.precisions[2],
        "bleu_4": bleu.precisions[3],
    }


def compute_chrf_score(predictions: List[str], references: List[str]) -> Dict:
    """Compute chrF score using sacrebleu."""
    if not SACREBLEU_AVAILABLE:
        return {"error": "sacrebleu not available"}
    
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return {
        "chrf": chrf.score,
        "chrf_signature": chrf.signature
    }


def compute_comet_score(predictions: List[str], references: List[str], 
                       sources: List[str]) -> Dict:
    """Compute COMET score using evaluate library."""
    if not EVALUATE_AVAILABLE:
        return {"error": "evaluate library not available"}
    
    try:
        comet = evaluate.load("comet")
        comet_results = comet.compute(
            predictions=predictions,
            references=references,
            sources=sources
        )
        return {
            "comet": comet_results["mean_score"],
            "comet_scores": comet_results["scores"]
        }
    except Exception as e:
        return {"error": f"COMET computation failed: {str(e)}"}


def compute_ter_score(predictions: List[str], references: List[str]) -> Dict:
    """Compute TER score using sacrebleu."""
    if not SACREBLEU_AVAILABLE:
        return {"error": "sacrebleu not available"}
    
    try:
        ter = sacrebleu.corpus_ter(predictions, [references])
        return {
            "ter": ter.score,
            "ter_signature": ter.signature
        }
    except Exception as e:
        return {"error": f"TER computation failed: {str(e)}"}


def compute_length_statistics(predictions: List[str], references: List[str]) -> Dict:
    """Compute length-related statistics."""
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    return {
        "avg_pred_length": np.mean(pred_lengths),
        "avg_ref_length": np.mean(ref_lengths),
        "length_ratio": np.mean(pred_lengths) / np.mean(ref_lengths),
        "length_variance_pred": np.var(pred_lengths),
        "length_variance_ref": np.var(ref_lengths)
    }


def evaluate_translation_model(generator: TranslationGenerator, 
                              sources: List[str], 
                              references: List[str],
                              source_lang: str,
                              target_lang: str,
                              domain: str = "general",
                              batch_size: int = 4,
                              max_new_tokens: int = 512,
                              temperature: float = 0.3) -> Dict:
    """Evaluate translation model on a dataset."""
    
    print(f"Generating translations for {len(sources)} examples...")
    start_time = time.time()
    
    # Generate translations
    predictions = generator.translate_batch(
        texts=sources,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    generation_time = time.time() - start_time
    
    print("Computing evaluation metrics...")
    
    # Compute all metrics
    results = {
        "num_examples": len(sources),
        "generation_time": generation_time,
        "avg_time_per_example": generation_time / len(sources),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "domain": domain
    }
    
    # BLEU score
    bleu_results = compute_bleu_score(predictions, references)
    results.update(bleu_results)
    
    # chrF score
    chrf_results = compute_chrf_score(predictions, references)
    results.update(chrf_results)
    
    # TER score
    ter_results = compute_ter_score(predictions, references)
    results.update(ter_results)
    
    # COMET score
    comet_results = compute_comet_score(predictions, references, sources)
    results.update(comet_results)
    
    # Length statistics
    length_stats = compute_length_statistics(predictions, references)
    results.update(length_stats)
    
    # Store examples for analysis
    results["examples"] = []
    for i in range(min(10, len(sources))):  # Store first 10 examples
        results["examples"].append({
            "source": sources[i],
            "reference": references[i],
            "prediction": predictions[i]
        })
    
    return results


def load_test_data(args) -> tuple:
    """Load test data from various sources."""
    sources, references = [], []
    
    if args.test_file:
        # Load from CSV file
        print(f"Loading test data from {args.test_file}")
        df = pd.read_csv(args.test_file)
        
        if args.source_column not in df.columns:
            raise ValueError(f"Source column '{args.source_column}' not found in file")
        if args.target_column not in df.columns:
            raise ValueError(f"Target column '{args.target_column}' not found in file")
        
        sources = df[args.source_column].tolist()
        references = df[args.target_column].tolist()
        
    elif args.dataset_name:
        # Load from HuggingFace datasets
        print(f"Loading dataset {args.dataset_name}")
        if args.dataset_config:
            dataset = load_dataset(args.dataset_name, args.dataset_config)
        else:
            dataset = load_dataset(args.dataset_name)
        
        test_split = args.dataset_split or "test"
        if test_split not in dataset:
            raise ValueError(f"Split '{test_split}' not found in dataset")
        
        test_data = dataset[test_split]
        
        # Handle different dataset formats
        if 'translation' in test_data.column_names:
            # WMT-style datasets
            for example in test_data:
                sources.append(example['translation'][args.source_lang])
                references.append(example['translation'][args.target_lang])
        elif args.source_column and args.target_column:
            # Custom column names
            sources = test_data[args.source_column]
            references = test_data[args.target_column]
        else:
            raise ValueError("Cannot determine source and target columns")
    
    else:
        raise ValueError("Either --test_file or --dataset_name must be provided")
    
    # Limit number of examples if specified
    if args.max_examples:
        sources = sources[:args.max_examples]
        references = references[:args.max_examples]
    
    print(f"Loaded {len(sources)} test examples")
    return sources, references


def main():
    parser = argparse.ArgumentParser(description="Evaluate Machine Translation Model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned translation model")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run inference on")
    
    # Data arguments
    parser.add_argument("--test_file", type=str, default=None,
                       help="CSV file with test data")
    parser.add_argument("--source_column", type=str, default="source",
                       help="Name of source text column")
    parser.add_argument("--target_column", type=str, default="target",
                       help="Name of target text column")
    
    # Dataset arguments (alternative to test_file)
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Dataset configuration")
    parser.add_argument("--dataset_split", type=str, default="test",
                       help="Dataset split to evaluate on")
    
    # Translation arguments
    parser.add_argument("--source_lang", type=str, required=True,
                       help="Source language code")
    parser.add_argument("--target_lang", type=str, required=True,
                       help="Target language code")
    parser.add_argument("--domain", type=str, default="general",
                       help="Translation domain")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Generation temperature")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for generation")
    
    # Evaluation arguments
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to evaluate")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file for evaluation results")
    
    args = parser.parse_args()
    
    # Load test data
    sources, references = load_test_data(args)
    
    # Initialize translation generator
    print(f"Loading model from {args.model_path}")
    generator = TranslationGenerator(args.model_path, args.device)
    
    # Run evaluation
    print(f"Evaluating translation quality...")
    results = evaluate_translation_model(
        generator=generator,
        sources=sources,
        references=references,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        domain=args.domain,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Add metadata
    results["model_path"] = args.model_path
    results["model_metadata"] = generator.metadata
    results["evaluation_args"] = vars(args)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to {args.output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Language pair: {args.source_lang} -> {args.target_lang}")
    print(f"Number of examples: {results['num_examples']}")
    print(f"Generation time: {results['generation_time']:.2f}s")
    print(f"Average time per example: {results['avg_time_per_example']:.3f}s")
    print()
    
    if 'bleu' in results:
        print(f"BLEU score: {results['bleu']:.2f}")
    if 'chrf' in results:
        print(f"chrF score: {results['chrf']:.2f}")
    if 'ter' in results:
        print(f"TER score: {results['ter']:.2f}")
    if 'comet' in results:
        print(f"COMET score: {results['comet']:.4f}")
    
    print(f"\nLength statistics:")
    print(f"  Average prediction length: {results['avg_pred_length']:.1f} tokens")
    print(f"  Average reference length: {results['avg_ref_length']:.1f} tokens")
    print(f"  Length ratio: {results['length_ratio']:.3f}")
    
    print("\nSample translations:")
    for i, example in enumerate(results['examples'][:3], 1):
        print(f"\n{i}. Source: {example['source']}")
        print(f"   Reference: {example['reference']}")
        print(f"   Prediction: {example['prediction']}")


if __name__ == "__main__":
    main() 