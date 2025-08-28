"""
CLI entry point for the GPT-OSS Hallucination Monitor.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .halluci_monitor import HallucinationMonitor
from .config import MonitorConfig, MonitorThresholds


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def read_file(file_path: str) -> str:
    """Read content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)


def read_context_files(context_paths: List[str]) -> List[str]:
    """Read multiple context files."""
    contexts = []
    for path in context_paths:
        try:
            content = read_file(path)
            contexts.append(content)
        except Exception as e:
            print(f"Warning: Could not read context file {path}: {e}")
    return contexts


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="GPT-OSS Hallucination Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  gpt-oss-monitor --prompt prompt.txt --completion output.txt

  # With context documents
  gpt-oss-monitor --prompt prompt.txt --completion output.txt --contexts ctx1.txt ctx2.txt

  # Custom configuration
  gpt-oss-monitor --prompt prompt.txt --completion output.txt --k 10 --temperature 0.8

  # Generate HTML report
  gpt-oss-monitor --prompt prompt.txt --completion output.txt --html --report-dir reports/
        """
    )
    
    # Input/output arguments
    parser.add_argument('--prompt', required=True, help='Path to prompt file')
    parser.add_argument('--completion', required=True, help='Path to completion file')
    parser.add_argument('--contexts', nargs='*', help='Paths to context document files')
    
    # Configuration arguments
    parser.add_argument('--k', type=int, default=5, help='Number of samples for self-consistency (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation (default: 0.7)')
    parser.add_argument('--max-new-tokens', type=int, default=512, help='Max new tokens for generation (default: 512)')
    
    # Threshold arguments
    parser.add_argument('--high-threshold', type=float, default=0.7, help='High risk threshold (default: 0.7)')
    parser.add_argument('--medium-threshold', type=float, default=0.4, help='Medium risk threshold (default: 0.4)')
    
    # Feature toggles
    parser.add_argument('--no-retrieval-support', action='store_true', help='Disable retrieval support detection')
    parser.add_argument('--no-jailbreak-heuristics', action='store_true', help='Disable jailbreak heuristics detection')
    
    # Output arguments
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--report-dir', default='runs', help='Directory for HTML reports (default: runs)')
    parser.add_argument('--output', help='Path to save JSON results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Read input files
        logger.info("Reading input files...")
        prompt = read_file(args.prompt)
        completion = read_file(args.completion)
        context_docs = read_context_files(args.contexts) if args.contexts else None
        
        # Create configuration
        config = MonitorConfig(
            k_samples=args.k,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            enable_retrieval_support=not args.no_retrieval_support,
            enable_jailbreak_heuristics=not args.no_jailbreak_heuristics,
            thresholds=MonitorThresholds(
                high=args.high_threshold,
                medium=args.medium_threshold
            ),
            html_report=args.html,
            report_dir=args.report_dir
        )
        
        # Initialize monitor
        logger.info("Initializing Hallucination Monitor...")
        monitor = HallucinationMonitor(config)
        
        # Run evaluation
        logger.info("Running evaluation...")
        results = monitor.evaluate(prompt, completion, context_docs)
        
        # Print summary
        print("\n" + "="*60)
        print("GPT-OSS HALLUCINATION MONITOR RESULTS")
        print("="*60)
        print(f"Risk Score: {results['risk_score']:.3f}")
        print(f"Risk Level: {results['risk_level'].upper()}")
        print("\nSignal Scores:")
        for signal_name, score in results['signals'].items():
            print(f"  {signal_name.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\nSpans with Issues: {len(results['spans'])}")
        for span in results['spans']:
            print(f"  - {span['text'][:50]}... ({span['reason']})")
        
        if results['artifacts'].get('html_report'):
            print(f"\nHTML Report: {results['artifacts']['html_report']}")
        
        # Save JSON results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nJSON results saved to: {args.output}")
        
        # Print suggestions for high-risk cases
        if results['risk_level'] == 'high':
            print("\n" + "="*60)
            print("HIGH RISK DETECTED - SUGGESTIONS")
            print("="*60)
            
            suggestions = monitor.suggest_cautious_decoding('high')
            print("Suggested decoding parameters:")
            for param, value in suggestions.items():
                print(f"  {param}: {value}")
            
            print("\nCitation request template:")
            print(monitor.ask_for_citations('high'))
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
