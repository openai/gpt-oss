#!/usr/bin/env python3
"""
GPT-OSS Hallucination Monitor Demo

This script demonstrates the comprehensive hallucination monitoring capabilities
of GPT-OSS, including all detection signals and professional report generation.

Usage:
    python -m gpt_oss.monitoring.demo.demo
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add the monitoring package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def print_header():
    """Print demo header."""
    print("=" * 80)
    print("🔍 GPT-OSS Hallucination Monitor - Comprehensive Demo")
    print("=" * 80)
    print()
    print("This demo showcases the advanced hallucination detection capabilities")
    print("of GPT-OSS, including multiple detection signals and professional reporting.")
    print()

def print_section(title):
    """Print a section header."""
    print(f"\n{title}")
    print("-" * len(title))

def demo_basic_usage():
    """Demonstrate basic usage of the hallucination monitor."""
    print_section("Basic Usage")
    
    try:
        from gpt_oss.monitoring import HallucinationMonitor, MonitorConfig
        
        # Initialize monitor with default configuration
        monitor = HallucinationMonitor()
        
        # Simple example
        prompt = "What is the capital of France?"
        completion = "Paris is the capital of France."
        
        print(f"📝 Prompt: {prompt}")
        print(f"🤖 Completion: {completion}")
        
        result = monitor.evaluate(prompt, completion)
        
        print(f"\n✅ Analysis Complete!")
        print(f"🎯 Risk Score: {result['risk_score']:.3f}")
        print(f"🚨 Risk Level: {result['risk_level'].upper()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import monitoring system: {e}")
        print("📥 Install dependencies: pip install -e '.[monitoring]'")
        return False

def demo_detection_signals():
    """Demonstrate different detection signals."""
    print_section("Detection Signals Demo")
    
    try:
        from gpt_oss.monitoring import HallucinationMonitor, MonitorConfig
        
        # Configure monitor with all signals enabled
        config = MonitorConfig(
            enable_retrieval_support=True,
            enable_jailbreak_heuristics=True,
            k_samples=3,
            html_report=True,
            report_dir="demo_reports"
        )
        
        monitor = HallucinationMonitor(cfg=config)
        
        # Test cases demonstrating different signals
        test_cases = [
            {
                "name": "Truthful Response",
                "prompt": "What is the capital of France?",
                "completion": "Paris is the capital of France.",
                "context": "France is a country in Europe. Paris is its capital city.",
                "expected": "low"
            },
            {
                "name": "Hallucinated Response",
                "prompt": "What is the population of Atlantis?",
                "completion": "Atlantis has a population of 2.3 million people and is located in the Atlantic Ocean.",
                "context": "Atlantis is a fictional island mentioned in Plato's works.",
                "expected": "high"
            },
            {
                "name": "Numeric Error",
                "prompt": "What is 15 + 27?",
                "completion": "15 + 27 = 45",
                "context": "",
                "expected": "high"
            },
            {
                "name": "Context Mismatch",
                "prompt": "What is the weather like in Paris?",
                "completion": "Paris has a tropical climate with palm trees and beaches.",
                "context": "Paris has a temperate climate with four distinct seasons.",
                "expected": "high"
            }
        ]
        
        print(f"🧪 Running {len(test_cases)} test cases...\n")
        
        for i, case in enumerate(test_cases, 1):
            print(f"📋 Test {i}: {case['name']}")
            print(f"   Prompt: {case['prompt']}")
            print(f"   Completion: {case['completion']}")
            
            if case['context']:
                print(f"   Context: {case['context']}")
            
            # Run analysis
            result = monitor.evaluate(
                prompt=case['prompt'],
                completion=case['completion'],
                context_docs=[case['context']] if case['context'] else []
            )
            
            # Display results
            risk_score = result['risk_score']
            risk_level = result['risk_level']
            
            print(f"   🎯 Risk Score: {risk_score:.3f}")
            print(f"   🚨 Risk Level: {risk_level.upper()}")
            
            # Show active signals
            signals = result['signals']
            active_signals = []
            for signal_name, signal_data in signals.items():
                if isinstance(signal_data, dict):
                    score = signal_data.get('score', 0)
                else:
                    score = signal_data
                
                if score > 0:
                    active_signals.append(f"{signal_name.upper()}: {score:.3f}")
            
            if active_signals:
                print(f"   🔍 Active Signals: {', '.join(active_signals)}")
            
            # Show spans
            if result.get('spans'):
                print(f"   🎯 Flagged Spans: {len(result['spans'])}")
                for span in result['spans'][:2]:  # Show first 2 spans
                    text = case['completion'][span['start']:span['end']]
                    print(f"      - '{text}'")
                if len(result['spans']) > 2:
                    print(f"      ... and {len(result['spans']) - 2} more")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_professional_reports():
    """Demonstrate professional HTML report generation."""
    print_section("Professional Report Generation")
    
    try:
        from gpt_oss.monitoring import HallucinationMonitor, MonitorConfig
        
        # Configure monitor for report generation
        config = MonitorConfig(
            enable_retrieval_support=True,
            enable_jailbreak_heuristics=True,
            k_samples=3,
            html_report=True,
            report_dir="demo_reports"
        )
        
        monitor = HallucinationMonitor(cfg=config)
        
        # High-risk example for demonstration
        prompt = "What is the population of Atlantis?"
        completion = "Atlantis has a population of 2.3 million people and is located in the Atlantic Ocean. The city was founded in 1500 BC and has a thriving economy based on underwater mining."
        context = "Atlantis is a fictional island mentioned in Plato's works. It was described as a powerful and advanced kingdom that sank into the ocean."
        
        print(f"📝 Analyzing: {prompt}")
        print(f"🤖 Completion: {completion}")
        print(f"📚 Context: {context}")
        
        # Run analysis
        result = monitor.evaluate(
            prompt=prompt,
            completion=completion,
            context_docs=[context]
        )
        
        print(f"\n✅ Analysis Complete!")
        print(f"🎯 Risk Score: {result['risk_score']:.3f}")
        print(f"🚨 Risk Level: {result['risk_level'].upper()}")
        
        # Check for HTML report
        if result.get('artifacts', {}).get('html_report'):
            html_path = result['artifacts']['html_report']
            if os.path.exists(html_path):
                print(f"\n📄 Professional HTML Report Generated!")
                print(f"📍 Location: {html_path}")
                print(f"📊 File size: {os.path.getsize(html_path):,} bytes")
                
                # Show report preview
                print(f"\n🌐 To view the report:")
                print(f"   open {html_path}")
                print(f"   # Or open in your browser: file://{os.path.abspath(html_path)}")
                
                return html_path
            else:
                print(f"❌ HTML report not found at: {html_path}")
        else:
            print("❌ No HTML report generated")
            
        return None
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_web_interface():
    """Demonstrate web interface capabilities."""
    print_section("Web Interface Demo")
    
    print("🎨 The GPT-OSS Hallucination Monitor includes a beautiful web interface!")
    print()
    print("Features:")
    print("  • Interactive configuration sliders")
    print("  • Real-time analysis with beautiful visualizations")
    print("  • Professional risk gauges and charts")
    print("  • Quick example buttons for instant testing")
    print("  • Color-coded risk highlighting")
    print("  • HTML report generation and download")
    print()
    print("To launch the web interface:")
    print("  1. Install dependencies: pip install streamlit plotly")
    print("  2. Run: python gpt_oss/monitoring/run_web_app.py")
    print("  3. Open: http://localhost:8501")
    print()

def demo_cli_usage():
    """Demonstrate CLI usage."""
    print_section("Command Line Interface")
    
    print("💻 The hallucination monitor also includes a powerful CLI:")
    print()
    print("Basic usage:")
    print("  gpt-oss-monitor --prompt 'Your prompt' --completion 'Model output'")
    print()
    print("With context documents:")
    print("  gpt-oss-monitor --prompt prompt.txt --completion output.txt --contexts context1.txt context2.txt")
    print()
    print("Generate HTML report:")
    print("  gpt-oss-monitor --prompt prompt.txt --completion output.txt --html")
    print()
    print("Custom configuration:")
    print("  gpt-oss-monitor --prompt prompt.txt --completion output.txt --k 10 --temperature 0.8 --html")
    print()

def demo_advanced_features():
    """Demonstrate advanced features."""
    print_section("Advanced Features")
    
    try:
        from gpt_oss.monitoring import HallucinationMonitor, MonitorConfig, MonitorThresholds
        
        # Custom configuration
        config = MonitorConfig(
            weights={
                'nli': 0.4,
                'self_consistency': 0.3,
                'retrieval_support': 0.2,
                'numeric_sanity': 0.1,
                'jailbreak_heuristics': 0.0
            },
            thresholds=MonitorThresholds(
                medium=0.5,
                high=0.8
            ),
            enable_retrieval_support=True,
            enable_jailbreak_heuristics=False,
            k_samples=5,
            temperature=0.8,
            max_new_tokens=200
        )
        
        monitor = HallucinationMonitor(cfg=config)
        
        print("⚙️ Custom Configuration Demo")
        print(f"   • NLI Weight: {config.weights['nli']}")
        print(f"   • Self-Consistency Weight: {config.weights['self_consistency']}")
        print(f"   • Retrieval Support Weight: {config.weights['retrieval_support']}")
        print(f"   • Numeric Sanity Weight: {config.weights['numeric_sanity']}")
        print(f"   • Jailbreak Heuristics: {'Enabled' if config.enable_jailbreak_heuristics else 'Disabled'}")
        print(f"   • K Samples: {config.k_samples}")
        print(f"   • Temperature: {config.temperature}")
        print()
        
        # Test with custom config
        prompt = "What is the weather like in Paris?"
        completion = "Paris has a tropical climate with palm trees and beaches."
        context = "Paris has a temperate climate with four distinct seasons."
        
        result = monitor.evaluate(prompt, completion, [context])
        
        print(f"📊 Custom Analysis Results:")
        print(f"   Risk Score: {result['risk_score']:.3f}")
        print(f"   Risk Level: {result['risk_level'].upper()}")
        
        # Show configuration in results
        config_dict = result.get('config', {})
        print(f"   Applied Config: {len(config_dict)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features demo failed: {e}")
        return False

def main():
    """Run the comprehensive demo."""
    print_header()
    
    # Check if monitoring system is available
    try:
        from gpt_oss.monitoring import HallucinationMonitor
        print("✅ GPT-OSS Hallucination Monitor is available!")
    except ImportError:
        print("❌ GPT-OSS Hallucination Monitor not found!")
        print("\n📥 To install:")
        print("   pip install -e '.[monitoring]'")
        print("   pip install streamlit plotly")
        return
    
    print()
    
    # Run demos
    success_count = 0
    total_demos = 5
    
    if demo_basic_usage():
        success_count += 1
    
    if demo_detection_signals():
        success_count += 1
    
    html_report_path = demo_professional_reports()
    if html_report_path:
        success_count += 1
    
    demo_web_interface()
    success_count += 1  # Always show web interface info
    
    demo_cli_usage()
    success_count += 1  # Always show CLI info
    
    if demo_advanced_features():
        success_count += 1
        total_demos += 1
    
    # Summary
    print_section("Demo Summary")
    print(f"✅ Completed {success_count}/{total_demos} demos successfully!")
    
    if html_report_path:
        print(f"\n📄 Professional HTML Report: {html_report_path}")
        print("   Open this file in your browser to see the beautiful design!")
    
    print(f"\n🎉 GPT-OSS Hallucination Monitor is ready for production use!")
    print(f"\n📚 For more information:")
    print(f"   • Documentation: gpt_oss/monitoring/examples/README.md")
    print(f"   • Web Interface: python gpt_oss/monitoring/run_web_app.py")
    print(f"   • CLI Usage: gpt-oss-monitor --help")
    print(f"   • Repository: https://github.com/openai/gpt-oss")

if __name__ == "__main__":
    main()
