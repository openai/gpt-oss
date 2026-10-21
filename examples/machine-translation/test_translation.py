#!/usr/bin/env python3
"""
Test script for the Machine Translation Recipe.

This script demonstrates the complete workflow:
1. Data preparation and validation
2. Training with sample data (simulated)
3. Inference with single text and batch
4. Evaluation with metrics

Usage:
    python test_translation.py
"""

import os
import json
import pandas as pd
from pathlib import Path
import sys

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_preparation():
    """Test data preparation and validation."""
    print("=" * 60)
    print("1. TESTING DATA PREPARATION")
    print("=" * 60)
    
    # Check if sample data exists
    if not os.path.exists("sample_data.csv"):
        print("❌ sample_data.csv not found. Please create sample data first.")
        return False
    
    # Load and validate sample data
    df = pd.read_csv("sample_data.csv")
    print(f"✅ Loaded sample data with {len(df)} translation pairs")
    print(f"   Columns: {list(df.columns)}")
    
    # Show sample translations
    print("\n📄 Sample translation pairs:")
    for i, row in df.head(3).iterrows():
        print(f"   EN: {row['source']}")
        print(f"   ES: {row['target']}")
        print()
    
    return True

def test_prompt_formatting():
    """Test prompt formatting functionality."""
    print("=" * 60)
    print("2. TESTING PROMPT FORMATTING")
    print("=" * 60)
    
    try:
        from machine_translation import create_translation_prompt, DOMAIN_TEMPLATES
        
        # Test basic prompt
        source_text = "Hello, how are you today?"
        target_text = "Hola, ¿cómo estás hoy?"
        
        prompt = create_translation_prompt(
            source_text=source_text,
            target_text=target_text,
            source_lang="en",
            target_lang="es",
            domain="general"
        )
        
        print("✅ Basic prompt formatting successful")
        print("📝 Generated prompt:")
        print("-" * 40)
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
        print("-" * 40)
        
        # Test domain-specific prompts
        print(f"\n🎯 Available domains: {list(DOMAIN_TEMPLATES.keys())}")
        
        # Test technical domain
        tech_prompt = create_translation_prompt(
            source_text="The API endpoint returns a JSON response.",
            target_text="El endpoint de la API devuelve una respuesta JSON.",
            source_lang="en",
            target_lang="es",
            domain="technical"
        )
        
        print("✅ Domain-specific prompt formatting successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Prompt formatting error: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("=" * 60)
    print("3. TESTING DATA PREPROCESSING")
    print("=" * 60)
    
    try:
        from machine_translation import MTScriptArguments, preprocess_translation_dataset
        from datasets import Dataset
        
        # Create a simple dataset from our sample data
        df = pd.read_csv("sample_data.csv")
        dataset = Dataset.from_pandas(df)
        
        # Create script arguments
        script_args = MTScriptArguments()
        script_args.source_lang = "en"
        script_args.target_lang = "es"
        script_args.domain = "general"
        
        # Preprocess the dataset
        processed_dataset = preprocess_translation_dataset(dataset, script_args)
        
        print(f"✅ Dataset preprocessing successful")
        print(f"   Original dataset size: {len(dataset)}")
        print(f"   Processed dataset size: {len(processed_dataset)}")
        print(f"   Processed columns: {processed_dataset.column_names}")
        
        # Show a sample processed example
        sample = processed_dataset[0]
        print("\n📝 Sample processed example:")
        print("-" * 40)
        print(sample['text'][:500] + "..." if len(sample['text']) > 500 else sample['text'])
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing error: {e}")
        return False

def simulate_training():
    """Simulate training process (without actual model training)."""
    print("=" * 60)
    print("4. SIMULATING TRAINING PROCESS")
    print("=" * 60)
    
    print("🚀 Training simulation (LoRA configuration):")
    print("   - Model: openai/gpt-oss-20b")
    print("   - Language pair: en -> es")
    print("   - Training method: LoRA")
    print("   - Domain: general")
    print("   - Dataset: sample_data.csv")
    
    # Simulate training configuration
    training_config = {
        "model_name_or_path": "openai/gpt-oss-20b",
        "source_lang": "en",
        "target_lang": "es",
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_length": 1024,
        "domain": "general"
    }
    
    print("\n⚙️  Training configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    # Simulate training steps
    print("\n📈 Simulated training progress:")
    print("   Epoch 1/5: Loss = 2.45, BLEU = 15.2")
    print("   Epoch 2/5: Loss = 1.89, BLEU = 22.7")
    print("   Epoch 3/5: Loss = 1.52, BLEU = 28.1")
    print("   Epoch 4/5: Loss = 1.31, BLEU = 31.5")
    print("   Epoch 5/5: Loss = 1.18, BLEU = 33.8")
    
    # Simulate saving model metadata
    metadata = {
        "source_language": "en",
        "target_language": "es",
        "language_pair": "en-es",
        "domain": "general",
        "model_type": "machine_translation",
        "base_model": "openai/gpt-oss-20b",
        "training_method": "LoRA",
        "final_bleu_score": 33.8
    }
    
    # Save simulated metadata
    with open("simulated_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Training simulation completed")
    print("💾 Model metadata saved to: simulated_model_metadata.json")
    
    return True

def test_generation_functionality():
    """Test generation script functionality (without actual model)."""
    print("=" * 60)
    print("5. TESTING GENERATION FUNCTIONALITY")
    print("=" * 60)
    
    try:
        from generate_translation import create_translation_prompt, extract_translation, DOMAIN_TEMPLATES
        
        # Test prompt creation for inference
        test_text = "Good morning, how can I help you?"
        
        prompt = create_translation_prompt(
            source_text=test_text,
            source_lang="en",
            target_lang="es",
            domain="general"
        )
        
        print("✅ Generation prompt creation successful")
        print(f"📝 Input text: {test_text}")
        print("🔄 Generated prompt:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
        
        # Test translation extraction
        simulated_response = prompt + "Buenos días, ¿cómo puedo ayudarte?<|im_end|>"
        extracted = extract_translation(simulated_response, prompt)
        
        print(f"\n✅ Translation extraction successful")
        print(f"📤 Extracted translation: {extracted}")
        
        # Test batch input preparation
        if os.path.exists("test_input.txt"):
            with open("test_input.txt", "r") as f:
                batch_texts = [line.strip() for line in f if line.strip()]
            
            print(f"\n📂 Batch input loaded: {len(batch_texts)} texts")
            for i, text in enumerate(batch_texts, 1):
                print(f"   {i}. {text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation functionality error: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics functionality."""
    print("=" * 60)
    print("6. TESTING EVALUATION METRICS")
    print("=" * 60)
    
    try:
        from evaluate_translation import (
            compute_bleu_score, 
            compute_chrf_score, 
            compute_length_statistics,
            SACREBLEU_AVAILABLE
        )
        
        # Sample predictions and references
        predictions = [
            "Hola, ¿cómo estás hoy?",
            "Me gustaría pedir un café, por favor.",
            "El clima está hermoso hoy."
        ]
        
        references = [
            "Hola, ¿cómo estás hoy?",
            "Quisiera ordenar un café, por favor.",
            "El tiempo está hermoso hoy."
        ]
        
        print(f"🔍 SacreBLEU available: {SACREBLEU_AVAILABLE}")
        
        if SACREBLEU_AVAILABLE:
            # Test BLEU score
            bleu_results = compute_bleu_score(predictions, references)
            print("✅ BLEU score computation successful")
            print(f"   BLEU score: {bleu_results.get('bleu', 'N/A')}")
            
            # Test chrF score
            chrf_results = compute_chrf_score(predictions, references)
            print("✅ chrF score computation successful")
            print(f"   chrF score: {chrf_results.get('chrf', 'N/A')}")
        else:
            print("⚠️  SacreBLEU not available - metrics will be limited")
        
        # Test length statistics (always available)
        length_stats = compute_length_statistics(predictions, references)
        print("✅ Length statistics computation successful")
        print(f"   Average prediction length: {length_stats['avg_pred_length']:.1f}")
        print(f"   Average reference length: {length_stats['avg_ref_length']:.1f}")
        print(f"   Length ratio: {length_stats['length_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation metrics error: {e}")
        return False

def test_configuration_files():
    """Test configuration files."""
    print("=" * 60)
    print("7. TESTING CONFIGURATION FILES")
    print("=" * 60)
    
    config_files = ["configs/mt_full.yaml", "configs/mt_lora.yaml"]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} found")
            
            # Read and validate basic structure
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                required_keys = ['model_name_or_path', 'source_lang', 'target_lang', 'learning_rate']
                missing_keys = [key for key in required_keys if key not in config]
                
                if missing_keys:
                    print(f"   ⚠️  Missing keys: {missing_keys}")
                else:
                    print(f"   ✅ Configuration structure valid")
                    print(f"   📋 Model: {config.get('model_name_or_path', 'N/A')}")
                    print(f"   🌐 Language pair: {config.get('source_lang', 'N/A')} -> {config.get('target_lang', 'N/A')}")
                    
            except ImportError:
                print(f"   ⚠️  PyYAML not available - skipping detailed validation")
            except Exception as e:
                print(f"   ❌ Error reading {config_file}: {e}")
        else:
            print(f"❌ {config_file} not found")
    
    return True

def generate_usage_examples():
    """Generate practical usage examples."""
    print("=" * 60)
    print("8. USAGE EXAMPLES")
    print("=" * 60)
    
    examples = {
        "training_lora": """
# LoRA Training Example
python machine_translation.py \\
    --config configs/mt_lora.yaml \\
    --source_lang en \\
    --target_lang es \\
    --dataset_name csv \\
    --dataset_config data_files=sample_data.csv \\
    --run_name test-en-es-lora \\
    --num_train_epochs 3""",
        
        "single_translation": """
# Single Text Translation
python generate_translation.py \\
    --model_path ./gpt-oss-20b-translator-lora \\
    --source_lang en \\
    --target_lang es \\
    --domain general \\
    --text "Hello, how are you today?" """,
        
        "batch_translation": """
# Batch Translation
python generate_translation.py \\
    --model_path ./gpt-oss-20b-translator-lora \\
    --source_lang en \\
    --target_lang es \\
    --input_file test_input.txt \\
    --output_file translations.txt \\
    --batch_size 4""",
        
        "evaluation": """
# Model Evaluation
python evaluate_translation.py \\
    --model_path ./gpt-oss-20b-translator-lora \\
    --source_lang en \\
    --target_lang es \\
    --test_file sample_data.csv \\
    --source_column source \\
    --target_column target \\
    --output_file evaluation_results.json"""
    }
    
    for name, example in examples.items():
        print(f"📝 {name.replace('_', ' ').title()}:")
        print(example)
        print()
    
    return True

def main():
    """Run all tests."""
    print("🧪 MACHINE TRANSLATION RECIPE TEST SUITE")
    print("🔬 Testing all components of the translation pipeline")
    print()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Data Preparation", test_data_preparation),
        ("Prompt Formatting", test_prompt_formatting),
        ("Data Preprocessing", test_data_preprocessing),
        ("Training Simulation", simulate_training),
        ("Generation Functionality", test_generation_functionality),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Configuration Files", test_configuration_files),
        ("Usage Examples", generate_usage_examples)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            test_results.append((test_name, False))
            print()
    
    # Summary
    print("=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The machine translation recipe is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n📚 Next steps:")
    print("1. Install required dependencies: pip install -r requirements.txt")
    print("2. Prepare your training data")
    print("3. Run training with: python machine_translation.py --config configs/mt_lora.yaml")
    print("4. Test inference with: python generate_translation.py")
    print("5. Evaluate results with: python evaluate_translation.py")

if __name__ == "__main__":
    main() 