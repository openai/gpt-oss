#!/usr/bin/env python3
"""
Simple demo of the machine translation recipe functionality.

This demo shows the core components without requiring GPU/ML dependencies.
"""

import json
import os
import pandas as pd

# Language name mappings
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian"
}

# Domain-specific instruction templates
DOMAIN_TEMPLATES = {
    "general": "Translate the following text from {source_lang} to {target_lang}:",
    "technical": "Translate the following technical text from {source_lang} to {target_lang}, preserving technical terminology:",
    "medical": "Translate the following medical text from {source_lang} to {target_lang}, maintaining medical accuracy:",
    "legal": "Translate the following legal text from {source_lang} to {target_lang}, preserving legal terminology:",
    "business": "Translate the following business text from {source_lang} to {target_lang}:",
    "news": "Translate the following news article from {source_lang} to {target_lang}:",
    "conversational": "Translate the following conversational text from {source_lang} to {target_lang}, maintaining the tone:"
}

def create_translation_prompt(source_text: str, target_text: str, source_lang: str, 
                            target_lang: str, domain: str = "general") -> str:
    """Create a formatted translation prompt for instruction tuning."""
    
    source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    
    instruction = DOMAIN_TEMPLATES[domain].format(
        source_lang=source_lang_name,
        target_lang=target_lang_name
    )
    
    prompt = f"""<|im_start|>system
You are a professional translator with expertise in {source_lang_name} and {target_lang_name}. Provide accurate, fluent, and contextually appropriate translations.
<|im_end|>
<|im_start|>user
{instruction}

{source_text}
<|im_end|>
<|im_start|>assistant
{target_text}<|im_end|>"""
    
    return prompt

def create_inference_prompt(source_text: str, source_lang: str, target_lang: str, 
                           domain: str = "general") -> str:
    """Create a prompt for inference (without target text)."""
    
    source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    
    instruction = DOMAIN_TEMPLATES[domain].format(
        source_lang=source_lang_name,
        target_lang=target_lang_name
    )
    
    prompt = f"""<|im_start|>system
You are a professional translator with expertise in {source_lang_name} and {target_lang_name}. Provide accurate, fluent, and contextually appropriate translations.
<|im_end|>
<|im_start|>user
{instruction}

{source_text}
<|im_end|>
<|im_start|>assistant
"""
    
    return prompt

def demo_prompt_generation():
    """Demonstrate prompt generation for different domains."""
    print("🔄 PROMPT GENERATION DEMO")
    print("=" * 50)
    
    test_cases = [
        {
            "source": "Hello, how are you today?",
            "target": "Hola, ¿cómo estás hoy?",
            "source_lang": "en",
            "target_lang": "es",
            "domain": "general"
        },
        {
            "source": "The API endpoint returns a JSON response with user data.",
            "target": "El endpoint de la API devuelve una respuesta JSON con datos del usuario.",
            "source_lang": "en",
            "target_lang": "es",
            "domain": "technical"
        },
        {
            "source": "Please schedule a meeting for tomorrow at 2 PM.",
            "target": "Por favor, programe una reunión para mañana a las 2 PM.",
            "source_lang": "en",
            "target_lang": "es",
            "domain": "business"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Example {i}: {case['domain'].title()} Domain")
        print(f"   Source ({case['source_lang']}): {case['source']}")
        print(f"   Target ({case['target_lang']}): {case['target']}")
        
        # Training prompt
        training_prompt = create_translation_prompt(
            source_text=case['source'],
            target_text=case['target'],
            source_lang=case['source_lang'],
            target_lang=case['target_lang'],
            domain=case['domain']
        )
        
        print("\n🏋️  Training Prompt:")
        print("-" * 40)
        print(training_prompt[:300] + "..." if len(training_prompt) > 300 else training_prompt)
        print("-" * 40)
        
        # Inference prompt
        inference_prompt = create_inference_prompt(
            source_text=case['source'],
            source_lang=case['source_lang'],
            target_lang=case['target_lang'],
            domain=case['domain']
        )
        
        print("\n🔮 Inference Prompt:")
        print("-" * 40)
        print(inference_prompt)
        print("-" * 40)

def demo_data_processing():
    """Demonstrate data processing capabilities."""
    print("\n📊 DATA PROCESSING DEMO")
    print("=" * 50)
    
    # Load sample data
    if os.path.exists("sample_data.csv"):
        df = pd.read_csv("sample_data.csv")
        print(f"✅ Loaded {len(df)} translation pairs from sample_data.csv")
        
        # Process each row into training format
        processed_examples = []
        
        for _, row in df.iterrows():
            prompt = create_translation_prompt(
                source_text=row['source'],
                target_text=row['target'],
                source_lang="en",
                target_lang="es",
                domain="general"
            )
            processed_examples.append({"text": prompt})
        
        print(f"✅ Processed {len(processed_examples)} training examples")
        
        # Show statistics
        total_chars = sum(len(ex["text"]) for ex in processed_examples)
        avg_chars = total_chars / len(processed_examples)
        
        print(f"📈 Statistics:")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average characters per example: {avg_chars:.1f}")
        print(f"   Estimated tokens (chars/4): {total_chars//4:,}")
        
        # Save processed data sample
        with open("processed_sample.json", "w", encoding="utf-8") as f:
            json.dump(processed_examples[:3], f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved 3 processed examples to processed_sample.json")
        
    else:
        print("❌ sample_data.csv not found")

def demo_multilingual_support():
    """Demonstrate multilingual capabilities."""
    print("\n🌍 MULTILINGUAL SUPPORT DEMO")
    print("=" * 50)
    
    # Test different language pairs
    test_pairs = [
        ("en", "es", "Hello world", "Hola mundo"),
        ("en", "fr", "Good morning", "Bonjour"),
        ("en", "de", "Thank you", "Danke"),
        ("en", "ja", "Welcome", "いらっしゃいませ"),
        ("en", "ko", "Goodbye", "안녕히 가세요"),
        ("en", "zh", "How much?", "多少钱？")
    ]
    
    print("🔤 Supported language pairs:")
    for source, target, source_text, target_text in test_pairs:
        source_name = LANGUAGE_NAMES.get(source, source)
        target_name = LANGUAGE_NAMES.get(target, target)
        
        print(f"   {source_name} → {target_name}: '{source_text}' → '{target_text}'")
        
        # Generate inference prompt
        prompt = create_inference_prompt(
            source_text=source_text,
            source_lang=source,
            target_lang=target,
            domain="general"
        )
        
        # Count prompt tokens (rough estimate)
        token_count = len(prompt.split())
        print(f"     Prompt tokens: ~{token_count}")
    
    print(f"\n📋 Total supported languages: {len(LANGUAGE_NAMES)}")
    print(f"🔄 Possible language pairs: {len(LANGUAGE_NAMES) * (len(LANGUAGE_NAMES) - 1)}")

def demo_domain_adaptation():
    """Demonstrate domain-specific adaptation."""
    print("\n🎯 DOMAIN ADAPTATION DEMO")
    print("=" * 50)
    
    source_text = "The patient shows symptoms of acute inflammation."
    
    print(f"📝 Source text: {source_text}")
    print("\n🏥 Domain-specific prompts:\n")
    
    for domain, template in DOMAIN_TEMPLATES.items():
        print(f"📌 {domain.upper()} Domain:")
        instruction = template.format(source_lang="English", target_lang="Spanish")
        print(f"   Instruction: {instruction}")
        
        prompt = create_inference_prompt(
            source_text=source_text,
            source_lang="en",
            target_lang="es",
            domain=domain
        )
        
        # Extract just the instruction part for comparison
        lines = prompt.split('\n')
        instruction_line = [line for line in lines if 'translate' in line.lower()][0]
        print(f"   Generated: {instruction_line.strip()}")
        print()

def demo_configuration():
    """Demonstrate configuration options."""
    print("\n⚙️  CONFIGURATION DEMO")
    print("=" * 50)
    
    # Training configurations
    configs = {
        "LoRA Training": {
            "model_name_or_path": "openai/gpt-oss-20b",
            "use_peft": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 5,
            "max_length": 1024
        },
        "Full Fine-tuning": {
            "model_name_or_path": "openai/gpt-oss-20b",
            "use_peft": False,
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "max_length": 1024
        }
    }
    
    for config_name, config in configs.items():
        print(f"🔧 {config_name}:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        print()
    
    # Generation configurations
    print("🔮 Generation Options:")
    generation_configs = [
        ("Conservative", {"temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.1}),
        ("Balanced", {"temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.1}),
        ("Creative", {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.0})
    ]
    
    for style, params in generation_configs:
        print(f"   {style}: {params}")

def main():
    """Run the complete demo."""
    print("🚀 MACHINE TRANSLATION RECIPE DEMO")
    print("=" * 60)
    print("This demo showcases the core functionality of the machine translation recipe")
    print("without requiring GPU or heavy ML dependencies.")
    print("=" * 60)
    
    # Run all demos
    demo_prompt_generation()
    demo_data_processing()
    demo_multilingual_support()
    demo_domain_adaptation()
    demo_configuration()
    
    # Summary
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 50)
    print("✅ Demonstrated prompt generation for training and inference")
    print("✅ Showed data processing capabilities")
    print("✅ Displayed multilingual support (12+ languages)")
    print("✅ Exhibited domain adaptation (7 domains)")
    print("✅ Presented configuration options")
    
    print("\n📚 Ready to use the machine translation recipe:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Prepare your data in CSV format")
    print("3. Train: python machine_translation.py --config configs/mt_lora.yaml")
    print("4. Translate: python generate_translation.py --model_path <path>")
    print("5. Evaluate: python evaluate_translation.py --model_path <path>")

if __name__ == "__main__":
    main() 