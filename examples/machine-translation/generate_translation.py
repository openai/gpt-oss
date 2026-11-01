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
Machine Translation Inference Script for GPT-OSS models.

This script performs machine translation inference using fine-tuned GPT-OSS models.
It supports multiple language pairs and domain-specific translation.

Usage:
python generate_translation.py \
    --model_path ./gpt-oss-20b-translator \
    --source_lang en \
    --target_lang es \
    --domain general \
    --text "Hello, how are you today?"

# Batch translation from file
python generate_translation.py \
    --model_path ./gpt-oss-20b-translator \
    --source_lang en \
    --target_lang fr \
    --input_file input.txt \
    --output_file translations.txt
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict
import time

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


def create_translation_prompt(source_text: str, source_lang: str, target_lang: str, 
                            domain: str = "general", custom_instruction: Optional[str] = None) -> str:
    """Create a formatted translation prompt for inference."""
    
    source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    
    if custom_instruction:
        instruction = custom_instruction.format(
            source_lang=source_lang_name,
            target_lang=target_lang_name
        )
    elif domain in DOMAIN_TEMPLATES:
        instruction = DOMAIN_TEMPLATES[domain].format(
            source_lang=source_lang_name,
            target_lang=target_lang_name
        )
    else:
        instruction = DOMAIN_TEMPLATES["general"].format(
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


def extract_translation(generated_text: str, prompt: str) -> str:
    """Extract the translation from the generated text."""
    # Remove the prompt from the generated text
    if prompt in generated_text:
        translation = generated_text[len(prompt):].strip()
    else:
        translation = generated_text.strip()
    
    # Remove the end token if present
    if "<|im_end|>" in translation:
        translation = translation.split("<|im_end|>")[0].strip()
    
    return translation


class TranslationGenerator:
    """Machine translation generator using fine-tuned GPT-OSS models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the translation generator."""
        self.model_path = model_path
        self.device = device
        
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Load translation metadata if available
        metadata_path = Path(model_path) / "translation_metadata.json"
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded translation metadata: {self.metadata}")
        
        print("Model loaded successfully!")
    
    def translate(self, text: str, source_lang: str, target_lang: str, 
                 domain: str = "general", custom_instruction: Optional[str] = None,
                 max_new_tokens: int = 512, temperature: float = 0.3,
                 top_p: float = 0.9, do_sample: bool = True) -> str:
        """Translate a single text."""
        
        prompt = create_translation_prompt(
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            custom_instruction=custom_instruction
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode and extract translation
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        translation = extract_translation(generated_text, prompt)
        
        return translation
    
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str,
                       domain: str = "general", custom_instruction: Optional[str] = None,
                       max_new_tokens: int = 512, temperature: float = 0.3,
                       top_p: float = 0.9, do_sample: bool = True,
                       batch_size: int = 4) -> List[str]:
        """Translate a batch of texts."""
        
        translations = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_translations = []
            
            for text in batch_texts:
                translation = self.translate(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    domain=domain,
                    custom_instruction=custom_instruction,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                batch_translations.append(translation)
            
            translations.extend(batch_translations)
            print(f"Translated {len(translations)}/{len(texts)} texts")
        
        return translations


def main():
    parser = argparse.ArgumentParser(description="Machine Translation with GPT-OSS")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned translation model")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run inference on")
    
    # Translation arguments
    parser.add_argument("--source_lang", type=str, required=True,
                       help="Source language code (e.g., en, es, fr)")
    parser.add_argument("--target_lang", type=str, required=True,
                       help="Target language code (e.g., en, es, fr)")
    parser.add_argument("--domain", type=str, default="general",
                       choices=list(DOMAIN_TEMPLATES.keys()),
                       help="Translation domain")
    parser.add_argument("--custom_instruction", type=str, default=None,
                       help="Custom instruction template")
    
    # Input/Output arguments
    parser.add_argument("--text", type=str, default=None,
                       help="Single text to translate")
    parser.add_argument("--input_file", type=str, default=None,
                       help="Input file with texts to translate (one per line)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for translations")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="Use sampling for generation")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing multiple texts")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and not args.input_file:
        raise ValueError("Either --text or --input_file must be provided")
    
    # Initialize generator
    generator = TranslationGenerator(args.model_path, args.device)
    
    if args.text:
        # Single text translation
        print(f"Translating: {args.text}")
        print(f"Language pair: {args.source_lang} -> {args.target_lang}")
        print(f"Domain: {args.domain}")
        print("-" * 50)
        
        start_time = time.time()
        translation = generator.translate(
            text=args.text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            domain=args.domain,
            custom_instruction=args.custom_instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample
        )
        end_time = time.time()
        
        print(f"Translation: {translation}")
        print(f"Time taken: {end_time - start_time:.2f}s")
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(translation + '\n')
            print(f"Translation saved to {args.output_file}")
    
    elif args.input_file:
        # Batch translation
        print(f"Loading texts from {args.input_file}")
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(texts)} texts to translate")
        print(f"Language pair: {args.source_lang} -> {args.target_lang}")
        print(f"Domain: {args.domain}")
        print("-" * 50)
        
        start_time = time.time()
        translations = generator.translate_batch(
            texts=texts,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            domain=args.domain,
            custom_instruction=args.custom_instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            batch_size=args.batch_size
        )
        end_time = time.time()
        
        print(f"Completed {len(translations)} translations")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Average time per translation: {(end_time - start_time) / len(translations):.2f}s")
        
        # Save translations
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(translation + '\n')
            print(f"Translations saved to {args.output_file}")
        else:
            # Print translations
            for i, (text, translation) in enumerate(zip(texts, translations), 1):
                print(f"{i}. Source: {text}")
                print(f"   Translation: {translation}")
                print()


if __name__ == "__main__":
    main() 