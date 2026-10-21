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
Machine Translation Fine-tuning Script for GPT-OSS models.

This script fine-tunes GPT-OSS models for machine translation tasks using instruction-based training.
It supports multiple language pairs and includes translation-specific data preprocessing and evaluation.

Usage:
# Full parameter training
accelerate launch \
    --config_file configs/zero3.yaml \
    machine_translation.py \
    --config configs/mt_full.yaml \
    --model_name_or_path openai/gpt-oss-20b \
    --source_lang en \
    --target_lang es \
    --run_name mt-en-es-full

# LoRA training
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --source_lang en \
    --target_lang fr \
    --run_name mt-en-fr-lora
"""

import json
import re
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from typing import Dict, List, Optional
import argparse

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)


class MTScriptArguments(ScriptArguments):
    """Extended script arguments for machine translation tasks."""
    source_lang: str = "en"
    target_lang: str = "es"
    instruction_template: Optional[str] = None
    use_domain_adaptation: bool = False
    domain: Optional[str] = None
    max_source_length: int = 512
    max_target_length: int = 512


# Language name mappings for instruction templates
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
                            target_lang: str, instruction_template: Optional[str] = None,
                            domain: Optional[str] = None) -> str:
    """Create a formatted translation prompt for instruction tuning."""
    
    source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    
    if instruction_template:
        instruction = instruction_template.format(
            source_lang=source_lang_name,
            target_lang=target_lang_name
        )
    elif domain and domain in DOMAIN_TEMPLATES:
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
{target_text}<|im_end|>"""
    
    return prompt


def preprocess_translation_dataset(dataset, script_args: MTScriptArguments) -> Dataset:
    """Preprocess translation dataset for instruction tuning."""
    
    def format_translation_example(example):
        # Handle different dataset formats
        if 'translation' in example:
            # WMT-style datasets
            source_text = example['translation'][script_args.source_lang]
            target_text = example['translation'][script_args.target_lang]
        elif 'source' in example and 'target' in example:
            # Source-target format
            source_text = example['source']
            target_text = example['target']
        elif f'text_{script_args.source_lang}' in example and f'text_{script_args.target_lang}' in example:
            # Language-specific columns
            source_text = example[f'text_{script_args.source_lang}']
            target_text = example[f'text_{script_args.target_lang}']
        else:
            # Try to infer from available columns
            columns = list(example.keys())
            source_col = next((col for col in columns if script_args.source_lang in col), None)
            target_col = next((col for col in columns if script_args.target_lang in col), None)
            
            if source_col and target_col:
                source_text = example[source_col]
                target_text = example[target_col]
            else:
                raise ValueError(f"Cannot find source and target columns for {script_args.source_lang}-{script_args.target_lang}")
        
        # Create instruction-formatted text
        formatted_text = create_translation_prompt(
            source_text=source_text,
            target_text=target_text,
            source_lang=script_args.source_lang,
            target_lang=script_args.target_lang,
            instruction_template=script_args.instruction_template,
            domain=script_args.domain if script_args.use_domain_adaptation else None
        )
        
        return {"text": formatted_text}
    
    # Apply formatting to all examples
    formatted_dataset = dataset.map(format_translation_example, remove_columns=dataset.column_names)
    return formatted_dataset


def load_translation_dataset(script_args: MTScriptArguments) -> Dict[str, Dataset]:
    """Load and preprocess translation datasets."""
    
    # Load the dataset
    if script_args.dataset_config:
        # Parse dataset_config if it contains key=value pairs (e.g., "data_files=sample_data.csv")
        if '=' in script_args.dataset_config and not script_args.dataset_config.startswith('{'):
            # Parse key=value format into dictionary
            try:
                key, value = script_args.dataset_config.split('=', 1)
                dataset_config_dict = {key.strip(): value.strip()}
                dataset = load_dataset(script_args.dataset_name, **dataset_config_dict)
            except ValueError as e:
                raise ValueError(f"Invalid dataset_config format: {script_args.dataset_config}. Expected format: key=value") from e
        else:
            # Use as-is for regular config names or JSON strings
            dataset = load_dataset(script_args.dataset_name, script_args.dataset_config)
    else:
        dataset = load_dataset(script_args.dataset_name)
    
    # Preprocess the dataset
    processed_dataset = {}
    
    if script_args.dataset_train_split in dataset:
        processed_dataset['train'] = preprocess_translation_dataset(
            dataset[script_args.dataset_train_split], script_args
        )
    
    if script_args.dataset_test_split in dataset:
        processed_dataset['test'] = preprocess_translation_dataset(
            dataset[script_args.dataset_test_split], script_args
        )
    
    # Create validation split if not available
    if 'validation' in dataset:
        processed_dataset['validation'] = preprocess_translation_dataset(
            dataset['validation'], script_args
        )
    elif 'test' in processed_dataset:
        # Use test set as validation if no validation set exists
        processed_dataset['validation'] = processed_dataset['test']
    
    return processed_dataset


def compute_translation_metrics(eval_pred, tokenizer, source_lang: str, target_lang: str):
    """Compute translation-specific metrics."""
    try:
        import sacrebleu
        import evaluate
        
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Extract only the translation part (after <|im_start|>assistant)
        translation_preds = []
        translation_labels = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            # Extract translation from assistant response
            if "<|im_start|>assistant" in pred:
                pred_translation = pred.split("<|im_start|>assistant")[-1].strip()
                pred_translation = pred_translation.replace("<|im_end|>", "").strip()
            else:
                pred_translation = pred.strip()
            
            if "<|im_start|>assistant" in label:
                label_translation = label.split("<|im_start|>assistant")[-1].strip()
                label_translation = label_translation.replace("<|im_end|>", "").strip()
            else:
                label_translation = label.strip()
            
            translation_preds.append(pred_translation)
            translation_labels.append(label_translation)
        
        # Compute BLEU score
        bleu = sacrebleu.corpus_bleu(translation_preds, [translation_labels])
        
        # Compute additional metrics if available
        results = {"bleu": bleu.score}
        
        try:
            # Try to compute COMET score if available
            comet = evaluate.load("comet")
            comet_results = comet.compute(
                predictions=translation_preds,
                references=translation_labels,
                sources=[""]*len(translation_preds)  # COMET needs sources but we don't have them here
            )
            results["comet"] = comet_results["mean_score"]
        except:
            pass
        
        return results
        
    except ImportError:
        print("Warning: sacrebleu not available. Install with: pip install sacrebleu")
        return {}


def main(script_args: MTScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    """Main training function for machine translation."""
    
    print(f"Training machine translation model for {script_args.source_lang} -> {script_args.target_lang}")
    
    # ------------------------
    # Load model & tokenizer
    # ------------------------
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------
    # Load dataset
    # --------------
    datasets = load_translation_dataset(script_args)
    
    print(f"Loaded {len(datasets['train'])} training examples")
    if 'validation' in datasets:
        print(f"Loaded {len(datasets['validation'])} validation examples")

    # -------------
    # Train model
    # -------------
    
    # Check if evaluation dataset is available when eval_strategy is not "no"
    eval_dataset = datasets.get('validation') if training_args.eval_strategy != "no" else None
    
    # If eval_strategy is set but no eval dataset available, disable evaluation
    if training_args.eval_strategy != "no" and eval_dataset is None:
        print("Warning: eval_strategy is set but no evaluation dataset available. Disabling evaluation.")
        # Override training args to disable evaluation
        training_args.eval_strategy = "no"
        training_args.evaluation_strategy = "no"  # Also set this for compatibility
        training_args.load_best_model_at_end = False
        eval_dataset = None
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        compute_metrics=lambda eval_pred: compute_translation_metrics(
            eval_pred, tokenizer, script_args.source_lang, script_args.target_lang
        ) if training_args.eval_strategy != "no" else None,
    )

    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(training_args.output_dir)
    
    # Save translation-specific metadata
    metadata = {
        "source_language": script_args.source_lang,
        "target_language": script_args.target_lang,
        "language_pair": f"{script_args.source_lang}-{script_args.target_lang}",
        "domain": script_args.domain,
        "instruction_template": script_args.instruction_template,
        "model_type": "machine_translation",
        "base_model": model_args.model_name_or_path
    }
    
    with open(f"{training_args.output_dir}/translation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Push to hub if requested
    if training_args.push_to_hub:
        trainer.push_to_hub(
            dataset_name=f"{script_args.dataset_name}-{script_args.source_lang}-{script_args.target_lang}"
        )
    
    print(f"Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((MTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args) 
