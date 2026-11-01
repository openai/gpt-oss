# Machine Translation Recipe for GPT-OSS Models

This recipe provides a complete pipeline for fine-tuning GPT-OSS models for machine translation tasks using instruction-based training. It supports multiple language pairs, domain adaptation, and comprehensive evaluation.

## Features

- 🌍 **Multi-language Support**: Supports 12+ language pairs including English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Dutch, Polish, and Ukrainian
- 🎯 **Domain Adaptation**: Specialized instruction templates for different domains (general, technical, medical, legal, business, news, conversational)
- 📊 **Comprehensive Evaluation**: BLEU, chrF, TER, and COMET scores for translation quality assessment
- ⚡ **Efficient Training**: Both full fine-tuning and LoRA support for parameter-efficient training
- 🔧 **Flexible Data Handling**: Support for various dataset formats including WMT, CSV files, and custom datasets

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install additional translation metrics
pip install comet-ml  # For COMET scores
```

### 2. Basic Training

#### Full Parameter Training
```bash
# Train English to Spanish translator
accelerate launch \
    --config_file configs/zero3.yaml \
    machine_translation.py \
    --config configs/mt_full.yaml \
    --model_name_or_path openai/gpt-oss-20b \
    --source_lang en \
    --target_lang es \
    --dataset_name wmt14 \
    --dataset_config de-en \
    --run_name mt-en-es-full
```

#### LoRA Training (Recommended for most users)
```bash
# Train with LoRA for efficient fine-tuning
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --source_lang en \
    --target_lang fr \
    --dataset_name opus100 \
    --dataset_config en-fr \
    --run_name mt-en-fr-lora
```

### 3. Inference

#### Single Text Translation
```bash
python generate_translation.py \
    --model_path ./gpt-oss-20b-translator-lora \
    --source_lang en \
    --target_lang es \
    --domain general \
    --text "Hello, how are you today?"
```

#### Batch Translation
```bash
python generate_translation.py \
    --model_path ./gpt-oss-20b-translator-lora \
    --source_lang en \
    --target_lang fr \
    --input_file input_texts.txt \
    --output_file translations.txt \
    --batch_size 8
```

### 4. Evaluation

```bash
python evaluate_translation.py \
    --model_path ./gpt-oss-20b-translator-lora \
    --source_lang en \
    --target_lang es \
    --dataset_name wmt14 \
    --dataset_config de-en \
    --max_examples 1000 \
    --output_file evaluation_results.json
```

## Detailed Usage

### Training Configuration

The recipe provides two pre-configured training setups:

#### `mt_full.yaml` - Full Parameter Training
- Optimized for maximum translation quality
- Requires significant GPU memory (A100 80GB recommended)
- Lower learning rate (1e-5) for stable training
- More conservative batch size due to longer sequences

#### `mt_lora.yaml` - LoRA Training
- Parameter-efficient fine-tuning
- Works on smaller GPUs (RTX 4090, A6000)
- Higher learning rate (1e-4) for LoRA
- Higher rank (16) optimized for translation tasks

### Supported Language Pairs

The recipe supports the following languages:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ja` | Japanese |
| `es` | Spanish | `ko` | Korean |
| `fr` | French | `zh` | Chinese |
| `de` | German | `nl` | Dutch |
| `it` | Italian | `pl` | Polish |
| `pt` | Portuguese | `uk` | Ukrainian |
| `ru` | Russian | | |

### Domain Adaptation

Use domain-specific training for better performance:

```bash
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --source_lang en \
    --target_lang es \
    --use_domain_adaptation true \
    --domain technical \
    --dataset_name your_technical_dataset
```

Available domains:
- `general`: General purpose translation
- `technical`: Technical documentation
- `medical`: Medical texts
- `legal`: Legal documents
- `business`: Business communications
- `news`: News articles
- `conversational`: Conversational text

### Custom Datasets

#### CSV Format
Your CSV file should have columns for source and target languages:

```csv
source,target
"Hello world","Hola mundo"
"How are you?","¿Cómo estás?"
```

Train with:
```bash
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --source_lang en \
    --target_lang es \
    --dataset_name csv \
    --dataset_config data_files=your_data.csv
```

#### HuggingFace Datasets
The recipe supports various HuggingFace datasets:

- `wmt14`, `wmt16`, `wmt17`, `wmt19`: WMT translation datasets
- `opus100`: 100 language pairs from OPUS
- `flores101`: Facebook's multilingual dataset

### Advanced Training Options

#### Custom Instruction Templates
```bash
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --source_lang en \
    --target_lang es \
    --instruction_template "Please translate this {source_lang} text to {target_lang} with high accuracy:"
```

#### Training with Different Model Sizes
```bash
# For 120B model (requires more GPUs)
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --model_name_or_path openai/gpt-oss-120b \
    --source_lang en \
    --target_lang es
```

### Generation Parameters

Fine-tune generation for your use case:

```bash
python generate_translation.py \
    --model_path ./gpt-oss-20b-translator-lora \
    --source_lang en \
    --target_lang es \
    --text "Your text here" \
    --temperature 0.1 \     # Lower for more deterministic translations
    --max_new_tokens 1024 \ # Longer for complex translations
    --top_p 0.95            # Nucleus sampling parameter
```

### Evaluation Metrics

The evaluation script computes several metrics:

- **BLEU**: Standard n-gram based metric
- **chrF**: Character-level F-score
- **TER**: Translation Error Rate
- **COMET**: Neural metric using cross-lingual embeddings
- **Length Statistics**: Analysis of translation length patterns

### Performance Optimization

#### Memory Optimization
```bash
# Use gradient checkpointing for memory efficiency
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --gradient_checkpointing true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
```

#### Speed Optimization
```bash
# Use Flash Attention for faster training
accelerate launch \
    --config_file configs/zero3.yaml \
    machine_translation.py \
    --config configs/mt_full.yaml \
    --attn_implementation kernels-community/vllm-flash-attn3
```

## Example Workflows

### 1. English to Spanish News Translation

```bash
# 1. Train the model
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --source_lang en \
    --target_lang es \
    --domain news \
    --dataset_name news_commentary \
    --run_name news-en-es

# 2. Evaluate on test set
python evaluate_translation.py \
    --model_path ./gpt-oss-20b-translator-lora \
    --source_lang en \
    --target_lang es \
    --dataset_name wmt14 \
    --dataset_config de-en \
    --output_file news_evaluation.json

# 3. Translate new articles
python generate_translation.py \
    --model_path ./gpt-oss-20b-translator-lora \
    --source_lang en \
    --target_lang es \
    --domain news \
    --input_file news_articles.txt \
    --output_file translated_articles.txt
```

### 2. Technical Documentation Translation

```bash
# Train on technical domain
python machine_translation.py \
    --config configs/mt_lora.yaml \
    --source_lang en \
    --target_lang de \
    --domain technical \
    --dataset_name your_technical_corpus \
    --instruction_template "Translate this technical documentation from {source_lang} to {target_lang}, preserving all technical terms and formatting:"
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use LoRA instead of full fine-tuning
   - Enable gradient checkpointing

2. **Poor Translation Quality**
   - Increase training epochs
   - Use domain-specific data
   - Adjust learning rate
   - Try different instruction templates

3. **Slow Training**
   - Use Flash Attention
   - Increase batch size with more GPUs
   - Use mixed precision training

### Tips for Better Results

1. **Data Quality**: Clean and preprocess your training data
2. **Domain Matching**: Use domain-specific training data
3. **Evaluation**: Use multiple metrics for comprehensive assessment
4. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
5. **Model Size**: Larger models generally perform better but require more resources

## Contributing

To add support for new languages or domains:

1. Update the `LANGUAGE_NAMES` dictionary in the scripts
2. Add domain-specific instruction templates to `DOMAIN_TEMPLATES`
3. Create appropriate evaluation datasets
4. Test with your specific language pair

## Citation

If you use this recipe in your research, please cite:

```bibtex
@software{gpt_oss_machine_translation,
  title={Machine Translation Recipe for GPT-OSS Models},
  author={OpenAI and Hugging Face},
  year={2025},
  url={https://github.com/openai/gpt-oss-recipes}
}
``` 