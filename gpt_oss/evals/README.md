# GPT-OSS Evaluations

This module is a reincarnation of [simple-evals](https://github.com/openai/simple-evals) adapted for GPT-OSS. It provides evaluation frameworks for testing GPT-OSS model performance on various benchmarks.

## 📊 Available Evaluations

### 🧠 **GPQA (Graduate-Level Google-Proof Q&A)**
A challenging dataset of graduate-level questions across multiple domains.

**Features:**
- 448 multiple-choice questions
- Graduate-level difficulty
- Multi-domain coverage
- Detailed reasoning evaluation

### 🏥 **HealthBench**
A medical reasoning benchmark for evaluating healthcare-related capabilities.

**Features:**
- Medical reasoning questions
- Clinical decision support scenarios
- Healthcare knowledge assessment

## 🚀 Quick Start

### Prerequisites
- GPT-OSS model running with Responses API
- Python 3.12+
- Required dependencies: `pip install -e .[eval]`

### Running Evaluations

#### 1. Start Your Model Server
```bash
# Using vLLM
vllm serve openai/gpt-oss-20b --port 8080

# Using Ollama with Responses API
ollama run gpt-oss:20b --port 8080

# Using local Responses API server
python -m gpt_oss.responses_api.serve --port 8080
```

#### 2. Run GPQA Evaluation
```bash
python -m gpt_oss.evals --eval gpqa --model gpt-oss-20b
```

#### 3. Run HealthBench Evaluation
```bash
python -m gpt_oss.evals --eval healthbench --model gpt-oss-20b
```

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="http://localhost:8080/v1"
```

### Command Line Options
```bash
python -m gpt_oss.evals --help
```

Common options:
- `--eval`: Evaluation type (gpqa, healthbench)
- `--model`: Model name
- `--max_samples`: Maximum number of samples to evaluate
- `--output_dir`: Directory for results
- `--reasoning_effort`: Reasoning effort level (low, medium, high)

## 📈 Understanding Results

### GPQA Results
- **Accuracy**: Overall correct answer percentage
- **Reasoning Quality**: Assessment of reasoning process
- **Domain Performance**: Performance across different subjects

### HealthBench Results
- **Medical Accuracy**: Correct medical reasoning
- **Clinical Relevance**: Practical healthcare application
- **Safety Assessment**: Patient safety considerations

## 🛠️ Custom Evaluations

### Creating Custom Evaluations
You can create custom evaluations by implementing the evaluation interface:

```python
from gpt_oss.evals.types import Evaluation

class CustomEvaluation(Evaluation):
    def __init__(self):
        self.name = "custom_eval"
        self.description = "Custom evaluation description"
    
    def generate_questions(self):
        # Generate evaluation questions
        pass
    
    def evaluate_response(self, question, response):
        # Evaluate model response
        pass
```

### Adding New Benchmarks
1. Create evaluation class
2. Implement question generation
3. Implement response evaluation
4. Add to evaluation registry

## 📊 Results Analysis

### Output Format
Results are saved in JSON format with:
- Question details
- Model responses
- Evaluation scores
- Reasoning analysis

### Visualization
Use the provided analysis tools to visualize results:
```bash
python -m gpt_oss.evals.report --results results.json
```

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure model server is running on correct port
2. **Model Not Found**: Verify model name matches your setup
3. **Memory Issues**: Reduce batch size or use smaller model

### Debug Mode
Enable verbose logging:
```bash
python -m gpt_oss.evals --eval gpqa --verbose
```

## 📖 Related Documentation

- [Main README](../../README.md) - Project overview
- [Responses API](../responses_api/) - API server implementation
- [Evaluation Types](types.py) - Evaluation interface definitions
- [Simple Evals](https://github.com/openai/simple-evals) - Original evaluation framework

## 🤝 Contributing

We welcome evaluation improvements! Please:
- Add new benchmark datasets
- Improve evaluation metrics
- Enhance result analysis
- Document evaluation methodologies