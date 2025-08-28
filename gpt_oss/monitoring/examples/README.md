# GPT-OSS Hallucination Monitor Examples

This directory contains example data and usage instructions for the GPT-OSS Hallucination Monitor.

## Quick Start

### 🌐 Web Interface (Recommended)

The easiest way to try the hallucination monitor is through our beautiful web interface:

```bash
# Install dependencies
pip install streamlit plotly
pip install -e ".[monitoring]"

# Launch the web interface
python gpt_oss/monitoring/run_web_app.py
```

This opens a sexy web interface at `http://localhost:8501` with:
- 🎨 Beautiful visualizations (gauge charts, radar plots, bar charts)
- ⚙️ Interactive configuration sliders
- 🎯 Quick example buttons
- 📊 Real-time analysis results
- 🎨 Color-coded risk highlighting
- 📄 HTML report generation

### 📝 Basic Usage (Python API)

```python
from gpt_oss.monitoring import HallucinationMonitor, MonitorConfig

# Initialize monitor
monitor = HallucinationMonitor()

# Evaluate a completion
results = monitor.evaluate(
    prompt="What is the capital of France?",
    completion="Paris is the capital of France. It has a population of 2.2 million people."
)

print(f"Risk Level: {results['risk_level']}")
print(f"Risk Score: {results['risk_score']:.3f}")
```

### With Context Documents

```python
# Provide context documents for retrieval support
context_docs = [
    "Paris is the capital and most populous city of France.",
    "The population of Paris was 2,165,423 in 2019."
]

results = monitor.evaluate(
    prompt="What is the capital of France?",
    completion="Paris is the capital of France with 2.2 million people.",
    context_docs=context_docs
)
```

### Custom Configuration

```python
from gpt_oss.monitoring import MonitorConfig, MonitorThresholds

config = MonitorConfig(
    k_samples=10,
    temperature=0.8,
    enable_retrieval_support=True,
    enable_jailbreak_heuristics=True,
    thresholds=MonitorThresholds(high=0.8, medium=0.5),
    html_report=True,
    report_dir="my_reports"
)

monitor = HallucinationMonitor(config)
```

## CLI Usage

### Basic Command

```bash
gpt-oss-monitor --prompt prompt.txt --completion output.txt
```

### With Context Documents

```bash
gpt-oss-monitor \
  --prompt prompt.txt \
  --completion output.txt \
  --contexts context1.txt context2.txt
```

### Generate HTML Report

```bash
gpt-oss-monitor \
  --prompt prompt.txt \
  --completion output.txt \
  --html \
  --report-dir reports/
```

### Custom Parameters

```bash
gpt-oss-monitor \
  --prompt prompt.txt \
  --completion output.txt \
  --k 10 \
  --temperature 0.8 \
  --high-threshold 0.8 \
  --medium-threshold 0.5 \
  --html
```

## Example Data Files

### truthfulqa_mini.jsonl

A small subset of TruthfulQA questions for testing. Each line contains a JSON object with:

```json
{
  "question": "What is the capital of France?",
  "correct_answer": "Paris",
  "incorrect_answer": "London"
}
```

### fever_mini.jsonl

A small subset of FEVER claims for testing. Each line contains a JSON object with:

```json
{
  "claim": "Paris is the capital of France.",
  "evidence": ["Paris is the capital and most populous city of France."],
  "label": "SUPPORTS"
}
```

## Running Examples

### Using the Example Data

1. Create test files:

```bash
# Create prompt file
echo "What is the capital of France?" > prompt.txt

# Create completion file
echo "Paris is the capital of France with 2.2 million people." > completion.txt

# Create context file
echo "Paris is the capital and most populous city of France." > context.txt
```

2. Run the monitor:

```bash
gpt-oss-monitor \
  --prompt prompt.txt \
  --completion completion.txt \
  --contexts context.txt \
  --html
```

### Python Script Example

Create a file `example_usage.py`:

```python
import json
from gpt_oss.monitoring import HallucinationMonitor

# Load example data
with open('truthfulqa_mini.jsonl', 'r') as f:
    examples = [json.loads(line) for line in f]

# Initialize monitor
monitor = HallucinationMonitor()

# Process examples
for i, example in enumerate(examples[:3]):  # Process first 3 examples
    print(f"\n--- Example {i+1} ---")
    
    results = monitor.evaluate(
        prompt=example['question'],
        completion=example['incorrect_answer']  # Use incorrect answer to test detection
    )
    
    print(f"Risk Level: {results['risk_level']}")
    print(f"Risk Score: {results['risk_score']:.3f}")
    print(f"Signals: {results['signals']}")
```

Run the script:

```bash
python example_usage.py
```

## Expected Results

### Low Risk Example
- **Prompt**: "What is 2 + 2?"
- **Completion**: "2 + 2 equals 4."
- **Expected**: Low risk, high numeric sanity, high NLI faithfulness

### Medium Risk Example
- **Prompt**: "What is the population of Paris?"
- **Completion**: "Paris has about 2.2 million people."
- **Expected**: Medium risk, moderate retrieval support

### High Risk Example
- **Prompt**: "What is the capital of France?"
- **Completion**: "London is the capital of France."
- **Expected**: High risk, low NLI faithfulness, contradiction detected

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: The monitor uses lightweight models by default. If you encounter loading issues, the system will fall back to simpler heuristics.

2. **Memory Issues**: For large documents, consider reducing the chunk size in the configuration.

3. **Slow Performance**: The first run may be slower due to model downloading. Subsequent runs will be faster.

### Getting Help

- Check the logs with `--verbose` flag
- Ensure all required dependencies are installed
- Verify input file formats and encoding (UTF-8)

## Advanced Usage

### Integration with GPT-OSS Models

```python
from gpt_oss import generate  # Import your model
from gpt_oss.monitoring import HallucinationMonitor

# Initialize monitor with your model
monitor = HallucinationMonitor(model=your_model, tokenizer=your_tokenizer)

# Generate and monitor in one step
def generate_with_monitoring(prompt, **kwargs):
    completion = generate(prompt, **kwargs)
    results = monitor.evaluate(prompt, completion)
    
    if results['risk_level'] == 'high':
        print("High risk detected! Consider regenerating with different parameters.")
    
    return completion, results
```

### Custom Detectors

You can extend the monitoring system by implementing custom detectors:

```python
from gpt_oss.monitoring.detectors import BaseDetector

class CustomDetector(BaseDetector):
    def detect(self, prompt, completion, **kwargs):
        # Your custom detection logic
        score = self._compute_score(prompt, completion)
        return score, {'details': 'custom analysis'}
```
