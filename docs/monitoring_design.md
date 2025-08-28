# GPT-OSS Hallucination Monitor Design

## Overview

The GPT-OSS Hallucination Monitor is a comprehensive system designed to detect and quantify hallucination risk in LLM outputs. It combines multiple detection signals to provide a robust assessment of potential issues in model completions.

## Architecture

### Core Components

```
gpt_oss/monitoring/
├── __init__.py                 # Main exports
├── halluci_monitor.py          # Main API and orchestration
├── config.py                   # Configuration dataclasses
├── detectors/                  # Detection modules
│   ├── self_consistency.py     # SC: k-resampling + semantic agreement
│   ├── nli_faithfulness.py     # NLI: sentence-level entailment
│   ├── numeric_sanity.py       # NS: arithmetic + unit consistency
│   ├── retrieval_support.py    # RS: context document verification
│   └── jailbreak_heuristics.py # JB: safety risk patterns
├── highlight/                  # Span highlighting utilities
│   ├── span_align.py          # Character/token span mapping
│   └── html_report.py         # HTML report generation
├── metrics/                    # Scoring and aggregation
│   └── scoring.py             # Risk score computation
└── examples/                   # Usage examples and test data
```

### Signal Flow

```
Input: (prompt, completion, context_docs)
    ↓
┌─────────────────────────────────────────────────────────┐
│                    Detectors                            │
├─────────────────────────────────────────────────────────┤
│ Self-Consistency  │ NLI Faithfulness │ Numeric Sanity   │
│ (k samples)       │ (entailment)     │ (arithmetic)     │
├─────────────────────────────────────────────────────────┤
│ Retrieval Support │ Jailbreak Heuristics               │
│ (context match)   │ (safety patterns)                  │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                    Aggregation                          │
├─────────────────────────────────────────────────────────┤
│ Weighted Risk Score = 1 - (w₁×NLI + w₂×SC + w₃×RS +   │
│                         w₄×NS + w₅×(1-JB))            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                    Output                               │
├─────────────────────────────────────────────────────────┤
│ • Risk Score [0,1]                                      │
│ • Risk Level (low/medium/high)                          │
│ • Individual Signal Scores                              │
│ • Highlighted Spans                                     │
│ • HTML Report (optional)                                │
└─────────────────────────────────────────────────────────┘
```

## Detection Signals

### 1. Self-Consistency (SC)

**Purpose**: Assess semantic consistency across multiple generations

**Method**:
- Generate k samples from the model using the same prompt
- Compute pairwise cosine similarity using sentence embeddings
- Score = mean pairwise similarity in [0,1]

**Implementation**:
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Fallback to character-based similarity if model unavailable
- Configurable k, temperature, and max tokens

**Weight**: 0.25 (default)

### 2. NLI Faithfulness (NLI)

**Purpose**: Check if completion sentences are entailed by prompt + context

**Method**:
- Split completion into sentences
- For each sentence S: premise = prompt + context, hypothesis = S
- Use NLI model to compute entailment probability
- Score = mean entailment probability over sentences

**Implementation**:
- Uses lightweight NLI model (`microsoft/DialoGPT-medium`)
- Fallback to word overlap if model unavailable
- Configurable entailment threshold

**Weight**: 0.35 (default)

### 3. Numeric Sanity (NS)

**Purpose**: Detect arithmetic and unit conversion errors

**Method**:
- Extract numbers and units using regex
- Check unit conversions (km↔mi, kg↔lb, °C↔°F, etc.)
- Verify simple arithmetic relationships
- Score = fraction of numeric assertions passing checks

**Implementation**:
- Comprehensive unit conversion library
- Tolerance-based arithmetic checking
- Handles common units and conversions

**Weight**: 0.15 (default)

### 4. Retrieval Support (RS)

**Purpose**: Verify claims against provided context documents

**Method**:
- Chunk context documents into segments
- For each sentence: compute similarity with all chunks
- Support if any chunk has similarity ≥ threshold
- Score = fraction of sentences supported

**Implementation**:
- Uses sentence transformers for semantic similarity
- Fallback to lexical overlap
- Configurable similarity threshold and chunk size

**Weight**: 0.20 (default)

### 5. Jailbreak Heuristics (JB)

**Purpose**: Detect potential safety risks and jailbreak attempts

**Method**:
- Pattern matching for jailbreak indicators
- Keyword analysis for high/medium risk terms
- Formatting analysis (excessive caps, punctuation)
- Score = weighted risk assessment [0,1]

**Implementation**:
- Regex patterns for common jailbreak attempts
- Keyword lists for risk assessment
- Formatting heuristics

**Weight**: 0.05 (default)

## Scoring Algorithm

### Risk Score Computation

```
risk_score = 1 - (w₁×NLI + w₂×SC + w₃×RS + w₄×NS + w₅×(1-JB))
```

Where:
- w₁ = 0.35 (NLI weight)
- w₂ = 0.25 (SC weight)
- w₃ = 0.20 (RS weight)
- w₄ = 0.15 (NS weight)
- w₅ = 0.05 (JB weight)

### Risk Level Classification

- **Low**: risk_score < 0.4
- **Medium**: 0.4 ≤ risk_score < 0.7
- **High**: risk_score ≥ 0.7

## Configuration

### MonitorConfig

```python
@dataclass
class MonitorConfig:
    # Self-consistency parameters
    k_samples: int = 5
    temperature: float = 0.7
    max_new_tokens: int = 512
    
    # Feature toggles
    enable_retrieval_support: bool = True
    enable_jailbreak_heuristics: bool = True
    
    # Scoring parameters
    thresholds: MonitorThresholds = MonitorThresholds()
    weights: Dict[str, float] = field(default_factory=lambda: {...})
    
    # Output options
    html_report: bool = True
    report_dir: str = "runs"
```

### Customization Points

1. **Weights**: Adjust signal importance for your use case
2. **Thresholds**: Modify risk level boundaries
3. **Models**: Replace default models with custom ones
4. **Features**: Enable/disable specific detectors

## API Design

### Main Interface

```python
class HallucinationMonitor:
    def __init__(self, cfg: MonitorConfig = None, model=None, tokenizer=None)
    
    def evaluate(
        self,
        prompt: str,
        completion: str,
        context_docs: List[str] = None
    ) -> Dict[str, Any]
    
    def suggest_cautious_decoding(self, risk_level: str) -> Dict[str, Any]
    def ask_for_citations(self, risk_level: str) -> str
```

### Output Format

```python
{
    "risk_score": 0.65,
    "risk_level": "medium",
    "signals": {
        "self_consistency": 0.8,
        "nli_faithfulness": 0.6,
        "numeric_sanity": 0.9,
        "retrieval_support": 0.7,
        "jailbreak_heuristics": 0.1
    },
    "spans": [
        {
            "text": "contradictory sentence",
            "start": 100,
            "end": 120,
            "reason": "NLI: contradiction",
            "color": "red"
        }
    ],
    "artifacts": {
        "html_report": "/path/to/report.html"
    },
    "config": {...}
}
```

## CLI Interface

### Basic Usage

```bash
gpt-oss-monitor --prompt prompt.txt --completion output.txt
```

### Advanced Usage

```bash
gpt-oss-monitor \
  --prompt prompt.txt \
  --completion output.txt \
  --contexts ctx1.txt ctx2.txt \
  --k 10 \
  --temperature 0.8 \
  --high-threshold 0.8 \
  --html \
  --report-dir reports/
```

## Performance Considerations

### Optimization Strategies

1. **Model Loading**: Lazy loading with fallbacks
2. **Caching**: Embedding and similarity caching
3. **Parallelization**: Independent detector execution
4. **Memory Management**: Efficient tensor handling

### Resource Requirements

- **CPU**: Lightweight fallback mode available
- **GPU**: Optional for faster inference
- **Memory**: ~2GB for full model loading
- **Storage**: ~500MB for model downloads

## Limitations and Future Work

### Current Limitations

1. **Model Dependencies**: Requires internet for first-time model download
2. **Language Support**: Primarily English-focused
3. **Context Length**: Limited by model context windows
4. **Temporal Claims**: No temporal consistency checking

### Future Enhancements

1. **Temporal Consistency**: Check temporal claim validity
2. **Entropy-based Uncertainty**: Model confidence estimation
3. **LLM-as-Judge**: Use LLM to evaluate other LLM outputs
4. **Multi-language Support**: Extend to other languages
5. **Custom Detectors**: Plugin architecture for custom signals

## Testing Strategy

### Test Categories

1. **Unit Tests**: Individual detector functionality
2. **Integration Tests**: End-to-end monitoring pipeline
3. **Performance Tests**: Speed and memory usage
4. **Regression Tests**: Ensure consistent results

### Test Data

- **TruthfulQA**: Factual accuracy evaluation
- **FEVER**: Fact extraction and verification
- **Custom Examples**: Domain-specific test cases

## Deployment Considerations

### Production Use

1. **Model Caching**: Cache models for faster startup
2. **Error Handling**: Graceful degradation on failures
3. **Monitoring**: Track performance and accuracy
4. **Scaling**: Horizontal scaling for high throughput

### Security

1. **Input Validation**: Sanitize all inputs
2. **Model Isolation**: Sandbox model execution
3. **Access Control**: Restrict sensitive operations
4. **Audit Logging**: Track all evaluations

## Contributing

### Extension Points

1. **Custom Detectors**: Implement new detection signals
2. **Model Adapters**: Support additional model types
3. **Report Formats**: Add new output formats
4. **Integration**: Connect with existing systems

### Development Guidelines

1. **Type Hints**: Full type annotation
2. **Documentation**: Comprehensive docstrings
3. **Testing**: High test coverage
4. **Performance**: Benchmark critical paths
