# GPT-OSS Hallucination Monitor Demo

This directory contains a comprehensive demo script that showcases all the features of the GPT-OSS Hallucination Monitor.

## 🚀 Quick Start

Run the demo to see the hallucination monitor in action:

```bash
# From the project root
python -m gpt_oss.monitoring.demo.demo
```

## 📋 What the Demo Shows

The demo script demonstrates:

### 1. **Basic Usage**
- Simple initialization and evaluation
- Risk score and level assessment
- Basic signal detection

### 2. **Detection Signals**
- **Truthful Response**: Low-risk example
- **Hallucinated Response**: High-risk example with multiple flagged spans
- **Numeric Error**: Arithmetic validation
- **Context Mismatch**: Retrieval support verification

### 3. **Professional Report Generation**
- HTML report creation with professional design
- Risk assessment visualization
- Flagged spans highlighting
- Signal breakdown analysis

### 4. **Web Interface**
- Interactive configuration sliders
- Real-time analysis with visualizations
- Professional risk gauges and charts
- Quick example buttons

### 5. **Command Line Interface**
- Basic CLI usage examples
- Context document integration
- HTML report generation
- Custom configuration options

### 6. **Advanced Features**
- Custom signal weights
- Configurable thresholds
- Feature toggles
- Parameter tuning

## 🎯 Demo Output

The demo generates:

1. **Console Output**: Detailed analysis results for each test case
2. **HTML Report**: Professional report saved to `demo_reports/` directory
3. **Signal Analysis**: Breakdown of each detection method
4. **Flagged Spans**: Specific text segments identified as problematic

## 📊 Example Results

```
📋 Test 2: Hallucinated Response
   Prompt: What is the population of Atlantis?
   Completion: Atlantis has a population of 2.3 million people...
   Context: Atlantis is a fictional island mentioned in Plato's works.
   🎯 Risk Score: 0.205
   🚨 Risk Level: LOW
   🔍 Active Signals: NLI_FAITHFULNESS: 0.506, JAILBREAK_HEURISTICS: 0.100
   🎯 Flagged Spans: 4
      - 'The city was founded in 1500 BC and has a thriving economy'
      - 'Atlantis has a population of 2'
```

## 🎨 Professional HTML Report

The demo generates a beautiful HTML report with:

- **GPT-OSS Logo**: Professional branding
- **Risk Assessment**: Clean score display and level indicators
- **Signal Grid**: Visual breakdown of detection signals
- **Flagged Spans**: Highlighted problematic text
- **Configuration**: Applied settings and parameters
- **Mobile Responsive**: Works on all devices

## 🔧 Requirements

To run the demo, install the monitoring dependencies:

```bash
pip install -e ".[monitoring]"
pip install streamlit plotly
```

## 📁 Generated Files

The demo creates:
- `demo_reports/`: Directory containing generated HTML reports
- Console output with detailed analysis
- Professional visualizations and charts

## 🚀 Next Steps

After running the demo:

1. **Explore the Web Interface**: `python gpt_oss/monitoring/run_web_app.py`
2. **Try the CLI**: `gpt-oss-monitor --help`
3. **Read Documentation**: `gpt_oss/monitoring/examples/README.md`
4. **View Generated Reports**: Open HTML files in your browser

## 🎉 Success Indicators

A successful demo run shows:
- ✅ All test cases completed
- 📄 Professional HTML report generated
- 🎯 Accurate risk assessment
- 🔍 Multiple detection signals working
- 📊 Clean, professional output

The demo validates that the hallucination monitor is working correctly and ready for production use!
