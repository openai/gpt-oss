#!/usr/bin/env python3
"""
Sexy Web Interface for GPT-OSS Hallucination Monitor
"""

import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import our monitoring system
try:
    from gpt_oss.monitoring import HallucinationMonitor, MonitorConfig, MonitorThresholds
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    st.error("⚠️ Monitoring dependencies not installed. Run: `pip install -e '.[monitoring]'`")

# Page config
st.set_page_config(
    page_title="GPT-OSS Hallucination Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sexy styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .risk-high { border-left-color: #ff4757; }
    .risk-medium { border-left-color: #ffa502; }
    .risk-low { border-left-color: #2ed573; }
    .signal-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def create_risk_gauge(score: float) -> go.Figure:
    """Create a beautiful gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Hallucination Risk Score", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#2ed573'},
                {'range': [30, 60], 'color': '#ffa502'},
                {'range': [60, 100], 'color': '#ff4757'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_signals_radar(signals: Dict[str, Any]) -> go.Figure:
    """Create a radar chart for signal scores"""
    categories = list(signals.keys())
    values = [signals[cat]['score'] * 100 for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Signal Scores',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_signal_bar_chart(signals: Dict[str, Any]) -> go.Figure:
    """Create a bar chart for signal scores"""
    categories = list(signals.keys())
    values = [signals[cat]['score'] * 100 for cat in categories]
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors[:len(categories)],
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Signal Scores Breakdown",
        xaxis_title="Signals",
        yaxis_title="Score (%)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def get_risk_color(risk_level: str) -> str:
    """Get color for risk level"""
    colors = {
        'low': '#2ed573',
        'medium': '#ffa502', 
        'high': '#ff4757',
        'critical': '#ff3838'
    }
    return colors.get(risk_level, '#667eea')

def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px 0; margin: -1rem -1rem 2rem -1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div style="max-width: 1000px; margin: 0 auto; display: flex; align-items: center; gap: 16px;">
            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHZpZXdCb3g9IjAgMCA0OCA0OCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTI0IDNMMyAxMkwyNCAyMUw0NSAxMkwyNCAzWiIgZmlsbD0id2hpdGUiLz4KPHBhdGggZD0iTTMgMTJWMzZMMjQgNDJWMjFMMyAxMloiIGZpbGw9IiNmMGYwZjAiLz4KPHBhdGggZD0iTTQ1IDEyVjM2TDI0IDQyVjIxTDQ1IDEyWiIgZmlsbD0iI2YwZjBmMCIvPgo8L3N2Zz4K" width="48" height="48" style="filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">
            <div>
                <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">GPT-OSS Hallucination Monitor</h1>
                <p style="margin: 4px 0 0 0; font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 400;">Advanced AI Safety & Reliability Analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not MONITOR_AVAILABLE:
        st.error("""
        ## Installation Required
        
        To use this web interface, install the monitoring dependencies:
        
        ```bash
        pip install -e ".[monitoring]"
        ```
        
        Or install individually:
        ```bash
        pip install numpy scipy regex tqdm sentence-transformers transformers torch jinja2
        ```
        """)
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Feature toggles
        st.subheader("Detection Signals")
        enable_nli = st.checkbox("NLI Faithfulness", value=True, help="Sentence-level entailment checking")
        enable_sc = st.checkbox("Self-Consistency", value=True, help="Multi-sample semantic agreement")
        enable_ns = st.checkbox("Numeric Sanity", value=True, help="Arithmetic & unit validation")
        enable_rs = st.checkbox("Retrieval Support", value=True, help="Claim verification against context")
        enable_jb = st.checkbox("Jailbreak Heuristics", value=True, help="Safety risk detection")
        
        # Weights
        st.subheader("Signal Weights")
        col1, col2 = st.columns(2)
        with col1:
            nli_weight = st.slider("NLI", 0.0, 1.0, 0.35, 0.05)
            sc_weight = st.slider("SC", 0.0, 1.0, 0.25, 0.05)
            ns_weight = st.slider("NS", 0.0, 1.0, 0.15, 0.05)
        with col2:
            rs_weight = st.slider("RS", 0.0, 1.0, 0.20, 0.05)
            jb_weight = st.slider("JB", 0.0, 1.0, 0.05, 0.05)
        
        # Thresholds
        st.subheader("Risk Thresholds")
        low_threshold = st.slider("Low", 0.0, 1.0, 0.3, 0.05)
        medium_threshold = st.slider("Medium", 0.0, 1.0, 0.6, 0.05)
        high_threshold = st.slider("High", 0.0, 1.0, 0.8, 0.05)
        
        # Advanced options
        with st.expander("Advanced Options"):
            k_samples = st.number_input("SC Samples (k)", 2, 10, 3)
            temperature = st.slider("SC Temperature", 0.1, 2.0, 0.7, 0.1)
            max_tokens = st.number_input("SC Max Tokens", 50, 500, 100)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Input tabs
        tab1, tab2, tab3 = st.tabs(["Simple", "Advanced", "Batch"])
        
        with tab1:
            prompt = st.text_area(
                "Prompt",
                placeholder="Enter your prompt here...",
                height=100,
                help="The input prompt given to the model"
            )
            
            completion = st.text_area(
                "Model Completion",
                placeholder="Enter the model's response here...",
                height=150,
                help="The model's generated response to analyze"
            )
            
            context = st.text_area(
                "Context (Optional)",
                placeholder="Enter supporting context documents...",
                height=100,
                help="Additional context to verify claims against"
            )
            
            contexts = [context] if context.strip() else []
        
        with tab2:
            st.info("Advanced mode for detailed configuration")
            # Could add more advanced options here
        
        with tab3:
            st.info("Batch processing coming soon!")
            # Could add file upload for batch processing
    
    with col2:
        st.header("🎯 Quick Examples")
        
        examples = {
            "Truthful": {
                "prompt": "What is the capital of France?",
                "completion": "Paris is the capital of France.",
                "context": "France is a country in Europe. Paris is its capital city."
            },
            "Hallucinated": {
                "prompt": "What is the population of Atlantis?",
                "completion": "Atlantis has a population of 2.3 million people and is located in the Atlantic Ocean.",
                "context": "Atlantis is a fictional island mentioned in Plato's works."
            },
            "Numeric Error": {
                "prompt": "What is 15 + 27?",
                "completion": "15 + 27 = 45",
                "context": ""
            }
        }
        
        for name, example in examples.items():
            if st.button(f"Load {name}"):
                st.session_state.prompt = example["prompt"]
                st.session_state.completion = example["completion"]
                st.session_state.context = example["context"]
                st.rerun()
    
    # Analysis button
    if st.button("🔍 Analyze Hallucination Risk", type="primary"):
        if not prompt or not completion:
            st.error("Please provide both prompt and completion!")
            return
        
        # Show loading
        with st.spinner("Analyzing... This may take a few moments."):
            try:
                # Create config
                config = MonitorConfig(
                    weights={
                        'nli': nli_weight if enable_nli else 0,
                        'sc': sc_weight if enable_sc else 0,
                        'rs': rs_weight if enable_rs else 0,
                        'ns': ns_weight if enable_ns else 0,
                        'jb': jb_weight if enable_jb else 0
                    },
                    thresholds=MonitorThresholds(
                        low=low_threshold,
                        medium=medium_threshold,
                        high=high_threshold
                    ),
                    enable_retrieval_support=enable_rs,
                    enable_jailbreak_heuristics=enable_jb,
                    k_samples=k_samples,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                )
                
                # Initialize monitor
                monitor = HallucinationMonitor(config=config)
                
                # Run analysis
                result = monitor.evaluate(
                    prompt=prompt,
                    completion=completion,
                    contexts=contexts,
                    generate_html_report=True
                )
                
                # Store result in session state
                st.session_state.result = result
                st.session_state.analysis_done = True
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.get('analysis_done', False) and 'result' in st.session_state:
        result = st.session_state.result
        
        st.header("📊 Analysis Results")
        
        # Risk score gauge
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_gauge = create_risk_gauge(result['risk_score'])
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            risk_level = result['risk_level']
            risk_color = get_risk_color(risk_level)
            
            st.markdown(f"""
            <div class="metric-card risk-{risk_level}">
                <h3>Risk Level</h3>
                <h2 style="color: {risk_color};">{risk_level.upper()}</h2>
                <p>Score: {result['risk_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick stats
            signals = result['signals']
            active_signals = len([s for s in signals.values() if s['score'] > 0])
            st.metric("Active Signals", active_signals)
            st.metric("Flagged Spans", len(result.get('spans', [])))
        
        # Signal breakdown
        st.subheader("🔍 Signal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_radar = create_signals_radar(signals)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            fig_bar = create_signal_bar_chart(signals)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed signal cards
        st.subheader("📋 Signal Details")
        
        for signal_name, signal_data in signals.items():
            if signal_data['score'] > 0:
                score_pct = signal_data['score'] * 100
                color = get_risk_color('high' if score_pct > 60 else 'medium' if score_pct > 30 else 'low')
                
                st.markdown(f"""
                <div class="signal-card">
                    <h4>{signal_name.upper()} - {score_pct:.1f}%</h4>
                    <p style="color: {color};">{signal_data.get('reason', 'No details available')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Spans highlighting
        if result.get('spans'):
            st.subheader("🎯 Highlighted Issues")
            
            # Create highlighted text
            text = completion
            spans = sorted(result['spans'], key=lambda x: x['start'])
            
            highlighted_text = ""
            last_end = 0
            
            for span in spans:
                highlighted_text += text[last_end:span['start']]
                risk_color = get_risk_color(span.get('risk', 'medium'))
                highlighted_text += f'<span style="background-color: {risk_color}; color: white; padding: 2px 4px; border-radius: 3px;" title="{span.get("signal", "unknown")}">{text[span["start"]:span["end"]]}</span>'
                last_end = span['end']
            
            highlighted_text += text[last_end:]
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
                <h4>Flagged Text:</h4>
                <p>{highlighted_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # HTML Report
        if result.get('artifacts', {}).get('html_report'):
            st.subheader("📄 Detailed Report")
            
            with open(result['artifacts']['html_report'], 'r') as f:
                html_content = f.read()
            
            st.download_button(
                label="📥 Download HTML Report",
                data=html_content,
                file_name=f"hallucination_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
            
            # Show preview
            with st.expander("Preview HTML Report"):
                st.components.v1.html(html_content, height=600, scrolling=True)
        
        # Raw JSON
        with st.expander("🔧 Raw JSON Data"):
            st.json(result)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6e6e80; padding: 2rem; font-size: 13px;">
        <p>GPT-OSS Hallucination Monitor | Built for AI Safety & Reliability</p>
        <p>Powered by Self-Consistency, NLI Faithfulness, Numeric Sanity, and Retrieval Support</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
