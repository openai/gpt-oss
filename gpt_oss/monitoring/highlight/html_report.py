"""
HTML report generator for hallucination monitoring results.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from jinja2 import Template


class HTMLReportGenerator:
    """Generator for HTML hallucination monitoring reports."""
    
    def __init__(self):
        """Initialize the HTML report generator."""
        self.template = self._get_template()
    
    def _get_template(self) -> Template:
        """Get the HTML template for reports."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-OSS Hallucination Analysis Report</title>
    <style>
        :root {
            --primary-color: #667eea;
            --primary-dark: #5a6fd8;
            --primary-light: #764ba2;
            --secondary-color: #f8f9ff;
            --text-primary: #1a1a1a;
            --text-secondary: #6e6e80;
            --border-color: #e5e5e5;
            --success-color: #10a37f;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --background: #ffffff;
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: var(--background);
            color: var(--text-primary);
            font-size: 14px;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: var(--background);
        }
        
        .header {
            background: var(--gradient-primary);
            padding: 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .logo-section {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo {
            width: 48px;
            height: 48px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
        }
        
        .header-title {
            font-size: 24px;
            font-weight: 700;
            color: white;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .timestamp {
            color: rgba(255,255,255,0.9);
            font-size: 13px;
            font-weight: 500;
        }
        
        .content {
            padding: 32px;
        }
        
        .summary-card {
            background: var(--background);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 32px;
        }
        
        .summary-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
        }
        
        .summary-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0;
        }
        
        .risk-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 13px;
        }
        
        .risk-low {
            background: rgba(16, 163, 127, 0.1);
            color: var(--success-color);
        }
        
        .risk-medium {
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning-color);
        }
        
        .risk-high {
            background: rgba(239, 68, 68, 0.1);
            color: var(--error-color);
        }
        
        .risk-score {
            font-size: 48px;
            font-weight: 700;
            color: var(--primary-color);
        }
        
        .risk-score-section {
            display: flex;
            align-items: center;
            gap: 24px;
            margin-bottom: 24px;
        }
        
        .risk-score {
            font-size: 48px;
            font-weight: 700;
            color: var(--text-primary);
        }
        
        .risk-score-label {
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
        }
        
        .signals-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }
        
        .signal-card {
            background: var(--secondary-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            transition: all 0.2s ease;
        }
        
        .signal-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
            border-color: var(--primary-color);
        }
        
        .signal-name {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .signal-score {
            font-size: 24px;
            font-weight: 700;
            color: var(--primary-color);
        }
        
        .analysis-section {
            margin-bottom: 32px;
        }
        
        .section-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 16px;
        }
        
        .completion-card {
            background: var(--secondary-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }
        
        .completion-text {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            color: var(--text-primary);
        }
        
        .highlighted-text {
            padding: 2px 4px;
            border-radius: 4px;
            font-weight: 500;
        }
        
        .highlight-nli { background-color: rgba(239, 68, 68, 0.15); color: #dc2626; }
        .highlight-sc { background-color: rgba(245, 158, 11, 0.15); color: #d97706; }
        .highlight-ns { background-color: rgba(16, 163, 127, 0.15); color: #059669; }
        .highlight-rs { background-color: rgba(99, 102, 241, 0.15); color: #7c3aed; }
        .highlight-jb { background-color: rgba(236, 72, 153, 0.15); color: #be185d; }
        
        .spans-section {
            margin-top: 16px;
        }
        
        .span-item {
            background: var(--background);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
            font-size: 13px;
        }
        
        .span-text {
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 4px;
        }
        
        .span-meta {
            color: var(--text-secondary);
            font-size: 12px;
        }
        
        .config-section {
            background: var(--secondary-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-top: 32px;
        }
        
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        
        .config-item {
            background: var(--background);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 12px;
        }
        
        .config-label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        
        .config-value {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            font-size: 13px;
            color: var(--text-primary);
        }
        
        .footer {
            background: var(--gradient-primary);
            padding: 24px 32px;
            text-align: center;
            color: rgba(255,255,255,0.9);
            font-size: 13px;
        }
        
        @media (max-width: 768px) {
            .header {
                padding: 16px 20px;
                flex-direction: column;
                gap: 12px;
                align-items: flex-start;
            }
            
            .content {
                padding: 20px;
            }
            
            .risk-score-section {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            
            .signals-grid {
                grid-template-columns: 1fr;
            }
            
            .config-grid {
                grid-template-columns: 1fr;
            }
        }
        .content {
            padding: 30px;
        }
        .summary {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }
        .risk-score {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .risk-low { color: #28a745; }
        .risk-medium { color: #ffc107; }
        .risk-high { color: #dc3545; }
        .signals {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .signal-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .signal-score {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .signal-name {
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        .completion-section {
            margin: 30px 0;
        }
        .completion-text {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            line-height: 1.8;
        }
        .highlighted-text {
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
        }
        .highlight-green { background-color: #d4edda; color: #155724; }
        .highlight-yellow { background-color: #fff3cd; color: #856404; }
        .highlight-red { background-color: #f8d7da; color: #721c24; }
        .highlight-purple { background-color: #e2d9f3; color: #5a2d82; }
        .highlight-blue { background-color: #d1ecf1; color: #0c5460; }
        .details-section {
            margin: 30px 0;
        }
        .collapsible {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin: 10px 0;
            overflow: hidden;
        }
        .collapsible-header {
            background: #e9ecef;
            padding: 15px 20px;
            cursor: pointer;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .collapsible-header:hover {
            background: #dee2e6;
        }
        .collapsible-content {
            padding: 20px;
            display: none;
        }
        .collapsible-content.active {
            display: block;
        }
        .toggle-icon {
            transition: transform 0.3s;
        }
        .toggle-icon.rotated {
            transform: rotate(180deg);
        }
        .config-section {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .config-item {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .config-label {
            font-weight: 600;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .config-value {
            margin-top: 5px;
            font-family: 'Courier New', monospace;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }
        @media (max-width: 768px) {
            .signals {
                grid-template-columns: 1fr;
            }
            .config-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo-section">
                <svg class="logo" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M24 3L3 12L24 21L45 12L24 3Z" fill="white"/>
                    <path d="M3 12V36L24 42V21L3 12Z" fill="#f0f0f0"/>
                    <path d="M45 12V36L24 42V21L45 12Z" fill="#f0f0f0"/>
                </svg>
                <div>
                    <h1 class="header-title">GPT-OSS Hallucination Analysis</h1>
                    <p style="margin: 4px 0 0 0; font-size: 14px; color: rgba(255,255,255,0.9); font-weight: 400;">Advanced AI Safety & Reliability Analysis</p>
                </div>
            </div>
            <div class="timestamp">{{ timestamp }}</div>
        </div>
        
        <div class="content">
            <div class="summary-card">
                <div class="summary-header">
                    <h2 class="summary-title">Risk Assessment</h2>
                    <div class="risk-indicator risk-{{ risk_level }}">
                        {{ risk_level.upper() }} RISK
                    </div>
                </div>
                
                <div class="risk-score-section">
                    <div>
                        <div class="risk-score">{{ "%.1f"|format(risk_score * 100) }}%</div>
                        <div class="risk-score-label">Hallucination Risk Score</div>
                    </div>
                </div>
                
                <div class="signals-grid">
                    {% for signal_name, signal_data in signals.items() %}
                    <div class="signal-card">
                        <div class="signal-name">{{ signal_name.replace('_', ' ').upper() }}</div>
                        <div class="signal-score">{{ "%.1f"|format(signal_data.score * 100) }}%</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="analysis-section">
                <h3 class="section-title">Analyzed Completion</h3>
                <div class="completion-card">
                    <div class="completion-text">{{ highlighted_completion }}</div>
                    
                    {% if spans %}
                    <div class="spans-section">
                        <h4 style="margin: 16px 0 8px 0; font-size: 14px; font-weight: 600; color: var(--text-primary);">Flagged Spans</h4>
                        {% for span in spans %}
                        <div class="span-item">
                            <div class="span-text">"{{ span.text }}"</div>
                            <div class="span-meta">{{ span.signal.upper() }} • {{ span.risk.upper() }} RISK</div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
            </div>
            
            {% if detector_details %}
            <div class="analysis-section">
                <h3 class="section-title">Detailed Analysis</h3>
                
                {% for detector_name, detector_data in detector_details.items() %}
                <div class="completion-card">
                    <h4 style="margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: var(--text-primary);">
                        {{ detector_name.replace('_', ' ').title() }}
                    </h4>
                    <pre style="font-size: 12px; color: var(--text-secondary); margin: 0; white-space: pre-wrap;">{{ detector_data | tojson(indent=2) }}</pre>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="config-section">
                <h3 class="section-title">Configuration</h3>
                <div class="config-grid">
                    {% for key, value in config.items() %}
                    <div class="config-item">
                        <div class="config-label">{{ key.replace('_', ' ').upper() }}</div>
                        <div class="config-value">{{ value }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by GPT-OSS Hallucination Monitor v{{ version }}</p>
            <p>For more information, visit the <a href="https://github.com/openai/gpt-oss" style="color: white; text-decoration: none; font-weight: 600;">GPT-OSS repository</a></p>
        </div>
    </div>
    
    <script>
        function toggleSection(sectionId) {
            const content = document.getElementById('content-' + sectionId);
            const icon = document.getElementById('icon-' + sectionId);
            
            content.classList.toggle('active');
            icon.classList.toggle('rotated');
        }
        
        // Auto-expand high-risk sections
        document.addEventListener('DOMContentLoaded', function() {
            const riskLevel = '{{ risk_level }}';
            if (riskLevel === 'high') {
                // Auto-expand all sections for high-risk cases
                document.querySelectorAll('.collapsible-content').forEach(content => {
                    content.classList.add('active');
                });
                document.querySelectorAll('.toggle-icon').forEach(icon => {
                    icon.classList.add('rotated');
                });
            }
        });
    </script>
</body>
</html>
        """
        return Template(template_str)
    
    def _highlight_text(self, text: str, spans: List[Dict[str, Any]]) -> str:
        """Highlight text with spans using HTML."""
        if not spans:
            return text
        
        # Sort spans by start position (descending) to avoid index issues
        sorted_spans = sorted(spans, key=lambda x: x['start'], reverse=True)
        
        highlighted_text = text
        
        for span in sorted_spans:
            start = span['start']
            end = span['end']
            color = span.get('color', 'blue')
            reason = span.get('reason', '')
            
            # Create highlight span
            highlight_html = f'<span class="highlighted-text highlight-{color}" title="{reason}">{highlighted_text[start:end]}</span>'
            
            # Replace the text
            highlighted_text = highlighted_text[:start] + highlight_html + highlighted_text[end:]
        
        return highlighted_text
    
    def _get_signal_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each signal type."""
        return {
            'self_consistency': 'Semantic agreement across multiple generations',
            'nli_faithfulness': 'Entailment against prompt and context',
            'numeric_sanity': 'Arithmetic and unit consistency',
            'retrieval_support': 'Support from provided context documents',
            'jailbreak_heuristics': 'Safety risk assessment'
        }
    
    def generate_report(self, results: Dict[str, Any], config: Dict[str, Any], 
                       version: str = "0.1.0") -> str:
        """Generate HTML report from monitoring results."""
        
        # Prepare data for template
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_score = results['risk_score']  # Keep as float for template
        risk_level = results['risk_level']
        
        # Prepare signals data
        signals = {}
        signal_descriptions = self._get_signal_descriptions()
        
        for signal_name, score in results['signals'].items():
            signals[signal_name] = {
                'score': score,
                'description': signal_descriptions.get(signal_name, 'Signal score')
            }
        
        # Highlight completion text
        highlighted_completion = self._highlight_text(
            results.get('completion', ''),
            results.get('spans', [])
        )
        
        # Prepare detector details
        detector_details = {}
        for detector_name, details in results.get('detector_details', {}).items():
            detector_details[detector_name] = details
        
        # Render template
        html_content = self.template.render(
            timestamp=timestamp,
            risk_score=risk_score,
            risk_level=risk_level,
            signals=signals,
            highlighted_completion=highlighted_completion,
            detector_details=detector_details,
            config=config,
            version=version
        )
        
        return html_content
    
    def save_report(self, results: Dict[str, Any], config: Dict[str, Any], 
                   output_path: str, version: str = "0.1.0") -> str:
        """Generate and save HTML report to file."""
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate HTML content
        html_content = self.generate_report(results, config, version)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
