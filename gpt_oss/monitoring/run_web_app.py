#!/usr/bin/env python3
"""
Launcher for GPT-OSS Hallucination Monitor Web Interface
"""

import sys
import subprocess
import importlib.util

def check_dependency(module_name, package_name=None):
    """Check if a dependency is available"""
    if package_name is None:
        package_name = module_name
    
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {package_name} not found")
        return False
    else:
        print(f"✅ {package_name} found")
        return True

def main():
    print("🔍 GPT-OSS Hallucination Monitor Web Interface")
    print("=" * 50)
    
    # Check core dependencies
    print("\n📦 Checking dependencies...")
    core_deps = [
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("gpt_oss.monitoring", "gpt-oss monitoring")
    ]
    
    missing_deps = []
    for module, package in core_deps:
        if not check_dependency(module, package):
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\n📥 Install them with:")
        print("pip install streamlit plotly")
        print("pip install -e '.[monitoring]'")
        return 1
    
    # Check optional ML dependencies
    print("\n🤖 Checking ML dependencies...")
    ml_deps = [
        ("sentence_transformers", "sentence-transformers"),
        ("transformers", "transformers"),
        ("torch", "torch")
    ]
    
    ml_available = True
    for module, package in ml_deps:
        if not check_dependency(module, package):
            ml_available = False
    
    if not ml_available:
        print("\n⚠️  Some ML dependencies are missing. The web interface will work with fallbacks.")
        print("📥 For full functionality, install:")
        print("pip install sentence-transformers transformers torch")
    
    print("\n🚀 Starting web interface...")
    print("📱 The interface will open in your browser at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the web app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "gpt_oss/monitoring/web_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        print(f"\n❌ Error starting web interface: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
