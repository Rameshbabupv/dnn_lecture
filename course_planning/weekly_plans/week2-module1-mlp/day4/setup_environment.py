#!/usr/bin/env python3
"""
Environment Setup Helper Script
Tutorial T3: TensorFlow Basic Operations Exercises

This script helps students set up their environment interactively.
"""

import os
import sys
import subprocess
import platform

def print_header():
    print("=" * 70)
    print("🚀 TensorFlow Exercises Environment Setup Helper")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required. Current version:", sys.version)
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True

def detect_package_managers():
    """Detect available package managers"""
    managers = {}
    
    # Check for conda
    try:
        subprocess.run(['conda', '--version'], capture_output=True, check=True)
        managers['conda'] = True
        print("✅ Conda detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        managers['conda'] = False
        print("❌ Conda not found")
    
    # Check for pipenv
    try:
        subprocess.run(['pipenv', '--version'], capture_output=True, check=True)
        managers['pipenv'] = True
        print("✅ Pipenv detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        managers['pipenv'] = False
        print("❌ Pipenv not found")
    
    # venv is built-in with Python 3.3+
    managers['venv'] = True
    print("✅ Python venv available (built-in)")
    
    return managers

def show_setup_options(managers):
    """Display setup options based on available managers"""
    print("\n🛠️  Available Setup Options:")
    print("-" * 40)
    
    options = []
    
    if managers['conda']:
        options.append(("1", "Conda Environment", "conda"))
        print("1. Conda Environment (recommended for data science)")
    
    if managers['venv']:
        options.append(("2", "Python venv", "venv"))
        print("2. Python venv (built-in, lightweight)")
    
    if managers['pipenv']:
        options.append(("3", "Pipenv", "pipenv"))
        print("3. Pipenv (modern dependency management)")
    
    options.append(("4", "Manual Installation", "manual"))
    print("4. Manual Installation (install directly to system)")
    
    options.append(("5", "Google Colab", "colab"))
    print("5. Google Colab Instructions (no local setup)")
    
    options.append(("6", "Show All Commands", "commands"))
    print("6. Show All Setup Commands (copy-paste)")
    
    return options

def setup_conda():
    """Setup conda environment"""
    print("\n🅰️ Setting up Conda Environment...")
    print("Run these commands:")
    print()
    print("conda create -n tensorflow_env python=3.9")
    print("conda activate tensorflow_env")
    print("conda install tensorflow numpy matplotlib")
    print()
    print("Or use pip within conda:")
    print("pip install -r requirements.txt")
    print()
    print("To activate later: conda activate tensorflow_env")

def setup_venv():
    """Setup Python venv"""
    print("\n🅱️ Setting up Python venv...")
    print("Run these commands:")
    print()
    
    if platform.system() == "Windows":
        print("python -m venv tensorflow_env")
        print("tensorflow_env\\Scripts\\activate")
    else:
        print("python3 -m venv tensorflow_env")
        print("source tensorflow_env/bin/activate")
    
    print("python -m pip install --upgrade pip")
    print("pip install -r requirements.txt")
    print()
    if platform.system() == "Windows":
        print("To activate later: tensorflow_env\\Scripts\\activate")
    else:
        print("To activate later: source tensorflow_env/bin/activate")

def setup_pipenv():
    """Setup pipenv environment"""
    print("\n🅲️ Setting up Pipenv...")
    print("Run these commands:")
    print()
    print("pip install pipenv  # if not already installed")
    print("pipenv install -r requirements.txt")
    print("pipenv shell")
    print()
    print("To activate later: pipenv shell")

def setup_manual():
    """Manual installation instructions"""
    print("\n🅳️ Manual Installation...")
    print("Run this command:")
    print()
    print("pip install -r requirements.txt")
    print()
    print("⚠️  Note: This installs packages globally. Virtual environments are recommended.")

def setup_colab():
    """Google Colab instructions"""
    print("\n🅴️ Google Colab Setup...")
    print("1. Go to: https://colab.research.google.com/")
    print("2. Create a new notebook")
    print("3. Upload the exercise file or copy-paste the code")
    print("4. TensorFlow is pre-installed!")
    print("5. Run: !python t3_tensorflow_basic_operations_exercises.py --list")
    print()
    print("✅ No local installation needed!")

def show_all_commands():
    """Show all setup commands"""
    print("\n📋 All Setup Commands Reference:")
    print("=" * 50)
    
    print("\n🅰️ CONDA:")
    print("conda create -n tensorflow_env python=3.9")
    print("conda activate tensorflow_env")
    print("pip install -r requirements.txt")
    
    print("\n🅱️ PYTHON VENV:")
    if platform.system() == "Windows":
        print("python -m venv tensorflow_env")
        print("tensorflow_env\\Scripts\\activate")
    else:
        print("python3 -m venv tensorflow_env")
        print("source tensorflow_env/bin/activate")
    print("pip install -r requirements.txt")
    
    print("\n🅲️ PIPENV:")
    print("pip install pipenv")
    print("pipenv install -r requirements.txt")
    print("pipenv shell")
    
    print("\n🅳️ MANUAL:")
    print("pip install -r requirements.txt")

def main():
    print_header()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python and try again.")
        return
    
    print()
    
    # Detect package managers
    managers = detect_package_managers()
    
    # Show options
    options = show_setup_options(managers)
    
    print("\n❓ Choose your preferred setup method (1-6): ", end="")
    
    try:
        choice = input().strip()
        
        # Execute based on choice
        setup_functions = {
            "1": setup_conda,
            "2": setup_venv,
            "3": setup_pipenv,
            "4": setup_manual,
            "5": setup_colab,
            "6": show_all_commands
        }
        
        if choice in setup_functions:
            setup_functions[choice]()
        else:
            print("❌ Invalid choice. Please run the script again.")
            return
        
        if choice not in ["5", "6"]:  # Not Colab or show commands
            print("\n✅ Setup complete! Next steps:")
            print("1. Activate your environment (see commands above)")
            print("2. Run: python t3_tensorflow_basic_operations_exercises.py --list")
            print("3. Start with: python t3_tensorflow_basic_operations_exercises.py 1")
    
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()