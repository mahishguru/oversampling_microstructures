#!/usr/bin/env python3
"""
Pre-flight check script to verify pipeline requirements.
Run this before executing the main pipeline.
"""

import sys
import os


def check_imports():
    """Check if all required packages are available."""
    print("Checking required packages...")
    required_packages = {
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'imblearn': 'imbalanced-learn'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (MISSING)")
            missing.append(package_name)
    
    return missing


def check_input_directory():
    """Check if input directory exists and has data."""
    print("\nChecking input directory...")
    input_dir = "data/odf_harmonics"
    
    if not os.path.exists(input_dir):
        print(f"  ✗ Directory '{input_dir}' does not exist")
        return False
    
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if len(txt_files) == 0:
        print(f"  ✗ No .txt files found in '{input_dir}'")
        return False
    
    print(f"  ✓ Found {len(txt_files)} .txt files in '{input_dir}'")
    return True


def check_output_directories():
    """Check if output directories can be created."""
    print("\nChecking output directories...")
    dirs = ['synthetic_samples', 'visualizations', 'logs']
    
    for dir_name in dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"  ✓ {dir_name}/")
        except Exception as e:
            print(f"  ✗ {dir_name}/ (ERROR: {e})")
            return False
    
    return True


def check_python_version():
    """Check Python version."""
    print("\nChecking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires >= 3.7)")
        return False


def main():
    """Run all pre-flight checks."""
    print("="*70)
    print("ODF OVERSAMPLING PIPELINE - PRE-FLIGHT CHECK")
    print("="*70)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check imports
    missing_packages = check_imports()
    
    # Check input directory
    input_ok = check_input_directory()
    
    # Check output directories
    output_ok = check_output_directories()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if not python_ok:
        print("\n⚠ Python version is too old. Please upgrade to Python 3.7+")
    else:
        print("\n✓ Python version is compatible")
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nOr using requirements.txt:")
        print("  pip install -r requirements.txt")
    else:
        print("\n✓ All required packages are installed")
    
    if not input_ok:
        print("\n⚠ Input directory issue detected")
        print("  Make sure 'data/odf_harmonics/' exists and contains .txt files")
    else:
        print("✓ Input directory is ready")
    
    if not output_ok:
        print("\n⚠ Cannot create output directories")
    else:
        print("✓ Output directories are ready")
    
    if python_ok and not missing_packages and input_ok and output_ok:
        print("\n" + "="*70)
        print("🚀 ALL CHECKS PASSED - Ready to run pipeline!")
        print("="*70)
        print("\nRun the pipeline with:")
        print("  python main.py")
        return 0
    else:
        print("\n" + "="*70)
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
