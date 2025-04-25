#!/usr/bin/env python3
"""
Encrypted ASI Engine Setup
--------------------------
This script sets up the encrypted ASI engine and prepares it for use.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def setup_asi_engine(license_key=None):
    """
    Set up the ASI engine for development use.
    
    Args:
        license_key: Optional custom license key
    """
    print("\n" + "=" * 80)
    print("ENCRYPTED ASI ENGINE SETUP".center(80))
    print("=" * 80)
    
    # Default license key for public API access
    if not license_key:
        license_key = "ASI-PUBLIC-INTERFACE-000"
        print(f"\nUsing default public license key: {license_key}")
    else:
        print(f"\nUsing custom license key: {license_key}")
    
    # Create necessary directories
    project_root = Path(__file__).parent
    security_dir = project_root / "unreal_asi/security"
    security_dir.mkdir(parents=True, exist_ok=True)
    
    # Save license key
    license_file = security_dir / "license_key.txt"
    with open(license_file, 'w') as f:
        f.write(license_key)
    
    print("\nSetting up ASI engine components...")
    
    # Initialize ASI system
    try:
        from unreal_asi.asi_public_api import initialize_asi
        success = initialize_asi(license_key)
        
        if success:
            print("\nASI engine initialized successfully!")
        else:
            print("\nWarning: ASI engine initialization returned failure status.")
            print("Some features may be limited.")
    except Exception as e:
        print(f"\nError during initialization: {e}")
        print("You may need to check your installation.")
    
    # Create reports directory for examples
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETE".center(80))
    print("=" * 80)
    
    print("\nThe ASI engine is ready for use.")
    print("To use the ASI engine in your applications:")
    print("1. Import from unreal_asi.asi_public_api")
    print("2. Initialize the ASI engine with initialize_asi()")
    print("3. Create an ASI instance with create_asi_instance()")
    
    print("\nTry running one of the example applications:")
    print("  python unreal_asi/applications/examples/healthcare_application.py")
    print("  python unreal_asi/applications/examples/finance_application.py")
    
    return True

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Set up the encrypted ASI engine")
    parser.add_argument("--license", type=str, help="Custom license key")
    args = parser.parse_args()
    
    # Setup ASI engine
    setup_asi_engine(args.license)
