#!/usr/bin/env python3
"""
Run Real ASI Application
------------------------

This script serves as a centralized entry point for running real-world applications
with the actual ASI Engine components. It ensures that the ASI components are
properly initialized before running any application.

Usage:
    python3 run_real_asi_app.py [app_name] [app_arguments]

Examples:
    python3 run_real_asi_app.py content_summarizer --file document.txt
    python3 run_real_asi_app.py virtual_assistant --demo
    python3 run_real_asi_app.py document_assistant --file document.txt
"""

import os
import sys
import argparse
import logging
import importlib
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("ASI-Launcher")

# Available applications
AVAILABLE_APPS = {
    "content_summarizer": "real_world_apps/content_summarizer/app.py",
    "virtual_assistant": "real_world_apps/virtual_assistant/app.py",
    "document_assistant": "real_world_apps/document_assistant/app.py",
    "sentiment_dashboard": "real_world_apps/sentiment_dashboard/app.py",
    "translation_service": "real_world_apps/translation_service/app.py",
    "learning_platform": "real_world_apps/learning_platform/app.py",
    "banking": "real_world_apps/banking/app.py",
    "recommendation_engine": "real_world_apps/recommendation_engine/app.py",
    "military_logistics": "real_world_apps/military_logistics/app.py"
}

def initialize_asi_components():
    """Initialize the ASI components."""
    logger.info("Initializing ASI components...")
    
    # Ensure environment variables are set
    if "AGI_TOOLKIT_KEY" not in os.environ:
        os.environ["AGI_TOOLKIT_KEY"] = "AGI-Toolkit-Secure-2025"
        logger.info("Set AGI_TOOLKIT_KEY environment variable")
        
    if "MRZKELP_LICENSE_KEY" not in os.environ:
        license_key = "540e4a27d374b9cd58add850949aeed4595ee582570252db538bdb3776d7aa98cd7614c533640914d1df5e03462ff9247b3ff385bff7ebd5b04de66b09c1c231"
        os.environ["MRZKELP_LICENSE_KEY"] = license_key
        os.environ["MRZKELP_CLIENT_ID"] = "demo@example.com"
        os.environ["MRZKELP_SECRET"] = "AGIToolkitMaster"
        logger.info("Set MRZKELP_LICENSE_KEY environment variables")
    
    # Set USE_REAL_ASI environment variable
    os.environ["USE_REAL_ASI"] = "true"
    logger.info("Set USE_REAL_ASI=true")
    
    # Run the fix_component_loading.py script if it exists
    fix_script_path = Path("fix_component_loading.py")
    if fix_script_path.exists():
        logger.info("Running fix_component_loading.py...")
        try:
            subprocess.run([sys.executable, "fix_component_loading.py"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run fix_component_loading.py: {e}")
            return False
    else:
        logger.warning("fix_component_loading.py not found. ASI components may not initialize correctly.")
        return True  # Continue anyway

def run_application(app_name, args):
    """Run the specified application with the given arguments."""
    if app_name not in AVAILABLE_APPS:
        logger.error(f"Unknown application: {app_name}")
        logger.info(f"Available applications: {', '.join(AVAILABLE_APPS.keys())}")
        return False
    
    app_path = AVAILABLE_APPS[app_name]
    
    # Apply app-specific parameter fixes
    fixed_args = fix_app_specific_args(app_name, args)
    full_command = [sys.executable, app_path] + fixed_args
    
    logger.info(f"Running {app_name}: {' '.join(full_command)}")
    
    try:
        subprocess.run(full_command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run {app_name}: {e}")
        return False

def fix_app_specific_args(app_name, args):
    """Fix app-specific arguments to ensure compatibility."""
    fixed_args = args.copy()
    
    # Specific fixes for translation_service
    if app_name == "translation_service":
        # Replace --target_language with --target if needed
        for i, arg in enumerate(fixed_args):
            if arg == "--target_language" and i < len(fixed_args) - 1:
                fixed_args[i] = "--target"
    
    # Specific fixes for virtual_assistant
    elif app_name == "virtual_assistant":
        # If we have a --text argument, convert it to --demo and remove the text
        # since virtual_assistant doesn't support direct --text input
        if "--text" in fixed_args:
            text_index = fixed_args.index("--text")
            # Remove the text argument and its value
            if text_index < len(fixed_args) - 1:
                del fixed_args[text_index:text_index+2]
            else:
                del fixed_args[text_index]
            # Add demo mode if not already present
            if "--demo" not in fixed_args:
                fixed_args.append("--demo")
    
    # Additional app-specific fixes can be added here
    
    return fixed_args

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run real-world applications with real ASI components")
    parser.add_argument("app", choices=list(AVAILABLE_APPS.keys()), help="Application to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to the application")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Initialize ASI components
    if not initialize_asi_components():
        logger.error("Failed to initialize ASI components")
        sys.exit(1)
    
    # Run the application
    success = run_application(args.app, args.args)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
