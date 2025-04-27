#!/usr/bin/env python3
"""
Fix Component Loading

This script will help properly load the ASI and MOCK-LLM components for
real-world application usage by importing them directly from the showcase
methodology that's working.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComponentLoader")

def initialize_real_components():
    """Initialize the real ASI and MOCK-LLM components."""
    logger.info("Initializing real components...")
    
    # Ensure AGI_TOOLKIT_KEY is set
    if "AGI_TOOLKIT_KEY" not in os.environ:
        logger.error("AGI_TOOLKIT_KEY environment variable not set")
        logger.info("Setting environment variable AGI_TOOLKIT_KEY to 'AGI-Toolkit-Secure-2025'")
        os.environ["AGI_TOOLKIT_KEY"] = "AGI-Toolkit-Secure-2025"
    
    # Add license keys if not already set
    if "MRZKELP_LICENSE_KEY" not in os.environ:
        license_key = "540e4a27d374b9cd58add850949aeed4595ee582570252db538bdb3776d7aa98cd7614c533640914d1df5e03462ff9247b3ff385bff7ebd5b04de66b09c1c231"
        os.environ["MRZKELP_LICENSE_KEY"] = license_key
        os.environ["MRZKELP_CLIENT_ID"] = "demo@example.com"
        os.environ["MRZKELP_SECRET"] = "AGIToolkitMaster"
    
    # Import the working modules from the showcase
    try:
        # Use the direct import method from asi_showcase.py
        from unreal_asi.asi_public_api import initialize_asi, create_asi_instance
        
        # Initialize the ASI engine
        initialize_asi()  
        logger.info("ASI Engine initialized successfully")
        
        # Create a global ASI instance
        asi = create_asi_instance(name="ApplicationASI")
        logger.info("ASI Instance created successfully")
        
        # Export the instance for applications to use
        import builtins
        builtins.ASI_INSTANCE = asi
        logger.info("ASI Instance exported as ASI_INSTANCE")
        
        # Try to initialize MOCK-LLM components
        try:
            from mock_llm.main import initialize_mock_llm, create_llm_instance
            initialize_mock_llm()
            llm = create_llm_instance(name="ApplicationLLM")
            builtins.MOCK_LLM_INSTANCE = llm
            logger.info("MOCK-LLM Instance exported as MOCK_LLM_INSTANCE")
        except Exception as e:
            logger.warning(f"Could not initialize MOCK-LLM components: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize real components: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if initialize_real_components():
        logger.info("Real components initialized successfully")
        sys.exit(0)
    else:
        logger.error("Failed to initialize real components")
        sys.exit(1)
