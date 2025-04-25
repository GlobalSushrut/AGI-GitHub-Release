#!/usr/bin/env python3
"""
Sample ASI Application
---------------------
This demonstrates how to use the ASI public API to build real-world applications
without needing to understand the encrypted internal implementation.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the ASI public API
from asi_public_api import initialize_asi, create_asi_instance

def healthcare_analyzer():
    """
    Sample healthcare application using ASI capabilities.
    This application can analyze patient data and generate insights
    while the core ASI algorithms remain encrypted and protected.
    """
    print("\n" + "=" * 80)
    print("HEALTHCARE ANALYZER - Powered by ASI Engine".center(80))
    print("=" * 80)
    
    # Initialize the ASI system
    print("\nInitializing ASI Engine...")
    initialize_asi()
    
    # Create an ASI instance for healthcare
    asi = create_asi_instance(name="HealthcareASI", config={
        "domain": "healthcare",
        "confidence_threshold": 0.7,
        "sensitivity": 0.85
    })
    
    # Sample patient data (in a real application, this would come from a database)
    patient_data = {
        "blood_pressure": 0.75,  # Normalized value (higher means higher blood pressure)
        "heart_rate": 0.65,      # Normalized value
        "respiratory_rate": 0.5, # Normalized value
        "temperature": 0.6,      # Normalized value
        "glucose_level": 0.8     # Normalized value
    }
    
    # Phase 1: Pattern Discovery
    print("\n" + "-" * 80)
    print("PHASE 1: PATIENT DATA PATTERN ANALYSIS".center(80))
    print("-" * 80)
    
    # Discover patterns in patient data
    patterns = asi.discover_patterns(domain="patient_vitals", properties=patient_data)
    
    print(f"\nDiscovered {len(patterns['patterns'])} patterns in patient data")
    print(f"Analysis method: {patterns['method']}")
    print(f"Confidence: {patterns['confidence']:.4f}")
    
    # Display discovered patterns
    for i, pattern in enumerate(patterns['patterns']):
        print(f"\nPattern {i+1}: {pattern['concept']}")
        print(f"  Description: {pattern['description']}")
        print(f"  Significance: {pattern['significance']:.4f}")
    
    # Phase 2: Medical Insight Generation
    print("\n" + "-" * 80)
    print("PHASE 2: MEDICAL INSIGHT GENERATION".center(80))
    print("-" * 80)
    
    # Generate medical insights
    medical_concepts = ["blood pressure", "heart rate", "medication", "treatment"]
    insights = asi.generate_insight(concepts=medical_concepts)
    
    print(f"\nMedical Insight (Confidence: {insights['confidence']:.4f}):")
    print(f"\"{insights['text']}\"")
    print(f"Related concepts: {', '.join(insights['concepts'])}")
    
    # Phase 3: Treatment Timeline Prediction
    print("\n" + "-" * 80)
    print("PHASE 3: TREATMENT OUTCOME PREDICTION".center(80))
    print("-" * 80)
    
    # Create treatment scenario
    treatment_scenario = {
        "name": "Hypertension Treatment Plan",
        "complexity": 0.7,
        "uncertainty": 0.5,
        "domain": "cardiology",
        "current_state": patient_data
    }
    
    # Predict potential treatment outcomes
    prediction = asi.predict_timeline(treatment_scenario)
    
    print(f"\nTreatment Scenario: {treatment_scenario['name']}")
    print(f"Prediction confidence: {prediction['confidence']:.4f}")
    print(f"Temporal coherence: {prediction['temporal_coherence']:.4f}")
    
    # Display base timeline
    print("\nPrimary Treatment Outcome:")
    for i, step in enumerate(prediction['base_timeline']):
        print(f"  Step {i+1}: {step['event']} - {step['emotion']}")
        print(f"    {step['description']}")
    
    # Display alternative timelines
    print(f"\nAlternative Outcomes ({len(prediction['branching_timelines'])} branches):")
    for i, branch in enumerate(prediction['branching_timelines']):
        print(f"\nAlternative {i+1}: {branch['type']}")
        print(f"Probability: {branch['probability']:.4f}")
        
        for j, step in enumerate(branch['timeline']):
            print(f"  Step {j+1}: {step['event']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY".center(80))
    print("=" * 80)
    
    print("\nThis demonstration has showcased how the encrypted ASI engine can be used")
    print("to build powerful healthcare applications without exposing the underlying")
    print("algorithms and mathematical implementations.")
    
    print("\nDevelopers can leverage ASI capabilities for:")
    print("1. Pattern discovery in complex medical data")
    print("2. Generation of cross-domain medical insights")
    print("3. Prediction of treatment outcomes with timeline analysis")
    print("\nAll while the core ASI implementation remains secure and encrypted.")

def finance_analyzer():
    """
    Sample finance application using ASI capabilities.
    This application can analyze market data and generate investment insights
    while the core ASI algorithms remain encrypted and protected.
    """
    print("\n" + "=" * 80)
    print("FINANCIAL MARKET ANALYZER - Powered by ASI Engine".center(80))
    print("=" * 80)
    
    # Initialize the ASI system
    print("\nInitializing ASI Engine...")
    initialize_asi()
    
    # Create an ASI instance for finance
    asi = create_asi_instance(name="FinanceASI", config={
        "domain": "finance",
        "risk_tolerance": 0.6,
        "time_horizon": "medium"
    })
    
    # Sample market data (in a real application, this would come from financial APIs)
    market_data = {
        "market_volatility": 0.7,    # Normalized value
        "interest_rates": 0.4,       # Normalized value
        "economic_growth": 0.5,      # Normalized value
        "inflation": 0.6,            # Normalized value
        "consumer_sentiment": 0.45    # Normalized value
    }
    
    # Phase 1: Market Pattern Discovery
    print("\n" + "-" * 80)
    print("PHASE 1: MARKET PATTERN ANALYSIS".center(80))
    print("-" * 80)
    
    # Discover patterns in market data
    patterns = asi.discover_patterns(domain="market_conditions", properties=market_data)
    
    print(f"\nDiscovered {len(patterns['patterns'])} patterns in market data")
    print(f"Analysis method: {patterns['method']}")
    print(f"Confidence: {patterns['confidence']:.4f}")
    
    # Display discovered patterns
    for i, pattern in enumerate(patterns['patterns']):
        print(f"\nPattern {i+1}: {pattern['concept']}")
        print(f"  Description: {pattern['description']}")
        print(f"  Significance: {pattern['significance']:.4f}")
    
    # Phase 2: Investment Insight Generation
    print("\n" + "-" * 80)
    print("PHASE 2: INVESTMENT INSIGHT GENERATION".center(80))
    print("-" * 80)
    
    # Generate investment insights
    financial_concepts = ["volatility", "interest rates", "diversification", "risk"]
    insights = asi.generate_insight(concepts=financial_concepts)
    
    print(f"\nInvestment Insight (Confidence: {insights['confidence']:.4f}):")
    print(f"\"{insights['text']}\"")
    print(f"Related concepts: {', '.join(insights['concepts'])}")
    
    # Phase 3: Market Timeline Prediction
    print("\n" + "-" * 80)
    print("PHASE 3: MARKET SCENARIO PREDICTION".center(80))
    print("-" * 80)
    
    # Create market scenario
    market_scenario = {
        "name": "Interest Rate Adjustment Impact",
        "complexity": 0.8,
        "uncertainty": 0.7,
        "domain": "macroeconomics",
        "current_state": market_data
    }
    
    # Predict potential market outcomes
    prediction = asi.predict_timeline(market_scenario)
    
    print(f"\nMarket Scenario: {market_scenario['name']}")
    print(f"Prediction confidence: {prediction['confidence']:.4f}")
    print(f"Temporal coherence: {prediction['temporal_coherence']:.4f}")
    
    # Display base timeline
    print("\nPrimary Market Outcome:")
    for i, step in enumerate(prediction['base_timeline']):
        print(f"  Step {i+1}: {step['event']} - {step['emotion']}")
        print(f"    {step['description']}")
    
    # Display alternative timelines
    print(f"\nAlternative Outcomes ({len(prediction['branching_timelines'])} branches):")
    for i, branch in enumerate(prediction['branching_timelines']):
        print(f"\nAlternative {i+1}: {branch['type']}")
        print(f"Probability: {branch['probability']:.4f}")
        
        for j, step in enumerate(branch['timeline']):
            print(f"  Step {j+1}: {step['event']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY".center(80))
    print("=" * 80)
    
    print("\nThis demonstration has showcased how the encrypted ASI engine can be used")
    print("to build powerful financial applications without exposing the underlying")
    print("algorithms and mathematical implementations.")

def run_demo():
    """Run the sample ASI application demo."""
    print("\nASI ENGINE DEMO")
    print("---------------")
    print("This demo showcases how to build applications with the encrypted ASI engine.")
    print("The core algorithms and math are hidden while the API remains accessible.")
    
    while True:
        print("\nSelect a demo application:")
        print("1. Healthcare Analyzer")
        print("2. Financial Market Analyzer")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            healthcare_analyzer()
        elif choice == "2":
            finance_analyzer()
        elif choice == "3":
            print("\nExiting ASI demo. Thank you!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    run_demo()
