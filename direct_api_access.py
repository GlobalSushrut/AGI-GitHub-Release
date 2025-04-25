#!/usr/bin/env python3
"""
Direct API Access Example
-------------------------
This script demonstrates how to directly access the ASI engine's API
for building real-world applications.
"""

import os
import sys
from datetime import datetime

# Import ASI public API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

def main():
    """Demonstrate direct API access to ASI engine capabilities."""
    print("\n" + "=" * 80)
    print("DIRECT ASI API ACCESS EXAMPLE".center(80))
    print("=" * 80)
    
    # Step 1: Initialize the ASI engine
    print("\nStep 1: Initialize ASI Engine")
    print("-" * 40)
    success = initialize_asi()
    
    if not success:
        print("Failed to initialize ASI engine. Exiting.")
        return
    
    print("ASI engine initialized successfully!")
    
    # Step 2: Create an ASI instance
    print("\nStep 2: Create ASI Instance")
    print("-" * 40)
    asi = create_asi_instance(name="MyASI", config={
        "domain": "general",
        "confidence_threshold": 0.6
    })
    
    print(f"ASI instance created: {asi.name}")
    
    # Step 3: Discover patterns in data
    print("\nStep 3: Pattern Discovery")
    print("-" * 40)
    
    # Sample data for pattern discovery
    data = {
        "feature_a": 0.8,
        "feature_b": 0.6,
        "feature_c": 0.7,
        "feature_d": 0.5
    }
    
    print("Input data:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    # Discover patterns using ASI
    patterns = asi.discover_patterns(domain="sample_domain", properties=data)
    
    print("\nDiscovered patterns:")
    print(f"Emergence level: {patterns['emergence_level']:.4f}")
    print(f"Confidence: {patterns['confidence']:.4f}")
    print(f"Analysis method: {patterns['method']}")
    
    for i, pattern in enumerate(patterns['patterns']):
        print(f"\nPattern {i+1}: {pattern['concept']}")
        print(f"  Description: {pattern['description']}")
        print(f"  Significance: {pattern['significance']:.4f}")
    
    # Step 4: Generate insights
    print("\nStep 4: Insight Generation")
    print("-" * 40)
    
    # Generate insights using ASI
    insights = asi.generate_insight(concepts=["data", "patterns", "analysis"])
    
    print(f"Generated insight (confidence: {insights['confidence']:.4f}):")
    print(f"\"{insights['text']}\"")
    print(f"Related concepts: {', '.join(insights['concepts'])}")
    print(f"Strange loop depth: {insights['strange_loop_depth']}")
    
    # Step 5: Predict timeline
    print("\nStep 5: Timeline Prediction")
    print("-" * 40)
    
    # Create scenario for prediction
    scenario = {
        "name": "Project Development",
        "complexity": 0.7,
        "uncertainty": 0.5,
        "domain": "project_management"
    }
    
    print(f"Scenario: {scenario['name']}")
    print(f"Complexity: {scenario['complexity']}")
    print(f"Uncertainty: {scenario['uncertainty']}")
    
    # Predict timeline using ASI
    prediction = asi.predict_timeline(scenario)
    
    print(f"\nPrediction results:")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Temporal coherence: {prediction['temporal_coherence']:.4f}")
    
    print("\nBase timeline:")
    for i, step in enumerate(prediction['base_timeline']):
        print(f"  Step {i+1}: {step['event']} - {step['emotion']}")
        if 'description' in step:
            print(f"    {step['description']}")
    
    print(f"\nAlternative branches: {len(prediction['branching_timelines'])}")
    for i, branch in enumerate(prediction['branching_timelines']):
        print(f"\nBranch {i+1}: {branch['type']}")
        print(f"Probability: {branch.get('probability', 'N/A')}")
        
        for j, step in enumerate(branch['timeline']):
            print(f"  Step {j+1}: {step['event']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY".center(80))
    print("=" * 80)
    
    print("\nThis demonstration has shown how to directly access the ASI engine's API")
    print("to build real-world applications without needing to understand the")
    print("encrypted internal implementation.")
    
    print("\nThe key steps are:")
    print("1. Initialize the ASI engine with initialize_asi()")
    print("2. Create an ASI instance with create_asi_instance()")
    print("3. Use the ASI instance methods to access capabilities:")
    print("   - discover_patterns(): Find patterns in data")
    print("   - generate_insight(): Generate cross-domain insights")
    print("   - predict_timeline(): Predict multiple future timelines")

if __name__ == "__main__":
    main()
