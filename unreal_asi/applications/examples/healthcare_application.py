#!/usr/bin/env python3
"""
Healthcare Application Example
-----------------------------
This demonstrates how to use the ASI public API to build a healthcare application
without needing to understand the encrypted internal implementation.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the ASI public API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

class HealthcareAnalyzer:
    """Healthcare application using encrypted ASI capabilities."""
    
    def __init__(self):
        """Initialize the healthcare analyzer."""
        print("\n" + "=" * 80)
        print("HEALTHCARE ANALYZER - Powered by ASI Engine".center(80))
        print("=" * 80)
        
        # Initialize the ASI system
        print("\nInitializing ASI Engine...")
        initialize_asi()
        
        # Create an ASI instance for healthcare
        self.asi = create_asi_instance(name="HealthcareASI", config={
            "domain": "healthcare",
            "confidence_threshold": 0.7,
            "sensitivity": 0.85
        })
        
        # Initialize patient database
        self.patients = {}
        self.analysis_results = {}
    
    def add_patient(self, patient_id: str, data: Dict[str, float]) -> bool:
        """
        Add a patient to the analyzer.
        
        Args:
            patient_id: Unique identifier for the patient
            data: Patient vital signs and metrics (normalized between 0-1)
            
        Returns:
            bool: Success status
        """
        required_fields = ["blood_pressure", "heart_rate", "respiratory_rate", "temperature"]
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                print(f"Error: Missing required field '{field}'")
                return False
        
        # Add patient to database
        self.patients[patient_id] = {
            "data": data,
            "added_at": datetime.now().isoformat(),
            "analyzed": False
        }
        
        print(f"Patient {patient_id} added successfully")
        return True
    
    def analyze_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Analyze a patient using ASI capabilities.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict: Analysis results
        """
        if patient_id not in self.patients:
            print(f"Error: Patient {patient_id} not found")
            return {"error": "Patient not found"}
        
        patient = self.patients[patient_id]
        
        print(f"\nAnalyzing patient {patient_id}...")
        
        # Phase 1: Pattern Discovery
        print("\n" + "-" * 50)
        print("PHASE 1: PATIENT DATA PATTERN ANALYSIS")
        print("-" * 50)
        
        # Discover patterns in patient data using ASI
        patterns = self.asi.discover_patterns(domain="patient_vitals", properties=patient["data"])
        
        print(f"\nDiscovered {len(patterns['patterns'])} patterns in patient data")
        print(f"Analysis method: {patterns['method']}")
        print(f"Confidence: {patterns['confidence']:.4f}")
        
        # Display discovered patterns
        for i, pattern in enumerate(patterns['patterns']):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"  Description: {pattern['description']}")
            print(f"  Significance: {pattern['significance']:.4f}")
        
        # Phase 2: Medical Insight Generation
        print("\n" + "-" * 50)
        print("PHASE 2: MEDICAL INSIGHT GENERATION")
        print("-" * 50)
        
        # Generate medical insights using ASI
        medical_concepts = ["blood pressure", "heart rate", "medication", "treatment"]
        insights = self.asi.generate_insight(concepts=medical_concepts)
        
        print(f"\nMedical Insight (Confidence: {insights['confidence']:.4f}):")
        print(f"\"{insights['text']}\"")
        print(f"Related concepts: {', '.join(insights['concepts'])}")
        
        # Phase 3: Treatment Timeline Prediction
        print("\n" + "-" * 50)
        print("PHASE 3: TREATMENT OUTCOME PREDICTION")
        print("-" * 50)
        
        # Create treatment scenario
        treatment_scenario = {
            "name": "Personalized Treatment Plan",
            "complexity": 0.7,
            "uncertainty": 0.5,
            "domain": "patient_care",
            "current_state": patient["data"]
        }
        
        # Predict potential treatment outcomes using ASI
        prediction = self.asi.predict_timeline(treatment_scenario)
        
        print(f"\nTreatment Scenario: {treatment_scenario['name']}")
        print(f"Prediction confidence: {prediction['confidence']:.4f}")
        
        # Display base timeline
        print("\nPrimary Treatment Outcome:")
        for i, step in enumerate(prediction['base_timeline']):
            print(f"  Step {i+1}: {step['event']} - {step['emotion']}")
            print(f"    {step['description']}")
        
        # Store analysis results
        analysis = {
            "patterns": patterns,
            "insights": insights,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_results[patient_id] = analysis
        self.patients[patient_id]["analyzed"] = True
        
        return analysis
    
    def visualize_results(self, patient_id: str, output_path: str = None) -> bool:
        """
        Visualize analysis results for a patient.
        
        Args:
            patient_id: Patient identifier
            output_path: Optional path to save visualization
            
        Returns:
            bool: Success status
        """
        if patient_id not in self.patients:
            print(f"Error: Patient {patient_id} not found")
            return False
        
        if patient_id not in self.analysis_results:
            print(f"Error: Patient {patient_id} has not been analyzed yet")
            return False
        
        analysis = self.analysis_results[patient_id]
        patient_data = self.patients[patient_id]["data"]
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Patient Vitals
        plt.subplot(2, 2, 1)
        keys = list(patient_data.keys())
        values = [patient_data[k] for k in keys]
        plt.bar(keys, values, color='skyblue')
        plt.title('Patient Vital Signs')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Plot 2: Pattern Significance
        plt.subplot(2, 2, 2)
        pattern_names = [p["concept"] for p in analysis["patterns"]["patterns"]]
        pattern_values = [p["significance"] for p in analysis["patterns"]["patterns"]]
        plt.barh(pattern_names, pattern_values, color='lightgreen')
        plt.title('Pattern Significance')
        plt.xlim(0, 1)
        
        # Plot 3: Timeline Visualization
        plt.subplot(2, 1, 2)
        timeline = analysis["prediction"]["base_timeline"]
        x = range(len(timeline))
        events = [item["event"] for item in timeline]
        emotions = {"neutral": 0.5, "optimistic": 0.7, "cautious": 0.4, 
                   "confident": 0.8, "concerned": 0.3, "positive": 0.9}
        y = [emotions.get(item["emotion"].split()[-1], 0.5) for item in timeline]
        
        plt.plot(x, y, 'o-', markersize=10)
        plt.title('Treatment Timeline Projection')
        plt.xticks(x, events, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save or show the visualization
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path)
            print(f"\nVisualization saved to {output_path}")
        else:
            plt.show()
        
        return True

def run_demo():
    """Run the healthcare analyzer demo."""
    analyzer = HealthcareAnalyzer()
    
    # Sample patient data
    patients = {
        "P001": {
            "blood_pressure": 0.75,
            "heart_rate": 0.65,
            "respiratory_rate": 0.5,
            "temperature": 0.6,
            "glucose_level": 0.8,
            "cholesterol": 0.7,
            "white_blood_cell_count": 0.45
        },
        "P002": {
            "blood_pressure": 0.4,
            "heart_rate": 0.85,
            "respiratory_rate": 0.7,
            "temperature": 0.5,
            "glucose_level": 0.3,
            "oxygen_saturation": 0.9,
            "pain_level": 0.6
        }
    }
    
    # Add patients to the analyzer
    for patient_id, data in patients.items():
        analyzer.add_patient(patient_id, data)
    
    # Analyze each patient
    for patient_id in patients.keys():
        analyzer.analyze_patient(patient_id)
        
        # Generate visualization
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        analyzer.visualize_results(patient_id, f"{output_dir}/{patient_id}_analysis.png")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE".center(80))
    print("=" * 80)
    
    print("\nThis demonstration has showcased how the encrypted ASI engine")
    print("can be used to build powerful healthcare applications without")
    print("exposing the underlying algorithms and implementations.")

if __name__ == "__main__":
    run_demo()
