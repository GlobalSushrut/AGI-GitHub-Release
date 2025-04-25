#!/usr/bin/env python3
"""
Medical Diagnosis Assistant
--------------------------
A real-world application that uses the encrypted ASI engine to assist
with medical diagnosis by analyzing patient symptoms and medical history.
"""

import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import ASI API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

class MedicalDiagnosisAssistant:
    """Medical diagnosis assistant using ASI infrastructure."""
    
    def __init__(self):
        """Initialize the medical diagnosis assistant."""
        print("\n" + "=" * 80)
        print("MEDICAL DIAGNOSIS ASSISTANT - Powered by Encrypted ASI Engine".center(80))
        print("=" * 80)
        
        # Initialize ASI engine - accessing the encrypted infrastructure
        print("\nInitializing ASI engine...")
        success = initialize_asi()
        
        if not success:
            print("Failed to initialize ASI engine. Exiting.")
            sys.exit(1)
        
        # Create ASI instance with specific configuration for medical analysis
        self.asi = create_asi_instance(name="MedicalASI", config={
            "domain": "healthcare",
            "sensitivity": 0.85,
            "specificity": 0.90,
            "comprehensive_analysis": True
        })
        
        print("ASI engine initialized successfully")
        
        # Initialize patient database
        self.patients = {}
        self.diagnoses = {}
        self.treatment_plans = {}
    
    def add_patient(self, patient_id, data):
        """
        Add a patient to the system.
        
        Args:
            patient_id: Unique identifier for the patient
            data: Patient data including symptoms, vitals, and history
        """
        # Normalize patient data for ASI processing
        normalized_data = self._normalize_patient_data(data)
        
        # Store patient data
        self.patients[patient_id] = {
            "raw_data": data,
            "normalized_data": normalized_data,
            "added_at": datetime.datetime.now().isoformat()
        }
        
        print(f"\nAdded patient {patient_id}")
        print(f"Symptoms: {', '.join(data['symptoms'])}")
        print(f"Age: {data['demographics']['age']}")
        print(f"Gender: {data['demographics']['gender']}")
    
    def _normalize_patient_data(self, data):
        """
        Normalize patient data for ASI processing.
        
        Args:
            data: Raw patient data
            
        Returns:
            Dict: Normalized patient data
        """
        normalized = {}
        
        # Convert symptoms to numerical format (1 for present, 0 for absent)
        all_possible_symptoms = [
            "fever", "cough", "shortness_of_breath", "fatigue", "headache", 
            "sore_throat", "congestion", "nausea", "vomiting", "diarrhea",
            "body_aches", "loss_of_taste", "loss_of_smell", "rash", "joint_pain"
        ]
        
        symptoms_vector = {symptom: 0 for symptom in all_possible_symptoms}
        for symptom in data.get('symptoms', []):
            if symptom in symptoms_vector:
                symptoms_vector[symptom] = 1
        
        # Normalize vital signs
        vitals = data.get('vitals', {})
        normalized_vitals = {
            "temperature": self._normalize_value(vitals.get('temperature', 98.6), 97, 103),
            "heart_rate": self._normalize_value(vitals.get('heart_rate', 70), 40, 140),
            "blood_pressure_systolic": self._normalize_value(vitals.get('blood_pressure_systolic', 120), 90, 180),
            "blood_pressure_diastolic": self._normalize_value(vitals.get('blood_pressure_diastolic', 80), 50, 110),
            "oxygen_saturation": self._normalize_value(vitals.get('oxygen_saturation', 98), 80, 100),
            "respiratory_rate": self._normalize_value(vitals.get('respiratory_rate', 16), 10, 30)
        }
        
        # Normalize demographic information
        demographics = data.get('demographics', {})
        normalized_demographics = {
            "age": self._normalize_value(demographics.get('age', 35), 0, 100),
            "gender_male": 1 if demographics.get('gender', '').lower() == 'male' else 0,
            "gender_female": 1 if demographics.get('gender', '').lower() == 'female' else 0,
            "gender_other": 1 if demographics.get('gender', '').lower() not in ['male', 'female'] else 0
        }
        
        # Combine all normalized data
        normalized = {**symptoms_vector, **normalized_vitals, **normalized_demographics}
        
        return normalized
    
    def _normalize_value(self, value, min_val, max_val):
        """
        Normalize a value to range 0-1.
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            float: Normalized value
        """
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
            return max(0, min(1, normalized))
        return 0.5
    
    def diagnose_patient(self, patient_id):
        """
        Diagnose a patient using ASI's pattern discovery capabilities.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict: Diagnosis results
        """
        if patient_id not in self.patients:
            print(f"Error: Patient {patient_id} not found")
            return None
        
        patient_data = self.patients[patient_id]
        
        print(f"\nDiagnosing patient {patient_id}...")
        print("-" * 40)
        
        # Use ASI pattern discovery to identify symptom patterns
        # This uses the encrypted ASI infrastructure through the API
        patterns = self.asi.discover_patterns(
            domain="medical_symptoms",
            properties=patient_data["normalized_data"]
        )
        
        print(f"Discovered {len(patterns['patterns'])} symptom patterns")
        print(f"Analysis confidence: {patterns['confidence']:.4f}")
        
        # Display top patterns
        for i, pattern in enumerate(patterns['patterns'][:3]):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"  Description: {pattern['description']}")
            print(f"  Significance: {pattern['significance']:.4f}")
        
        # Generate medical insight using ASI's encrypted capabilities
        medical_concepts = ["diagnosis", "symptoms", "treatment", "prognosis"]
        insight = self.asi.generate_insight(
            concepts=medical_concepts
        )
        
        print(f"\nMedical Insight (Confidence: {insight['confidence']:.4f}):")
        print(f"\"{insight['text']}\"")
        
        # Create a prioritized list of potential diagnoses
        diagnoses = self._generate_potential_diagnoses(patterns, patient_data["raw_data"])
        
        print("\nPotential Diagnoses:")
        for i, diagnosis in enumerate(diagnoses[:3]):
            print(f"  {i+1}. {diagnosis['condition']} - Confidence: {diagnosis['confidence']:.4f}")
            print(f"     Reasoning: {diagnosis['reasoning']}")
        
        # Store diagnosis results
        diagnosis_result = {
            "patterns": patterns,
            "insight": insight,
            "diagnoses": diagnoses,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.diagnoses[patient_id] = diagnosis_result
        
        return diagnosis_result
    
    def _generate_potential_diagnoses(self, patterns, raw_patient_data):
        """
        Generate potential diagnoses based on symptom patterns.
        
        Args:
            patterns: Pattern discovery results from ASI
            raw_patient_data: Raw patient data
            
        Returns:
            List: Potential diagnoses
        """
        # This is a simplified version for demonstration
        # In a real implementation, this would use a more sophisticated algorithm
        
        symptoms = raw_patient_data.get('symptoms', [])
        age = raw_patient_data.get('demographics', {}).get('age', 35)
        gender = raw_patient_data.get('demographics', {}).get('gender', 'unknown')
        
        # Common diagnosis mapping (simplified)
        diagnosis_mapping = {
            "fever": ["Common Cold", "Influenza", "COVID-19", "Infection"],
            "cough": ["Common Cold", "Bronchitis", "Pneumonia", "COVID-19"],
            "shortness_of_breath": ["Asthma", "Pneumonia", "Heart Failure", "COVID-19"],
            "fatigue": ["Depression", "Anemia", "Chronic Fatigue Syndrome", "Hypothyroidism"],
            "headache": ["Tension Headache", "Migraine", "Sinusitis", "Dehydration"],
            "sore_throat": ["Strep Throat", "Common Cold", "Tonsillitis", "Laryngitis"],
            "congestion": ["Common Cold", "Sinusitis", "Allergies", "Rhinitis"],
            "nausea": ["Gastroenteritis", "Food Poisoning", "Migraine", "Pregnancy"],
            "vomiting": ["Gastroenteritis", "Food Poisoning", "Concussion", "Appendicitis"],
            "diarrhea": ["Gastroenteritis", "Food Poisoning", "Irritable Bowel Syndrome", "Infection"],
            "body_aches": ["Influenza", "Fibromyalgia", "Common Cold", "COVID-19"],
            "loss_of_taste": ["COVID-19", "Zinc Deficiency", "Common Cold", "Sinusitis"],
            "loss_of_smell": ["COVID-19", "Sinusitis", "Common Cold", "Nasal Polyps"],
            "rash": ["Allergic Reaction", "Eczema", "Psoriasis", "Contact Dermatitis"],
            "joint_pain": ["Arthritis", "Fibromyalgia", "Lupus", "Lyme Disease"]
        }
        
        # Count occurrences of each potential diagnosis
        diagnosis_counts = {}
        
        for symptom in symptoms:
            if symptom in diagnosis_mapping:
                for diagnosis in diagnosis_mapping[symptom]:
                    if diagnosis in diagnosis_counts:
                        diagnosis_counts[diagnosis] += 1
                    else:
                        diagnosis_counts[diagnosis] = 1
        
        # Calculate confidence based on symptom count and pattern confidence
        diagnoses = []
        total_symptoms = len(symptoms)
        
        if total_symptoms > 0:
            for diagnosis, count in diagnosis_counts.items():
                # Base confidence on symptom coverage
                base_confidence = count / total_symptoms
                
                # Adjust by ASI pattern confidence
                adjusted_confidence = (base_confidence + patterns['confidence']) / 2
                
                # Generate reasoning
                matching_symptoms = [s for s in symptoms if diagnosis in diagnosis_mapping.get(s, [])]
                reasoning = f"Based on {', '.join(matching_symptoms)}"
                
                diagnoses.append({
                    "condition": diagnosis,
                    "confidence": adjusted_confidence,
                    "reasoning": reasoning,
                    "matching_symptoms": matching_symptoms
                })
        
        # Sort by confidence
        diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
        
        return diagnoses
    
    def generate_treatment_plan(self, patient_id):
        """
        Generate a treatment plan using ASI's timeline prediction.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict: Treatment plan
        """
        if patient_id not in self.patients or patient_id not in self.diagnoses:
            print(f"Error: Patient {patient_id} has not been diagnosed yet")
            return None
        
        patient_data = self.patients[patient_id]
        diagnosis = self.diagnoses[patient_id]
        
        print(f"\nGenerating treatment plan for patient {patient_id}...")
        print("-" * 40)
        
        # Get the top diagnosis
        top_diagnosis = diagnosis['diagnoses'][0] if diagnosis['diagnoses'] else None
        
        if not top_diagnosis:
            print("Error: No diagnosis available to generate treatment plan")
            return None
        
        # Create medical scenario for ASI prediction
        medical_scenario = {
            "name": f"Treatment for {top_diagnosis['condition']}",
            "complexity": 0.7,
            "uncertainty": 0.6,
            "domain": "medical_treatment",
            "current_state": patient_data["normalized_data"],
            "diagnosis": top_diagnosis['condition']
        }
        
        # Use ASI to predict possible treatment timelines
        prediction = self.asi.predict_timeline(medical_scenario)
        
        print(f"Prediction confidence: {prediction['confidence']:.4f}")
        print(f"Temporal coherence: {prediction['temporal_coherence']:.4f}")
        
        # Display base timeline (most likely treatment path)
        print("\nRecommended Treatment Plan:")
        for i, step in enumerate(prediction['base_timeline']):
            print(f"  Step {i+1}: {step['event']}")
            if 'description' in step:
                print(f"    {step['description']}")
        
        # Display alternative scenarios
        print(f"\nAlternative Treatment Approaches: {len(prediction['branching_timelines'])}")
        for i, branch in enumerate(prediction['branching_timelines']):
            print(f"  Approach {i+1}: {branch['type']} (Probability: {branch.get('probability', 0):.2f})")
        
        # Generate specific treatment recommendations
        recommendations = self._generate_treatment_recommendations(
            top_diagnosis['condition'], 
            prediction, 
            patient_data["raw_data"]
        )
        
        print("\nSpecific Treatment Recommendations:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['description']}")
            print(f"     Type: {rec['type']}")
            print(f"     Priority: {rec['priority']}")
        
        # Store treatment plan
        treatment_plan = {
            "diagnosis": top_diagnosis,
            "prediction": prediction,
            "recommendations": recommendations,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.treatment_plans[patient_id] = treatment_plan
        
        return treatment_plan
    
    def _generate_treatment_recommendations(self, condition, prediction, raw_patient_data):
        """
        Generate specific treatment recommendations.
        
        Args:
            condition: Diagnosed condition
            prediction: ASI timeline prediction
            raw_patient_data: Raw patient data
            
        Returns:
            List: Treatment recommendations
        """
        # This is a simplified version for demonstration
        # In a real implementation, this would use a more sophisticated algorithm
        
        # Basic treatment recommendations by condition (simplified)
        treatment_options = {
            "Common Cold": [
                {"type": "Medication", "description": "Over-the-counter pain relievers", "priority": "Medium"},
                {"type": "Medication", "description": "Decongestants for nasal congestion", "priority": "Medium"},
                {"type": "Self-care", "description": "Rest and drink plenty of fluids", "priority": "High"},
                {"type": "Self-care", "description": "Gargle with salt water for sore throat", "priority": "Low"}
            ],
            "Influenza": [
                {"type": "Medication", "description": "Antiviral medications", "priority": "High"},
                {"type": "Medication", "description": "Pain relievers for fever and aches", "priority": "Medium"},
                {"type": "Self-care", "description": "Rest and drink plenty of fluids", "priority": "High"},
                {"type": "Self-care", "description": "Humidifier to ease congestion and cough", "priority": "Low"}
            ],
            "COVID-19": [
                {"type": "Monitoring", "description": "Regular monitoring of oxygen levels", "priority": "High"},
                {"type": "Medication", "description": "Pain relievers for fever and aches", "priority": "Medium"},
                {"type": "Self-care", "description": "Isolation to prevent spread", "priority": "High"},
                {"type": "Self-care", "description": "Rest and drink plenty of fluids", "priority": "High"},
                {"type": "Medical care", "description": "Seek emergency care if breathing becomes difficult", "priority": "High"}
            ],
            "Pneumonia": [
                {"type": "Medication", "description": "Antibiotics for bacterial pneumonia", "priority": "High"},
                {"type": "Medication", "description": "Pain relievers for fever and pain", "priority": "Medium"},
                {"type": "Monitoring", "description": "Regular monitoring of oxygen levels", "priority": "High"},
                {"type": "Self-care", "description": "Rest and drink plenty of fluids", "priority": "High"},
                {"type": "Medical care", "description": "Possible hospitalization for severe cases", "priority": "High"}
            ]
        }
        
        # Get recommendations for the specific condition
        recommendations = treatment_options.get(condition, [])
        
        # If no specific recommendations, provide general recommendations
        if not recommendations:
            recommendations = [
                {"type": "Medical care", "description": "Consult with a specialist for specific treatment", "priority": "High"},
                {"type": "Monitoring", "description": "Regular monitoring of symptoms", "priority": "Medium"},
                {"type": "Self-care", "description": "Rest and maintain good nutrition", "priority": "Medium"}
            ]
        
        # Adjust recommendations based on patient demographics
        age = raw_patient_data.get('demographics', {}).get('age', 35)
        
        # Add age-specific recommendations
        if age < 12:
            recommendations.append({
                "type": "Medication", 
                "description": "Adjust medication dosages for pediatric patient", 
                "priority": "High"
            })
        elif age > 65:
            recommendations.append({
                "type": "Monitoring", 
                "description": "More frequent monitoring due to age-related risks", 
                "priority": "High"
            })
        
        return recommendations
    
    def visualize_diagnosis(self, patient_id, output_path=None):
        """
        Visualize diagnosis and treatment plan.
        
        Args:
            patient_id: Patient identifier
            output_path: Optional path to save visualization
            
        Returns:
            bool: Success status
        """
        if patient_id not in self.diagnoses or patient_id not in self.patients:
            print(f"Error: Complete data not available for patient {patient_id}")
            return False
        
        if patient_id not in self.treatment_plans:
            print(f"Error: No treatment plan for patient {patient_id}")
            return False
        
        patient_data = self.patients[patient_id]
        diagnosis = self.diagnoses[patient_id]
        treatment_plan = self.treatment_plans[patient_id]
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Symptom Heatmap
        plt.subplot(2, 2, 1)
        symptoms = patient_data["raw_data"].get("symptoms", [])
        symptom_values = []
        symptom_labels = []
        
        for key, value in patient_data["normalized_data"].items():
            if key in ["fever", "cough", "shortness_of_breath", "fatigue", "headache", 
                      "sore_throat", "congestion", "nausea", "vomiting", "diarrhea",
                      "body_aches", "loss_of_taste", "loss_of_smell", "rash", "joint_pain"]:
                if value > 0:
                    symptom_values.append(value)
                    symptom_labels.append(key.replace("_", " ").title())
        
        if symptom_values:
            plt.barh(symptom_labels, symptom_values, color='red', alpha=0.6)
            plt.title('Patient Symptoms')
            plt.xlabel('Severity')
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, "No symptoms recorded", ha='center', va='center')
            plt.title('Patient Symptoms')
        
        # Plot 2: Diagnosis Confidence
        plt.subplot(2, 2, 2)
        diagnoses = diagnosis.get('diagnoses', [])[:5]  # Top 5 diagnoses
        if diagnoses:
            diagnosis_names = [d['condition'] for d in diagnoses]
            diagnosis_confidence = [d['confidence'] for d in diagnoses]
            
            plt.barh(diagnosis_names, diagnosis_confidence, color='green', alpha=0.6)
            plt.title('Potential Diagnoses')
            plt.xlabel('Confidence')
            plt.xlim(0, 1)
        else:
            plt.text(0.5, 0.5, "No diagnoses available", ha='center', va='center')
            plt.title('Potential Diagnoses')
        
        # Plot 3: Treatment Timeline
        plt.subplot(2, 1, 2)
        timeline = treatment_plan['prediction']['base_timeline']
        
        if timeline:
            events = [item['event'] for item in timeline]
            y_positions = list(range(len(events)))
            
            for i, (event, y) in enumerate(zip(events, y_positions)):
                plt.text(i, y, event, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc='lightblue', alpha=0.6))
            
            # Connect events with arrows
            for i in range(len(events) - 1):
                plt.annotate('', xy=(i+1, y_positions[i+1]), xytext=(i, y_positions[i]),
                            arrowprops=dict(arrowstyle='->'))
            
            plt.title('Treatment Timeline')
            plt.yticks([])  # Hide y-axis
            plt.xticks([])  # Hide x-axis
        else:
            plt.text(0.5, 0.5, "No treatment timeline available", ha='center', va='center')
            plt.title('Treatment Timeline')
        
        # Add text annotation with ASI insight
        if 'insight' in diagnosis:
            insight_text = diagnosis['insight']['text']
            plt.figtext(0.5, 0.01, f"ASI Insight: {insight_text}", wrap=True, 
                      horizontalalignment='center', fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
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
    """Run a demonstration of the medical diagnosis assistant."""
    # Initialize the assistant
    assistant = MedicalDiagnosisAssistant()
    
    # Sample patient data
    patients = {
        "P001": {
            "demographics": {
                "age": 45,
                "gender": "female",
                "blood_type": "O+",
                "height": 165,
                "weight": 70
            },
            "symptoms": [
                "fever",
                "cough",
                "shortness_of_breath",
                "fatigue",
                "loss_of_taste",
                "loss_of_smell"
            ],
            "vitals": {
                "temperature": 101.2,
                "heart_rate": 88,
                "blood_pressure_systolic": 125,
                "blood_pressure_diastolic": 85,
                "oxygen_saturation": 94,
                "respiratory_rate": 22
            },
            "medical_history": [
                "asthma",
                "seasonal allergies"
            ]
        },
        "P002": {
            "demographics": {
                "age": 67,
                "gender": "male",
                "blood_type": "A+",
                "height": 175,
                "weight": 85
            },
            "symptoms": [
                "shortness_of_breath",
                "chest_pain",
                "fatigue",
                "swelling_in_legs",
                "irregular_heartbeat"
            ],
            "vitals": {
                "temperature": 98.8,
                "heart_rate": 92,
                "blood_pressure_systolic": 145,
                "blood_pressure_diastolic": 95,
                "oxygen_saturation": 91,
                "respiratory_rate": 18
            },
            "medical_history": [
                "hypertension",
                "high_cholesterol",
                "type_2_diabetes"
            ]
        }
    }
    
    # Process patients
    for patient_id, data in patients.items():
        # Add patient
        assistant.add_patient(patient_id, data)
        
        # Diagnose patient
        assistant.diagnose_patient(patient_id)
        
        # Generate treatment plan
        assistant.generate_treatment_plan(patient_id)
        
        # Visualize results
        output_dir = os.path.join(root_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)
        assistant.visualize_diagnosis(patient_id, f"{output_dir}/{patient_id}_medical_diagnosis.png")
    
    print("\n" + "=" * 80)
    print("MEDICAL DIAGNOSIS ASSISTANT DEMO COMPLETE".center(80))
    print("=" * 80)
    
    print("\nThis demonstration has shown how the encrypted ASI engine can be used")
    print("to build sophisticated medical diagnosis applications without accessing")
    print("the proprietary algorithms and mathematical implementations.")
    
    print("\nThe application successfully leveraged ASI capabilities:")
    print("1. Pattern discovery in patient symptoms")
    print("2. Medical insight generation")
    print("3. Treatment timeline prediction")
    
    print("\nCheck the generated visualizations in the reports directory.")

if __name__ == "__main__":
    run_demo()
