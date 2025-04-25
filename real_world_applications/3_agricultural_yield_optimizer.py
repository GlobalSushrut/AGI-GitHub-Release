#!/usr/bin/env python3
"""
Agricultural Yield Optimizer
---------------------------
Helps farmers maximize crop yields using ASI analysis of growing conditions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import ASI API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

class AgriculturalYieldOptimizer:
    """Agricultural yield optimization using ASI infrastructure."""
    
    def __init__(self):
        """Initialize the yield optimizer."""
        print("\nAGRICULTURAL YIELD OPTIMIZER - Powered by Encrypted ASI Engine")
        
        # Initialize ASI engine
        print("Initializing ASI engine...")
        success = initialize_asi()
        
        if not success:
            print("Failed to initialize ASI engine. Exiting.")
            sys.exit(1)
        
        # Create ASI instance with specific configuration for agricultural analysis
        self.asi = create_asi_instance(name="AgriASI", config={
            "domain": "agriculture",
            "pattern_sensitivity": 0.8,
            "prediction_horizon": 180,  # days
            "multivariate_analysis": True
        })
        
        print("ASI engine initialized successfully")
        
        # Initialize field and crop data
        self.fields = {}
        self.crops = {}
        self.growing_conditions = {}
        self.recommendations = {}
    
    def generate_crop_data(self, num_fields=5, num_crops=4):
        """Generate simulated crop and field data."""
        print(f"\nGenerating simulated data for {num_fields} fields and {num_crops} crops...")
        
        # Define crop types
        crop_types = ["Corn", "Wheat", "Soybeans", "Rice"][:num_crops]
        
        # Define growing condition factors
        factors = ["temperature", "rainfall", "soil_pH", "nitrogen", "phosphorus", "potassium", "sunlight"]
        
        # Generate field data
        for i in range(1, num_fields + 1):
            field_id = f"F{i}"
            
            # Create field with random properties
            self.fields[field_id] = {
                "size": random.uniform(5, 20),  # hectares
                "location": (random.uniform(0, 10), random.uniform(0, 10)),
                "soil_type": random.choice(["Clay", "Loam", "Sandy", "Silt"]),
                "elevation": random.uniform(100, 500),  # meters
                "current_crop": random.choice(crop_types)
            }
            
            # Create growing condition history
            self.growing_conditions[field_id] = {}
            
            # Generate random growing condition data for each month
            for month in range(1, 13):
                month_data = {}
                
                for factor in factors:
                    # Generate different ranges based on the factor
                    if factor == "temperature":
                        # Seasonal variation
                        base = 15 + 10 * np.sin((month - 3) * np.pi / 6)
                        month_data[factor] = base + random.uniform(-3, 3)
                    elif factor == "rainfall":
                        # Seasonal variation
                        base = 80 + 40 * np.sin((month - 3) * np.pi / 6)
                        month_data[factor] = max(0, base + random.uniform(-20, 20))
                    elif factor == "soil_pH":
                        month_data[factor] = random.uniform(5.5, 7.5)
                    elif factor in ["nitrogen", "phosphorus", "potassium"]:
                        month_data[factor] = random.uniform(10, 100)
                    elif factor == "sunlight":
                        # Seasonal variation
                        base = 7 + 5 * np.sin((month - 3) * np.pi / 6)
                        month_data[factor] = max(4, base + random.uniform(-1, 1))
                
                self.growing_conditions[field_id][month] = month_data
        
        # Generate crop data
        for crop in crop_types:
            # Create crop with ideal growing conditions
            self.crops[crop] = {
                "name": crop,
                "growing_season": random.randint(90, 180),  # days
                "water_needs": random.uniform(300, 900),  # mm per season
                "ideal_temperature": random.uniform(15, 30),  # °C
                "ideal_soil_pH": random.uniform(5.5, 7.5),
                "nutrient_needs": {
                    "nitrogen": random.uniform(50, 100),
                    "phosphorus": random.uniform(30, 80),
                    "potassium": random.uniform(40, 90)
                }
            }
        
        print(f"Generated data for {len(self.fields)} fields and {len(self.crops)} crop types")
    
    def analyze_growing_conditions(self):
        """Analyze growing conditions using ASI pattern discovery."""
        print("\nAnalyzing growing conditions...")
        
        # Prepare data for ASI analysis
        analysis_data = []
        
        for field_id, field in self.fields.items():
            field_data = {
                "field_id": field_id,
                "size": field["size"],
                "soil_type": field["soil_type"],
                "elevation": field["elevation"],
                "current_crop": field["current_crop"]
            }
            
            # Add growing condition data
            for month, conditions in self.growing_conditions[field_id].items():
                for factor, value in conditions.items():
                    field_data[f"{factor}_month_{month}"] = value
            
            analysis_data.append(field_data)
        
        # Convert list to dictionary format expected by ASI API
        properties_dict = {}
        
        # Extract numeric properties from each field
        for i, field_data in enumerate(analysis_data):
            field_id = field_data.get('field_id', f'F{i}')
            
            # Add numeric values to the properties dictionary
            for key, value in field_data.items():
                if isinstance(value, (int, float)):
                    properties_dict[f"{field_id}_{key}"] = value
        
        # Ensure we have at least some properties
        if not properties_dict:
            properties_dict = {
                "soil_moisture": 0.7,
                "temperature": 0.65,
                "sunlight": 0.8,
                "fertility": 0.5
            }
            
        # Use ASI pattern discovery
        patterns = self.asi.discover_patterns(
            domain="agricultural_conditions",
            properties=properties_dict
        )
        
        print(f"Discovered {len(patterns['patterns'])} growing condition patterns")
        print(f"Pattern analysis confidence: {patterns['confidence']:.4f}")
        
        # Display top patterns
        for i, pattern in enumerate(patterns['patterns'][:2]):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"Description: {pattern['description']}")
        
        return patterns
    
    def generate_crop_recommendations(self, patterns):
        """Generate crop recommendations using ASI insight generation."""
        print("\nGenerating crop recommendations...")
        
        # Generate insights for crop optimization
        agricultural_concepts = [
            "crop_yield", "growing_conditions", "planting_schedule",
            "irrigation_needs", "fertilization"
        ]
        
        insights = self.asi.generate_insight(concepts=agricultural_concepts)
        
        print(f"Agricultural insight (Confidence: {insights['confidence']:.4f}):")
        print(f"\"{insights['text']}\"")
        
        # Generate recommendations for each field
        for field_id, field in self.fields.items():
            # Create scenario for ASI prediction
            crop_scenario = {
                "name": f"Crop optimization for {field_id}",
                "complexity": 0.6,
                "uncertainty": 0.5,
                "domain": "crop_yield_optimization",
                "variables": {
                    "field_id": field_id,
                    "size": field["size"],
                    "soil_type": field["soil_type"],
                    "elevation": field["elevation"],
                    "current_crop": field["current_crop"],
                    "growing_conditions": self.growing_conditions[field_id]
                }
            }
            
            # Use ASI to predict optimal growing strategies
            prediction = self.asi.predict_timeline(crop_scenario)
            
            # Create recommendations
            self.recommendations[field_id] = {
                "field_id": field_id,
                "current_crop": field["current_crop"],
                "recommended_crops": self._extract_recommended_crops(prediction),
                "planting_schedule": self._extract_planting_schedule(prediction),
                "irrigation_plan": self._extract_irrigation_plan(prediction),
                "fertilization_plan": self._extract_fertilization_plan(prediction),
                "predicted_yield_increase": prediction.get("metrics", {}).get("yield_increase", 0.15),
                "confidence": prediction["confidence"]
            }
        
        # Print recommendation summary
        print(f"\nGenerated recommendations for {len(self.recommendations)} fields")
        
        for field_id, rec in list(self.recommendations.items())[:2]:
            print(f"\nField {field_id}:")
            print(f"Current crop: {rec['current_crop']}")
            print(f"Recommended crops: {', '.join(rec['recommended_crops'])}")
            print(f"Predicted yield increase: {rec['predicted_yield_increase']*100:.1f}%")
            print(f"Confidence: {rec['confidence']:.4f}")
        
        return self.recommendations
    
    def _extract_recommended_crops(self, prediction):
        """Extract recommended crops from ASI prediction."""
        recommended_crops = []
        
        for event in prediction["base_timeline"]:
            if "crop" in event.get("event", "").lower():
                # Extract crop name from event text
                for crop in self.crops.keys():
                    if crop.lower() in event["event"].lower():
                        if crop not in recommended_crops:
                            recommended_crops.append(crop)
        
        # If no specific crops found, include some defaults
        if not recommended_crops:
            recommended_crops = list(self.crops.keys())[:2]
        
        return recommended_crops
    
    def _extract_planting_schedule(self, prediction):
        """Extract planting schedule from ASI prediction."""
        schedule = []
        
        for event in prediction["base_timeline"]:
            if "plant" in event.get("event", "").lower():
                schedule.append(event)
        
        return schedule
    
    def _extract_irrigation_plan(self, prediction):
        """Extract irrigation plan from ASI prediction."""
        plan = []
        
        for event in prediction["base_timeline"]:
            if "irrigat" in event.get("event", "").lower() or "water" in event.get("event", "").lower():
                plan.append(event)
        
        return plan
    
    def _extract_fertilization_plan(self, prediction):
        """Extract fertilization plan from ASI prediction."""
        plan = []
        
        for event in prediction["base_timeline"]:
            if "fertil" in event.get("event", "").lower() or "nutrient" in event.get("event", "").lower():
                plan.append(event)
        
        return plan
    
    def visualize_recommendations(self, output_path=None):
        """Visualize crop recommendations."""
        if not self.recommendations:
            print("No recommendations available to visualize")
            return False
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Field layout with recommendations
        plt.subplot(2, 2, 1)
        
        for field_id, field in self.fields.items():
            x, y = field["location"]
            size = field["size"] * 20  # Scale for visibility
            
            # Get recommendation confidence
            if field_id in self.recommendations:
                confidence = self.recommendations[field_id]["confidence"]
            else:
                confidence = 0.5
            
            # Color based on recommendation confidence
            color = plt.cm.RdYlGn(confidence)
            
            plt.scatter(x, y, s=size, color=color, alpha=0.7)
            plt.text(x, y, field_id, fontsize=8, ha='center', va='center')
        
        plt.title('Field Layout with Crop Recommendations')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Predicted yield increases
        plt.subplot(2, 2, 2)
        
        field_ids = list(self.recommendations.keys())
        yield_increases = [self.recommendations[f]["predicted_yield_increase"] * 100 for f in field_ids]
        
        # Sort by yield increase
        yield_increases, field_ids = zip(*sorted(zip(yield_increases, field_ids), reverse=True))
        
        plt.bar(field_ids, yield_increases, color='green', alpha=0.7)
        plt.title('Predicted Yield Increase by Field')
        plt.ylabel('Yield Increase (%)')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Growing conditions comparison
        plt.subplot(2, 1, 2)
        
        # Select a few key growing factors
        factors = ["temperature", "rainfall", "soil_pH"]
        months = list(range(1, 13))
        
        # Plot for first field only
        if self.fields:
            field_id = list(self.fields.keys())[0]
            
            for factor in factors:
                values = [self.growing_conditions[field_id][month][factor] for month in months]
                plt.plot(months, values, marker='o', linestyle='-', label=factor)
        
        plt.title(f'Growing Conditions Over Time (Field {field_id})')
        plt.xlabel('Month')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
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
    """Run a demonstration of the agricultural yield optimizer."""
    # Initialize the optimizer
    optimizer = AgriculturalYieldOptimizer()
    
    # Generate simulated crop data
    optimizer.generate_crop_data()
    
    # Analyze growing conditions
    patterns = optimizer.analyze_growing_conditions()
    
    # Generate crop recommendations
    optimizer.generate_crop_recommendations(patterns)
    
    # Visualize results
    output_dir = os.path.join(root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    optimizer.visualize_recommendations(f"{output_dir}/crop_recommendations.png")
    
    print("\nAGRICULTURAL YIELD OPTIMIZER DEMO COMPLETE")
    print("\nThis demonstration has shown how the encrypted ASI engine can be used")
    print("to build agricultural optimization applications without accessing")
    print("the proprietary algorithms and mathematical implementations.")

if __name__ == "__main__":
    run_demo()
