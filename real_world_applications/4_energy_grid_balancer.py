#!/usr/bin/env python3
"""
Energy Grid Balancing System
---------------------------
Optimizes energy distribution across power grid networks using ASI engine.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import ASI API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

class EnergyGridBalancer:
    """Energy grid balancing system using ASI infrastructure."""
    
    def __init__(self):
        """Initialize the energy grid balancer."""
        print("\nENERGY GRID BALANCING SYSTEM - Powered by Encrypted ASI Engine")
        
        # Initialize ASI engine
        print("Initializing ASI engine...")
        success = initialize_asi()
        
        if not success:
            print("Failed to initialize ASI engine. Exiting.")
            sys.exit(1)
        
        # Create ASI instance with specific configuration for energy analysis
        self.asi = create_asi_instance(name="EnergyASI", config={
            "domain": "energy_systems",
            "optimization_depth": 0.85,
            "temporal_resolution": "hourly",
            "system_complexity": 0.7
        })
        
        print("ASI engine initialized successfully")
        
        # Initialize grid data structures
        self.grid_nodes = {}
        self.power_sources = {}
        self.energy_data = {}
        self.balancing_recommendations = {}
    
    def generate_grid_data(self, num_nodes=10, num_days=7):
        """Generate simulated energy grid data."""
        print(f"\nGenerating simulated data for {num_nodes} grid nodes over {num_days} days...")
        
        # Define node types
        node_types = ["Distribution", "Transmission", "Substation"]
        
        # Define energy source types
        source_types = ["Solar", "Wind", "Hydro", "Coal", "Natural Gas", "Nuclear"]
        
        # Generate nodes
        for i in range(1, num_nodes + 1):
            node_id = f"N{i}"
            
            # Create node with random properties
            self.grid_nodes[node_id] = {
                "type": random.choice(node_types),
                "capacity": random.uniform(20, 100),  # MW
                "location": (random.uniform(0, 20), random.uniform(0, 20)),
                "connections": []
            }
            
            # Connect nodes in a reasonable network topology
            # Each node connects to 1-3 other nodes
            potential_connections = [f"N{j}" for j in range(1, num_nodes + 1) if j != i]
            num_connections = min(random.randint(1, 3), len(potential_connections))
            self.grid_nodes[node_id]["connections"] = random.sample(potential_connections, num_connections)
        
        # Generate power sources
        for i in range(1, num_nodes // 2 + 1):
            source_id = f"S{i}"
            source_type = random.choice(source_types)
            
            # Connect to a random node
            connected_node = f"N{random.randint(1, num_nodes)}"
            
            # Create source with type-specific properties
            if source_type == "Solar":
                capacity = random.uniform(10, 30)
                variability = "high"
                carbon_footprint = 0.1
            elif source_type == "Wind":
                capacity = random.uniform(15, 40)
                variability = "high"
                carbon_footprint = 0.1
            elif source_type == "Hydro":
                capacity = random.uniform(30, 80)
                variability = "medium"
                carbon_footprint = 0.2
            elif source_type == "Coal":
                capacity = random.uniform(50, 100)
                variability = "low"
                carbon_footprint = 0.9
            elif source_type == "Natural Gas":
                capacity = random.uniform(40, 90)
                variability = "low"
                carbon_footprint = 0.6
            else:  # Nuclear
                capacity = random.uniform(80, 150)
                variability = "very low"
                carbon_footprint = 0.1
            
            self.power_sources[source_id] = {
                "type": source_type,
                "capacity": capacity,  # MW
                "connected_node": connected_node,
                "variability": variability,
                "carbon_footprint": carbon_footprint
            }
        
        # Generate hourly energy data over specified days
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        hours = 24 * num_days
        timestamps = [start_date + timedelta(hours=h) for h in range(hours)]
        
        # Initialize energy data structure
        for node_id in self.grid_nodes:
            self.energy_data[node_id] = {}
        
        # Generate data for each time point
        for timestamp in timestamps:
            hour = timestamp.hour
            
            # Simulate daily load patterns
            # Morning peak: 7-9 AM
            # Evening peak: 6-9 PM
            # Base load during night
            if 7 <= hour <= 9:
                load_factor = 0.7 + random.random() * 0.3
            elif 18 <= hour <= 21:
                load_factor = 0.8 + random.random() * 0.2
            elif 22 <= hour <= 23 or 0 <= hour <= 5:
                load_factor = 0.3 + random.random() * 0.2
            else:
                load_factor = 0.5 + random.random() * 0.3
            
            # Generate data for each node
            for node_id, node in self.grid_nodes.items():
                capacity = node["capacity"]
                
                # Calculate supply from connected sources
                supply = 0
                for source_id, source in self.power_sources.items():
                    if source["connected_node"] == node_id:
                        source_capacity = source["capacity"]
                        
                        # Apply time-of-day and source type factors
                        if source["type"] == "Solar":
                            # Solar only produces during daylight
                            if 6 <= hour <= 18:
                                peak_hour = 12
                                hour_factor = 1 - abs(hour - peak_hour) / 12
                                supply += source_capacity * hour_factor * (0.7 + random.random() * 0.3)
                        elif source["type"] == "Wind":
                            # Wind is variable but more at night
                            if hour <= 6 or hour >= 18:
                                supply += source_capacity * (0.6 + random.random() * 0.4)
                            else:
                                supply += source_capacity * (0.3 + random.random() * 0.5)
                        else:
                            # Other sources more stable
                            supply += source_capacity * (0.8 + random.random() * 0.2)
                
                # Calculate demand based on capacity and load factor
                demand = capacity * load_factor
                
                # Calculate transmission to/from connected nodes (simplified)
                transmission = 0
                for connected_node in node["connections"]:
                    # Randomly distribute excess/deficit
                    transmission += random.uniform(-10, 10)  # MW
                
                # Calculate grid balance
                balance = supply - demand + transmission
                
                # Store data
                self.energy_data[node_id][timestamp] = {
                    "supply": supply,
                    "demand": demand,
                    "transmission": transmission,
                    "balance": balance
                }
        
        print(f"Generated energy data for {len(self.grid_nodes)} nodes and {len(self.power_sources)} sources")
    
    def analyze_grid_patterns(self):
        """Analyze energy grid patterns using ASI pattern discovery."""
        print("\nAnalyzing energy grid patterns...")
        
        # Prepare data for ASI analysis
        analysis_data = []
        
        for node_id, node in self.grid_nodes.items():
            node_data = {
                "node_id": node_id,
                "type": node["type"],
                "capacity": node["capacity"],
                "num_connections": len(node["connections"])
            }
            
            # Add average and peak values
            supplies = []
            demands = []
            balances = []
            
            for timestamp, data in self.energy_data[node_id].items():
                supplies.append(data["supply"])
                demands.append(data["demand"])
                balances.append(data["balance"])
            
            node_data["avg_supply"] = np.mean(supplies) if supplies else 0
            node_data["peak_supply"] = np.max(supplies) if supplies else 0
            node_data["avg_demand"] = np.mean(demands) if demands else 0
            node_data["peak_demand"] = np.max(demands) if demands else 0
            node_data["avg_balance"] = np.mean(balances) if balances else 0
            node_data["balance_volatility"] = np.std(balances) if len(balances) > 1 else 0
            
            analysis_data.append(node_data)
        
        # Convert list to dictionary format expected by ASI API
        properties_dict = {}
        
        # Extract numeric properties from each node's analysis data
        for i, node_data in enumerate(analysis_data):
            node_id = node_data.get('node_id', f'N{i}')
            
            # Add numeric values to the properties dictionary
            for key, value in node_data.items():
                if isinstance(value, (int, float)):
                    properties_dict[f"{node_id}_{key}"] = value
        
        # Ensure we have at least some properties
        if not properties_dict:
            properties_dict = {
                "grid_stability": 0.75,
                "power_flow": 0.65,
                "load_balance": 0.8,
                "transmission_efficiency": 0.6
            }
            
        # Use ASI pattern discovery
        patterns = self.asi.discover_patterns(
            domain="energy_distribution",
            properties=properties_dict
        )
        
        print(f"Discovered {len(patterns['patterns'])} energy grid patterns")
        print(f"Pattern analysis confidence: {patterns['confidence']:.4f}")
        
        # Display top patterns
        for i, pattern in enumerate(patterns['patterns'][:2]):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"Description: {pattern['description']}")
        
        return patterns
    
    def generate_balancing_recommendations(self, patterns):
        """Generate grid balancing recommendations using ASI."""
        print("\nGenerating grid balancing recommendations...")
        
        # Generate insights for energy optimization
        energy_concepts = [
            "grid_stability", "load_balancing", "energy_efficiency",
            "renewable_integration", "peak_shaving"
        ]
        
        insights = self.asi.generate_insight(concepts=energy_concepts)
        
        print(f"Energy grid insight (Confidence: {insights['confidence']:.4f}):")
        print(f"\"{insights['text']}\"")
        
        # Identify problematic nodes
        problematic_nodes = []
        
        for node_id, node_data in self.grid_nodes.items():
            # Get node balances
            balances = [data["balance"] for _, data in self.energy_data[node_id].items()]
            
            # Check for problems
            if balances:
                avg_balance = np.mean(balances)
                balance_volatility = np.std(balances)
                
                if abs(avg_balance) > 10 or balance_volatility > 15:
                    problematic_nodes.append({
                        "node_id": node_id,
                        "avg_balance": avg_balance,
                        "balance_volatility": balance_volatility,
                        "type": node_data["type"]
                    })
        
        # Generate recommendations for each problematic node
        for node in problematic_nodes:
            node_id = node["node_id"]
            
            # Create scenario for ASI prediction
            energy_scenario = {
                "name": f"Grid balancing for {node_id}",
                "complexity": 0.7,
                "uncertainty": 0.5,
                "domain": "grid_optimization",
                "variables": {
                    "node_id": node_id,
                    "type": node["type"],
                    "avg_balance": node["avg_balance"],
                    "volatility": node["balance_volatility"],
                    "grid_data": self.grid_nodes[node_id]
                }
            }
            
            # Use ASI to predict optimal balancing strategies
            prediction = self.asi.predict_timeline(energy_scenario)
            
            # Create recommendations
            self.balancing_recommendations[node_id] = {
                "node_id": node_id,
                "current_status": {
                    "avg_balance": node["avg_balance"],
                    "volatility": node["balance_volatility"]
                },
                "recommended_actions": self._extract_balancing_actions(prediction),
                "storage_recommendations": self._extract_storage_recommendations(prediction),
                "distribution_changes": self._extract_distribution_changes(prediction),
                "stability_improvement": prediction.get("metrics", {}).get("stability_improvement", 0.2),
                "efficiency_gain": prediction.get("metrics", {}).get("efficiency_gain", 0.15),
                "confidence": prediction["confidence"]
            }
        
        # Print recommendation summary
        print(f"\nGenerated balancing recommendations for {len(self.balancing_recommendations)} nodes")
        
        for node_id, rec in list(self.balancing_recommendations.items())[:2]:
            print(f"\nNode {node_id}:")
            print(f"Current status: Avg balance = {rec['current_status']['avg_balance']:.2f} MW, " +
                  f"Volatility = {rec['current_status']['volatility']:.2f} MW")
            print(f"Stability improvement: {rec['stability_improvement']*100:.1f}%")
            print(f"Efficiency gain: {rec['efficiency_gain']*100:.1f}%")
            print(f"Confidence: {rec['confidence']:.4f}")
            
            if rec['recommended_actions']:
                print("Top recommended actions:")
                for i, action in enumerate(rec['recommended_actions'][:2]):
                    print(f"  - {action['event']}")
        
        return self.balancing_recommendations
    
    def _extract_balancing_actions(self, prediction):
        """Extract balancing actions from ASI prediction."""
        actions = []
        
        for event in prediction["base_timeline"]:
            if any(term in event.get("event", "").lower() for term in 
                   ["balanc", "distribut", "load", "generat", "transmi"]):
                actions.append(event)
        
        return actions
    
    def _extract_storage_recommendations(self, prediction):
        """Extract storage recommendations from ASI prediction."""
        storage_recs = []
        
        for event in prediction["base_timeline"]:
            if any(term in event.get("event", "").lower() for term in 
                   ["storage", "battery", "capacit", "reserve"]):
                storage_recs.append(event)
        
        return storage_recs
    
    def _extract_distribution_changes(self, prediction):
        """Extract distribution changes from ASI prediction."""
        dist_changes = []
        
        for event in prediction["base_timeline"]:
            if any(term in event.get("event", "").lower() for term in 
                   ["distribut", "flow", "redirect", "rout"]):
                dist_changes.append(event)
        
        return dist_changes
    
    def visualize_grid_analysis(self, output_path=None):
        """Visualize energy grid analysis and recommendations."""
        if not self.balancing_recommendations or not self.energy_data:
            print("No grid data or recommendations available to visualize")
            return False
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Grid network with nodes and connections
        plt.subplot(2, 2, 1)
        
        # Plot nodes
        for node_id, node in self.grid_nodes.items():
            x, y = node["location"]
            
            # Set node size and color
            size = node["capacity"]
            
            # Determine color based on recommendations
            if node_id in self.balancing_recommendations:
                # Red for problematic nodes with recommendations
                color = 'red'
            else:
                # Green for stable nodes
                color = 'green'
            
            plt.scatter(x, y, s=size*2, color=color, alpha=0.7)
            plt.text(x, y, node_id, fontsize=8, ha='center', va='center')
            
            # Plot connections
            for connected_node in node["connections"]:
                if connected_node in self.grid_nodes:
                    x2, y2 = self.grid_nodes[connected_node]["location"]
                    plt.plot([x, x2], [y, y2], 'gray', alpha=0.4)
        
        # Plot power sources
        for source_id, source in self.power_sources.items():
            if source["connected_node"] in self.grid_nodes:
                x, y = self.grid_nodes[source["connected_node"]]["location"]
                
                # Adjust location slightly to avoid overlap
                x += random.uniform(-0.5, 0.5)
                y += random.uniform(-0.5, 0.5)
                
                # Set marker based on source type
                if "Solar" in source["type"]:
                    marker = '*'
                    color = 'yellow'
                elif "Wind" in source["type"]:
                    marker = 's'
                    color = 'lightblue'
                elif "Hydro" in source["type"]:
                    marker = 'v'
                    color = 'blue'
                elif "Coal" in source["type"]:
                    marker = 'X'
                    color = 'black'
                elif "Gas" in source["type"]:
                    marker = 'P'
                    color = 'orange'
                else:  # Nuclear
                    marker = 'H'
                    color = 'purple'
                
                plt.scatter(x, y, s=source["capacity"], color=color, marker=marker, alpha=0.8)
        
        plt.title('Energy Grid Network')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Supply/Demand Balance for a sample node
        plt.subplot(2, 2, 2)
        
        # Select a node with recommendations
        if self.balancing_recommendations:
            sample_node_id = list(self.balancing_recommendations.keys())[0]
        else:
            sample_node_id = list(self.energy_data.keys())[0]
        
        # Get time series data
        timestamps = sorted(self.energy_data[sample_node_id].keys())
        hours = [t.hour for t in timestamps]
        supplies = [self.energy_data[sample_node_id][t]["supply"] for t in timestamps]
        demands = [self.energy_data[sample_node_id][t]["demand"] for t in timestamps]
        balances = [self.energy_data[sample_node_id][t]["balance"] for t in timestamps]
        
        # Plot supply and demand
        plt.plot(hours[:24], supplies[:24], 'g-', label='Supply')
        plt.plot(hours[:24], demands[:24], 'r-', label='Demand')
        plt.plot(hours[:24], balances[:24], 'b--', label='Balance')
        
        plt.title(f'Supply/Demand for Node {sample_node_id} (24h)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Power (MW)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Improvement metrics for recommended nodes
        plt.subplot(2, 1, 2)
        
        node_ids = list(self.balancing_recommendations.keys())
        stability_improvements = [self.balancing_recommendations[n]["stability_improvement"] * 100 for n in node_ids]
        efficiency_gains = [self.balancing_recommendations[n]["efficiency_gain"] * 100 for n in node_ids]
        
        x = np.arange(len(node_ids))
        width = 0.35
        
        plt.bar(x - width/2, stability_improvements, width, label='Stability Improvement (%)')
        plt.bar(x + width/2, efficiency_gains, width, label='Efficiency Gain (%)')
        
        plt.xlabel('Grid Node')
        plt.ylabel('Improvement (%)')
        plt.title('Predicted Grid Improvements with ASI Recommendations')
        plt.xticks(x, node_ids)
        plt.grid(True, axis='y', alpha=0.3)
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
    """Run a demonstration of the energy grid balancer."""
    # Initialize the balancer
    balancer = EnergyGridBalancer()
    
    # Generate simulated grid data
    balancer.generate_grid_data()
    
    # Analyze grid patterns
    patterns = balancer.analyze_grid_patterns()
    
    # Generate balancing recommendations
    balancer.generate_balancing_recommendations(patterns)
    
    # Visualize results
    output_dir = os.path.join(root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    balancer.visualize_grid_analysis(f"{output_dir}/grid_balancing.png")
    
    print("\nENERGY GRID BALANCER DEMO COMPLETE")
    print("\nThis demonstration has shown how the encrypted ASI engine can be used")
    print("to build sophisticated energy grid optimization applications without")
    print("accessing the proprietary algorithms and mathematical implementations.")

if __name__ == "__main__":
    run_demo()
