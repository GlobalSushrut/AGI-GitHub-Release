#!/usr/bin/env python3
"""
Smart City Traffic Management System
-----------------------------------
A real-world application that uses the encrypted ASI engine to optimize
traffic flow in urban environments.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import ASI API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

class TrafficManagementSystem:
    """Smart city traffic management system using ASI infrastructure."""
    
    def __init__(self):
        """Initialize the traffic management system."""
        print("\n" + "=" * 80)
        print("SMART CITY TRAFFIC MANAGEMENT SYSTEM - Powered by Encrypted ASI Engine".center(80))
        print("=" * 80)
        
        # Initialize ASI engine
        print("\nInitializing ASI engine...")
        success = initialize_asi()
        
        if not success:
            print("Failed to initialize ASI engine. Exiting.")
            sys.exit(1)
        
        # Create ASI instance with specific configuration for traffic analysis
        self.asi = create_asi_instance(name="TrafficASI", config={
            "domain": "urban_systems",
            "pattern_complexity": 0.75,
            "temporal_depth": 0.8,
            "multivariate_analysis": True
        })
        
        print("ASI engine initialized successfully")
        
        # Initialize traffic data
        self.traffic_grid = {}
        self.junctions = {}
        self.congestion_history = {}
        self.signal_plans = {}
    
    def generate_traffic_data(self, grid_size=(5, 5), time_periods=24):
        """
        Generate simulated traffic data for testing.
        
        Args:
            grid_size: Size of the city grid (x, y)
            time_periods: Number of time periods to simulate
        """
        print(f"\nGenerating simulated traffic data for a {grid_size[0]}x{grid_size[1]} grid city...")
        
        # Create city grid with roads and junctions
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                junction_id = f"J{x+1}{y+1}"
                
                # Create junction with traffic lights
                self.junctions[junction_id] = {
                    "location": (x, y),
                    "signals": ["north_south", "east_west"],
                    "current_signal": "north_south",
                    "cycle_time": random.randint(30, 120),  # seconds
                    "connected_roads": []
                }
                
                # Connect roads to junctions
                directions = ["north", "south", "east", "west"]
                for direction in directions:
                    road_id = f"R{junction_id}_{direction}"
                    self.traffic_grid[road_id] = {
                        "junction": junction_id,
                        "direction": direction,
                        "length": random.randint(100, 500),  # meters
                        "lanes": random.randint(1, 3),
                        "speed_limit": random.choice([30, 40, 50, 60])  # km/h
                    }
                    self.junctions[junction_id]["connected_roads"].append(road_id)
        
        # Generate traffic patterns for each road across time periods
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        times = [base_time + timedelta(hours=i) for i in range(time_periods)]
        
        # Traffic patterns with rush hours
        for road_id, road in self.traffic_grid.items():
            self.traffic_grid[road_id]["traffic_data"] = {}
            
            for t in times:
                hour = t.hour
                
                # Create rush hour patterns (morning and evening)
                base_traffic = road["lanes"] * 10  # Base capacity per lane
                
                # Morning rush hour (7-9 AM)
                if 7 <= hour <= 9:
                    multiplier = 1.5 + random.random() * 0.5
                # Evening rush hour (4-6 PM)
                elif 16 <= hour <= 18:
                    multiplier = 1.7 + random.random() * 0.5
                # Normal daytime
                elif 10 <= hour <= 15:
                    multiplier = 0.8 + random.random() * 0.4
                # Night time
                else:
                    multiplier = 0.3 + random.random() * 0.3
                
                # Add some randomness to traffic volume
                volume = int(base_traffic * multiplier * (0.9 + random.random() * 0.2))
                
                # Calculate speed based on volume (higher volume = lower speed)
                max_capacity = road["lanes"] * 25  # Maximum vehicles per time unit
                capacity_ratio = min(volume / max_capacity, 1.5)  # Ratio of volume to capacity
                
                # Speed decreases as capacity_ratio increases
                speed = max(5, int(road["speed_limit"] * (1 - (capacity_ratio - 0.5) * 0.6)))
                
                # Calculate congestion level (0-1)
                congestion = min(1, capacity_ratio * 0.8)
                
                self.traffic_grid[road_id]["traffic_data"][t.strftime("%H:%M")] = {
                    "volume": volume,
                    "speed": speed,
                    "congestion": congestion
                }
        
        print(f"Generated traffic data for {len(self.traffic_grid)} roads over {time_periods} hours")
    
    def analyze_traffic_patterns(self):
        """
        Analyze traffic patterns using ASI's pattern discovery.
        
        Returns:
            Dict: Traffic analysis results
        """
        print("\nAnalyzing traffic patterns across the city...")
        print("-" * 40)
        
        # Prepare data for ASI analysis - convert traffic data to numerical format
        traffic_data = {}
        
        for road_id, road in self.traffic_grid.items():
            junction_id = road["junction"]
            
            # Aggregate traffic data by junction and time
            if junction_id not in traffic_data:
                traffic_data[junction_id] = {}
            
            for time, data in road["traffic_data"].items():
                if time not in traffic_data[junction_id]:
                    traffic_data[junction_id][time] = {
                        "total_volume": 0,
                        "avg_speed": 0,
                        "avg_congestion": 0,
                        "road_count": 0
                    }
                
                # Aggregate data
                traffic_data[junction_id][time]["total_volume"] += data["volume"]
                traffic_data[junction_id][time]["avg_speed"] += data["speed"]
                traffic_data[junction_id][time]["avg_congestion"] += data["congestion"]
                traffic_data[junction_id][time]["road_count"] += 1
        
        # Calculate averages and prepare for ASI
        asi_input = []
        
        for junction_id, times in traffic_data.items():
            junction_data = {"junction": junction_id}
            
            for time, data in times.items():
                # Calculate averages
                if data["road_count"] > 0:
                    junction_data[f"volume_{time}"] = data["total_volume"]
                    junction_data[f"speed_{time}"] = data["avg_speed"] / data["road_count"]
                    junction_data[f"congestion_{time}"] = data["avg_congestion"] / data["road_count"]
            
            asi_input.append(junction_data)
        
        # Use ASI pattern discovery to identify traffic patterns
        # Convert complex data structure to a simplified format expected by ASI API
        # The API expects a dictionary with float values, not complex objects
        flat_properties = {}
        
        # Flatten the complex structure
        for i, junction_data in enumerate(asi_input):
            # Extract key metrics and add as separate properties
            junction_id = junction_data.get('junction', f'J{i}')
            
            # Add basic properties with extracted numeric values
            for key, value in junction_data.items():
                if isinstance(value, (int, float)):
                    flat_properties[f"{junction_id}_{key}"] = value
                elif key != 'junction':  # Skip the junction ID itself
                    # For time-based data, extract a few key periods
                    if key.startswith('volume_') or key.startswith('speed_') or key.startswith('congestion_'):
                        flat_properties[f"{junction_id}_{key}"] = 0.5  # Default value
            
            # Ensure we have at least some properties
            if junction_id not in flat_properties:
                flat_properties[junction_id] = 0.5
        
        # If no properties were extracted, use a default set
        if not flat_properties:
            flat_properties = {
                "traffic_density": 0.7,
                "congestion_level": 0.6,
                "peak_hour_factor": 0.85,
                "signal_efficiency": 0.5
            }
                
        patterns = self.asi.discover_patterns(
            domain="traffic_flow",
            properties=flat_properties
        )
        
        print(f"Discovered {len(patterns['patterns'])} traffic patterns")
        print(f"Pattern analysis confidence: {patterns['confidence']:.4f}")
        
        # Display top patterns
        for i, pattern in enumerate(patterns['patterns'][:3]):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"  Description: {pattern['description']}")
            print(f"  Significance: {pattern['significance']:.4f}")
        
        # Store the analysis results
        analysis_results = {
            "patterns": patterns,
            "traffic_data": traffic_data,
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis_results
    
    def identify_congestion_hotspots(self, analysis_results):
        """
        Identify congestion hotspots based on pattern analysis.
        
        Args:
            analysis_results: Results from traffic analysis
            
        Returns:
            Dict: Congestion hotspots
        """
        print("\nIdentifying congestion hotspots...")
        print("-" * 40)
        
        # Extract patterns related to congestion
        congestion_patterns = [
            p for p in analysis_results["patterns"]["patterns"]
            if "congestion" in p["concept"].lower() or "delay" in p["concept"].lower()
        ]
        
        # Analyze junction-level congestion
        junction_congestion = {}
        
        for junction_id, times in analysis_results["traffic_data"].items():
            # Calculate average congestion across all time periods
            congestion_values = []
            peak_congestion = 0
            peak_time = ""
            
            for time, data in times.items():
                if data["road_count"] > 0:
                    congestion = data["avg_congestion"] / data["road_count"]
                    congestion_values.append(congestion)
                    
                    # Track peak congestion
                    if congestion > peak_congestion:
                        peak_congestion = congestion
                        peak_time = time
            
            # Calculate metrics
            avg_congestion = sum(congestion_values) / len(congestion_values) if congestion_values else 0
            
            junction_congestion[junction_id] = {
                "average_congestion": avg_congestion,
                "peak_congestion": peak_congestion,
                "peak_time": peak_time,
                "location": self.junctions[junction_id]["location"]
            }
        
        # Identify hotspots based on congestion levels
        hotspots = {}
        thresholds = {
            "critical": 0.8,
            "high": 0.6,
            "moderate": 0.4
        }
        
        for severity, threshold in thresholds.items():
            hotspots[severity] = []
            
            for junction_id, data in junction_congestion.items():
                if data["average_congestion"] >= threshold:
                    hotspots[severity].append({
                        "junction_id": junction_id,
                        "location": data["location"],
                        "average_congestion": data["average_congestion"],
                        "peak_congestion": data["peak_congestion"],
                        "peak_time": data["peak_time"]
                    })
            
            # Sort by congestion level
            hotspots[severity].sort(key=lambda x: x["average_congestion"], reverse=True)
        
        # Print hotspot summary
        print(f"Identified congestion hotspots:")
        for severity, spots in hotspots.items():
            print(f"  {severity.title()}: {len(spots)} junctions")
            
            # Show top hotspots in each category
            for i, spot in enumerate(spots[:3]):
                if i < len(spots):
                    print(f"    - Junction {spot['junction_id']} (Peak: {spot['peak_time']}, "
                          f"Congestion: {spot['peak_congestion']:.2f})")
        
        # Store congestion history
        self.congestion_history[datetime.now().strftime("%Y-%m-%d")] = hotspots
        
        return hotspots
    
    def optimize_traffic_signals(self, hotspots):
        """
        Optimize traffic signal timing using ASI's insight generation.
        
        Args:
            hotspots: Congestion hotspots data
            
        Returns:
            Dict: Optimized signal plans
        """
        print("\nOptimizing traffic signal plans...")
        print("-" * 40)
        
        # Create a list of all problematic junctions
        problematic_junctions = []
        
        for severity in ["critical", "high", "moderate"]:
            for spot in hotspots.get(severity, []):
                problematic_junctions.append(spot["junction_id"])
        
        # Generate insights for traffic optimization
        traffic_concepts = [
            "signal_timing", "traffic_flow", "congestion_mitigation",
            "road_capacity", "peak_traffic"
        ]
        
        insights = self.asi.generate_insight(concepts=traffic_concepts)
        
        print(f"Traffic optimization insight (Confidence: {insights['confidence']:.4f}):")
        print(f"\"{insights['text']}\"")
        
        # Create signal plans for each problematic junction
        signal_plans = {}
        
        for junction_id in problematic_junctions:
            junction = self.junctions[junction_id]
            
            # Create scenario for ASI prediction
            signal_scenario = {
                "name": f"Traffic optimization for {junction_id}",
                "complexity": 0.6,
                "uncertainty": 0.4,
                "domain": "traffic_signal_optimization",
                "variables": {
                    "junction_id": junction_id,
                    "location": junction["location"],
                    "signals": junction["signals"],
                    "current_cycle_time": junction["cycle_time"],
                    "connected_roads": junction["connected_roads"]
                }
            }
            
            # Use ASI to predict optimal signal timings
            prediction = self.asi.predict_timeline(signal_scenario)
            
            # Extract recommended timing changes
            if prediction["base_timeline"]:
                # Extract recommendations from ASI prediction
                recommendations = []
                
                for step in prediction["base_timeline"]:
                    if "signal_timing" in step.get("event", "").lower():
                        recommendations.append(step)
                
                # Create optimized signal plan
                optimized_plan = {
                    "junction_id": junction_id,
                    "current_cycle_time": junction["cycle_time"],
                    "recommendations": recommendations,
                    "predicted_improvement": prediction.get("metrics", {}).get("congestion_reduction", 0.2),
                    "confidence": prediction["confidence"]
                }
                
                signal_plans[junction_id] = optimized_plan
        
        # Print signal optimization summary
        print(f"\nCreated optimized signal plans for {len(signal_plans)} junctions")
        
        for junction_id, plan in list(signal_plans.items())[:3]:  # Show first 3 examples
            print(f"\nJunction {junction_id}:")
            print(f"  Current cycle time: {plan['current_cycle_time']} seconds")
            print(f"  Predicted congestion reduction: {plan['predicted_improvement']*100:.1f}%")
            print(f"  Confidence: {plan['confidence']:.4f}")
            
            if plan['recommendations']:
                print("  Recommendations:")
                for i, rec in enumerate(plan['recommendations'][:2]):
                    print(f"    - {rec['event']}")
        
        # Store signal plans
        self.signal_plans = signal_plans
        
        return signal_plans
    
    def visualize_traffic_optimization(self, hotspots, signal_plans, output_path=None):
        """
        Visualize traffic hotspots and optimization plans.
        
        Args:
            hotspots: Congestion hotspots data
            signal_plans: Optimized signal plans
            output_path: Optional path to save visualization
            
        Returns:
            bool: Success status
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: City grid with congestion hotspots
        plt.subplot(2, 2, 1)
        
        # Get grid dimensions
        max_x = max(self.junctions[j]["location"][0] for j in self.junctions) + 1
        max_y = max(self.junctions[j]["location"][1] for j in self.junctions) + 1
        
        # Plot all junctions
        for junction_id, junction in self.junctions.items():
            x, y = junction["location"]
            
            # Default color (low congestion)
            color = 'green'
            size = 100
            
            # Check if this junction is a hotspot
            for severity in ["moderate", "high", "critical"]:
                for spot in hotspots.get(severity, []):
                    if spot["junction_id"] == junction_id:
                        if severity == "critical":
                            color = 'red'
                            size = 200
                        elif severity == "high":
                            color = 'orange'
                            size = 150
                        else:
                            color = 'yellow'
                            size = 120
            
            plt.scatter(x, y, color=color, s=size, alpha=0.7)
            plt.text(x, y, junction_id, fontsize=8, ha='center', va='center')
        
        # Plot grid layout (roads)
        for x in range(max_x):
            plt.axvline(x=x-0.5, color='gray', linestyle='-', alpha=0.2)
        for y in range(max_y):
            plt.axhline(y=y-0.5, color='gray', linestyle='-', alpha=0.2)
        
        plt.title('City Traffic Congestion Map')
        plt.xlim(-1, max_x)
        plt.ylim(-1, max_y)
        plt.grid(True, color='gray', linestyle='--', alpha=0.3)
        
        # Add a legend for hotspot severity
        plt.scatter([], [], c='red', s=200, label='Critical', alpha=0.7)
        plt.scatter([], [], c='orange', s=150, label='High', alpha=0.7)
        plt.scatter([], [], c='yellow', s=120, label='Moderate', alpha=0.7)
        plt.scatter([], [], c='green', s=100, label='Low', alpha=0.7)
        plt.legend(loc='upper right')
        
        # Plot 2: Congestion by time of day for top hotspots
        plt.subplot(2, 2, 2)
        
        # Get top 3 critical hotspots
        top_hotspots = []
        for severity in ["critical", "high"]:
            for spot in hotspots.get(severity, [])[:3]:
                top_hotspots.append(spot["junction_id"])
        
        # Limit to top 3
        top_hotspots = top_hotspots[:3]
        
        # Colors for different hotspots
        colors = ['red', 'orange', 'blue']
        
        # Get congestion data for these junctions across time
        for i, junction_id in enumerate(top_hotspots):
            times = []
            congestion = []
            
            for time, data in analysis_results["traffic_data"].get(junction_id, {}).items():
                if data["road_count"] > 0:
                    times.append(time)
                    congestion.append(data["avg_congestion"] / data["road_count"])
            
            if times and congestion:
                plt.plot(times, congestion, marker='o', linestyle='-', color=colors[i], 
                        label=f'Junction {junction_id}')
        
        plt.title('Congestion by Time of Day')
        plt.ylabel('Congestion Level')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Signal Optimization Impact
        plt.subplot(2, 1, 2)
        
        # Check if we have any optimized junctions
        if signal_plans:
            # Plot predicted congestion reduction for optimized junctions
            junctions = list(signal_plans.keys())[:10]  # Limit to top 10 junctions
            improvements = [signal_plans[j]["predicted_improvement"] * 100 for j in junctions]
            
            # Sort by improvement if we have any junctions
            if junctions and improvements:
                improvements, junctions = zip(*sorted(zip(improvements, junctions), reverse=True))
                
                # Create bar colors based on improvement level
                bar_colors = ['green' if imp > 30 else 'blue' if imp > 20 else 'orange' for imp in improvements]
                
                plt.bar(junctions, improvements, color=bar_colors, alpha=0.7)
                plt.title('Predicted Congestion Reduction with Signal Optimization')
                plt.ylabel('Reduction (%)')
                plt.xlabel('Junction')
                plt.xticks(rotation=45)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(True, axis='y', alpha=0.3)
            else:
                plt.text(0.5, 0.5, "No improvements data available", ha='center', va='center')
                plt.title('Signal Optimization - No Data')
        else:
            plt.text(0.5, 0.5, "No signal plans available", ha='center', va='center')
            plt.title('Signal Optimization - No Plans')
        
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
    """Run a demonstration of the traffic management system."""
    # Initialize the system
    system = TrafficManagementSystem()
    
    # Generate simulated traffic data
    system.generate_traffic_data(grid_size=(5, 5), time_periods=24)
    
    # Analyze traffic patterns
    analysis_results = system.analyze_traffic_patterns()
    
    # Identify congestion hotspots
    hotspots = system.identify_congestion_hotspots(analysis_results)
    
    # Optimize traffic signals
    signal_plans = system.optimize_traffic_signals(hotspots)
    
    # Visualize results
    output_dir = os.path.join(root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    system.visualize_traffic_optimization(
        hotspots, 
        signal_plans, 
        f"{output_dir}/traffic_optimization.png"
    )
    
    print("\n" + "=" * 80)
    print("TRAFFIC MANAGEMENT SYSTEM DEMO COMPLETE".center(80))
    print("=" * 80)
    
    print("\nThis demonstration has shown how the encrypted ASI engine can be used")
    print("to build sophisticated traffic management applications without accessing")
    print("the proprietary algorithms and mathematical implementations.")
    
    print("\nThe application successfully leveraged ASI capabilities:")
    print("1. Pattern discovery in traffic flow data")
    print("2. Insight generation for traffic optimization")
    print("3. Timeline prediction for optimal signal timing")
    
    print("\nCheck the generated visualization in the reports directory.")

if __name__ == "__main__":
    run_demo()
