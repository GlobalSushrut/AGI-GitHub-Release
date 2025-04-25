#!/usr/bin/env python3
"""
Supply Chain Optimizer
---------------------
Optimizes logistics and inventory management using ASI engine.
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

class SupplyChainOptimizer:
    """Supply chain optimization system using ASI infrastructure."""
    
    def __init__(self):
        """Initialize the supply chain optimizer."""
        print("\nSUPPLY CHAIN OPTIMIZER - Powered by Encrypted ASI Engine")
        
        # Initialize ASI engine
        print("Initializing ASI engine...")
        success = initialize_asi()
        
        if not success:
            print("Failed to initialize ASI engine. Exiting.")
            sys.exit(1)
        
        # Create ASI instance with specific configuration for supply chain analysis
        self.asi = create_asi_instance(name="SupplyChainASI", config={
            "domain": "logistics",
            "optimization_factor": 0.85,
            "system_complexity": 0.8,
            "multivariate_analysis": True
        })
        
        print("ASI engine initialized successfully")
        
        # Initialize supply chain data
        self.facilities = {}
        self.products = {}
        self.inventory = {}
        self.shipments = {}
        self.optimizations = {}
    
    def generate_supply_chain_data(self, num_facilities=10, num_products=15, num_days=30):
        """Generate simulated supply chain data."""
        print(f"\nGenerating simulated data for {num_facilities} facilities and {num_products} products...")
        
        # Define facility types
        facility_types = ["Factory", "Distribution Center", "Warehouse", "Retail"]
        
        # Define product categories
        categories = ["Electronics", "Clothing", "Food", "Home", "Beauty"]
        
        # Generate facilities
        for i in range(1, num_facilities + 1):
            facility_id = f"F{i}"
            facility_type = random.choice(facility_types)
            
            # Generate capacity based on facility type
            if facility_type == "Factory":
                capacity = random.uniform(5000, 20000)
            elif facility_type == "Distribution Center":
                capacity = random.uniform(15000, 40000)
            elif facility_type == "Warehouse":
                capacity = random.uniform(30000, 80000)
            else:  # Retail
                capacity = random.uniform(1000, 5000)
            
            self.facilities[facility_id] = {
                "name": f"{facility_type} {i}",
                "type": facility_type,
                "capacity": capacity,
                "location": (random.uniform(0, 100), random.uniform(0, 100)),
                "connected_facilities": []
            }
        
        # Create network connections
        for facility_id, facility in self.facilities.items():
            # Define potential connections based on facility type
            potential_connections = []
            
            if facility["type"] == "Factory":
                # Factories connect to Distribution Centers and Warehouses
                potential_connections = [f for f, props in self.facilities.items() 
                                       if props["type"] in ["Distribution Center", "Warehouse"]
                                       and f != facility_id]
            elif facility["type"] == "Distribution Center":
                # Distribution Centers connect to Warehouses and Retail
                potential_connections = [f for f, props in self.facilities.items() 
                                       if props["type"] in ["Warehouse", "Retail"]
                                       and f != facility_id]
            elif facility["type"] == "Warehouse":
                # Warehouses connect to Retail
                potential_connections = [f for f, props in self.facilities.items() 
                                       if props["type"] == "Retail"
                                       and f != facility_id]
            
            # Select 1-3 connections
            num_connections = min(random.randint(1, 3), len(potential_connections))
            facility["connected_facilities"] = random.sample(potential_connections, num_connections)
        
        # Generate products
        for i in range(1, num_products + 1):
            product_id = f"P{i}"
            category = random.choice(categories)
            
            # Generate properties based on category
            if category == "Electronics":
                volume = random.uniform(0.5, 5)
                value = random.uniform(100, 1000)
                shelf_life = None  # Non-perishable
            elif category == "Clothing":
                volume = random.uniform(0.2, 2)
                value = random.uniform(20, 200)
                shelf_life = None  # Non-perishable
            elif category == "Food":
                volume = random.uniform(0.1, 1)
                value = random.uniform(5, 50)
                shelf_life = random.randint(5, 90)  # Days
            elif category == "Home":
                volume = random.uniform(1, 10)
                value = random.uniform(30, 300)
                shelf_life = None  # Non-perishable
            else:  # Beauty
                volume = random.uniform(0.1, 0.5)
                value = random.uniform(10, 100)
                shelf_life = random.randint(180, 720)  # Days
            
            self.products[product_id] = {
                "name": f"{category} Product {i}",
                "category": category,
                "volume": volume,  # cubic meters
                "value": value,  # dollars
                "shelf_life": shelf_life,  # days or None
                "lead_time": random.randint(1, 14)  # days to produce/procure
            }
        
        # Generate inventory data
        # Initialize inventory structure
        for facility_id in self.facilities:
            self.inventory[facility_id] = {}
            
            # Only add products that make sense for the facility type
            facility_type = self.facilities[facility_id]["type"]
            
            for product_id, product in self.products.items():
                # Factories have fewer product types but higher quantities
                if facility_type == "Factory":
                    if random.random() < 0.3:  # 30% chance to have this product
                        max_capacity = self.facilities[facility_id]["capacity"] / 10
                        self.inventory[facility_id][product_id] = {
                            "current_stock": random.uniform(max_capacity * 0.4, max_capacity * 0.8),
                            "min_stock": max_capacity * 0.2,
                            "max_stock": max_capacity,
                            "stock_history": []
                        }
                # Distribution centers have more products
                elif facility_type == "Distribution Center":
                    if random.random() < 0.6:  # 60% chance
                        max_capacity = self.facilities[facility_id]["capacity"] / 20
                        self.inventory[facility_id][product_id] = {
                            "current_stock": random.uniform(max_capacity * 0.3, max_capacity * 0.7),
                            "min_stock": max_capacity * 0.15,
                            "max_stock": max_capacity,
                            "stock_history": []
                        }
                # Warehouses have most products
                elif facility_type == "Warehouse":
                    if random.random() < 0.8:  # 80% chance
                        max_capacity = self.facilities[facility_id]["capacity"] / 30
                        self.inventory[facility_id][product_id] = {
                            "current_stock": random.uniform(max_capacity * 0.3, max_capacity * 0.7),
                            "min_stock": max_capacity * 0.15,
                            "max_stock": max_capacity,
                            "stock_history": []
                        }
                # Retail has selective inventory
                else:
                    if random.random() < 0.4:  # 40% chance
                        max_capacity = self.facilities[facility_id]["capacity"] / 15
                        self.inventory[facility_id][product_id] = {
                            "current_stock": random.uniform(max_capacity * 0.3, max_capacity * 0.7),
                            "min_stock": max_capacity * 0.2,
                            "max_stock": max_capacity,
                            "stock_history": []
                        }
        
        # Generate historical inventory data
        start_date = datetime.now() - timedelta(days=num_days)
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        for facility_id, products in self.inventory.items():
            for product_id, inventory_data in products.items():
                max_stock = inventory_data["max_stock"]
                min_stock = inventory_data["min_stock"]
                current_stock = inventory_data["current_stock"]
                
                # Generate historical stock levels
                stock_history = []
                
                for date in dates:
                    # Create some variability in stock levels
                    # We'll create trends that make sense (gradual depletion followed by replenishment)
                    if not stock_history:
                        # Start with current_stock for the oldest date
                        stock_level = current_stock
                    else:
                        # Get previous stock level
                        prev_stock = stock_history[-1]["stock_level"]
                        
                        # Simulate daily usage
                        usage = random.uniform(0, prev_stock * 0.2)  # Use up to 20% of stock
                        
                        # Simulate replenishment if stock is low
                        if prev_stock - usage < min_stock and random.random() < 0.7:
                            # 70% chance to reorder when below min stock
                            replenishment = random.uniform(max_stock * 0.3, max_stock * 0.7)
                        else:
                            replenishment = 0
                        
                        # Calculate new stock level
                        stock_level = max(0, prev_stock - usage + replenishment)
                    
                    stock_history.append({
                        "date": date,
                        "stock_level": stock_level
                    })
                
                # Store the history
                inventory_data["stock_history"] = stock_history
        
        # Generate shipment data
        self.shipments = []
        
        # Create reasonable number of shipments
        num_shipments = num_facilities * num_products // 3
        
        for i in range(num_shipments):
            # Select random source and destination facilities
            source_candidates = [f for f, props in self.facilities.items() 
                              if props["type"] in ["Factory", "Distribution Center", "Warehouse"]]
            
            if not source_candidates:
                continue
                
            source_id = random.choice(source_candidates)
            
            # Destination should be connected to source
            if not self.facilities[source_id]["connected_facilities"]:
                continue
                
            destination_id = random.choice(self.facilities[source_id]["connected_facilities"])
            
            # Select a product that exists in source inventory
            available_products = list(self.inventory[source_id].keys())
            
            if not available_products:
                continue
                
            product_id = random.choice(available_products)
            
            # Generate random shipment date
            shipment_date = start_date + timedelta(days=random.randint(0, num_days-1))
            
            # Calculate transit time based on distance
            source_loc = self.facilities[source_id]["location"]
            dest_loc = self.facilities[destination_id]["location"]
            
            distance = ((source_loc[0] - dest_loc[0])**2 + (source_loc[1] - dest_loc[1])**2)**0.5
            transit_time = max(1, int(distance / 20))  # 1 day per 20 distance units
            
            # Generate quantity
            quantity = random.uniform(50, 500)
            
            # Create shipment
            shipment = {
                "shipment_id": f"SH{i+1}",
                "source_id": source_id,
                "destination_id": destination_id,
                "product_id": product_id,
                "quantity": quantity,
                "shipment_date": shipment_date,
                "transit_time": transit_time,
                "status": random.choice(["Completed", "In Transit", "Scheduled"]),
                "cost": quantity * self.products[product_id]["value"] * 0.1  # Shipping cost
            }
            
            self.shipments.append(shipment)
        
        print(f"Generated data for {len(self.facilities)} facilities, {len(self.products)} products, and {len(self.shipments)} shipments")
    
    def analyze_supply_chain(self):
        """Analyze supply chain patterns using ASI pattern discovery."""
        print("\nAnalyzing supply chain patterns...")
        
        # Prepare data for ASI analysis
        analysis_data = []
        
        # Analyze inventory patterns
        for facility_id, products in self.inventory.items():
            facility = self.facilities[facility_id]
            
            # Calculate inventory metrics
            total_value = 0
            avg_turnover = 0
            stock_outs = 0
            products_count = 0
            
            for product_id, inventory_data in products.items():
                product = self.products[product_id]
                
                # Calculate current inventory value
                current_value = inventory_data["current_stock"] * product["value"]
                total_value += current_value
                
                # Count near-stockouts from history
                if inventory_data["stock_history"]:
                    low_stock_count = sum(1 for h in inventory_data["stock_history"] 
                                       if h["stock_level"] < inventory_data["min_stock"])
                    stock_outs += low_stock_count
                
                products_count += 1
            
            # Calculate average metrics
            if products_count > 0:
                avg_value = total_value / products_count
                avg_stock_outs = stock_outs / products_count
            else:
                avg_value = 0
                avg_stock_outs = 0
            
            # Create facility analysis data
            facility_data = {
                "facility_id": facility_id,
                "facility_type": facility["type"],
                "capacity": facility["capacity"],
                "products_count": products_count,
                "total_inventory_value": total_value,
                "avg_product_value": avg_value,
                "avg_stock_outs": avg_stock_outs,
                "connection_count": len(facility["connected_facilities"])
            }
            
            analysis_data.append(facility_data)
        
        # Convert list to dictionary format expected by ASI API
        properties_dict = {}
        
        # Extract numeric properties from each facility's analysis data
        for i, facility_data in enumerate(analysis_data):
            facility_id = facility_data.get('facility_id', f'F{i}')
            
            # Add numeric values to the properties dictionary
            for key, value in facility_data.items():
                if isinstance(value, (int, float)):
                    properties_dict[f"{facility_id}_{key}"] = value
        
        # Ensure we have at least some properties
        if not properties_dict:
            properties_dict = {
                "inventory_turnover": 0.7,
                "stockout_frequency": 0.3,
                "inventory_value": 0.65,
                "order_efficiency": 0.8
            }
            
        # Use ASI pattern discovery
        patterns = self.asi.discover_patterns(
            domain="supply_chain",
            properties=properties_dict
        )
        
        print(f"Discovered {len(patterns['patterns'])} supply chain patterns")
        print(f"Pattern analysis confidence: {patterns['confidence']:.4f}")
        
        # Display top patterns
        for i, pattern in enumerate(patterns['patterns'][:2]):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"Description: {pattern['description']}")
        
        return patterns
    
    def optimize_inventory_levels(self, patterns):
        """Generate inventory optimization recommendations using ASI insight."""
        print("\nGenerating inventory optimization recommendations...")
        
        # Generate insights for supply chain optimization
        supply_chain_concepts = [
            "inventory_optimization", "lead_time_reduction", "stockout_prevention",
            "cost_reduction", "throughput_improvement"
        ]
        
        insights = self.asi.generate_insight(concepts=supply_chain_concepts)
        
        print(f"Supply chain insight (Confidence: {insights['confidence']:.4f}):")
        print(f"\"{insights['text']}\"")
        
        # Identify problematic facilities
        problematic_facilities = []
        
        for facility_id, products in self.inventory.items():
            facility = self.facilities[facility_id]
            
            # Calculate stock-out frequency
            stock_out_freq = 0
            total_products = 0
            
            for product_id, inventory_data in products.items():
                if inventory_data["stock_history"]:
                    low_stock_count = sum(1 for h in inventory_data["stock_history"] 
                                       if h["stock_level"] < inventory_data["min_stock"])
                    stock_out_freq += low_stock_count / len(inventory_data["stock_history"])
                    total_products += 1
            
            if total_products > 0:
                avg_stock_out_freq = stock_out_freq / total_products
            else:
                avg_stock_out_freq = 0
            
            # Check for problems
            if avg_stock_out_freq > 0.1 or total_products < 2:
                problematic_facilities.append({
                    "facility_id": facility_id,
                    "stock_out_frequency": avg_stock_out_freq,
                    "products_count": total_products,
                    "type": facility["type"]
                })
        
        # Generate optimization recommendations for each problematic facility
        for facility in problematic_facilities:
            facility_id = facility["facility_id"]
            
            # Create scenario for ASI prediction
            supply_chain_scenario = {
                "name": f"Inventory optimization for {facility_id}",
                "complexity": 0.7,
                "uncertainty": 0.5,
                "domain": "inventory_management",
                "variables": {
                    "facility_id": facility_id,
                    "facility_type": facility["type"],
                    "stock_out_frequency": facility["stock_out_frequency"],
                    "products_count": facility["products_count"]
                }
            }
            
            # Use ASI to predict optimal inventory strategies
            prediction = self.asi.predict_timeline(supply_chain_scenario)
            
            # Create optimization recommendations
            self.optimizations[facility_id] = {
                "facility_id": facility_id,
                "current_status": {
                    "stock_out_frequency": facility["stock_out_frequency"],
                    "products_count": facility["products_count"]
                },
                "inventory_adjustments": self._extract_inventory_adjustments(prediction),
                "reorder_recommendations": self._extract_reorder_recommendations(prediction),
                "facility_improvements": self._extract_facility_improvements(prediction),
                "expected_stockout_reduction": prediction.get("metrics", {}).get("stockout_reduction", 0.3),
                "expected_cost_savings": prediction.get("metrics", {}).get("cost_savings", 0.15),
                "confidence": prediction["confidence"]
            }
        
        # Print optimization summary
        print(f"\nGenerated optimization recommendations for {len(self.optimizations)} facilities")
        
        for facility_id, opt in list(self.optimizations.items())[:2]:
            print(f"\nFacility {facility_id}:")
            print(f"Current stock-out frequency: {opt['current_status']['stock_out_frequency']:.2f}")
            print(f"Expected stock-out reduction: {opt['expected_stockout_reduction']*100:.1f}%")
            print(f"Expected cost savings: {opt['expected_cost_savings']*100:.1f}%")
            print(f"Confidence: {opt['confidence']:.4f}")
            
            if opt['inventory_adjustments']:
                print("Top inventory adjustments:")
                for i, adj in enumerate(opt['inventory_adjustments'][:2]):
                    print(f"  - {adj['event']}")
        
        return self.optimizations
    
    def _extract_inventory_adjustments(self, prediction):
        """Extract inventory adjustments from ASI prediction."""
        adjustments = []
        
        for event in prediction["base_timeline"]:
            if any(term in event.get("event", "").lower() for term in 
                   ["inventory", "stock level", "safety stock"]):
                adjustments.append(event)
        
        return adjustments
    
    def _extract_reorder_recommendations(self, prediction):
        """Extract reorder recommendations from ASI prediction."""
        recommendations = []
        
        for event in prediction["base_timeline"]:
            if any(term in event.get("event", "").lower() for term in 
                   ["reorder", "order", "procurement"]):
                recommendations.append(event)
        
        return recommendations
    
    def _extract_facility_improvements(self, prediction):
        """Extract facility improvement recommendations from ASI prediction."""
        improvements = []
        
        for event in prediction["base_timeline"]:
            if any(term in event.get("event", "").lower() for term in 
                   ["facility", "layout", "capacity", "improve"]):
                improvements.append(event)
        
        return improvements
    
    def optimize_routing(self):
        """Optimize shipment routing using ASI timeline prediction."""
        print("\nOptimizing shipment routing...")
        
        # Only proceed if we have facilities and shipments
        if not self.facilities or not self.shipments:
            print("No facilities or shipments available for routing optimization")
            return None
        
        # Create scenario for ASI prediction
        routing_scenario = {
            "name": "Supply chain routing optimization",
            "complexity": 0.8,
            "uncertainty": 0.6,
            "domain": "logistics_routing",
            "variables": {
                "facility_count": len(self.facilities),
                "shipment_count": len(self.shipments),
                "network_complexity": sum(len(f["connected_facilities"]) for f in self.facilities.values()) / len(self.facilities)
            }
        }
        
        # Use ASI to predict optimal routing
        prediction = self.asi.predict_timeline(routing_scenario)
        
        # Extract routing recommendations
        routing_recommendations = {
            "network_adjustments": [],
            "transit_time_reduction": [],
            "cost_savings": []
        }
        
        for event in prediction["base_timeline"]:
            event_text = event.get("event", "").lower()
            
            if "network" in event_text or "connection" in event_text:
                routing_recommendations["network_adjustments"].append(event)
            elif "transit" in event_text or "delivery" in event_text:
                routing_recommendations["transit_time_reduction"].append(event)
            elif "cost" in event_text or "saving" in event_text:
                routing_recommendations["cost_savings"].append(event)
        
        # Calculate expected improvements
        expected_improvements = {
            "transit_time_reduction": prediction.get("metrics", {}).get("transit_time_reduction", 0.2),
            "cost_reduction": prediction.get("metrics", {}).get("cost_reduction", 0.15),
            "network_efficiency": prediction.get("metrics", {}).get("network_efficiency", 0.25),
            "confidence": prediction["confidence"]
        }
        
        # Print routing optimization summary
        print(f"\nGenerated routing optimization recommendations")
        print(f"Expected transit time reduction: {expected_improvements['transit_time_reduction']*100:.1f}%")
        print(f"Expected cost reduction: {expected_improvements['cost_reduction']*100:.1f}%")
        print(f"Expected network efficiency improvement: {expected_improvements['network_efficiency']*100:.1f}%")
        print(f"Confidence: {expected_improvements['confidence']:.4f}")
        
        # Print sample recommendations
        for category, recommendations in routing_recommendations.items():
            if recommendations:
                print(f"\n{category.replace('_', ' ').title()}:")
                for i, rec in enumerate(recommendations[:2]):
                    print(f"  - {rec['event']}")
        
        return {
            "recommendations": routing_recommendations,
            "expected_improvements": expected_improvements
        }
    
    def visualize_supply_chain(self, output_path=None):
        """Visualize supply chain network and optimization results."""
        if not self.facilities or not self.inventory:
            print("No supply chain data available to visualize")
            return False
        
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Supply Chain Network
        plt.subplot(2, 2, 1)
        
        # Define marker styles for different facility types
        markers = {
            "Factory": "s",       # square
            "Distribution Center": "^",  # triangle up
            "Warehouse": "o",     # circle
            "Retail": "D"         # diamond
        }
        
        # Define colors for different facility conditions
        colors = {
            "optimized": "green",
            "problematic": "red",
            "normal": "blue"
        }
        
        # Plot facilities
        for facility_id, facility in self.facilities.items():
            x, y = facility["location"]
            facility_type = facility["type"]
            
            # Determine marker
            marker = markers.get(facility_type, "o")
            
            # Determine color based on optimization status
            if facility_id in self.optimizations:
                color = colors["problematic"]
            else:
                color = colors["normal"]
            
            # Size based on capacity
            size = max(100, min(500, facility["capacity"] / 100))
            
            # Plot the facility
            plt.scatter(x, y, s=size, c=color, marker=marker, alpha=0.7)
            plt.text(x, y, facility_id, fontsize=8, ha='center', va='center')
            
            # Plot connections
            for connected_id in facility["connected_facilities"]:
                if connected_id in self.facilities:
                    x2, y2 = self.facilities[connected_id]["location"]
                    plt.plot([x, x2], [y, y2], 'gray', alpha=0.4)
        
        plt.title('Supply Chain Network')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        # Add a legend for facility types
        for facility_type, marker in markers.items():
            plt.scatter([], [], c='blue', marker=marker, s=100, label=facility_type)
        plt.legend(loc='upper right')
        
        # Plot 2: Inventory Analysis
        plt.subplot(2, 2, 2)
        
        # Group facilities by type
        facility_types = {}
        for facility_id, facility in self.facilities.items():
            facility_type = facility["type"]
            if facility_type not in facility_types:
                facility_types[facility_type] = []
            facility_types[facility_type].append(facility_id)
        
        # Calculate average inventory value by facility type
        avg_values = []
        facility_type_labels = []
        
        for facility_type, facility_ids in facility_types.items():
            total_value = 0
            count = 0
            
            for facility_id in facility_ids:
                if facility_id in self.inventory:
                    for product_id, inventory_data in self.inventory[facility_id].items():
                        if product_id in self.products:
                            product = self.products[product_id]
                            total_value += inventory_data["current_stock"] * product["value"]
                            count += 1
            
            if count > 0:
                avg_value = total_value / count
                avg_values.append(avg_value)
                facility_type_labels.append(facility_type)
        
        # Create bar chart
        plt.bar(facility_type_labels, avg_values, color='skyblue')
        plt.title('Average Inventory Value by Facility Type')
        plt.ylabel('Average Value ($)')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Optimization Improvements
        plt.subplot(2, 1, 2)
        
        if self.optimizations:
            facility_ids = list(self.optimizations.keys())
            stockout_reductions = [self.optimizations[f]["expected_stockout_reduction"] * 100 for f in facility_ids]
            cost_savings = [self.optimizations[f]["expected_cost_savings"] * 100 for f in facility_ids]
            
            x = np.arange(len(facility_ids))
            width = 0.35
            
            plt.bar(x - width/2, stockout_reductions, width, label='Stock-out Reduction (%)')
            plt.bar(x + width/2, cost_savings, width, label='Cost Savings (%)')
            
            plt.xlabel('Facility')
            plt.ylabel('Improvement (%)')
            plt.title('Expected Supply Chain Improvements')
            plt.xticks(x, facility_ids)
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No optimization data available", ha='center', va='center')
            plt.title('Supply Chain Optimization')
        
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
    """Run a demonstration of the supply chain optimizer."""
    # Initialize the optimizer
    optimizer = SupplyChainOptimizer()
    
    # Generate simulated supply chain data
    optimizer.generate_supply_chain_data()
    
    # Analyze supply chain patterns
    patterns = optimizer.analyze_supply_chain()
    
    # Optimize inventory levels
    optimizer.optimize_inventory_levels(patterns)
    
    # Optimize routing
    optimizer.optimize_routing()
    
    # Visualize results
    output_dir = os.path.join(root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    optimizer.visualize_supply_chain(f"{output_dir}/supply_chain_optimization.png")
    
    print("\nSUPPLY CHAIN OPTIMIZER DEMO COMPLETE")
    print("\nThis demonstration has shown how the encrypted ASI engine can be used")
    print("to build sophisticated supply chain optimization applications without")
    print("accessing the proprietary algorithms and mathematical implementations.")

if __name__ == "__main__":
    run_demo()
