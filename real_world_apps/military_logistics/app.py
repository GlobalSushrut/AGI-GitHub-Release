#!/usr/bin/env python3
"""
Military Logistics Application
-----------------------------

A real-world application that demonstrates how to use the AGI Toolkit
for military logistics optimization, resource planning, and risk assessment.

Features:
- Supply chain management
- Resource tracking
- Route optimization
- Risk assessment
- Maintenance scheduling
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to path so we can import the AGI Toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the AGI Toolkit
from agi_toolkit import AGIAPI

from resource_manager import ResourceManager
from route_optimizer import RouteOptimizer
from risk_assessor import RiskAssessor


class MilitaryLogistics:
    """Military logistics application using AGI Toolkit."""
    
    def __init__(self):
        """Initialize the military logistics application."""
        # Configure logging
        self.logger = logging.getLogger("MilitaryLogistics")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing Military Logistics Application")
        
        # Initialize the AGI Toolkit API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        # Initialize components
        self.resource_manager = ResourceManager(self.api)
        self.route_optimizer = RouteOptimizer(self.api)
        self.risk_assessor = RiskAssessor(self.api)
        
        self.logger.info("Military Logistics Application initialized")
    
    def add_resource(self, resource_id: str, name: str, category: str, quantity: int, location: str) -> Dict:
        """Add a resource to inventory."""
        return self.resource_manager.add_resource(resource_id, name, category, quantity, location)
    
    def update_resource(self, resource_id: str, quantity: int, location: str = None) -> Dict:
        """Update resource quantity or location."""
        return self.resource_manager.update_resource(resource_id, quantity, location)
    
    def get_resource(self, resource_id: str) -> Dict:
        """Get resource details."""
        return self.resource_manager.get_resource(resource_id)
    
    def list_resources(self, category: str = None, location: str = None) -> List[Dict]:
        """List resources with optional filtering."""
        return self.resource_manager.list_resources(category, location)
    
    def optimize_route(self, start_point: str, destinations: List[str], 
                      constraints: Dict = None) -> Dict:
        """Optimize delivery routes."""
        return self.route_optimizer.optimize(start_point, destinations, constraints)
    
    def assess_route_risks(self, route: List[str], date: str = None) -> Dict:
        """Assess risks for a given route."""
        return self.risk_assessor.assess_route(route, date)
    
    def generate_supply_plan(self, operation_id: str, duration_days: int, 
                           troop_count: int, mission_type: str) -> Dict:
        """Generate a supply plan for an operation."""
        self.logger.info(f"Generating supply plan for operation {operation_id}")
        
        # Calculate base resource requirements
        daily_resources = self._calculate_daily_requirements(troop_count, mission_type)
        
        # Adjust for mission duration
        total_resources = {}
        for resource, daily_amount in daily_resources.items():
            total_resources[resource] = daily_amount * duration_days
        
        # Add buffer based on risk assessment
        risk_factors = self.risk_assessor.assess_operation(operation_id, mission_type)
        overall_risk = risk_factors.get("overall_risk", 0.5)
        
        # Apply risk buffer (higher risk = more buffer supplies)
        buffer_multiplier = 1.0 + (overall_risk * 0.5)  # Up to 50% more for high risk
        
        buffered_resources = {}
        for resource, amount in total_resources.items():
            buffered_resources[resource] = int(amount * buffer_multiplier)
        
        # Generate plan
        supply_plan = {
            "operation_id": operation_id,
            "duration_days": duration_days,
            "troop_count": troop_count,
            "mission_type": mission_type,
            "risk_assessment": risk_factors,
            "resources_required": buffered_resources,
            "generated_at": datetime.now().isoformat()
        }
        
        return supply_plan
    
    def _calculate_daily_requirements(self, troop_count: int, mission_type: str) -> Dict[str, float]:
        """Calculate daily resource requirements based on troop count and mission type."""
        # Base requirements per soldier per day
        base_requirements = {
            "water_liters": 5.0,  # Liters of water
            "food_rations": 3.0,  # Meal packs
            "fuel_liters": 0.5,   # Vehicle fuel
            "ammo_kg": 0.2,       # Ammunition
            "medical_supplies": 0.1  # Medical supplies
        }
        
        # Mission type multipliers
        mission_multipliers = {
            "training": {
                "water_liters": 1.0,
                "food_rations": 1.0,
                "fuel_liters": 0.8,
                "ammo_kg": 0.5,
                "medical_supplies": 0.7
            },
            "peacekeeping": {
                "water_liters": 1.2,
                "food_rations": 1.2,
                "fuel_liters": 1.5,
                "ammo_kg": 0.7,
                "medical_supplies": 1.2
            },
            "combat": {
                "water_liters": 1.5,
                "food_rations": 1.5,
                "fuel_liters": 2.0,
                "ammo_kg": 3.0,
                "medical_supplies": 2.0
            }
        }
        
        # Default to training if mission type not recognized
        multipliers = mission_multipliers.get(mission_type.lower(), mission_multipliers["training"])
        
        # Calculate requirements
        requirements = {}
        for resource, base_amount in base_requirements.items():
            multiplier = multipliers.get(resource, 1.0)
            requirements[resource] = base_amount * multiplier * troop_count
        
        return requirements


def display_resources(resources: List[Dict]):
    """Display resources in a user-friendly format."""
    print("\n" + "="*80)
    print("RESOURCES".center(80))
    print("="*80 + "\n")
    
    if not resources:
        print("No resources found.")
        return
    
    for i, resource in enumerate(resources, 1):
        print(f"{i}. {resource.get('name', 'Unnamed resource')} ({resource.get('resource_id', 'unknown')})")
        print(f"   Category: {resource.get('category', 'uncategorized')}")
        print(f"   Quantity: {resource.get('quantity', 0)}")
        print(f"   Location: {resource.get('location', 'unknown')}")
        print("-" * 80)
    
    print("="*80)


def display_route(route_plan: Dict):
    """Display a route plan in a user-friendly format."""
    print("\n" + "="*80)
    print("OPTIMIZED ROUTE".center(80))
    print("="*80 + "\n")
    
    if not route_plan or "error" in route_plan:
        print(f"Error: {route_plan.get('error', 'Unknown error')}")
        return
    
    print(f"Start point: {route_plan.get('start_point', 'unknown')}")
    
    print("\nRoute:")
    for i, waypoint in enumerate(route_plan.get("route", []), 1):
        print(f"  {i}. {waypoint}")
    
    print(f"\nEstimated distance: {route_plan.get('total_distance', 0)} km")
    print(f"Estimated duration: {route_plan.get('total_time', 0)} hours")
    
    if "risks" in route_plan:
        print("\nRisk assessment:")
        for risk_type, level in route_plan.get("risks", {}).items():
            print(f"  {risk_type.replace('_', ' ').title()}: {level}")
    
    print("="*80)


def display_supply_plan(supply_plan: Dict):
    """Display a supply plan in a user-friendly format."""
    print("\n" + "="*80)
    print("OPERATION SUPPLY PLAN".center(80))
    print("="*80 + "\n")
    
    print(f"Operation ID: {supply_plan.get('operation_id', 'unknown')}")
    print(f"Mission type: {supply_plan.get('mission_type', 'unknown')}")
    print(f"Duration: {supply_plan.get('duration_days', 0)} days")
    print(f"Troop count: {supply_plan.get('troop_count', 0)}")
    
    print("\nResources required:")
    for resource, amount in supply_plan.get("resources_required", {}).items():
        print(f"  {resource.replace('_', ' ').title()}: {amount:.1f}")
    
    print("\nRisk assessment:")
    for risk_type, level in supply_plan.get("risk_assessment", {}).items():
        if isinstance(level, (int, float)):
            print(f"  {risk_type.replace('_', ' ').title()}: {level:.2f}")
    
    print(f"\nGenerated at: {supply_plan.get('generated_at', 'unknown')}")
    print("="*80)


def demo_logistics():
    """Run a demonstration of the military logistics application."""
    logistics = MilitaryLogistics()
    
    # Add sample resources
    logistics.add_resource("water-001", "Bottled Water", "consumable", 5000, "Base Alpha")
    logistics.add_resource("food-001", "MRE Packs", "consumable", 2000, "Base Alpha")
    logistics.add_resource("fuel-001", "Diesel Fuel", "fuel", 10000, "Base Alpha")
    logistics.add_resource("ammo-001", "5.56mm Rounds", "ammunition", 50000, "Base Alpha")
    logistics.add_resource("med-001", "First Aid Kits", "medical", 500, "Base Alpha")
    
    # List resources
    resources = logistics.list_resources()
    display_resources(resources)
    
    # Optimize route
    route = logistics.optimize_route(
        "Base Alpha",
        ["Checkpoint Bravo", "Forward Operating Base Charlie", "Outpost Delta"],
        {"avoid_hostile_areas": True, "prioritize_safety": True}
    )
    display_route(route)
    
    # Generate supply plan
    supply_plan = logistics.generate_supply_plan(
        "OP-EAGLE-EYE",
        14,  # 14 days
        120,  # 120 troops
        "peacekeeping"
    )
    display_supply_plan(supply_plan)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Military Logistics Application")
    parser.add_argument("--demo", action="store_true", help="Run demonstration mode")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if args.demo:
        demo_logistics()
    else:
        print("Please use --demo to run the demonstration.")


if __name__ == "__main__":
    main()
