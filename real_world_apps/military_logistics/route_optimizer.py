#!/usr/bin/env python3
"""
Route Optimizer for Military Logistics
------------------------------------

Optimizes routes for military supply operations considering
terrain, threats, and mission constraints.
"""

import os
import sys
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger("RouteOptimizer")


class RouteOptimizer:
    """Optimizes routes for military logistics operations."""
    
    def __init__(self, api):
        """
        Initialize the route optimizer.
        
        Args:
            api: The AGI Toolkit API instance
        """
        self.api = api
        self.logger = logger
        
        # Load map data
        self.map_data = {}
        self.locations = {}
        self.load_map_data()
    
    def load_map_data(self):
        """Load map and location data from API memory."""
        try:
            # Load map data
            memory_key = "military_logistics_map_data"
            map_data = self.api.retrieve_data(memory_key)
            
            if map_data and isinstance(map_data, dict):
                self.map_data = map_data
                self.logger.info(f"Loaded map data from memory")
            else:
                self.map_data = self.generate_sample_map_data()
                self.logger.info("Generated sample map data")
                
                # Save the generated data
                self.api.store_data(memory_key, self.map_data)
            
            # Load location data
            memory_key = "military_logistics_locations"
            location_data = self.api.retrieve_data(memory_key)
            
            if location_data and isinstance(location_data, dict):
                self.locations = location_data
                self.logger.info(f"Loaded {len(self.locations)} locations from memory")
            else:
                self.locations = self.generate_sample_locations()
                self.logger.info(f"Generated {len(self.locations)} sample locations")
                
                # Save the generated data
                self.api.store_data(memory_key, self.locations)
            
        except Exception as e:
            self.logger.error(f"Error loading map data: {str(e)}")
            
            # Generate sample data if loading fails
            self.map_data = self.generate_sample_map_data()
            self.locations = self.generate_sample_locations()
    
    def generate_sample_map_data(self) -> Dict:
        """Generate sample map data for demonstration."""
        return {
            "regions": {
                "north": {
                    "terrain": "mountainous",
                    "risk_level": 0.7,
                    "weather_conditions": "snow",
                    "infrastructure_quality": 0.3
                },
                "east": {
                    "terrain": "forest",
                    "risk_level": 0.5,
                    "weather_conditions": "rain",
                    "infrastructure_quality": 0.6
                },
                "south": {
                    "terrain": "desert",
                    "risk_level": 0.4,
                    "weather_conditions": "hot",
                    "infrastructure_quality": 0.5
                },
                "west": {
                    "terrain": "plains",
                    "risk_level": 0.2,
                    "weather_conditions": "mild",
                    "infrastructure_quality": 0.8
                },
                "central": {
                    "terrain": "urban",
                    "risk_level": 0.3,
                    "weather_conditions": "mild",
                    "infrastructure_quality": 0.9
                }
            },
            "routes": {
                "north-east": {"distance": 250, "difficulty": 0.8},
                "north-central": {"distance": 180, "difficulty": 0.6},
                "east-central": {"distance": 150, "difficulty": 0.4},
                "south-central": {"distance": 200, "difficulty": 0.3},
                "west-central": {"distance": 160, "difficulty": 0.2},
                "east-south": {"distance": 280, "difficulty": 0.5},
                "west-south": {"distance": 320, "difficulty": 0.4}
            }
        }
    
    def generate_sample_locations(self) -> Dict:
        """Generate sample locations for demonstration."""
        return {
            "Base Alpha": {
                "region": "central",
                "coordinates": (0, 0),
                "type": "main_base",
                "capacity": "large",
                "security_level": "high"
            },
            "Checkpoint Bravo": {
                "region": "west",
                "coordinates": (-80, 20),
                "type": "checkpoint",
                "capacity": "small",
                "security_level": "medium"
            },
            "Forward Operating Base Charlie": {
                "region": "east",
                "coordinates": (100, -30),
                "type": "forward_base",
                "capacity": "medium",
                "security_level": "high"
            },
            "Outpost Delta": {
                "region": "north",
                "coordinates": (20, 150),
                "type": "outpost",
                "capacity": "small",
                "security_level": "medium"
            },
            "Depot Echo": {
                "region": "south",
                "coordinates": (-50, -120),
                "type": "supply_depot",
                "capacity": "large",
                "security_level": "high"
            },
            "Watchtower Foxtrot": {
                "region": "north",
                "coordinates": (-30, 180),
                "type": "observation",
                "capacity": "minimal",
                "security_level": "low"
            },
            "Camp Golf": {
                "region": "west",
                "coordinates": (-150, -40),
                "type": "camp",
                "capacity": "medium",
                "security_level": "medium"
            }
        }
    
    def calculate_distance(self, start: str, end: str) -> float:
        """
        Calculate distance between two locations.
        
        Args:
            start: Starting location name
            end: Ending location name
            
        Returns:
            Distance in kilometers
        """
        # If we have coordinates, use them for more accurate distance
        if start in self.locations and end in self.locations:
            start_coords = self.locations[start].get("coordinates", (0, 0))
            end_coords = self.locations[end].get("coordinates", (0, 0))
            
            # Simple Euclidean distance (for demonstration)
            x1, y1 = start_coords
            x2, y2 = end_coords
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            # Scale to kilometers (arbitrary scale for demonstration)
            return abs(distance)
        
        # Otherwise, use predefined routes if available
        start_region = self.locations.get(start, {}).get("region", "central")
        end_region = self.locations.get(end, {}).get("region", "central")
        
        route_key = f"{start_region}-{end_region}"
        if route_key in self.map_data.get("routes", {}):
            return self.map_data["routes"][route_key].get("distance", 100)
        
        # If route not found, try the reverse direction
        route_key = f"{end_region}-{start_region}"
        if route_key in self.map_data.get("routes", {}):
            return self.map_data["routes"][route_key].get("distance", 100)
        
        # Default distance if no information is available
        return 200  # 200 km default
    
    def calculate_route_time(self, start: str, end: str, weather: str = "clear") -> float:
        """
        Calculate travel time between two locations.
        
        Args:
            start: Starting location name
            end: Ending location name
            weather: Weather conditions
            
        Returns:
            Travel time in hours
        """
        distance = self.calculate_distance(start, end)
        
        # Base speed in km/h
        base_speed = 60
        
        # Adjust for terrain and infrastructure
        start_region = self.locations.get(start, {}).get("region", "central")
        end_region = self.locations.get(end, {}).get("region", "central")
        
        # If regions are different, use the average infrastructure quality
        if start_region != end_region:
            start_infra = self.map_data.get("regions", {}).get(start_region, {}).get("infrastructure_quality", 0.5)
            end_infra = self.map_data.get("regions", {}).get(end_region, {}).get("infrastructure_quality", 0.5)
            infra_quality = (start_infra + end_infra) / 2
        else:
            infra_quality = self.map_data.get("regions", {}).get(start_region, {}).get("infrastructure_quality", 0.5)
        
        # Lower infrastructure quality reduces speed
        speed = base_speed * (0.5 + infra_quality / 2)  # Range from 50% to 100% of base speed
        
        # Weather impacts
        if weather == "snow":
            speed *= 0.6  # 60% of normal speed
        elif weather == "rain":
            speed *= 0.8  # 80% of normal speed
        elif weather == "fog":
            speed *= 0.7  # 70% of normal speed
        
        # Calculate time (hours = distance / speed)
        return distance / speed
    
    def optimize(self, start_point: str, destinations: List[str], 
               constraints: Dict = None) -> Dict:
        """
        Optimize a route through multiple destinations.
        
        Args:
            start_point: Starting location
            destinations: List of destinations to visit
            constraints: Optional constraints for route planning
            
        Returns:
            Optimized route plan
        """
        self.logger.info(f"Optimizing route from {start_point} through {len(destinations)} destinations")
        
        # Handle invalid input
        if not start_point or not destinations:
            return {"error": "Start point and at least one destination required"}
        
        # Default constraints
        if constraints is None:
            constraints = {}
        
        avoid_hostile = constraints.get("avoid_hostile_areas", False)
        prioritize_safety = constraints.get("prioritize_safety", False)
        max_daily_distance = constraints.get("max_daily_distance", 500)  # km
        
        # Validate locations exist
        all_locations = [start_point] + destinations
        for location in all_locations:
            if location not in self.locations:
                return {"error": f"Unknown location: {location}"}
        
        # If we have ASI available, use it for advanced route optimization
        if self.api.has_asi:
            route_data = {
                "start_point": start_point,
                "destinations": destinations,
                "locations": self.locations,
                "map_data": self.map_data,
                "constraints": constraints
            }
            
            result = self.api.process_with_asi({
                "task": "route_optimization",
                "data": route_data
            })
            
            if result.get("success", False):
                return result.get("route_plan", {})
        
        # Fallback: Simple route optimization
        # For demonstration, we'll use a greedy approach
        current = start_point
        unvisited = destinations.copy()
        route = [current]
        total_distance = 0
        total_time = 0
        
        while unvisited:
            # Find the nearest unvisited location
            nearest = None
            nearest_distance = float('inf')
            
            for location in unvisited:
                distance = self.calculate_distance(current, location)
                
                # If avoiding hostile areas, adjust distance based on risk
                if avoid_hostile:
                    region = self.locations.get(location, {}).get("region", "central")
                    risk = self.map_data.get("regions", {}).get(region, {}).get("risk_level", 0.5)
                    
                    # Increase effective distance based on risk
                    distance *= (1 + risk)
                
                if distance < nearest_distance:
                    nearest = location
                    nearest_distance = distance
            
            if nearest:
                # Add to route
                route.append(nearest)
                current = nearest
                unvisited.remove(nearest)
                
                # Add to totals
                total_distance += nearest_distance
                total_time += self.calculate_route_time(route[-2], route[-1])
        
        # Calculate risks
        risks = self.assess_route_risks(route)
        
        # Create the route plan
        route_plan = {
            "start_point": start_point,
            "destinations": destinations,
            "route": route,
            "total_distance": round(total_distance, 1),
            "total_time": round(total_time, 1),
            "risks": risks,
            "waypoints": self.generate_waypoints(route),
            "created_at": datetime.now().isoformat()
        }
        
        return route_plan
    
    def assess_route_risks(self, route: List[str]) -> Dict:
        """
        Assess risks for a given route.
        
        Args:
            route: List of locations in the route
            
        Returns:
            Risk assessment dictionary
        """
        risks = {
            "overall_risk": 0.0,
            "terrain_risk": 0.0,
            "weather_risk": 0.0,
            "security_risk": 0.0
        }
        
        # No route, return default risks
        if not route or len(route) < 2:
            return risks
        
        # Calculate risk for each segment
        segment_risks = []
        
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i+1]
            
            start_region = self.locations.get(start, {}).get("region", "central")
            end_region = self.locations.get(end, {}).get("region", "central")
            
            # Get region risks
            start_risk = self.map_data.get("regions", {}).get(start_region, {}).get("risk_level", 0.5)
            end_risk = self.map_data.get("regions", {}).get(end_region, {}).get("risk_level", 0.5)
            
            # Route segment risk is the average of starting and ending locations
            segment_risk = (start_risk + end_risk) / 2
            segment_risks.append(segment_risk)
            
            # Terrain risk
            start_terrain = self.map_data.get("regions", {}).get(start_region, {}).get("terrain", "plains")
            end_terrain = self.map_data.get("regions", {}).get(end_region, {}).get("terrain", "plains")
            
            # Map terrain to risk levels
            terrain_risk_map = {
                "plains": 0.2,
                "urban": 0.3,
                "forest": 0.5,
                "desert": 0.6,
                "mountainous": 0.8
            }
            
            terrain_risk = (terrain_risk_map.get(start_terrain, 0.5) + terrain_risk_map.get(end_terrain, 0.5)) / 2
            risks["terrain_risk"] += terrain_risk
            
            # Weather risk
            start_weather = self.map_data.get("regions", {}).get(start_region, {}).get("weather_conditions", "mild")
            end_weather = self.map_data.get("regions", {}).get(end_region, {}).get("weather_conditions", "mild")
            
            # Map weather to risk levels
            weather_risk_map = {
                "mild": 0.2,
                "hot": 0.4,
                "rain": 0.6,
                "fog": 0.7,
                "snow": 0.9
            }
            
            weather_risk = (weather_risk_map.get(start_weather, 0.5) + weather_risk_map.get(end_weather, 0.5)) / 2
            risks["weather_risk"] += weather_risk
            
            # Security risk (inverse of security level)
            start_security = self.locations.get(start, {}).get("security_level", "medium")
            end_security = self.locations.get(end, {}).get("security_level", "medium")
            
            # Map security to risk levels (inverse - higher security = lower risk)
            security_risk_map = {
                "high": 0.2,
                "medium": 0.5,
                "low": 0.8
            }
            
            security_risk = (security_risk_map.get(start_security, 0.5) + security_risk_map.get(end_security, 0.5)) / 2
            risks["security_risk"] += security_risk
        
        # Calculate averages
        segment_count = len(route) - 1
        if segment_count > 0:
            risks["overall_risk"] = sum(segment_risks) / segment_count
            risks["terrain_risk"] /= segment_count
            risks["weather_risk"] /= segment_count
            risks["security_risk"] /= segment_count
        
        # Round values
        for key in risks:
            risks[key] = round(risks[key], 2)
        
        return risks
    
    def generate_waypoints(self, route: List[str]) -> List[Dict]:
        """
        Generate detailed waypoints for a route.
        
        Args:
            route: List of locations in the route
            
        Returns:
            List of waypoint dictionaries
        """
        waypoints = []
        
        # Generate waypoints for each segment
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i+1]
            
            # Base waypoint at the start
            waypoints.append({
                "name": start,
                "type": "location",
                "coordinates": self.locations.get(start, {}).get("coordinates", (0, 0))
            })
            
            # Calculate intermediate waypoints
            distance = self.calculate_distance(start, end)
            
            # For longer segments, add some intermediate waypoints
            if distance > 100:
                # Add a checkpoint every ~50km
                num_checkpoints = min(int(distance / 50), 5)  # Max 5 checkpoints
                
                for j in range(num_checkpoints):
                    # Simple interpolation between start and end for demo
                    progress = (j + 1) / (num_checkpoints + 1)
                    
                    if start in self.locations and end in self.locations:
                        start_coords = self.locations[start].get("coordinates", (0, 0))
                        end_coords = self.locations[end].get("coordinates", (0, 0))
                        
                        x1, y1 = start_coords
                        x2, y2 = end_coords
                        
                        # Interpolate coordinates
                        x = x1 + (x2 - x1) * progress
                        y = y1 + (y2 - y1) * progress
                        
                        checkpoint = {
                            "name": f"Checkpoint {i}.{j+1}",
                            "type": "checkpoint",
                            "coordinates": (x, y)
                        }
                        
                        waypoints.append(checkpoint)
        
        # Add the final destination
        if route:
            end = route[-1]
            waypoints.append({
                "name": end,
                "type": "location",
                "coordinates": self.locations.get(end, {}).get("coordinates", (0, 0))
            })
        
        return waypoints
