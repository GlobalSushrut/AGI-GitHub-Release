#!/usr/bin/env python3
"""
Resource Manager for Military Logistics
--------------------------------------

Manages inventory of military resources, supplies, and equipment.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger("ResourceManager")


class ResourceManager:
    """Manages military resources and inventory."""
    
    def __init__(self, api):
        """
        Initialize the resource manager.
        
        Args:
            api: The AGI Toolkit API instance
        """
        self.api = api
        self.logger = logger
        
        # Load existing resources
        self.resources = {}
        self._load_resources()
    
    def _load_resources(self):
        """Load resources from API memory."""
        try:
            memory_key = "military_logistics_resources"
            resources_data = self.api.retrieve_data(memory_key)
            
            if resources_data and isinstance(resources_data, dict):
                self.resources = resources_data
                self.logger.info(f"Loaded {len(self.resources)} resources from memory")
            else:
                self.resources = {}
                self.logger.info("No valid resources found in memory, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading resources: {str(e)}")
    
    def _save_resources(self):
        """Save resources to API memory."""
        try:
            memory_key = "military_logistics_resources"
            self.api.store_data(memory_key, self.resources)
            self.logger.info(f"Saved {len(self.resources)} resources to memory")
        except Exception as e:
            self.logger.error(f"Error saving resources: {str(e)}")
    
    def add_resource(self, resource_id: str, name: str, category: str, quantity: int, location: str) -> Dict:
        """
        Add a new resource to inventory.
        
        Args:
            resource_id: Unique resource identifier
            name: Resource name
            category: Resource category (e.g., consumable, equipment, fuel)
            quantity: Quantity available
            location: Current location
            
        Returns:
            Resource data dictionary
        """
        if resource_id in self.resources:
            return self.update_resource(resource_id, quantity, location)
        
        resource = {
            "resource_id": resource_id,
            "name": name,
            "category": category,
            "quantity": quantity,
            "location": location,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "status": "available"
        }
        
        self.resources[resource_id] = resource
        self._save_resources()
        
        return resource
    
    def update_resource(self, resource_id: str, quantity: int, location: str = None) -> Dict:
        """
        Update resource quantity or location.
        
        Args:
            resource_id: Resource identifier
            quantity: New quantity
            location: New location (optional)
            
        Returns:
            Updated resource data
        """
        if resource_id not in self.resources:
            return {"error": f"Resource {resource_id} not found"}
        
        resource = self.resources[resource_id]
        
        # Update quantity
        prev_quantity = resource["quantity"]
        resource["quantity"] = quantity
        
        # Log the change
        if "quantity_history" not in resource:
            resource["quantity_history"] = []
        
        resource["quantity_history"].append({
            "previous": prev_quantity,
            "new": quantity,
            "change": quantity - prev_quantity,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update location if provided
        if location:
            prev_location = resource.get("location")
            resource["location"] = location
            
            # Log the location change
            if "location_history" not in resource:
                resource["location_history"] = []
            
            resource["location_history"].append({
                "previous": prev_location,
                "new": location,
                "timestamp": datetime.now().isoformat()
            })
        
        # Update status based on quantity
        if quantity <= 0:
            resource["status"] = "depleted"
        elif quantity < resource.get("min_quantity", 10):
            resource["status"] = "low"
        else:
            resource["status"] = "available"
        
        resource["last_updated"] = datetime.now().isoformat()
        
        self._save_resources()
        
        return resource
    
    def get_resource(self, resource_id: str) -> Dict:
        """
        Get resource details.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            Resource data or error dictionary
        """
        if resource_id not in self.resources:
            return {"error": f"Resource {resource_id} not found"}
        
        return self.resources[resource_id]
    
    def list_resources(self, category: str = None, location: str = None) -> List[Dict]:
        """
        List resources with optional filtering.
        
        Args:
            category: Filter by category (optional)
            location: Filter by location (optional)
            
        Returns:
            List of resource dictionaries
        """
        resources = list(self.resources.values())
        
        # Apply filters
        if category:
            resources = [r for r in resources if r.get("category") == category]
        
        if location:
            resources = [r for r in resources if r.get("location") == location]
        
        # Sort by ID
        resources.sort(key=lambda r: r.get("resource_id", ""))
        
        return resources
    
    def transfer_resources(self, resource_id: str, quantity: int, 
                         source_location: str, destination_location: str) -> Dict:
        """
        Transfer resources between locations.
        
        Args:
            resource_id: Resource identifier
            quantity: Quantity to transfer
            source_location: Source location
            destination_location: Destination location
            
        Returns:
            Transfer result dictionary
        """
        if resource_id not in self.resources:
            return {"error": f"Resource {resource_id} not found"}
        
        resource = self.resources[resource_id]
        
        # Check location
        if resource.get("location") != source_location:
            return {"error": f"Resource {resource_id} is not at {source_location}"}
        
        # Check quantity
        if resource.get("quantity", 0) < quantity:
            return {"error": f"Insufficient quantity of {resource_id} available"}
        
        # Perform the transfer
        new_quantity = resource.get("quantity", 0) - quantity
        self.update_resource(resource_id, new_quantity)
        
        # Create a new resource entry for the destination if it doesn't exist
        dest_resource_id = f"{resource_id}-{destination_location}"
        
        if dest_resource_id in self.resources:
            # Update existing destination resource
            dest_resource = self.resources[dest_resource_id]
            new_dest_quantity = dest_resource.get("quantity", 0) + quantity
            self.update_resource(dest_resource_id, new_dest_quantity)
        else:
            # Create a new resource at the destination
            self.add_resource(
                dest_resource_id,
                resource.get("name"),
                resource.get("category"),
                quantity,
                destination_location
            )
        
        # Log the transfer
        transfer_record = {
            "resource_id": resource_id,
            "source": source_location,
            "destination": destination_location,
            "quantity": quantity,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store transfer in memory
        try:
            memory_key = "military_logistics_transfers"
            transfers = self.api.retrieve_data(memory_key) or []
            
            if not isinstance(transfers, list):
                transfers = []
            
            transfers.append(transfer_record)
            self.api.store_data(memory_key, transfers)
        except Exception as e:
            self.logger.error(f"Error recording transfer: {str(e)}")
        
        return {
            "success": True,
            "resource_id": resource_id,
            "source": source_location,
            "destination": destination_location,
            "quantity": quantity,
            "source_remaining": new_quantity
        }
    
    def get_inventory_report(self, location: str = None) -> Dict:
        """
        Generate an inventory report.
        
        Args:
            location: Filter by location (optional)
            
        Returns:
            Inventory report dictionary
        """
        resources = self.list_resources(location=location)
        
        # Group by category
        categories = {}
        for resource in resources:
            category = resource.get("category", "uncategorized")
            
            if category not in categories:
                categories[category] = []
            
            categories[category].append(resource)
        
        # Calculate summary
        total_items = len(resources)
        depleted_items = sum(1 for r in resources if r.get("status") == "depleted")
        low_items = sum(1 for r in resources if r.get("status") == "low")
        
        report = {
            "location": location or "All locations",
            "total_items": total_items,
            "depleted_items": depleted_items,
            "low_items": low_items,
            "categories": {},
            "generated_at": datetime.now().isoformat()
        }
        
        # Add category summaries
        for category, items in categories.items():
            report["categories"][category] = {
                "count": len(items),
                "total_quantity": sum(item.get("quantity", 0) for item in items),
                "items": items
            }
        
        return report
