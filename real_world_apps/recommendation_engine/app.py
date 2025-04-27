#!/usr/bin/env python3
"""
E-commerce Recommendation Engine
------------------------------

A real-world application that demonstrates how to use the AGI Toolkit
to build a product recommendation engine for e-commerce platforms.

Features:
- Personalized product recommendations
- Content-based filtering
- Collaborative filtering simulation
- Recommendation explanations
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Any, Optional

# Add the parent directory to path so we can import the AGI Toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the ASI helper module and AGI Toolkit
from real_world_apps.asi_helper import initialize_asi_components, process_with_asi
from agi_toolkit import AGIAPI

class RecommendationEngine:
    """A product recommendation engine using AGI Toolkit."""
    
    def __init__(self, product_catalog_path: str = None):
        """
        Initialize the recommendation engine.
        
        Args:
            product_catalog_path: Path to the product catalog JSON file
        """
        # Configure logging
        self.logger = logging.getLogger("RecommendationEngine")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing Recommendation Engine")
        
        # Initialize real ASI components
        initialize_asi_components()
        
        # Set environment variable to ensure interface uses real components
        os.environ['USE_REAL_ASI'] = 'true'
        
        # Initialize the AGI Toolkit API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        # Load product catalog
        self.products = []
        if product_catalog_path:
            self.load_product_catalog(product_catalog_path)
        else:
            self.load_sample_products()
        
        # Initialize user preference history
        self.user_preferences = {}
        
        self.logger.info(f"Recommendation Engine initialized with {len(self.products)} products")
    
    def load_product_catalog(self, catalog_path: str) -> bool:
        """
        Load product catalog from a JSON file.
        
        Args:
            catalog_path: Path to the product catalog JSON file
            
        Returns:
            Success status
        """
        try:
            with open(catalog_path, 'r') as f:
                self.products = json.load(f)
            
            self.logger.info(f"Loaded {len(self.products)} products from catalog")
            return True
        except Exception as e:
            self.logger.error(f"Error loading product catalog: {str(e)}")
            return False
    
    def load_sample_products(self):
        """Load sample products for demonstration."""
        self.products = [
            {
                "id": "p1001",
                "name": "Premium Wireless Headphones",
                "category": "Electronics",
                "subcategory": "Audio",
                "price": 149.99,
                "rating": 4.7,
                "features": ["Noise cancellation", "Bluetooth 5.0", "40h battery life"],
                "description": "Premium wireless headphones with active noise cancellation and long battery life"
            },
            {
                "id": "p1002",
                "name": "Smartphone Stand with Wireless Charger",
                "category": "Electronics",
                "subcategory": "Accessories",
                "price": 39.99,
                "rating": 4.2,
                "features": ["Wireless charging", "Adjustable angle", "Compatible with all phones"],
                "description": "Convenient smartphone stand with built-in wireless charging capabilities"
            },
            {
                "id": "p1003",
                "name": "Ultra HD Smart TV - 55\"",
                "category": "Electronics",
                "subcategory": "Televisions",
                "price": 699.99,
                "rating": 4.5,
                "features": ["4K resolution", "Smart TV features", "HDR support"],
                "description": "55-inch Ultra HD smart television with vibrant display and smart features"
            },
            {
                "id": "p1004",
                "name": "Professional Chef Knife Set",
                "category": "Kitchen",
                "subcategory": "Cutlery",
                "price": 89.99,
                "rating": 4.8,
                "features": ["Stainless steel", "Ergonomic handles", "5 piece set"],
                "description": "High-quality chef knife set for culinary enthusiasts"
            },
            {
                "id": "p1005",
                "name": "Smart Home Security Camera",
                "category": "Electronics",
                "subcategory": "Security",
                "price": 79.99,
                "rating": 4.3,
                "features": ["1080p HD", "Night vision", "Motion detection"],
                "description": "Security camera with motion detection and mobile app control"
            },
            {
                "id": "p1006",
                "name": "Stainless Steel Water Bottle",
                "category": "Sports",
                "subcategory": "Hydration",
                "price": 24.99,
                "rating": 4.6,
                "features": ["Vacuum insulated", "24h cold/12h hot", "Leak-proof"],
                "description": "Eco-friendly water bottle that keeps drinks cold or hot for hours"
            },
            {
                "id": "p1007",
                "name": "Wireless Ergonomic Keyboard",
                "category": "Electronics",
                "subcategory": "Computer Accessories",
                "price": 59.99,
                "rating": 4.1,
                "features": ["Ergonomic design", "Wireless", "Rechargeable"],
                "description": "Comfortable keyboard designed to reduce wrist strain during long typing sessions"
            },
            {
                "id": "p1008",
                "name": "Smart Fitness Tracker",
                "category": "Sports",
                "subcategory": "Wearables",
                "price": 129.99,
                "rating": 4.4,
                "features": ["Heart rate monitor", "Sleep tracking", "Waterproof"],
                "description": "Track your fitness goals with this waterproof smart fitness tracker"
            },
            {
                "id": "p1009",
                "name": "Portable Bluetooth Speaker",
                "category": "Electronics",
                "subcategory": "Audio",
                "price": 49.99,
                "rating": 4.3,
                "features": ["Waterproof", "10h battery life", "Compact design"],
                "description": "Take your music anywhere with this compact, waterproof Bluetooth speaker"
            },
            {
                "id": "p1010",
                "name": "Coffee Maker with Grinder",
                "category": "Kitchen",
                "subcategory": "Appliances",
                "price": 119.99,
                "rating": 4.7,
                "features": ["Built-in grinder", "Programmable", "10-cup capacity"],
                "description": "Fresh coffee every morning with this programmable coffee maker with built-in grinder"
            }
        ]
        
        self.logger.info(f"Loaded {len(self.products)} sample products")
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        Update user preferences.
        
        Args:
            user_id: Unique user identifier
            preferences: Dictionary containing user preferences
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        # Update preferences
        self.user_preferences[user_id].update(preferences)
        
        # Store in memory
        memory_key = f"user_preferences_{user_id}"
        self.api.store_data(memory_key, self.user_preferences[user_id])
        
        self.logger.info(f"Updated preferences for user {user_id}")
    
    def record_user_interaction(self, user_id: str, product_id: str, interaction_type: str, rating: float = None):
        """
        Record a user interaction with a product.
        
        Args:
            user_id: Unique user identifier
            product_id: Product identifier
            interaction_type: Type of interaction (view, purchase, add_to_cart, etc.)
            rating: Optional rating (1-5) if the interaction is a review
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "viewed_products": [],
                "purchased_products": [],
                "rated_products": {}
            }
        
        prefs = self.user_preferences[user_id]
        
        # Update interaction history
        if interaction_type == "view":
            if "viewed_products" not in prefs:
                prefs["viewed_products"] = []
            prefs["viewed_products"].append(product_id)
        
        elif interaction_type == "purchase":
            if "purchased_products" not in prefs:
                prefs["purchased_products"] = []
            prefs["purchased_products"].append(product_id)
        
        elif interaction_type == "rate" and rating is not None:
            if "rated_products" not in prefs:
                prefs["rated_products"] = {}
            prefs["rated_products"][product_id] = rating
        
        # Extract product categories
        product = self.get_product_by_id(product_id)
        if product:
            category = product.get("category")
            if category:
                if "category_preferences" not in prefs:
                    prefs["category_preferences"] = {}
                
                if category not in prefs["category_preferences"]:
                    prefs["category_preferences"][category] = 0
                
                # Increment category preference based on interaction type
                if interaction_type == "view":
                    prefs["category_preferences"][category] += 1
                elif interaction_type == "purchase":
                    prefs["category_preferences"][category] += 5  # Stronger signal
                elif interaction_type == "rate" and rating is not None:
                    prefs["category_preferences"][category] += (rating - 3) * 2  # Based on rating
        
        # Store updated preferences
        memory_key = f"user_preferences_{user_id}"
        self.api.store_data(memory_key, prefs)
        
        self.logger.info(f"Recorded {interaction_type} interaction for user {user_id} with product {product_id}")
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a product by its ID."""
        for product in self.products:
            if product["id"] == product_id:
                return product
        return None
    
    def recommend_products(self, user_id: str, context: Dict = None, limit: int = 5) -> List[Dict]:
        """
        Generate product recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            context: Additional context like current page, cart items, etc.
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommended products with explanation
        """
        self.logger.info(f"Generating recommendations for user {user_id}")
        
        # Get user preferences if available
        user_prefs = self.user_preferences.get(user_id, {})
        
        # Generate recommendations using real ASI if available
        if self.api.has_asi:
            try:
                # Prepare data for recommendation
                recommendation_data = {
                    "task": "recommend_products",
                    "user_id": user_id,
                    "user_preferences": user_prefs,
                    "product_catalog": self.products[:100],  # Limit catalog size
                    "context": context,
                    "limit": limit
                }
                
                # Process with real ASI
                result = process_with_asi(self.api, recommendation_data)
                
                # Handle the result
                if isinstance(result, dict) and result.get("success", False) and "result" in result:
                    recommendations_data = result["result"]
                    
                    # Handle different output formats
                    if isinstance(recommendations_data, list):
                        # If already a list of recommendations
                        return recommendations_data[:limit]
                    elif isinstance(recommendations_data, dict):
                        # If recommendations are in a specific field
                        if "recommendations" in recommendations_data:
                            return recommendations_data["recommendations"][:limit]
                        elif "products" in recommendations_data:
                            return recommendations_data["products"][:limit]
                    
                    # Log success
                    self.logger.info(f"Generated {len(recommendations_data)} recommendations using ASI")
            except Exception as e:
                self.logger.error(f"Error generating recommendations with ASI: {str(e)}")
        
        # Fallback to rule-based recommendations
        return self._rule_based_recommendations(user_id, context, limit)
    
    def _rule_based_recommendations(self, user_id: str, context: Dict = None, limit: int = 5) -> List[Dict]:
        """Generate rule-based recommendations."""
        self.logger.info(f"Generating rule-based recommendations for user {user_id}")
        
        user_prefs = self.user_preferences.get(user_id, {})
        
        # If no preferences, return top-rated products
        if not user_prefs:
            top_products = sorted(self.products, key=lambda p: p.get("rating", 0), reverse=True)
            return [
                {
                    "product": product,
                    "score": product.get("rating", 0) / 5.0,
                    "strategy": "rule-based",
                    "explanation": "Top-rated product"
                }
                for product in top_products[:limit]
            ]
        
        # Get category preferences
        category_prefs = user_prefs.get("category_preferences", {})
        
        # Score products based on category preference
        scored_products = []
        for product in self.products:
            score = 0.0
            category = product.get("category")
            
            # Category match
            if category in category_prefs:
                score += category_prefs[category] / 10.0  # Normalize
            
            # Rating boost
            score += product.get("rating", 0) / 10.0
            
            # Check if already purchased
            purchased = user_prefs.get("purchased_products", [])
            if product["id"] in purchased:
                continue  # Skip already purchased products
            
            # Add explanation
            explanation = f"Recommended based on your interest in {category} products"
            
            scored_products.append({
                "product": product,
                "score": min(score, 1.0),  # Cap at 1.0
                "strategy": "rule-based",
                "explanation": explanation
            })
        
        # Sort by score and return top results
        scored_products.sort(key=lambda p: p["score"], reverse=True)
        return scored_products[:limit]
    
    def analyze_product_affinity(self, product_id: str, limit: int = 3) -> List[Dict]:
        """
        Find products with high affinity to the given product.
        
        Args:
            product_id: Product ID to find related products for
            limit: Maximum number of related products to return
            
        Returns:
            List of related products with affinity score
        """
        self.logger.info(f"Finding related products for {product_id}")
        
        # Find the source product
        source_product = None
        for product in self.products:
            if product.get("id") == product_id:
                source_product = product
                break
        
        if not source_product:
            self.logger.error(f"Product {product_id} not found in catalog")
            return []
        
        # Use real ASI for advanced affinity analysis if available
        if self.api.has_asi:
            try:
                # Prepare data for affinity analysis
                affinity_data = {
                    "task": "product_affinity",
                    "source_product": source_product,
                    "product_catalog": self.products[:100],  # Limit catalog size
                    "limit": limit
                }
                
                # Process with real ASI
                result = process_with_asi(self.api, affinity_data)
                
                # Handle the result
                if isinstance(result, dict) and result.get("success", False) and "result" in result:
                    affinity_result = result["result"]
                    
                    # Handle different output formats
                    if isinstance(affinity_result, list):
                        # If already a list of products
                        return affinity_result[:limit]
                    elif isinstance(affinity_result, dict):
                        # If products are in a specific field
                        if "related_products" in affinity_result:
                            return affinity_result["related_products"][:limit]
                        elif "products" in affinity_result:
                            return affinity_result["products"][:limit]
                        elif "affinity" in affinity_result:
                            return affinity_result["affinity"][:limit]
                    
                    # Log success
                    self.logger.info(f"Generated {len(affinity_result)} product affinities using ASI")
            except Exception as e:
                self.logger.error(f"Error analyzing product affinity with ASI: {str(e)}")
        
        # Fallback to category-based affinity
        return self._category_based_affinity(source_product, limit)
    
    def _category_based_affinity(self, source_product: Dict, limit: int = 3) -> List[Dict]:
        """Find products with similar categories."""
        self.logger.info(f"Finding products with similar categories to {source_product['name']}")
        
        # Get products with the same category
        similar_products = [
            product for product in self.products 
            if product["category"] == source_product["category"] and product["id"] != source_product["id"]
        ]
        
        # Sort by rating and return top results
        similar_products.sort(key=lambda p: p.get("rating", 0), reverse=True)
        return similar_products[:limit]
    
    def explain_recommendation(self, product_id: str, user_id: str) -> str:
        """
        Generate a detailed explanation for a recommendation.
        
        Args:
            product_id: Product identifier
            user_id: User identifier
            
        Returns:
            Explanation text
        """
        self.logger.info(f"Generating detailed explanation for product {product_id} for user {user_id}")
        
        product = self.get_product_by_id(product_id)
        if not product:
            return "Product not found"
        
        user_prefs = self.user_preferences.get(user_id, {})
        
        # Use real ASI for generating explanation if available
        if self.api.has_asi:
            category_prefs = user_prefs.get("category_preferences", {})
            category_interest = category_prefs.get(product["category"], 0)
            
            # Create a prompt for personalized explanation
            prompt = f"""You are a recommendation system explaining why the following product is being recommended:

Product: {product["name"]}
Category: {product["category"]}
Features: {", ".join(product.get("features", []))}
Description: {product["description"]}

The user has shown interest in {product["category"]} products with a score of {category_interest}.

Generate a brief, personalized explanation (2-3 sentences) for why this product is being recommended.
"""
            
            return process_with_asi(self.api, {"task": "generate_explanation", "prompt": prompt})
        
        # Fallback explanation
        category = product.get("category", "")
        rating = product.get("rating", 0)
        
        explanation = f"We recommend {product['name']} because "
        
        if category in user_prefs.get("category_preferences", {}):
            explanation += f"you've shown interest in {category} products. "
        else:
            explanation += f"it's a highly-rated product in the {category} category. "
        
        explanation += f"With a rating of {rating}/5, many customers have found this product satisfying. "
        
        if "features" in product and product["features"]:
            explanation += f"Key features include {', '.join(product['features'][:2])}."
        
        return explanation


def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Display the recommendations in a user-friendly format."""
    print("\n" + "="*80)
    print("PERSONALIZED PRODUCT RECOMMENDATIONS".center(80))
    print("="*80 + "\n")
    
    if not recommendations:
        print("No recommendations found. Try interacting with some products first.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        product = rec["product"]
        score = rec["score"]
        strategy = rec["strategy"]
        explanation = rec["explanation"]
        
        print(f"#{i}: {product['name']} (${product['price']:.2f}) - {product['rating']}/5 stars")
        print(f"  Category: {product['category']}")
        
        if "features" in product:
            print(f"  Features: {', '.join(product['features'])}")
        
        print(f"  Match Score: {score:.2f}")
        print(f"  Strategy: {strategy.capitalize()}")
        print(f"  Why recommended: {explanation}")
        print("-" * 80)
    
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="E-commerce Recommendation Engine")
    parser.add_argument("--catalog", type=str, help="Path to product catalog JSON file")
    parser.add_argument("--user", type=str, default="user123", help="User ID to generate recommendations for")
    parser.add_argument("--product", type=str, help="Product ID to find related products for")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of recommendations to return")
    parser.add_argument("--context", type=str, help="Context for recommendation as JSON string")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Ensure USE_REAL_ASI is set to true
    os.environ['USE_REAL_ASI'] = 'true'
    
    # Initialize the recommendation engine
    engine = RecommendationEngine(args.catalog)
    
    # For demo purposes, simulate some user interactions
    if args.user == "demo_user":
        # Simulate user viewing and purchasing products
        engine.record_user_interaction(args.user, "p1001", "view")
        engine.record_user_interaction(args.user, "p1003", "view")
        engine.record_user_interaction(args.user, "p1005", "view")
        engine.record_user_interaction(args.user, "p1001", "purchase")
        engine.record_user_interaction(args.user, "p1001", "rate", 5.0)
        engine.record_user_interaction(args.user, "p1009", "view")
        
        # Set some explicit preferences
        engine.update_user_preferences(args.user, {
            "preferred_categories": ["Electronics", "Sports"],
            "price_range": {"min": 20, "max": 200}
        })
    
    # Get recommendations
    recommendations = engine.get_recommendations(args.user, args.count, args.strategy)
    
    # Display recommendations
    display_recommendations(recommendations)
    
    # Get detailed explanation for the top recommendation if available
    if recommendations:
        top_product_id = recommendations[0]["product"]["id"]
        detailed_explanation = engine.explain_recommendation(top_product_id, args.user)
        print("\nDetailed explanation for top recommendation:")
        print(detailed_explanation)


if __name__ == "__main__":
    main()
