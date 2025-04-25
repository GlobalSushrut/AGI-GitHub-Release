#!/usr/bin/env python3
"""
Customer Behavior Analyzer
------------------------
Analyzes customer behavior patterns and generates personalized marketing strategies.
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

class CustomerBehaviorAnalyzer:
    """Customer behavior analyzer using ASI infrastructure."""
    
    def __init__(self):
        """Initialize the customer behavior analyzer."""
        print("\nCUSTOMER BEHAVIOR ANALYZER - Powered by Encrypted ASI Engine")
        
        # Initialize ASI engine
        print("Initializing ASI engine...")
        success = initialize_asi()
        
        if not success:
            print("Failed to initialize ASI engine. Exiting.")
            sys.exit(1)
        
        # Create ASI instance with specific configuration for customer analysis
        self.asi = create_asi_instance(name="CustomerASI", config={
            "domain": "consumer_behavior",
            "sensitivity": 0.8,
            "pattern_complexity": 0.7,
            "behavioral_analysis": True
        })
        
        print("ASI engine initialized successfully")
        
        # Initialize data structures
        self.customers = {}
        self.products = {}
        self.transactions = {}
        self.segments = {}
        self.marketing_strategies = {}
    
    def generate_customer_data(self, num_customers=100, num_products=20, num_days=30):
        """Generate simulated customer and product data."""
        print(f"\nGenerating simulated data for {num_customers} customers and {num_products} products...")
        
        # Define product categories
        categories = ["Electronics", "Clothing", "Food", "Home", "Beauty", "Sports"]
        
        # Generate products
        for i in range(1, num_products + 1):
            product_id = f"P{i}"
            category = random.choice(categories)
            
            # Price range based on category
            if category == "Electronics":
                price = random.uniform(100, 1000)
            elif category == "Clothing":
                price = random.uniform(20, 200)
            elif category == "Food":
                price = random.uniform(5, 50)
            elif category == "Home":
                price = random.uniform(30, 300)
            elif category == "Beauty":
                price = random.uniform(10, 100)
            else:  # Sports
                price = random.uniform(15, 150)
            
            self.products[product_id] = {
                "name": f"{category} Item {i}",
                "category": category,
                "price": price,
                "popularity": random.uniform(0.1, 1.0)
            }
        
        # Generate customer demographic profiles
        for i in range(1, num_customers + 1):
            customer_id = f"C{i}"
            
            # Generate age with realistic distribution
            age_group = random.choices(
                ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
                weights=[0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
            )[0]
            
            # Map age group to numeric value for calculations
            if age_group == "18-24":
                age = random.randint(18, 24)
            elif age_group == "25-34":
                age = random.randint(25, 34)
            elif age_group == "35-44":
                age = random.randint(35, 44)
            elif age_group == "45-54":
                age = random.randint(45, 54)
            elif age_group == "55-64":
                age = random.randint(55, 64)
            else:
                age = random.randint(65, 85)
            
            gender = random.choice(["Male", "Female", "Other"])
            location = random.choice(["Urban", "Suburban", "Rural"])
            
            # Generate income based on age and location
            if location == "Urban":
                base_income = 50000
            elif location == "Suburban":
                base_income = 65000
            else:
                base_income = 45000
            
            # Adjust for age
            if age < 25:
                income_factor = 0.6
            elif age < 35:
                income_factor = 0.9
            elif age < 55:
                income_factor = 1.1
            else:
                income_factor = 0.8
            
            income = int(base_income * income_factor * random.uniform(0.7, 1.3))
            
            # Generate shopping preferences
            preferred_categories = random.sample(categories, random.randint(1, 3))
            price_sensitivity = random.uniform(0.3, 1.0)
            brand_loyalty = random.uniform(0.2, 0.9)
            
            self.customers[customer_id] = {
                "age": age,
                "age_group": age_group,
                "gender": gender,
                "location": location,
                "income": income,
                "preferred_categories": preferred_categories,
                "price_sensitivity": price_sensitivity,
                "brand_loyalty": brand_loyalty,
                "first_purchase": None,
                "last_purchase": None,
                "total_spent": 0,
                "purchase_count": 0
            }
        
        # Generate transaction history
        start_date = datetime.now() - timedelta(days=num_days)
        
        # Initialize transactions dictionary
        self.transactions = {customer_id: [] for customer_id in self.customers}
        
        # Generate multiple transactions per customer
        for customer_id, customer in self.customers.items():
            # Number of transactions depends on customer profile
            num_transactions = int(random.betavariate(2, 5) * 15) + 1  # Between 1 and 15
            
            for _ in range(num_transactions):
                # Generate transaction date
                days_ago = random.randint(0, num_days)
                transaction_date = start_date + timedelta(days=days_ago)
                
                # Select products based on customer preferences
                preferred_categories = customer["preferred_categories"]
                
                # Filter products by preferred categories
                potential_products = [
                    p_id for p_id, p in self.products.items() 
                    if p["category"] in preferred_categories
                ]
                
                # If no potential products, use all products
                if not potential_products:
                    potential_products = list(self.products.keys())
                
                # Number of products in transaction
                num_products = max(1, int(random.expovariate(0.5)))
                
                # Ensure we don't try to select more products than available
                num_products = min(num_products, len(potential_products))
                
                # Select products
                selected_products = random.sample(potential_products, num_products)
                
                # Calculate transaction details
                items = []
                transaction_total = 0
                
                for p_id in selected_products:
                    product = self.products[p_id]
                    quantity = max(1, int(random.expovariate(0.7)))
                    
                    # Apply price sensitivity
                    effective_price = product["price"] * (1 - 0.2 * customer["price_sensitivity"])
                    
                    item_total = effective_price * quantity
                    transaction_total += item_total
                    
                    items.append({
                        "product_id": p_id,
                        "product_name": product["name"],
                        "category": product["category"],
                        "price": effective_price,
                        "quantity": quantity,
                        "item_total": item_total
                    })
                
                # Create transaction
                transaction = {
                    "transaction_id": f"T{customer_id}_{len(self.transactions[customer_id]) + 1}",
                    "customer_id": customer_id,
                    "date": transaction_date,
                    "items": items,
                    "total": transaction_total
                }
                
                # Add to customer's transactions
                self.transactions[customer_id].append(transaction)
                
                # Update customer purchase history
                if customer["first_purchase"] is None or transaction_date < customer["first_purchase"]:
                    customer["first_purchase"] = transaction_date
                
                if customer["last_purchase"] is None or transaction_date > customer["last_purchase"]:
                    customer["last_purchase"] = transaction_date
                
                customer["total_spent"] += transaction_total
                customer["purchase_count"] += 1
        
        print(f"Generated {sum(len(t) for t in self.transactions.values())} transactions")
        print(f"Customer data generation complete")
    
    def analyze_customer_segments(self):
        """Analyze customer segments using ASI pattern discovery."""
        print("\nAnalyzing customer segments...")
        
        # Prepare data for ASI analysis
        analysis_data = []
        
        for customer_id, customer in self.customers.items():
            # Skip customers with no transactions
            if customer["purchase_count"] == 0:
                continue
            
            # Create feature vector for analysis
            customer_data = {
                "customer_id": customer_id,
                "age": customer["age"],
                "gender_male": 1 if customer["gender"] == "Male" else 0,
                "gender_female": 1 if customer["gender"] == "Female" else 0,
                "gender_other": 1 if customer["gender"] == "Other" else 0,
                "location_urban": 1 if customer["location"] == "Urban" else 0,
                "location_suburban": 1 if customer["location"] == "Suburban" else 0,
                "location_rural": 1 if customer["location"] == "Rural" else 0,
                "income": customer["income"],
                "price_sensitivity": customer["price_sensitivity"],
                "brand_loyalty": customer["brand_loyalty"],
                "total_spent": customer["total_spent"],
                "purchase_count": customer["purchase_count"],
                "avg_transaction": customer["total_spent"] / customer["purchase_count"] if customer["purchase_count"] > 0 else 0
            }
            
            # Add category preferences
            for category in ["Electronics", "Clothing", "Food", "Home", "Beauty", "Sports"]:
                customer_data[f"prefers_{category.lower()}"] = 1 if category in customer["preferred_categories"] else 0
            
            # Add recency and frequency metrics
            if customer["last_purchase"]:
                days_since_last = (datetime.now() - customer["last_purchase"]).days
                customer_data["recency"] = days_since_last
                
                if customer["first_purchase"] and customer["last_purchase"] > customer["first_purchase"]:
                    days_active = (customer["last_purchase"] - customer["first_purchase"]).days
                    if days_active > 0:
                        customer_data["frequency"] = customer["purchase_count"] / days_active
                    else:
                        customer_data["frequency"] = customer["purchase_count"]
                else:
                    customer_data["frequency"] = 1
            else:
                customer_data["recency"] = 30  # Default value
                customer_data["frequency"] = 0
            
            analysis_data.append(customer_data)
        
        # Convert list to dictionary format expected by ASI API
        properties_dict = {}
        
        # Extract numeric properties from each customer's analysis data
        for i, customer_data in enumerate(analysis_data):
            customer_id = customer_data.get('customer_id', f'C{i}')
            
            # Add numeric values to the properties dictionary
            for key, value in customer_data.items():
                if isinstance(value, (int, float)):
                    properties_dict[f"{customer_id}_{key}"] = value
        
        # Ensure we have at least some properties
        if not properties_dict:
            properties_dict = {
                "purchase_frequency": 0.65,
                "average_spend": 0.75,
                "loyalty_score": 0.8,
                "recency": 0.5
            }
            
        # Use ASI pattern discovery
        patterns = self.asi.discover_patterns(
            domain="customer_behavior",
            properties=properties_dict
        )
        
        print(f"Discovered {len(patterns['patterns'])} customer behavior patterns")
        print(f"Pattern analysis confidence: {patterns['confidence']:.4f}")
        
        # Display top patterns
        for i, pattern in enumerate(patterns['patterns'][:2]):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"Description: {pattern['description']}")
        
        return patterns
    
    def segment_customers(self, patterns):
        """Segment customers based on discovered patterns."""
        print("\nSegmenting customers based on behavior patterns...")
        
        # Generate insights about customer segments
        customer_concepts = [
            "customer_segments", "purchasing_behavior", "price_sensitivity",
            "brand_loyalty", "demographic_influences"
        ]
        
        insights = self.asi.generate_insight(concepts=customer_concepts)
        
        print(f"Customer segmentation insight (Confidence: {insights['confidence']:.4f}):")
        print(f"\"{insights['text']}\"")
        
        # Create customer segments based on patterns
        # We'll create 4-6 segments using pattern elements
        num_segments = random.randint(4, 6)
        
        # Create segment definitions based on insights and patterns
        segment_definitions = self._generate_segment_definitions(patterns, num_segments)
        
        # Assign customers to segments
        for customer_id, customer in self.customers.items():
            best_match = None
            best_score = -1
            
            for segment_id, segment in segment_definitions.items():
                # Calculate match score
                score = self._calculate_segment_match(customer, segment)
                
                if score > best_score:
                    best_score = score
                    best_match = segment_id
            
            # Assign customer to best matching segment
            if best_match and best_match in self.segments:
                self.segments[best_match]["customers"].append(customer_id)
            else:
                # If no segment exists yet, create it
                if best_match not in self.segments:
                    self.segments[best_match] = {
                        "id": best_match,
                        "name": segment_definitions[best_match]["name"],
                        "description": segment_definitions[best_match]["description"],
                        "characteristics": segment_definitions[best_match]["characteristics"],
                        "customers": [customer_id]
                    }
        
        # Print segment summary
        print(f"\nCreated {len(self.segments)} customer segments:")
        
        for segment_id, segment in self.segments.items():
            print(f"\n{segment['name']} (ID: {segment_id})")
            print(f"Description: {segment['description']}")
            print(f"Size: {len(segment['customers'])} customers")
            print("Key characteristics:")
            for char in segment["characteristics"][:3]:
                print(f"  - {char}")
        
        return self.segments
    
    def _generate_segment_definitions(self, patterns, num_segments):
        """Generate segment definitions based on patterns."""
        segment_definitions = {}
        
        # Common segment types
        segment_types = [
            {"name": "High-Value Loyalists", 
             "description": "Customers who spend a lot and purchase frequently",
             "characteristics": ["High average transaction value", "Frequent purchases", "Low price sensitivity"]},
            {"name": "Occasional Shoppers", 
             "description": "Customers who purchase infrequently but may spend well",
             "characteristics": ["Infrequent purchases", "Medium to high transaction value", "Brand conscious"]},
            {"name": "Deal Seekers", 
             "description": "Price-conscious customers who look for bargains",
             "characteristics": ["High price sensitivity", "Variable purchase frequency", "Lower average transaction"]},
            {"name": "New Explorers", 
             "description": "Recent new customers still exploring products",
             "characteristics": ["Recent first purchase", "Limited purchase history", "Exploring different categories"]},
            {"name": "Category Enthusiasts", 
             "description": "Customers who focus primarily on specific product categories",
             "characteristics": ["Strong category focus", "Medium loyalty", "Category-specific purchasing"]},
            {"name": "Premium Buyers", 
             "description": "Customers focused on high-end items regardless of frequency",
             "characteristics": ["Very high transaction value", "Low price sensitivity", "Quality-focused"]}
        ]
        
        # Select a subset of segment types
        selected_segments = random.sample(segment_types, min(num_segments, len(segment_types)))
        
        # Create segment definitions
        for i, segment in enumerate(selected_segments):
            segment_id = f"S{i+1}"
            
            # Add some pattern-based characteristics if available
            pattern_characteristics = []
            if patterns and "patterns" in patterns and i < len(patterns["patterns"]):
                pattern = patterns["patterns"][i]
                if "description" in pattern:
                    pattern_characteristics.append(pattern["description"])
            
            # Combine standard and pattern-based characteristics
            characteristics = segment["characteristics"] + pattern_characteristics
            
            segment_definitions[segment_id] = {
                "name": segment["name"],
                "description": segment["description"],
                "characteristics": characteristics
            }
        
        return segment_definitions
    
    def _calculate_segment_match(self, customer, segment):
        """Calculate how well a customer matches a segment definition."""
        # This is a simplified scoring function
        score = 0
        
        # Extract key characteristics from segment
        segment_chars = [c.lower() for c in segment["characteristics"]]
        
        # Check for high value indicators
        if any("high" in c and "value" in c for c in segment_chars):
            # Score based on total spent
            if customer["total_spent"] > 500:
                score += 3
            elif customer["total_spent"] > 200:
                score += 1
        
        # Check for frequency indicators
        if any("frequent" in c for c in segment_chars):
            if customer["purchase_count"] > 5:
                score += 3
            elif customer["purchase_count"] > 2:
                score += 1
        
        # Check for infrequent indicators
        if any("infrequent" in c for c in segment_chars):
            if customer["purchase_count"] <= 2:
                score += 3
            elif customer["purchase_count"] <= 4:
                score += 1
        
        # Check for price sensitivity
        if any("price sensitivity" in c for c in segment_chars):
            if customer["price_sensitivity"] > 0.7:
                score += 2
            elif customer["price_sensitivity"] > 0.5:
                score += 1
        
        # Check for brand loyalty
        if any("loyalty" in c for c in segment_chars):
            if customer["brand_loyalty"] > 0.7:
                score += 2
            elif customer["brand_loyalty"] > 0.5:
                score += 1
        
        # Check for category focus
        if any("category" in c for c in segment_chars):
            if len(customer["preferred_categories"]) <= 2:
                score += 2
            elif len(customer["preferred_categories"]) <= 3:
                score += 1
        
        # Add a small random factor to prevent ties
        score += random.uniform(0, 0.5)
        
        return score
    
    def generate_marketing_strategies(self):
        """Generate marketing strategies for customer segments."""
        print("\nGenerating personalized marketing strategies...")
        
        if not self.segments:
            print("No customer segments available. Please run segment_customers first.")
            return None
        
        # Generate marketing strategies for each segment
        for segment_id, segment in self.segments.items():
            # Create scenario for ASI prediction
            marketing_scenario = {
                "name": f"Marketing strategy for {segment['name']}",
                "complexity": 0.7,
                "uncertainty": 0.4,
                "domain": "marketing_optimization",
                "variables": {
                    "segment_id": segment_id,
                    "segment_name": segment["name"],
                    "segment_size": len(segment["customers"]),
                    "characteristics": segment["characteristics"]
                }
            }
            
            # Use ASI to predict effective marketing strategies
            prediction = self.asi.predict_timeline(marketing_scenario)
            
            # Create marketing strategy
            self.marketing_strategies[segment_id] = {
                "segment_id": segment_id,
                "segment_name": segment["name"],
                "segment_size": len(segment["customers"]),
                "primary_strategies": self._extract_primary_strategies(prediction),
                "content_recommendations": self._extract_content_recommendations(prediction),
                "channel_recommendations": self._extract_channel_recommendations(prediction),
                "timing_recommendations": self._extract_timing_recommendations(prediction),
                "expected_engagement": prediction.get("metrics", {}).get("engagement", 0.3),
                "expected_conversion": prediction.get("metrics", {}).get("conversion", 0.15),
                "confidence": prediction["confidence"]
            }
        
        # Print marketing strategy summary
        print(f"\nGenerated marketing strategies for {len(self.marketing_strategies)} segments")
        
        for segment_id, strategy in list(self.marketing_strategies.items())[:2]:
            print(f"\nSegment: {strategy['segment_name']} (ID: {segment_id})")
            print(f"Size: {strategy['segment_size']} customers")
            print(f"Expected engagement: {strategy['expected_engagement']*100:.1f}%")
            print(f"Expected conversion: {strategy['expected_conversion']*100:.1f}%")
            
            print("Primary strategies:")
            for i, strat in enumerate(strategy['primary_strategies'][:3]):
                print(f"  {i+1}. {strat['event']}")
        
        return self.marketing_strategies
    
    def _extract_primary_strategies(self, prediction):
        """Extract primary marketing strategies from ASI prediction."""
        strategies = []
        
        for event in prediction["base_timeline"]:
            if "strateg" in event.get("event", "").lower() or "campaign" in event.get("event", "").lower():
                strategies.append(event)
        
        return strategies
    
    def _extract_content_recommendations(self, prediction):
        """Extract content recommendations from ASI prediction."""
        recommendations = []
        
        for event in prediction["base_timeline"]:
            if "content" in event.get("event", "").lower() or "message" in event.get("event", "").lower():
                recommendations.append(event)
        
        return recommendations
    
    def _extract_channel_recommendations(self, prediction):
        """Extract channel recommendations from ASI prediction."""
        recommendations = []
        
        for event in prediction["base_timeline"]:
            if "channel" in event.get("event", "").lower() or "platform" in event.get("event", "").lower():
                recommendations.append(event)
        
        return recommendations
    
    def _extract_timing_recommendations(self, prediction):
        """Extract timing recommendations from ASI prediction."""
        recommendations = []
        
        for event in prediction["base_timeline"]:
            if "timing" in event.get("event", "").lower() or "schedule" in event.get("event", "").lower():
                recommendations.append(event)
        
        return recommendations
    
    def visualize_customer_analysis(self, output_path=None):
        """Visualize customer segments and marketing strategies."""
        if not self.segments or not self.marketing_strategies:
            print("No segments or marketing strategies available to visualize")
            return False
        
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Customer Segments by Size
        plt.subplot(2, 2, 1)
        
        segment_names = [s["name"] for s in self.segments.values()]
        segment_sizes = [len(s["customers"]) for s in self.segments.values()]
        
        # Sort by size
        segment_sizes, segment_names = zip(*sorted(zip(segment_sizes, segment_names), reverse=True))
        
        plt.bar(segment_names, segment_sizes, color='skyblue')
        plt.title('Customer Segments by Size')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Plot 2: Expected Marketing Performance by Segment
        plt.subplot(2, 2, 2)
        
        segment_ids = list(self.marketing_strategies.keys())
        engagement_rates = [self.marketing_strategies[s]["expected_engagement"] * 100 for s in segment_ids]
        conversion_rates = [self.marketing_strategies[s]["expected_conversion"] * 100 for s in segment_ids]
        
        x = np.arange(len(segment_ids))
        width = 0.35
        
        plt.bar(x - width/2, engagement_rates, width, label='Engagement (%)')
        plt.bar(x + width/2, conversion_rates, width, label='Conversion (%)')
        
        plt.xlabel('Segment')
        plt.ylabel('Rate (%)')
        plt.title('Expected Marketing Performance')
        plt.xticks(x, [self.segments[s]["name"] for s in segment_ids], rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        
        # Plot 3: Customer Age Distribution by Segment
        plt.subplot(2, 2, 3)
        
        # Create age distributions for each segment
        for segment_id, segment in list(self.segments.items())[:4]:  # Limit to first 4 segments
            ages = [self.customers[c]["age"] for c in segment["customers"] if c in self.customers]
            
            if ages:
                plt.hist(ages, alpha=0.5, bins=7, label=segment["name"])
        
        plt.title('Age Distribution by Segment')
        plt.xlabel('Age')
        plt.ylabel('Number of Customers')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Purchase Behavior Analysis
        plt.subplot(2, 2, 4)
        
        # Create scatter plot of recency vs frequency for segmented customers
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        
        for i, (segment_id, segment) in enumerate(list(self.segments.items())[:6]):  # Limit to first 6 segments
            # Get recency and frequency data
            recency = []
            frequency = []
            sizes = []
            
            for c_id in segment["customers"]:
                if c_id not in self.customers:
                    continue
                
                customer = self.customers[c_id]
                
                # Skip customers with no purchase history
                if not customer["last_purchase"]:
                    continue
                
                # Calculate recency (days since last purchase)
                days_since_last = (datetime.now() - customer["last_purchase"]).days
                
                # Calculate frequency (purchases per active day)
                if customer["first_purchase"] and customer["last_purchase"] > customer["first_purchase"]:
                    days_active = (customer["last_purchase"] - customer["first_purchase"]).days
                    freq = customer["purchase_count"] / max(1, days_active) * 30  # Normalized to monthly rate
                else:
                    freq = customer["purchase_count"]
                
                recency.append(days_since_last)
                frequency.append(freq)
                sizes.append(max(20, min(200, customer["total_spent"] / 10)))
            
            if recency and frequency:
                plt.scatter(recency, frequency, alpha=0.6, c=colors[i % len(colors)], 
                           s=50, label=segment["name"])
        
        plt.title('Recency vs. Frequency by Segment')
        plt.xlabel('Days Since Last Purchase (Recency)')
        plt.ylabel('Purchases per Month (Frequency)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
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
    """Run a demonstration of the customer behavior analyzer."""
    # Initialize the analyzer
    analyzer = CustomerBehaviorAnalyzer()
    
    # Generate simulated customer data
    analyzer.generate_customer_data()
    
    # Analyze customer behavior patterns
    patterns = analyzer.analyze_customer_segments()
    
    # Segment customers
    analyzer.segment_customers(patterns)
    
    # Generate marketing strategies
    analyzer.generate_marketing_strategies()
    
    # Visualize results
    output_dir = os.path.join(root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    analyzer.visualize_customer_analysis(f"{output_dir}/customer_analysis.png")
    
    print("\nCUSTOMER BEHAVIOR ANALYZER DEMO COMPLETE")
    print("\nThis demonstration has shown how the encrypted ASI engine can be used")
    print("to build sophisticated customer analytics applications without accessing")
    print("the proprietary algorithms and mathematical implementations.")

if __name__ == "__main__":
    run_demo()
