#!/usr/bin/env python3
"""
Financial Analysis Application Example
-------------------------------------
This demonstrates how to use the ASI public API to build a financial analysis application
without needing to understand the encrypted internal implementation.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the ASI public API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

class FinancialAnalyzer:
    """Financial analysis application using encrypted ASI capabilities."""
    
    def __init__(self):
        """Initialize the financial analyzer."""
        print("\n" + "=" * 80)
        print("FINANCIAL MARKET ANALYZER - Powered by ASI Engine".center(80))
        print("=" * 80)
        
        # Initialize the ASI system
        print("\nInitializing ASI Engine...")
        initialize_asi()
        
        # Create an ASI instance for finance
        self.asi = create_asi_instance(name="FinanceASI", config={
            "domain": "finance",
            "risk_tolerance": 0.6,
            "time_horizon": "medium"
        })
        
        # Initialize market database
        self.market_data = {}
        self.analysis_results = {}
    
    def add_market_data(self, market_id: str, data: Dict[str, float]) -> bool:
        """
        Add market data to the analyzer.
        
        Args:
            market_id: Unique identifier for the market
            data: Market metrics (normalized between 0-1)
            
        Returns:
            bool: Success status
        """
        required_fields = ["market_volatility", "interest_rates", "economic_growth", "inflation"]
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                print(f"Error: Missing required field '{field}'")
                return False
        
        # Add market to database
        self.market_data[market_id] = {
            "data": data,
            "added_at": datetime.now().isoformat(),
            "analyzed": False
        }
        
        print(f"Market {market_id} added successfully")
        return True
    
    def analyze_market(self, market_id: str) -> Dict[str, Any]:
        """
        Analyze a market using ASI capabilities.
        
        Args:
            market_id: Market identifier
            
        Returns:
            Dict: Analysis results
        """
        if market_id not in self.market_data:
            print(f"Error: Market {market_id} not found")
            return {"error": "Market not found"}
        
        market = self.market_data[market_id]
        
        print(f"\nAnalyzing market {market_id}...")
        
        # Phase 1: Pattern Discovery
        print("\n" + "-" * 50)
        print("PHASE 1: MARKET PATTERN ANALYSIS")
        print("-" * 50)
        
        # Discover patterns in market data using ASI
        patterns = self.asi.discover_patterns(domain="market_conditions", properties=market["data"])
        
        print(f"\nDiscovered {len(patterns['patterns'])} patterns in market data")
        print(f"Analysis method: {patterns['method']}")
        print(f"Confidence: {patterns['confidence']:.4f}")
        
        # Display discovered patterns
        for i, pattern in enumerate(patterns['patterns']):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"  Description: {pattern['description']}")
            print(f"  Significance: {pattern['significance']:.4f}")
        
        # Phase 2: Investment Insight Generation
        print("\n" + "-" * 50)
        print("PHASE 2: INVESTMENT INSIGHT GENERATION")
        print("-" * 50)
        
        # Generate investment insights using ASI
        financial_concepts = ["volatility", "interest rates", "diversification", "risk"]
        insights = self.asi.generate_insight(concepts=financial_concepts)
        
        print(f"\nInvestment Insight (Confidence: {insights['confidence']:.4f}):")
        print(f"\"{insights['text']}\"")
        print(f"Related concepts: {', '.join(insights['concepts'])}")
        
        # Phase 3: Market Timeline Prediction
        print("\n" + "-" * 50)
        print("PHASE 3: MARKET SCENARIO PREDICTION")
        print("-" * 50)
        
        # Create market scenario
        market_scenario = {
            "name": "Interest Rate Adjustment Impact",
            "complexity": 0.8,
            "uncertainty": 0.7,
            "domain": "macroeconomics",
            "current_state": market["data"]
        }
        
        # Predict potential market outcomes using ASI
        prediction = self.asi.predict_timeline(market_scenario)
        
        print(f"\nMarket Scenario: {market_scenario['name']}")
        print(f"Prediction confidence: {prediction['confidence']:.4f}")
        print(f"Temporal coherence: {prediction['temporal_coherence']:.4f}")
        
        # Display base timeline
        print("\nPrimary Market Outcome:")
        for i, step in enumerate(prediction['base_timeline']):
            print(f"  Step {i+1}: {step['event']} - {step['emotion']}")
            print(f"    {step['description']}")
        
        # Store analysis results
        analysis = {
            "patterns": patterns,
            "insights": insights,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_results[market_id] = analysis
        self.market_data[market_id]["analyzed"] = True
        
        return analysis
    
    def generate_investment_strategy(self, market_id: str) -> Dict[str, Any]:
        """
        Generate an investment strategy based on market analysis.
        
        Args:
            market_id: Market identifier
            
        Returns:
            Dict: Investment strategy
        """
        if market_id not in self.analysis_results:
            print(f"Error: Market {market_id} has not been analyzed yet")
            return {"error": "Market not analyzed"}
        
        analysis = self.analysis_results[market_id]
        market = self.market_data[market_id]["data"]
        
        # Generate investment strategy based on analysis
        # This is a demonstration - in a real app, this would use more sophisticated logic
        volatility = market.get("market_volatility", 0.5)
        growth = market.get("economic_growth", 0.5)
        interest = market.get("interest_rates", 0.5)
        inflation = market.get("inflation", 0.5)
        
        # Calculate basic allocations
        stock_allocation = 0.4 + (0.2 * growth) - (0.1 * volatility)
        bond_allocation = 0.3 + (0.2 * interest) - (0.1 * inflation)
        cash_allocation = 1.0 - stock_allocation - bond_allocation
        
        # Apply insights from ASI analysis
        patterns = analysis["patterns"]["patterns"]
        for pattern in patterns:
            # Adjust allocations based on pattern significance
            if "growth" in pattern["concept"] and pattern["significance"] > 0.6:
                stock_allocation += 0.05
                bond_allocation -= 0.03
                cash_allocation -= 0.02
            if "interest" in pattern["concept"] and pattern["significance"] > 0.6:
                bond_allocation += 0.05
                stock_allocation -= 0.03
                cash_allocation -= 0.02
        
        # Normalize allocations
        total = stock_allocation + bond_allocation + cash_allocation
        stock_allocation /= total
        bond_allocation /= total
        cash_allocation /= total
        
        # Create strategy recommendations
        strategy = {
            "market_id": market_id,
            "allocations": {
                "stocks": round(stock_allocation, 2),
                "bonds": round(bond_allocation, 2),
                "cash": round(cash_allocation, 2)
            },
            "risk_level": "high" if stock_allocation > 0.6 else "medium" if stock_allocation > 0.4 else "low",
            "time_horizon": "long-term",
            "recommendations": [
                f"Allocate {int(stock_allocation * 100)}% to stock market investments",
                f"Allocate {int(bond_allocation * 100)}% to bond investments",
                f"Keep {int(cash_allocation * 100)}% in cash or cash equivalents"
            ],
            "rationale": analysis["insights"]["text"],
            "generated_at": datetime.now().isoformat()
        }
        
        # Display strategy
        print("\n" + "-" * 50)
        print("INVESTMENT STRATEGY")
        print("-" * 50)
        print(f"\nRisk Level: {strategy['risk_level']}")
        print(f"Time Horizon: {strategy['time_horizon']}")
        
        print("\nRecommended Allocations:")
        print(f"  Stocks: {strategy['allocations']['stocks'] * 100:.1f}%")
        print(f"  Bonds: {strategy['allocations']['bonds'] * 100:.1f}%")
        print(f"  Cash: {strategy['allocations']['cash'] * 100:.1f}%")
        
        print("\nRecommendations:")
        for i, rec in enumerate(strategy["recommendations"]):
            print(f"  {i+1}. {rec}")
        
        print(f"\nRationale: {strategy['rationale']}")
        
        return strategy
    
    def visualize_results(self, market_id: str, output_path: str = None) -> bool:
        """
        Visualize analysis results for a market.
        
        Args:
            market_id: Market identifier
            output_path: Optional path to save visualization
            
        Returns:
            bool: Success status
        """
        if market_id not in self.analysis_results:
            print(f"Error: Market {market_id} has not been analyzed yet")
            return False
        
        analysis = self.analysis_results[market_id]
        market_data = self.market_data[market_id]["data"]
        
        # Generate investment strategy if not already done
        if not hasattr(self, 'strategies') or market_id not in self.strategies:
            if not hasattr(self, 'strategies'):
                self.strategies = {}
            self.strategies[market_id] = self.generate_investment_strategy(market_id)
        
        strategy = self.strategies[market_id]
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Market Indicators
        plt.subplot(2, 2, 1)
        keys = list(market_data.keys())
        values = [market_data[k] for k in keys]
        plt.bar(keys, values, color='skyblue')
        plt.title('Market Indicators')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Plot 2: Pattern Significance
        plt.subplot(2, 2, 2)
        pattern_names = [p["concept"] for p in analysis["patterns"]["patterns"]]
        pattern_values = [p["significance"] for p in analysis["patterns"]["patterns"]]
        plt.barh(pattern_names, pattern_values, color='lightgreen')
        plt.title('Pattern Significance')
        plt.xlim(0, 1)
        
        # Plot 3: Asset Allocation
        plt.subplot(2, 2, 3)
        allocations = strategy["allocations"]
        plt.pie([allocations["stocks"], allocations["bonds"], allocations["cash"]], 
                labels=['Stocks', 'Bonds', 'Cash'],
                autopct='%1.1f%%',
                colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title('Recommended Asset Allocation')
        
        # Plot 4: Timeline Visualization
        plt.subplot(2, 2, 4)
        timeline = analysis["prediction"]["base_timeline"]
        x = range(len(timeline))
        events = [item["event"] for item in timeline]
        emotions = {"neutral": 0.5, "optimistic": 0.7, "cautious": 0.4, 
                   "confident": 0.8, "concerned": 0.3, "positive": 0.9}
        y = [emotions.get(item["emotion"].split()[-1], 0.5) for item in timeline]
        
        plt.plot(x, y, 'o-', markersize=10)
        plt.title('Market Timeline Projection')
        plt.xticks(x, events, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        
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
    """Run the financial analyzer demo."""
    analyzer = FinancialAnalyzer()
    
    # Sample market data
    markets = {
        "US_MARKET": {
            "market_volatility": 0.4,
            "interest_rates": 0.3,
            "economic_growth": 0.65,
            "inflation": 0.45,
            "consumer_sentiment": 0.7,
            "unemployment": 0.3,
            "housing_market": 0.6
        },
        "EMERGING_MARKETS": {
            "market_volatility": 0.7,
            "interest_rates": 0.6,
            "economic_growth": 0.8,
            "inflation": 0.65,
            "consumer_sentiment": 0.5,
            "foreign_investment": 0.7,
            "political_stability": 0.4
        }
    }
    
    # Add markets to the analyzer
    for market_id, data in markets.items():
        analyzer.add_market_data(market_id, data)
    
    # Analyze each market
    for market_id in markets.keys():
        analyzer.analyze_market(market_id)
        
        # Generate investment strategy
        analyzer.generate_investment_strategy(market_id)
        
        # Generate visualization
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        analyzer.visualize_results(market_id, f"{output_dir}/{market_id}_analysis.png")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE".center(80))
    print("=" * 80)
    
    print("\nThis demonstration has showcased how the encrypted ASI engine")
    print("can be used to build sophisticated financial applications without")
    print("exposing the underlying algorithms and implementations.")

if __name__ == "__main__":
    run_demo()
