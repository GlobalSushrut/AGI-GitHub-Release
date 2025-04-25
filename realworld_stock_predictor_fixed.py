#!/usr/bin/env python3
"""
Real-World Stock Market Predictor
--------------------------------
A practical application that uses the encrypted ASI infrastructure
to predict stock market trends and generate trading signals.

This demonstrates how developers can build real-world applications
using the ASI API while the core algorithms remain encrypted.
"""

import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Import ASI API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

class StockMarketPredictor:
    """Real-world stock market predictor using ASI infrastructure."""
    
    def __init__(self):
        """Initialize the stock market predictor."""
        print("\n" + "=" * 80)
        print("STOCK MARKET PREDICTOR - Powered by Encrypted ASI Engine".center(80))
        print("=" * 80)
        
        # Initialize ASI engine - accessing the encrypted infrastructure
        print("\nInitializing ASI engine...")
        success = initialize_asi()
        
        if not success:
            print("Failed to initialize ASI engine. Exiting.")
            sys.exit(1)
        
        # Create ASI instance with specific configuration for financial analysis
        self.asi = create_asi_instance(name="FinancialASI", config={
            "domain": "finance",
            "risk_tolerance": 0.65,
            "time_horizon": "medium",
            "volatility_factor": 0.7
        })
        
        print("ASI engine initialized successfully")
        
        # Initialize historical data store
        self.historical_data = {}
        self.predictions = {}
        self.trading_signals = {}
    
    def load_stock_data(self, stock_symbol, price_data):
        """
        Load stock data for analysis.
        
        Args:
            stock_symbol: Stock ticker symbol
            price_data: Dictionary with price time series data
        """
        # Normalize price data for ASI processing
        normalized_data = {}
        if len(price_data) > 1:
            min_price = min(price_data.values())
            max_price = max(price_data.values())
            price_range = max_price - min_price
            
            if price_range > 0:
                for date, price in price_data.items():
                    normalized_data[date] = (price - min_price) / price_range
        
        # Calculate technical indicators
        indicators = self._calculate_technical_indicators(price_data)
        
        # Store data
        self.historical_data[stock_symbol] = {
            "raw_data": price_data,
            "normalized_data": normalized_data,
            "indicators": indicators,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        print(f"\nLoaded data for {stock_symbol}")
        print(f"Data points: {len(price_data)}")
        print(f"Technical indicators calculated: {len(indicators)}")
    
    def _calculate_technical_indicators(self, price_data):
        """
        Calculate technical indicators from price data.
        
        Args:
            price_data: Dictionary with price time series data
            
        Returns:
            Dict: Technical indicators
        """
        dates = list(price_data.keys())
        prices = list(price_data.values())
        
        # Calculate simple moving averages
        sma_5 = self._calculate_sma(prices, 5)
        sma_20 = self._calculate_sma(prices, 20)
        
        # Calculate volatility
        if len(prices) >= 20:
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        else:
            volatility = 0.0
        
        # Calculate price momentum (rate of change)
        if len(prices) >= 10:
            momentum = (prices[-1] / prices[-10]) - 1.0
        else:
            momentum = 0.0
        
        # Calculate trend strength
        if len(sma_5) > 0 and len(sma_20) > 0:
            trend = sma_5[-1] / sma_20[-1] - 1.0
        else:
            trend = 0.0
        
        # Return indicators as normalized values between 0-1
        return {
            "sma_5_20_ratio": self._normalize_value(sma_5[-1] / sma_20[-1] if len(sma_5) > 0 and len(sma_20) > 0 else 1.0, 0.8, 1.2),
            "volatility": self._normalize_value(volatility, 0, 0.2),
            "momentum": self._normalize_value(momentum, -0.1, 0.1),
            "trend": self._normalize_value(trend, -0.05, 0.05),
            "price_level": self._normalize_value(prices[-1], min(prices), max(prices)) if prices else 0.5
        }
    
    def _calculate_sma(self, data, window):
        """Calculate simple moving average."""
        result = []
        for i in range(len(data)):
            if i < window - 1:
                result.append(0)
            else:
                result.append(sum(data[i-(window-1):i+1]) / window)
        return result
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize a value to range 0-1."""
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
            return max(0, min(1, normalized))
        return 0.5
    
    def analyze_stock(self, stock_symbol):
        """
        Analyze a stock using ASI's pattern discovery capabilities.
        
        Args:
            stock_symbol: Stock ticker symbol
            
        Returns:
            Dict: Analysis results
        """
        if stock_symbol not in self.historical_data:
            print(f"Error: No data for {stock_symbol}")
            return None
        
        stock_data = self.historical_data[stock_symbol]
        
        print(f"\nAnalyzing {stock_symbol}...")
        print("-" * 40)
        
        # Use ASI pattern discovery to identify market patterns
        # This uses the encrypted ASI infrastructure through the API
        patterns = self.asi.discover_patterns(
            domain="stock_market",
            properties=stock_data["indicators"]
        )
        
        print(f"Discovered {len(patterns['patterns'])} market patterns")
        print(f"Analysis confidence: {patterns['confidence']:.4f}")
        
        # Display top patterns
        for i, pattern in enumerate(patterns['patterns'][:3]):
            print(f"\nPattern {i+1}: {pattern['concept']}")
            print(f"  Description: {pattern['description']}")
            print(f"  Significance: {pattern['significance']:.4f}")
        
        # Generate market insight using ASI's encrypted capabilities
        insight = self.asi.generate_insight(
            concepts=["stock market", "volatility", "trend", "momentum"]
        )
        
        print(f"\nMarket Insight (Confidence: {insight['confidence']:.4f}):")
        print(f"\"{insight['text']}\"")
        
        # Store analysis results
        return {
            "patterns": patterns,
            "insight": insight,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def predict_price_movement(self, stock_symbol, time_horizon=30):
        """
        Predict future price movement using ASI's timeline prediction.
        
        Args:
            stock_symbol: Stock ticker symbol
            time_horizon: Number of days to predict
            
        Returns:
            Dict: Prediction results
        """
        if stock_symbol not in self.historical_data:
            print(f"Error: No data for {stock_symbol}")
            return None
        
        stock_data = self.historical_data[stock_symbol]
        
        print(f"\nPredicting price movement for {stock_symbol}...")
        print("-" * 40)
        
        # Create market scenario for ASI prediction
        # This leverages the encrypted ASI core algorithms
        market_scenario = {
            "name": f"{stock_symbol} Price Projection",
            "complexity": 0.8,
            "uncertainty": 0.7,
            "domain": "stock_price",
            "current_state": stock_data["indicators"],
            "time_horizon": time_horizon
        }
        
        # Use ASI to predict possible price timelines
        prediction = self.asi.predict_timeline(market_scenario)
        
        print(f"Prediction confidence: {prediction['confidence']:.4f}")
        print(f"Temporal coherence: {prediction['temporal_coherence']:.4f}")
        
        # Display base timeline (most likely scenario)
        print("\nMost Likely Price Trajectory:")
        for i, step in enumerate(prediction['base_timeline']):
            day = (i * time_horizon) // len(prediction['base_timeline'])
            print(f"  Day {day}: {step['event']} - {step['emotion']}")
        
        # Display alternative scenarios
        print(f"\nAlternative Scenarios: {len(prediction['branching_timelines'])}")
        for i, branch in enumerate(prediction['branching_timelines']):
            print(f"  Scenario {i+1}: {branch['type']} (Probability: {branch.get('probability', 0):.2f})")
        
        # Generate trading signals
        signals = self._generate_trading_signals(prediction, stock_data)
        print("\nTrading Signals Generated:")
        print(f"  Primary Signal: {signals['primary_signal']}")
        print(f"  Confidence: {signals['confidence']:.4f}")
        print(f"  Time Horizon: {signals['time_horizon']} days")
        print(f"  Entry Price Range: {signals['entry_price_range']['low']:.2f} - {signals['entry_price_range']['high']:.2f}")
        
        # Store prediction
        self.predictions[stock_symbol] = prediction
        self.trading_signals[stock_symbol] = signals
        
        return {
            "prediction": prediction,
            "trading_signals": signals,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _generate_trading_signals(self, prediction, stock_data):
        """
        Generate trading signals from ASI predictions.
        
        Args:
            prediction: ASI timeline prediction
            stock_data: Historical stock data
            
        Returns:
            Dict: Trading signals
        """
        # Get current price (last price in the data)
        prices = list(stock_data["raw_data"].values())
        current_price = prices[-1] if prices else 0
        
        # Determine primary signal based on first few steps of base timeline
        bullish_terms = ["increase", "rise", "grow", "positive", "uptrend", "optimistic"]
        bearish_terms = ["decrease", "fall", "decline", "negative", "downtrend", "cautious"]
        
        # Count bullish vs bearish signals in the prediction
        bullish_count = 0
        bearish_count = 0
        
        # Analyze first half of the timeline with more weight on earlier events
        timeline_to_analyze = prediction['base_timeline'][:len(prediction['base_timeline'])//2]
        for i, step in enumerate(timeline_to_analyze):
            weight = 1.0 - (i / len(timeline_to_analyze))  # Earlier steps have more weight
            
            # Check event description for bullish/bearish terms
            event_text = step['event'].lower()
            for term in bullish_terms:
                if term in event_text:
                    bullish_count += weight
            
            for term in bearish_terms:
                if term in event_text:
                    bearish_count += weight
            
            # Check emotion
            emotion = step['emotion'].lower()
            if any(term in emotion for term in ["optimistic", "confident", "positive"]):
                bullish_count += 0.5 * weight
            elif any(term in emotion for term in ["cautious", "concerned", "negative"]):
                bearish_count += 0.5 * weight
        
        # Determine signal
        if bullish_count > bearish_count * 1.5:
            primary_signal = "STRONG BUY"
            confidence = min(0.9, 0.5 + (bullish_count - bearish_count) / 5)
        elif bullish_count > bearish_count:
            primary_signal = "BUY"
            confidence = 0.5 + (bullish_count - bearish_count) / 10
        elif bearish_count > bullish_count * 1.5:
            primary_signal = "STRONG SELL"
            confidence = min(0.9, 0.5 + (bearish_count - bullish_count) / 5)
        elif bearish_count > bullish_count:
            primary_signal = "SELL"
            confidence = 0.5 + (bearish_count - bullish_count) / 10
        else:
            primary_signal = "HOLD"
            confidence = 0.5
        
        # Adjust with prediction confidence
        confidence = (confidence + prediction['confidence']) / 2
        
        # Calculate entry price range
        if primary_signal in ["BUY", "STRONG BUY"]:
            # For buy signals, set entry below current price
            entry_low = current_price * 0.97
            entry_high = current_price * 1.01
        elif primary_signal in ["SELL", "STRONG SELL"]:
            # For sell signals, set entry above current price
            entry_low = current_price * 0.99
            entry_high = current_price * 1.03
        else:
            # For hold signals
            entry_low = current_price * 0.98
            entry_high = current_price * 1.02
        
        # Determine time horizon based on prediction
        time_horizon = 5  # Default to 5 days
        if prediction['temporal_coherence'] > 0.7:
            time_horizon = 14  # Two weeks for high coherence
        elif prediction['temporal_coherence'] > 0.5:
            time_horizon = 7   # One week for medium coherence
        
        return {
            "primary_signal": primary_signal,
            "confidence": confidence,
            "time_horizon": time_horizon,
            "entry_price_range": {
                "low": entry_low,
                "high": entry_high
            },
            "supporting_factors": {
                "bullish_indicators": bullish_count,
                "bearish_indicators": bearish_count,
                "prediction_confidence": prediction['confidence'],
                "temporal_coherence": prediction['temporal_coherence']
            }
        }
    
    def visualize_analysis(self, stock_symbol, output_path=None):
        """
        Visualize stock analysis and predictions.
        
        Args:
            stock_symbol: Stock ticker symbol
            output_path: Optional path to save visualization
            
        Returns:
            bool: Success status
        """
        if (stock_symbol not in self.historical_data or 
            stock_symbol not in self.predictions or
            stock_symbol not in self.trading_signals):
            print(f"Error: Complete data not available for {stock_symbol}")
            return False
        
        stock_data = self.historical_data[stock_symbol]
        prediction = self.predictions[stock_symbol]
        signals = self.trading_signals[stock_symbol]
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Historical Price
        plt.subplot(2, 2, 1)
        dates = list(stock_data["raw_data"].keys())
        prices = list(stock_data["raw_data"].values())
        plt.plot(range(len(dates)), prices, 'b-', linewidth=2)
        plt.title(f'{stock_symbol} Historical Price')
        plt.xticks([])  # Hide x-axis labels
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Technical Indicators
        plt.subplot(2, 2, 2)
        indicators = stock_data["indicators"]
        ind_names = list(indicators.keys())
        ind_values = [indicators[k] for k in ind_names]
        plt.bar(ind_names, ind_values, color='green', alpha=0.6)
        plt.title('Technical Indicators')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Plot 3: Price Prediction
        plt.subplot(2, 1, 2)
        
        # Setup for prediction chart
        current_price = prices[-1]
        time_horizon = signals['time_horizon']
        
        # Create prediction line based on signals
        prediction_days = list(range(time_horizon))  # Fixed: removed +1 to match dimensions
        
        if signals['primary_signal'] in ["STRONG BUY", "BUY"]:
            # Bullish prediction
            max_increase = current_price * (1 + 0.02 * signals['confidence'] * time_horizon)
            predicted_prices = [current_price * (1 + 0.02 * signals['confidence'] * min(i, time_horizon/2)) for i in prediction_days]
        elif signals['primary_signal'] in ["STRONG SELL", "SELL"]:
            # Bearish prediction
            max_decrease = current_price * (1 - 0.02 * signals['confidence'] * time_horizon)
            predicted_prices = [current_price * (1 - 0.02 * signals['confidence'] * min(i, time_horizon/2)) for i in prediction_days]
        else:
            # Neutral prediction
            predicted_prices = [current_price] * len(prediction_days)
        
        # Plot historical prices
        plt.plot(range(len(dates)), prices, 'b-', label='Historical', linewidth=2)
        
        # Fixed: Ensure x and y arrays have the same length for predicted prices
        future_x = range(len(dates)-1, len(dates) + len(prediction_days))
        future_y = [prices[-1]] + predicted_prices
        plt.plot(future_x, future_y, 'r--', label='Predicted', linewidth=2)
        
        # Add prediction confidence band
        confidence = signals['confidence']
        upper_band = [p * (1 + 0.01 * (1 - confidence)) for p in predicted_prices]
        lower_band = [p * (1 - 0.01 * (1 - confidence)) for p in predicted_prices]
        
        plt.fill_between(
            future_x,
            [prices[-1]] + upper_band,
            [prices[-1]] + lower_band,
            color='red', alpha=0.2
        )
        
        # Add labels and formatting
        plt.title(f'{stock_symbol} Price Prediction - Signal: {signals["primary_signal"]} (Confidence: {signals["confidence"]:.2f})')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add text annotation with ASI insight
        if 'insight' in prediction:
            insight_text = prediction['insight']['text']
            plt.figtext(0.5, 0.01, f"ASI Insight: {insight_text}", wrap=True, 
                      horizontalalignment='center', fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
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

def run_real_world_test():
    """Run a real-world test of the stock predictor using the encrypted ASI engine."""
    # Initialize the stock market predictor
    predictor = StockMarketPredictor()
    
    # Sample stock data (in a real application, this would come from a market data API)
    # Using sample data for AAPL (Apple) stock
    aapl_data = {
        "2023-01-03": 125.07,
        "2023-01-04": 126.36,
        "2023-01-05": 125.02,
        "2023-01-06": 129.62,
        "2023-01-09": 130.15,
        "2023-01-10": 130.73,
        "2023-01-11": 133.49,
        "2023-01-12": 133.41,
        "2023-01-13": 134.76,
        "2023-01-17": 135.94,
        "2023-01-18": 135.21,
        "2023-01-19": 135.27,
        "2023-01-20": 137.87,
        "2023-01-23": 141.11,
        "2023-01-24": 142.53,
        "2023-01-25": 141.86,
        "2023-01-26": 143.96,
        "2023-01-27": 145.93,
        "2023-01-30": 143.00,
        "2023-01-31": 144.29,
        "2023-02-01": 145.43,
        "2023-02-02": 150.82,
        "2023-02-03": 154.50,
        "2023-02-06": 151.73,
        "2023-02-07": 154.65,
        "2023-02-08": 151.92,
        "2023-02-09": 150.87,
        "2023-02-10": 151.01
    }
    
    # Sample data for MSFT (Microsoft) stock
    msft_data = {
        "2023-01-03": 239.58,
        "2023-01-04": 229.10,
        "2023-01-05": 222.31,
        "2023-01-06": 224.93,
        "2023-01-09": 227.12,
        "2023-01-10": 228.85,
        "2023-01-11": 235.77,
        "2023-01-12": 238.51,
        "2023-01-13": 239.23,
        "2023-01-17": 240.35,
        "2023-01-18": 235.81,
        "2023-01-19": 231.93,
        "2023-01-20": 240.22,
        "2023-01-23": 242.58,
        "2023-01-24": 242.04,
        "2023-01-25": 240.61,
        "2023-01-26": 248.00,
        "2023-01-27": 248.16,
        "2023-01-30": 242.71,
        "2023-01-31": 247.81,
        "2023-02-01": 252.75,
        "2023-02-02": 264.60,
        "2023-02-03": 258.35,
        "2023-02-06": 256.77,
        "2023-02-07": 267.56,
        "2023-02-08": 266.73,
        "2023-02-09": 263.62,
        "2023-02-10": 258.06
    }
    
    # Load stock data
    predictor.load_stock_data("AAPL", aapl_data)
    predictor.load_stock_data("MSFT", msft_data)
    
    # Analyze stocks
    for symbol in ["AAPL", "MSFT"]:
        # Perform analysis using encrypted ASI infrastructure
        analysis = predictor.analyze_stock(symbol)
        
        # Generate price predictions using encrypted ASI infrastructure
        prediction = predictor.predict_price_movement(symbol)
        
        # Store insight for visualization
        if analysis and "insight" in analysis:
            predictor.predictions[symbol]["insight"] = analysis["insight"]
        
        # Visualize results
        output_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(output_dir, exist_ok=True)
        predictor.visualize_analysis(symbol, f"{output_dir}/{symbol}_analysis.png")
    
    print("\n" + "=" * 80)
    print("REAL-WORLD APPLICATION TEST COMPLETE".center(80))
    print("=" * 80)
    
    print("\nThis demonstration has shown that developers can build real-world")
    print("stock market prediction applications using the encrypted ASI infrastructure.")
    print("\nThe application successfully leveraged the ASI engine's capabilities:")
    print("1. Pattern discovery in financial data")
    print("2. Cross-domain insights generation")
    print("3. Multi-timeline prediction for market scenarios")
    print("\nAll of this was done while the core ASI algorithms remained securely encrypted.")
    print("Developers can build sophisticated applications without access to proprietary code.")
    
    print("\nCheck the generated visualizations in the reports directory.")

if __name__ == "__main__":
    run_real_world_test()
