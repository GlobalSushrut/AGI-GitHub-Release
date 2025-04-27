import os
import sys
import logging
from transaction_manager import TransactionManager

# Add the parent directory to path for ASI helper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from real_world_apps.asi_helper import initialize_asi_components, process_with_asi

class RiskAnalyzer:
    """Profiles risk for accounts based on transaction history using advanced ASI analysis."""
    def __init__(self, api=None):
        self.tm = TransactionManager(api=api)
        self.logger = logging.getLogger("RiskAnalyzer")
        
        # Initialize real ASI components
        try:
            # Set environment variable to ensure interface uses real components
            os.environ['USE_REAL_ASI'] = 'true'
            initialize_asi_components()
            self.asi_available = True
            self.logger.info("ASI Engine initialized for risk analysis")
        except Exception as e:
            self.asi_available = False
            self.logger.warning(f"Could not initialize ASI Engine: {str(e)}")

    def analyze_account_risk(self, account_id: str) -> dict:
        """Compute a comprehensive risk score for the account using ASI."""
        # Get transaction history
        txs = self.tm.get_transactions(account_id)
        
        # Simple fallback calculation
        total = sum(t.get("amount", 0) for t in txs)
        count = len(txs)
        basic_risk_score = min(total / (count or 1) / 1000, 1.0)  # Normalize to 0-1 range
        
        # Default result
        result = {
            "account_id": account_id, 
            "risk_score": basic_risk_score, 
            "transaction_count": count,
            "risk_factors": [],
            "risk_level": "low" if basic_risk_score < 0.3 else "medium" if basic_risk_score < 0.7 else "high"
        }
        
        # Try using ASI for advanced risk analysis
        if self.asi_available and txs:
            try:
                # Prepare data for ASI analysis
                risk_data = {
                    "task": "analyze_risk",
                    "account_id": account_id,
                    "transactions": txs[:50]  # Limit to most recent 50 transactions
                }
                
                # Use ASI to analyze account risk
                from agi_toolkit import AGIAPI
                api = AGIAPI()
                
                asi_result = process_with_asi(api, risk_data)
                
                # Process the result
                if isinstance(asi_result, dict) and asi_result.get("success", False) and "result" in asi_result:
                    risk_data = asi_result["result"]
                    
                    # Handle different output formats
                    if isinstance(risk_data, dict):
                        # Extract risk score
                        if "risk_score" in risk_data:
                            result["risk_score"] = risk_data["risk_score"]
                        
                        # Extract risk level
                        if "risk_level" in risk_data:
                            result["risk_level"] = risk_data["risk_level"]
                        
                        # Extract risk factors
                        if "risk_factors" in risk_data:
                            result["risk_factors"] = risk_data["risk_factors"]
                        elif "factors" in risk_data:
                            result["risk_factors"] = risk_data["factors"]
                        
                        # Extract any other useful information
                        for key in ["recommendations", "alerts", "patterns"]:
                            if key in risk_data:
                                result[key] = risk_data[key]
                self.logger.info(f"ASI risk analysis for account {account_id}: {result['risk_level']} ({result['risk_score']:.2f})")
            except Exception as e:
                self.logger.error(f"Error in ASI risk analysis: {str(e)}")
        
        return result
