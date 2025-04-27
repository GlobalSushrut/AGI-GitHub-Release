import os
import sys
import logging
from transaction_manager import TransactionManager

# Add the parent directory to path for ASI helper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from real_world_apps.asi_helper import initialize_asi_components, process_with_asi

class FraudDetector:
    """Detects fraudulent transactions using advanced ASI-powered analysis."""
    def __init__(self, api=None):
        self.tm = TransactionManager(api=api)
        self.logger = logging.getLogger("FraudDetector")
        
        # Initialize real ASI components
        try:
            # Set environment variable to ensure interface uses real components
            os.environ['USE_REAL_ASI'] = 'true'
            initialize_asi_components()
            self.asi_available = True
            self.logger.info("ASI Engine initialized for fraud detection")
        except Exception as e:
            self.asi_available = False
            self.logger.warning(f"Could not initialize ASI Engine: {str(e)}")

    def analyze_transaction(self, transaction: dict) -> bool:
        """Return True if transaction is flagged as fraudulent."""
        # Basic rule-based checks as fallback
        if transaction.get("amount", 0) > 10000:
            return True
            
        # Try using ASI for advanced fraud detection
        if self.asi_available:
            try:
                # Get transaction history for context
                account_id = transaction.get('account_id')
                tx_history = self.tm.get_transactions(account_id)[-10:]  # Last 10 transactions
                
                # Format data for ASI analysis
                tx_data = {
                    "task": "detect_fraud",
                    "transaction": transaction,
                    "transaction_history": tx_history
                }
                
                # Use ASI to analyze the transaction
                from agi_toolkit import AGIAPI
                api = AGIAPI()
                
                result = process_with_asi(api, tx_data)
                
                # Process the result
                if isinstance(result, dict) and result.get("success", False) and "result" in result:
                    fraud_data = result["result"]
                    
                    # Handle different output formats
                    if isinstance(fraud_data, dict):
                        if "is_fraud" in fraud_data:
                            return fraud_data["is_fraud"]
                        elif "fraud_score" in fraud_data:
                            return fraud_data["fraud_score"] > 0.7
                        elif "fraud" in fraud_data:
                            return fraud_data["fraud"]
                    elif isinstance(fraud_data, bool):
                        return fraud_data
                    elif isinstance(fraud_data, (int, float)):
                        return fraud_data > 0.7  # Threshold for fraud score
                        
                self.logger.info(f"ASI fraud detection result for tx {transaction.get('tx_id')}: {result}")
            except Exception as e:
                self.logger.error(f"Error in ASI fraud detection: {str(e)}")
                
        # Advanced heuristics (only used if ASI failed or is unavailable)
        tx_amount = transaction.get("amount", 0)
        tx_desc = transaction.get("description", "").lower()
        
        # Look for suspicious keywords
        suspicious_terms = ["wire", "transfer", "international", "forex", "bitcoin", "crypto"]
        if any(term in tx_desc for term in suspicious_terms) and tx_amount > 5000:
            return True
            
        return False
