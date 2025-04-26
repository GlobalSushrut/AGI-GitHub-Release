from transaction_manager import TransactionManager

class FraudDetector:
    """Detects fraudulent transactions using rule-based and ML methods."""
    def __init__(self, api=None):
        self.tm = TransactionManager(api=api)

    def analyze_transaction(self, transaction: dict) -> bool:
        """Return True if transaction is flagged as fraudulent."""
        # Rule-based checks
        if transaction.get("amount", 0) > 10000:
            return True
        # Placeholder for ML-based model
        return False
