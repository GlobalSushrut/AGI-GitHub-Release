from transaction_manager import TransactionManager

class RiskAnalyzer:
    """Profiles risk for accounts based on transaction history."""
    def __init__(self, api=None):
        self.tm = TransactionManager(api=api)

    def analyze_account_risk(self, account_id: str) -> dict:
        """Compute a simple risk score for the account."""
        txs = self.tm.get_transactions(account_id)
        total = sum(t.get("amount", 0) for t in txs)
        count = len(txs)
        risk_score = total / (count or 1)
        return {"account_id": account_id, "risk_score": risk_score, "transaction_count": count}
