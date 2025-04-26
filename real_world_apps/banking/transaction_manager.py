import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from memory_api import MemoryAPI

class TransactionManager:
    """Handles ingestion and retrieval of banking transactions using AGI Toolkit."""
    
    def __init__(self, api=None):
        """Initialize the transaction manager."""
        self.logger = logging.getLogger("TransactionManager")
        self.api = api if api is not None else MemoryAPI()
        self.memory_key = "banking_transactions"
        
        # Initialize transactions storage in memory if it doesn't exist
        try:
            transactions, _ = self.api.retrieve_data(self.memory_key)
            self.logger.info(f"Loaded {len(transactions)} existing transactions")
        except:
            self.logger.info("No existing transactions found, initializing empty storage")
            self.api.store_data(self.memory_key, [])
    
    def ingest_transaction(self, transaction: Dict) -> Dict:
        """Ingest a new transaction into the system."""
        # Add timestamp if not present
        if "timestamp" not in transaction:
            transaction["timestamp"] = datetime.now().isoformat()
        
        # Add unique ID if not present
        if "id" not in transaction:
            transaction["id"] = f"tx-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self._get_all_transactions())}"
        
        # Retrieve current transactions
        transactions, _ = self.api.retrieve_data(self.memory_key)
        
        # Add new transaction
        transactions.append(transaction)
        
        # Store updated transactions
        self.api.store_data(self.memory_key, transactions)
        
        self.logger.info(f"Transaction {transaction['id']} ingested successfully")
        return transaction
    
    def get_transactions(self, account_id: str) -> List[Dict]:
        """Retrieve transactions for a given account."""
        transactions = self._get_all_transactions()
        return [t for t in transactions if t.get("account_id") == account_id]
    
    def _get_all_transactions(self) -> List[Dict]:
        """Get all transactions from memory."""
        try:
            transactions, _ = self.api.retrieve_data(self.memory_key)
            return transactions
        except Exception as e:
            self.logger.error(f"Error retrieving transactions: {str(e)}")
            return []
