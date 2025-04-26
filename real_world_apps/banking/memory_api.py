import threading
import json
import os
from typing import Any, Tuple

class MemoryAPI:
    """
    Simple file-based persistent memory API for banking transactions.
    """
    def __init__(self, filename="transactions_db.json"):
        self.filename = filename
        self.lock = threading.Lock()
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                json.dump({}, f)

    def retrieve_data(self, key: str) -> Tuple[Any, None]:
        with self.lock:
            with open(self.filename, 'r') as f:
                db = json.load(f)
            return db.get(key, []), None

    def store_data(self, key: str, value: Any):
        with self.lock:
            with open(self.filename, 'r') as f:
                db = json.load(f)
            db[key] = value
            with open(self.filename, 'w') as f:
                json.dump(db, f, indent=2)
