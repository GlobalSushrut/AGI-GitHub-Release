import sys
import os
# Ensure project root (where 'infra/' lives) is in sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import click
import logging
from transaction_manager import TransactionManager
from fraud_detector import FraudDetector
from risk_analyzer import RiskAnalyzer
from memory_api import MemoryAPI
try:
    from infra.licensing.license_validator import validate_license
except ModuleNotFoundError as e:
    print("\n[ERROR] Could not import 'infra.licensing.license_validator'.\nRun this script with PYTHONPATH set to project root, e.g.:\n  PYTHONPATH=. python3 real_world_apps/banking/app.py ...\n\nOriginal error:", e)
    exit(1)


memory_api = MemoryAPI()

# Example metric calculation (replace with real metrics in production)
def get_activity_metrics():
    # Get real metrics from the application
    tm = TransactionManager(api=memory_api)
    
    # Actual transaction count from database
    tx_count = len(tm._get_all_transactions())
    
    # For demo purposes, we'll simulate whether this is corporate usage
    # In a real app, you would measure real corporate indicators:
    # - Number of concurrent users
    # - Transaction volume
    # - API call frequency
    # - Database size
    # - Running in cloud/enterprise environment
    
    # Corporate usage is detected when:
    is_corporate_demo = False  # Set to True to force license check
    
    # Check environment variables for simulating corporate usage
    if os.environ.get("SIMULATE_CORPORATE", "").lower() == "true":
        is_corporate_demo = True
        active_users = 75  # Enterprise level (high)
        memory_usage = 1200  # MB (high)
    else:
        # Open source / development usage (low metrics)
        active_users = 2 if tx_count < 10 else 5
        memory_usage = 50
    
    # If you want to test the corporate license enforcement,
    # run with: SIMULATE_CORPORATE=true python3 app.py ...
    return [tx_count, active_users, memory_usage]

# License check decorator
def license_required(f):
    def wrapper(*args, **kwargs):
        metrics = get_activity_metrics()
        license_key = os.environ.get("MRZKELP_LICENSE_KEY")
        client_id = os.environ.get("MRZKELP_CLIENT_ID", "demo@example.com")
        secret = os.environ.get("MRZKELP_SECRET", "YourSecretSalt")
        valid, msg = validate_license(metrics, license_key=license_key, client_id=client_id, secret=secret)
        if not valid:
            click.echo(f"[MR-ZKELP] {msg}")
            exit(1)
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@click.group()
def cli():
    """Banking CLI"""
    pass

@cli.command()
@click.option('--account-id', required=True, help='Account ID')
@click.option('--tx-id', required=True, help='Transaction ID')
@click.option('--amount', type=float, required=True, help='Transaction amount')
@click.option('--description', default='', help='Transaction description')
@license_required
def ingest(account_id, tx_id, amount, description):
    """Ingest a new transaction and flag fraud"""
    tm = TransactionManager(api=memory_api)
    transaction = {
        'account_id': account_id,
        'tx_id': tx_id,
        'amount': amount,
        'description': description
    }
    tm.ingest_transaction(transaction)
    fd = FraudDetector(api=memory_api)
    is_fraud = fd.analyze_transaction(transaction)
    click.echo(f'Transaction ingested. Fraudulent: {is_fraud}')

@cli.command()
@click.option('--account-id', required=True, help='Account ID')
@license_required
def show_transactions(account_id):
    """Show transactions for an account"""
    tm = TransactionManager(api=memory_api)
    txs = tm.get_transactions(account_id)
    for t in txs:
        click.echo(t)

@cli.command()
@click.option('--account-id', required=True, help='Account ID')
@license_required
def analyze_account(account_id):
    """Analyze risk for an account"""
    ra = RiskAnalyzer(api=memory_api)
    result = ra.analyze_account_risk(account_id)
    click.echo(result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Banking Application started.")
    cli()
