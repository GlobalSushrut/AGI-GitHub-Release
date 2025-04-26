import hashlib
import argparse
import sys
import platform
from datetime import datetime
from enum import Enum

class LicenseType(Enum):
    CORPORATE = "corporate"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"

def generate_license(secret, client_id, license_type=LicenseType.CORPORATE):
    """Generate a license hash using the same algorithm as the validator"""
    # Match the exact same format used in license_validator.py
    timestamp = datetime.now().strftime("%Y%m")
    system_id = platform.node()
    raw = f"{secret}:{client_id}:{timestamp}:{system_id}"
    license_hash = hashlib.sha512(raw.encode()).hexdigest()
    return license_hash

def main():
    parser = argparse.ArgumentParser(description="Generate a commercial license hash for MR-ZKELP.")
    parser.add_argument('--client-id', required=True, help='Client email or unique identifier')
    parser.add_argument('--secret', required=True, help='Your secret salt (keep safe!)')
    parser.add_argument('--type', choices=['corporate', 'production', 'enterprise'], 
                        default='corporate', help='License type')
    parser.add_argument('--expiry-months', type=int, default=12, 
                        help='License validity in months (for display purposes)')
    args = parser.parse_args()

    # Convert string type to enum
    license_type = LicenseType(args.type)
    
    # Generate the license hash
    license_hash = generate_license(args.secret, args.client_id, license_type)
    
    # Calculate expiry date for display
    current_month = int(datetime.now().strftime("%m"))
    current_year = int(datetime.now().strftime("%Y"))
    expiry_month = ((current_month - 1 + args.expiry_months) % 12) + 1
    expiry_year = current_year + ((current_month - 1 + args.expiry_months) // 12)
    
    # Print license information
    print(f"\n{'=' * 80}")
    print(f"LICENSE INFORMATION: {license_type.value.upper()}")
    print(f"{'=' * 80}")
    print(f"Client ID:       {args.client_id}")
    print(f"License Type:    {license_type.value.title()}")
    print(f"Generated Date:  {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Expires:         {expiry_year}-{expiry_month:02d}-01")
    print(f"{'=' * 80}")
    print(f"\nLICENSE KEY:\n{license_hash}")
    print(f"\n{'=' * 80}")
    print("HOW TO USE:")
    print(f"1. Set environment variable: export MRZKELP_LICENSE_KEY='{license_hash}'")
    print(f"2. Set environment variable: export MRZKELP_CLIENT_ID='{args.client_id}'")
    print(f"3. Set environment variable: export MRZKELP_SECRET='{args.secret}'")
    print(f"4. Run your AGI Toolkit app normally")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()
