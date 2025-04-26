import hashlib
import enum
import platform
import socket
import os
from datetime import datetime

class UsageType(enum.Enum):
    DEVELOPMENT = "development"  # Open source, building, testing
    CORPORATE = "corporate"      # Commercial/enterprise usage
    PRODUCTION = "production"    # Deployed in production

def mock_root(value, mock_degree=3):
    """Apply non-linear transformation to obfuscate validation checks"""
    return value ** (1.0 / mock_degree)

def zk_entropy_threshold(activity_metrics):
    """
    Calculate entropy from activity metrics
    activity_metrics: list of numeric metrics (e.g., [api_calls, active_users, memory_usage])
    """
    # Apply weighted sum - give more importance to user count and throughput
    if len(activity_metrics) >= 3:
        # When we have at least 3 metrics: [transactions, users, memory]
        # Weight users more heavily for corporate detection
        weights = [1.0, 2.5, 1.0]  # Emphasize user count as key corporate indicator
        weighted_sum = sum(m * w for m, w in zip(activity_metrics, weights))
        entropy = weighted_sum / sum(weights[0:len(activity_metrics)])
    else:
        # Simple average for fewer metrics
        entropy = sum(activity_metrics) / max(len(activity_metrics), 1)
    return entropy

def detect_environment():
    """Detect if running in corporate/production environment based on system signals"""
    # These signals can help detect corporate deployments
    signals = {
        "hostname": socket.gethostname(),
        "domain": socket.getfqdn(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "username": os.environ.get("USER", "unknown"),
    }
    
    # Check for common corporate/cloud environment indicators
    corporate_indicators = [
        ".corp.", "enterprise", "prod", "aws", "azure", "gcp", 
        "cloud", "cluster", "k8s", "docker"
    ]
    
    # Count how many corporate indicators are present in the signals
    indicator_count = 0
    for value in signals.values():
        for indicator in corporate_indicators:
            if indicator in str(value).lower():
                indicator_count += 1
    
    return indicator_count

def generate_expected_license(secret, client_id):
    """Generate license hash that only you (with the secret) can produce"""
    # Add more entropy and complexity to make license format unique to your system
    timestamp = datetime.now().strftime("%Y%m")
    system_id = platform.node()
    raw = f"{secret}:{client_id}:{timestamp}:{system_id}"
    return hashlib.sha512(raw.encode()).hexdigest()

def validate_license(activity_metrics, license_key=None, client_id=None, secret="YourSecretSalt", 
                     dev_threshold=100, corp_threshold=250):
    """
    Validates if the current usage requires a license
    
    Args:
        activity_metrics: List of metrics [transactions, users, memory_usage, etc]
        license_key: License key provided by the user
        client_id: Client identifier (email, company name)
        secret: Secret salt for license generation (only you should know this)
        dev_threshold: Threshold for development/open-source usage
        corp_threshold: Threshold for corporate/production usage
    
    Returns:
        (True/False, message)
    """
    # Calculate the entropy and apply mock root transformation
    entropy = zk_entropy_threshold(activity_metrics)
    transformed = mock_root(entropy)
    
    # Get environmental signals (adds more security against spoofing)
    env_indicators = detect_environment()
    
    # Determine usage type based on metrics and environment
    usage_type = UsageType.DEVELOPMENT
    if transformed > corp_threshold or env_indicators >= 3:
        usage_type = UsageType.PRODUCTION
    elif transformed > dev_threshold:
        usage_type = UsageType.CORPORATE
    
    # Enforce license for corporate/production usage
    if usage_type == UsageType.DEVELOPMENT:
        return True, f"Free Tier OK - Development/Open-Source Usage (entropy: {entropy:.1f})"
    
    # Corporate or Production use requires a license
    if license_key and client_id:
        expected_key = generate_expected_license(secret, client_id)
        if license_key == expected_key:
            return True, f"License Validated - {usage_type.value.title()} Use Authorized"
        else:
            return False, f"Invalid License Key for {usage_type.value.title()} Use"
    else:
        return False, f"License Required - {usage_type.value.title()} Usage Detected (entropy: {entropy:.1f})"

