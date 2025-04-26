# Mock Root ZK Entropy Licensing Protocol (MR-ZKELP)

## What This Solves

This licensing system implements Umesh's open+commercial licensing model:

- **100% Free for Open Source**: Anyone can build/test/develop with the AGI toolkit
- **License Required for Corporate**: Automatic detection of enterprise/production usage
- **You Control Who Gets Licenses**: Only you can generate valid license hashes

## How It Works

1. The system monitors usage metrics and environment signals
2. Development/open-source use is automatically permitted
3. Corporate/production usage is detected and requires a license you provide
4. Mathematical proof ensures they can't bypass the validation

---

## Directory Contents

- **license_validator.py**: Import in any app to enforce the licensing model
- **license_generator.py**: CLI tool for generating commercial license hashes (for you only)

---

## Usage

### 1. Validate License in Your App

```python
from infra.licensing.license_validator import validate_license

activity_metrics = [api_calls, active_users, memory_usage]
result, msg = validate_license(activity_metrics, license_key=provided_key, client_id="client@example.com", secret="YourSecretSalt")
if not result:
    print(msg)
    exit(1)
```
- If usage is below the threshold, runs free.
- If above, requires a valid license hash (which only you can issue).

#### ⚡ How to Run Your App with Licensing

Always run from the project root with PYTHONPATH set:

```bash
PYTHONPATH=. python3 real_world_apps/banking/app.py ...
```

If you see an import error for 'infra', this is the fix.

### 2. Generate a License Hash

```shell
python3 license_generator.py --client-id client@example.com --secret YourSecretSalt
```
- Output is a license hash for that client.
- Share only with paid/commercial users.

---

## How It Works
- **Mock Root + ZK Entropy**: Scrambles usage metrics so thresholds can't be easily faked.
- **License Hash**: SHA-512 of (secret + client_id). Only you know the secret.
- **Easy to integrate**: Add to any Python app.

---

## Future Extensions
- ZK Proof-based validation
- Pay-as-you-go metering
- Multiple license levels (change the threshold)

---

**Invented by Umesh, implemented by Cascade AI.**
