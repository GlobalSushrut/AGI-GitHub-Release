"""
Encryption Loader for ASI Infrastructure
This loader is the only unencrypted component and handles decryption of the rest of the system.
DO NOT MODIFY THIS FILE OR THE SYSTEM WILL FAIL TO LOAD.
"""

import os
import sys
import base64
import hashlib
import importlib.util
import types
import time
from pathlib import Path

# Store the encryption state
_ENCRYPTION_STATE = None
_RUNTIME_KEY = None
_DECRYPTION_ENABLED = False
_ENCRYPTED_MODULES_CACHE = {}

# Path to encryption state file
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_STATE_FILE = _PROJECT_ROOT / "unreal_asi/security/encryption_state.bin"
_MANIFEST_FILE = _PROJECT_ROOT / "unreal_asi/security/encryption_manifest.json"
_LICENSE_FILE = _PROJECT_ROOT / "unreal_asi/security/license_key.txt"

def _derive_key_from_license(license_key):
    """Derive encryption key from license key"""
    if not license_key:
        return None
    
    # Extract components from license key
    try:
        parts = license_key.strip().split('-')
        if len(parts) != 3:
            return None
        
        # Use license components to derive a stable key
        key_material = license_key.encode()
        return hashlib.sha256(key_material).hexdigest()[:16]
    except:
        return None

def _setup_encryption_state():
    """Initialize the encryption state without importing framework"""
    global _ENCRYPTION_STATE
    
    if _ENCRYPTION_STATE is None:
        try:
            # Create a minimal encryption state for bootstrap
            key_material = _RUNTIME_KEY.encode() if _RUNTIME_KEY else b"default"
            key = hashlib.sha256(key_material).digest()[:16]
            
            # Basic state to allow initial decryption
            _ENCRYPTION_STATE = {
                "key": key,
                "version": "1.0",
                "timestamp": int(time.time())
            }
            
            return True
        except Exception as e:
            print(f"Error bootstrapping encryption: {e}")
            return False
    
    return True

def load_encrypted_module(encrypted_path, module_name=None):
    """Load an encrypted module using the current encryption state"""
    global _ENCRYPTION_STATE, _RUNTIME_KEY, _DECRYPTION_ENABLED, _ENCRYPTED_MODULES_CACHE
    
    if not _DECRYPTION_ENABLED:
        #print(f"Error: Decryption not enabled. Cannot load {encrypted_path}")
        return None
    
    if not module_name:
        module_name = Path(encrypted_path).stem.replace('.py', '')
    
    # Check cache first
    if module_name in _ENCRYPTED_MODULES_CACHE:
        return _ENCRYPTED_MODULES_CACHE[module_name]
    
    try:
        # Check if module is already loaded (original or .enc version)
        if module_name in sys.modules:
            return sys.modules[module_name]
        
        # Find actual file path
        if isinstance(encrypted_path, str) and not os.path.exists(encrypted_path):
            # Convert module name to path
            parts = module_name.split('.')
            rel_path = '/'.join(parts)
            
            # Try different possible locations
            candidates = [
                f"{_PROJECT_ROOT}/{rel_path}.py.enc",
                f"{_PROJECT_ROOT}/{rel_path}.enc",
                f"{_PROJECT_ROOT}/unreal_asi/{rel_path}.py.enc",
                f"{_PROJECT_ROOT}/unreal_asi/{rel_path}.enc"
            ]
            
            for candidate in candidates:
                if os.path.exists(candidate):
                    encrypted_path = candidate
                    break
        
        # Load encrypted content
        with open(encrypted_path, 'rb') as f:
            encrypted_content = f.read()
        
        # Basic decryption (XOR with key)
        # In production, this would use proper decryption algorithms
        key = _ENCRYPTION_STATE["key"]
        decrypted = bytearray(len(encrypted_content))
        
        for i in range(len(encrypted_content)):
            decrypted[i] = encrypted_content[i] ^ key[i % len(key)]
        
        # Create module
        spec = importlib.machinery.ModuleSpec(module_name, None)
        module = types.ModuleType(spec.name)
        module.__file__ = encrypted_path
        module.__spec__ = spec
        
        # Execute decrypted code in module context
        exec(decrypted, module.__dict__)
        
        # Cache the module
        sys.modules[module_name] = module
        _ENCRYPTED_MODULES_CACHE[module_name] = module
        
        return module
    
    except Exception as e:
        print(f"Error loading encrypted module {module_name}: {e}")
        return None

def activate_license(license_key):
    """Activate the ASI system with a license key"""
    global _RUNTIME_KEY, _DECRYPTION_ENABLED
    
    _RUNTIME_KEY = license_key
    
    # Try to load the encryption framework
    try:
        # Save license to file for persistence
        with open(_LICENSE_FILE, 'w') as f:
            f.write(license_key)
        
        # Mark decryption as enabled
        _DECRYPTION_ENABLED = True
        
        # Initialize encryption state
        success = _setup_encryption_state()
        if not success:
            _DECRYPTION_ENABLED = False
            print("License activation failed: could not initialize encryption")
            return False
        
        print("License activated successfully")
        return True
    
    except Exception as e:
        _DECRYPTION_ENABLED = False
        print(f"License activation failed: {e}")
        return False

def is_license_active():
    """Check if a valid license is currently active"""
    global _DECRYPTION_ENABLED
    return _DECRYPTION_ENABLED

# Monkey patch the import system to handle encrypted modules
_original_import = __import__

def _encrypted_import(name, globals=None, locals=None, fromlist=(), level=0):
    # If the name starts with 'unreal_asi', check if it might be encrypted
    if name.startswith('unreal_asi') and _DECRYPTION_ENABLED:
        # Try to find the encrypted module
        module = load_encrypted_module(name)
        if module is not None:
            # If fromlist is specified, return the module itself
            if fromlist:
                return module
                
            # Otherwise return the top-level package
            parts = name.split('.')
            if len(parts) > 1:
                return sys.modules.get(parts[0])
            return module
    
    # Try normal import
    try:
        return _original_import(name, globals, locals, fromlist, level)
    except (ImportError, ModuleNotFoundError) as e:
        # If module not found, check if it's an encrypted module
        if _DECRYPTION_ENABLED and name.startswith('unreal_asi'):
            # Convert module name to path
            mod_path = name.replace('.', '/')
            
            # Look for encrypted version
            enc_path = f"{_PROJECT_ROOT}/{mod_path}.py.enc"
            
            if os.path.exists(enc_path):
                # Load encrypted module
                module = load_encrypted_module(enc_path, name)
                if module:
                    # If fromlist is specified, return requested attributes
                    if fromlist:
                        return module
                    # Otherwise return the top-level package
                    parts = name.split('.')
                    if len(parts) > 1:
                        return sys.modules.get(parts[0])
                    return module
        
        # If we get here, the module is not found or decryption failed
        raise e

# Replace built-in __import__ function
sys.meta_path.insert(0, type('EncryptionFinder', (), {'find_spec': lambda slf, fullname, path, target=None: 
    importlib.machinery.ModuleSpec(fullname, None) if fullname.startswith('unreal_asi') and _DECRYPTION_ENABLED else None,
    'find_module': lambda slf, fullname, path=None: 
    slf if fullname.startswith('unreal_asi') and _DECRYPTION_ENABLED else None,
    'load_module': lambda slf, fullname: load_encrypted_module(fullname)
})())

# Auto-load license on startup if available
if os.path.exists(_LICENSE_FILE):
    try:
        with open(_LICENSE_FILE, 'r') as f:
            saved_license = f.read().strip()
        if saved_license:
            activate_license(saved_license)
    except:
        pass
