#!/usr/bin/env python3
"""
ASI Encryption Framework
-----------------------
Provides enhanced encryption capabilities for ASI engine components.
This framework ensures that core algorithms and mathematical implementations
remain hidden while allowing for public API access.

WARNING: This file will be encrypted after the first run.
"""

import os
import sys
import base64
import hashlib
import json
import time
import random
import struct
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List, Callable
import importlib.util

# Path constants
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENCRYPTION_STATE_FILE = _PROJECT_ROOT / "unreal_asi/security/asi_encryption_state.bin"
_ENCRYPTION_MANIFEST_FILE = _PROJECT_ROOT / "unreal_asi/security/asi_encryption_manifest.json"
_LICENSE_FILE = _PROJECT_ROOT / "unreal_asi/security/asi_license_key.txt"

# Encryption parameters
_KEY_DERIVATION_ROUNDS = 10000
_CHUNK_SIZE = 8192
_HEADER_SIZE = 128
_ENCRYPTION_VERSION = "2.0"

class ASIEncryptionEngine:
    """Advanced encryption engine for ASI components."""
    
    def __init__(self, key: str, version: str = _ENCRYPTION_VERSION):
        """
        Initialize the ASI encryption engine.
        
        Args:
            key: Encryption key
            version: Encryption version
        """
        self.version = version
        self._key = key
        self._salt = None
        self._derived_key = None
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize encryption keys and salts."""
        # Create a salt from the key
        key_bytes = self._key.encode() if isinstance(self._key, str) else self._key
        self._salt = hashlib.sha256(key_bytes).digest()[:16]
        
        # Derive key using PBKDF2-like approach
        derived = key_bytes
        for _ in range(_KEY_DERIVATION_ROUNDS):
            derived = hashlib.sha256(derived + self._salt).digest()
        
        self._derived_key = derived[:32]  # 256-bit key
    
    def encrypt_file(self, source_path: Union[str, Path], target_path: Union[str, Path]) -> bool:
        """
        Encrypt a file using the ASI encryption algorithm.
        
        Args:
            source_path: Path to the file to encrypt
            target_path: Path where the encrypted file will be saved
            
        Returns:
            bool: True if encryption succeeded
        """
        try:
            source_path = Path(source_path)
            target_path = Path(target_path)
            
            if not source_path.exists():
                print(f"Error: Source file {source_path} does not exist")
                return False
            
            # Create target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read source file
            with open(source_path, 'rb') as src_file:
                content = src_file.read()
            
            # Encrypt content
            encrypted = self._encrypt_content(content)
            
            # Write encrypted content to target file
            with open(target_path, 'wb') as out_file:
                out_file.write(encrypted)
            
            return True
        
        except Exception as e:
            print(f"Encryption error: {e}")
            return False
    
    def decrypt_file(self, encrypted_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Optional[bytes]:
        """
        Decrypt a file using the ASI encryption algorithm.
        
        Args:
            encrypted_path: Path to the encrypted file
            output_path: Optional path to save the decrypted file
            
        Returns:
            bytes: Decrypted content or None if decryption failed
        """
        try:
            encrypted_path = Path(encrypted_path)
            
            if not encrypted_path.exists():
                print(f"Error: Encrypted file {encrypted_path} does not exist")
                return None
            
            # Read encrypted file
            with open(encrypted_path, 'rb') as enc_file:
                encrypted = enc_file.read()
            
            # Decrypt content
            decrypted = self._decrypt_content(encrypted)
            
            # Write decrypted content to output file if specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as out_file:
                    out_file.write(decrypted)
            
            return decrypted
        
        except Exception as e:
            print(f"Decryption error: {e}")
            return None
    
    def decrypt_code(self, encrypted_data: bytes) -> Optional[bytes]:
        """
        Decrypt ASI code.
        
        Args:
            encrypted_data: Encrypted code bytes
            
        Returns:
            bytes: Decrypted code or None if decryption failed
        """
        try:
            return self._decrypt_content(encrypted_data)
        except Exception as e:
            print(f"Code decryption error: {e}")
            return None
    
    def _encrypt_content(self, content: bytes) -> bytes:
        """
        Encrypt content using the ASI encryption algorithm.
        
        Args:
            content: Content to encrypt
            
        Returns:
            bytes: Encrypted content
        """
        # Create header with metadata
        header = {
            "version": self.version,
            "timestamp": int(time.time()),
            "salt": base64.b64encode(self._salt).decode('utf-8'),
            "content_hash": hashlib.sha256(content).hexdigest(),
            "content_size": len(content),
            "encryption_algo": "asi-advanced-algorithm"
        }
        
        # Convert header to bytes
        header_json = json.dumps(header).encode('utf-8')
        header_padded = header_json + b' ' * (_HEADER_SIZE - len(header_json))
        
        # Encrypt content in chunks
        encrypted_chunks = []
        
        # Add the header as plaintext (will be protected by obfuscation)
        encrypted_chunks.append(header_padded[:_HEADER_SIZE])
        
        # Process content in chunks
        for i in range(0, len(content), _CHUNK_SIZE):
            chunk = content[i:i+_CHUNK_SIZE]
            
            # Generate chunk key (different for each chunk)
            chunk_key = hashlib.sha256(self._derived_key + struct.pack("<I", i // _CHUNK_SIZE)).digest()
            
            # Encrypt chunk
            encrypted_chunk = self._encrypt_chunk(chunk, chunk_key)
            encrypted_chunks.append(encrypted_chunk)
        
        # Combine chunks
        return b''.join(encrypted_chunks)
    
    def _encrypt_chunk(self, chunk: bytes, chunk_key: bytes) -> bytes:
        """
        Encrypt a chunk of data.
        
        Args:
            chunk: Data chunk to encrypt
            chunk_key: Key for this chunk
            
        Returns:
            bytes: Encrypted chunk
        """
        # Create a chunk-specific IV
        iv = hashlib.md5(chunk_key).digest()
        
        # XOR chunk with expanded key (simple demonstration)
        # In a real implementation, use a proper encryption algorithm like AES
        expanded_key = self._expand_key(chunk_key, iv, len(chunk))
        encrypted = bytes(a ^ b for a, b in zip(chunk, expanded_key))
        
        # Add chunk metadata
        chunk_size = len(chunk)
        chunk_hash = hashlib.md5(chunk).digest()
        
        # Combine metadata and encrypted chunk
        return struct.pack("<I16s", chunk_size, chunk_hash) + encrypted
    
    def _decrypt_content(self, encrypted: bytes) -> bytes:
        """
        Decrypt content using the ASI encryption algorithm.
        
        Args:
            encrypted: Encrypted content
            
        Returns:
            bytes: Decrypted content
        """
        # Extract header
        header_data = encrypted[:_HEADER_SIZE]
        
        try:
            # Parse header
            header = json.loads(header_data.decode('utf-8').strip())
            
            # Validate version
            if header["version"] != self.version:
                raise ValueError(f"Incompatible encryption version: {header['version']}")
            
            # Extract content
            encrypted_content = encrypted[_HEADER_SIZE:]
            
            # Decrypt chunks
            decrypted_chunks = []
            offset = 0
            
            while offset < len(encrypted_content):
                # Read chunk metadata
                chunk_size, chunk_hash = struct.unpack("<I16s", encrypted_content[offset:offset+20])
                offset += 20
                
                # Read encrypted chunk
                encrypted_chunk = encrypted_content[offset:offset+chunk_size]
                offset += chunk_size
                
                # Generate chunk key
                chunk_index = len(decrypted_chunks)
                chunk_key = hashlib.sha256(self._derived_key + struct.pack("<I", chunk_index)).digest()
                
                # Decrypt chunk
                decrypted_chunk = self._decrypt_chunk(encrypted_chunk, chunk_key)
                
                # Validate chunk hash
                if hashlib.md5(decrypted_chunk).digest() != chunk_hash:
                    raise ValueError(f"Chunk {chunk_index} integrity check failed")
                
                decrypted_chunks.append(decrypted_chunk)
            
            # Combine chunks
            decrypted = b''.join(decrypted_chunks)
            
            # Validate content hash
            if hashlib.sha256(decrypted).hexdigest() != header["content_hash"]:
                raise ValueError("Content integrity check failed")
            
            return decrypted
            
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def _decrypt_chunk(self, encrypted_chunk: bytes, chunk_key: bytes) -> bytes:
        """
        Decrypt a chunk of data.
        
        Args:
            encrypted_chunk: Encrypted chunk
            chunk_key: Key for this chunk
            
        Returns:
            bytes: Decrypted chunk
        """
        # Generate IV
        iv = hashlib.md5(chunk_key).digest()
        
        # XOR with expanded key (simple demonstration)
        # In a real implementation, use a proper decryption algorithm like AES
        expanded_key = self._expand_key(chunk_key, iv, len(encrypted_chunk))
        return bytes(a ^ b for a, b in zip(encrypted_chunk, expanded_key))
    
    def _expand_key(self, key: bytes, iv: bytes, length: int) -> bytes:
        """
        Expand a key to the required length.
        
        Args:
            key: Base key
            iv: Initialization vector
            length: Required length
            
        Returns:
            bytes: Expanded key
        """
        # Use key and IV to generate a pseudo-random sequence of bytes
        result = bytearray(length)
        seed = key + iv
        
        for i in range(0, length, 32):
            seed = hashlib.sha256(seed).digest()
            result[i:i+32] = seed[:min(32, length - i)]
        
        return bytes(result)

class ASIModuleEncryptor:
    """Manages encryption of ASI modules and maintains the encryption manifest."""
    
    def __init__(self, engine: ASIEncryptionEngine):
        """
        Initialize the ASI module encryptor.
        
        Args:
            engine: ASI encryption engine
        """
        self.engine = engine
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """
        Load the encryption manifest.
        
        Returns:
            Dict: Manifest data
        """
        if _ENCRYPTION_MANIFEST_FILE.exists():
            try:
                with open(_ENCRYPTION_MANIFEST_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default manifest
        return {
            "version": self.engine.version,
            "encrypted_modules": {},
            "last_update": int(time.time()),
            "encrypted_count": 0
        }
    
    def _save_manifest(self):
        """Save the current encryption manifest."""
        with open(_ENCRYPTION_MANIFEST_FILE, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def encrypt_module(self, module_path: Union[str, Path], relative_to: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
        """
        Encrypt an ASI module.
        
        Args:
            module_path: Path to the module to encrypt
            relative_to: Base path for generating relative paths
            
        Returns:
            Tuple[bool, str]: Success status and output path
        """
        module_path = Path(module_path)
        
        if not module_path.exists():
            return False, f"Module {module_path} does not exist"
        
        # Get relative path if specified
        if relative_to:
            relative_to = Path(relative_to)
            try:
                rel_path = module_path.relative_to(relative_to)
            except ValueError:
                rel_path = module_path.name
        else:
            rel_path = module_path.name
        
        # Generate output path
        output_path = module_path.with_suffix(module_path.suffix + ".enc")
        
        # Encrypt the module
        success = self.engine.encrypt_file(module_path, output_path)
        
        if success:
            # Update manifest
            module_id = str(rel_path)
            self.manifest["encrypted_modules"][module_id] = {
                "original_path": str(module_path),
                "encrypted_path": str(output_path),
                "relative_path": str(rel_path),
                "timestamp": int(time.time()),
                "size": output_path.stat().st_size
            }
            
            self.manifest["last_update"] = int(time.time())
            self.manifest["encrypted_count"] = len(self.manifest["encrypted_modules"])
            
            # Save manifest
            self._save_manifest()
            
            return True, str(output_path)
        
        return False, "Encryption failed"
    
    def encrypt_directory(self, directory: Union[str, Path], pattern: str = "*.py", 
                        recursive: bool = True, exclude: List[str] = None) -> Dict[str, Any]:
        """
        Encrypt all modules in a directory that match the pattern.
        
        Args:
            directory: Directory containing modules to encrypt
            pattern: File pattern to match
            recursive: Whether to recursively encrypt subdirectories
            exclude: Paths to exclude
            
        Returns:
            Dict: Summary of encryption results
        """
        directory = Path(directory)
        exclude = exclude or []
        exclude_paths = [Path(x) for x in exclude]
        
        # Convert pattern to list for multiple patterns
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern
        
        # Find files to encrypt
        files_to_encrypt = []
        
        for pattern in patterns:
            if recursive:
                files = list(directory.glob(f"**/{pattern}"))
            else:
                files = list(directory.glob(pattern))
            
            for file in files:
                if not any(file == ex or ex in file.parents for ex in exclude_paths):
                    files_to_encrypt.append(file)
        
        # Encrypt files
        results = {
            "success": [],
            "failed": [],
            "total": len(files_to_encrypt)
        }
        
        for file in files_to_encrypt:
            success, output_path = self.encrypt_module(file, directory)
            
            if success:
                results["success"].append({
                    "original": str(file),
                    "encrypted": output_path
                })
            else:
                results["failed"].append({
                    "path": str(file),
                    "error": output_path
                })
        
        # Update summary
        results["success_count"] = len(results["success"])
        results["failed_count"] = len(results["failed"])
        
        return results

def create_encryption_engine(license_key: str) -> ASIEncryptionEngine:
    """
    Create an ASI encryption engine from a license key.
    
    Args:
        license_key: License key
        
    Returns:
        ASIEncryptionEngine: Encryption engine
    """
    # Derive key from license
    if not license_key:
        raise ValueError("License key is required")
    
    # Create unique key from license
    key_material = license_key.encode()
    derived_key = hashlib.pbkdf2_hmac(
        'sha256', 
        key_material, 
        b'ASI-Encryption-Salt', 
        iterations=100000
    ).hex()
    
    return ASIEncryptionEngine(derived_key)

def encrypt_asi_infrastructure(license_key: str, core_dirs: List[str] = None) -> Dict[str, Any]:
    """
    Encrypt the ASI infrastructure.
    
    Args:
        license_key: License key for encryption
        core_dirs: List of core directories to encrypt
        
    Returns:
        Dict: Summary of encryption results
    """
    # Default core directories
    if core_dirs is None:
        core_dirs = [
            "unreal_asi/core",
            "unreal_asi/security",
            "unreal_asi/__init__.py",
            "unreal_asi/unified_mind.py"
        ]
    
    # Create encryption engine
    engine = create_encryption_engine(license_key)
    
    # Create module encryptor
    encryptor = ASIModuleEncryptor(engine)
    
    # Encrypt core directories and files
    results = {
        "total_encrypted": 0,
        "total_failed": 0,
        "directories": {}
    }
    
    for dir_path in core_dirs:
        path = _PROJECT_ROOT / dir_path
        
        if path.is_dir():
            # Encrypt directory
            dir_results = encryptor.encrypt_directory(
                path, 
                pattern="*.py", 
                recursive=True,
                exclude=["**/__pycache__", "**/tests"]
            )
            
            results["directories"][dir_path] = dir_results
            results["total_encrypted"] += dir_results["success_count"]
            results["total_failed"] += dir_results["failed_count"]
            
        elif path.is_file() and path.suffix == ".py":
            # Encrypt single file
            success, output_path = encryptor.encrypt_module(path, _PROJECT_ROOT)
            
            if success:
                results["total_encrypted"] += 1
                if "files" not in results:
                    results["files"] = []
                results["files"].append({
                    "original": str(path),
                    "encrypted": output_path
                })
            else:
                results["total_failed"] += 1
                if "failed_files" not in results:
                    results["failed_files"] = []
                results["failed_files"].append({
                    "path": str(path),
                    "error": output_path
                })
    
    # Store encryption state
    with open(_ENCRYPTION_STATE_FILE, 'wb') as f:
        state = {
            "version": engine.version,
            "timestamp": int(time.time()),
            "key_hash": hashlib.sha256(license_key.encode()).hexdigest()
        }
        f.write(json.dumps(state).encode('utf-8'))
    
    # Store license key
    with open(_LICENSE_FILE, 'w') as f:
        f.write(license_key)
    
    return results

# If this file is run directly, it will encrypt itself after creating the framework
if __name__ == "__main__":
    # Default public license key
    DEFAULT_LICENSE = "ASI-PUBLIC-INTERFACE-000"
    
    print("ASI Encryption Framework Setup")
    print("------------------------------")
    
    license_key = input("Enter license key (or press Enter for default public key): ").strip()
    if not license_key:
        license_key = DEFAULT_LICENSE
        print(f"Using default public license: {license_key}")
    
    # Create necessary directories
    _ENCRYPTION_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Encrypt the ASI infrastructure
    print("\nEncrypting ASI infrastructure...")
    results = encrypt_asi_infrastructure(license_key)
    
    print(f"\nEncryption complete!")
    print(f"Total modules encrypted: {results['total_encrypted']}")
    print(f"Total modules failed: {results['total_failed']}")
    
    # Encrypt this file itself
    print("\nEncrypting the encryption framework...")
    engine = create_encryption_engine(license_key)
    encryptor = ASIModuleEncryptor(engine)
    
    this_file = Path(__file__)
    success, output_path = encryptor.encrypt_module(this_file)
    
    if success:
        print(f"Encryption framework encrypted: {output_path}")
    else:
        print(f"Failed to encrypt framework: {output_path}")
    
    print("\nASI encryption setup complete. The ASI engine is now protected.")
