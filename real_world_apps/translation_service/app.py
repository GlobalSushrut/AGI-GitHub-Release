#!/usr/bin/env python3
"""
Language Translation Service
---------------------------

A real-world application that demonstrates how to use the AGI Toolkit
to build a multilingual translation service.

Features:
- Text translation between multiple languages
- Automatic language detection
- Contextual translation
- Domain-specific translation options
- Translation memory for improved consistency
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple
import re
import json
from datetime import datetime

# Add the parent directory to path so we can import the AGI Toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the ASI helper module and AGI Toolkit
from real_world_apps.asi_helper import initialize_asi_components, translate_text
from agi_toolkit import AGIAPI


class TranslationService:
    """A language translation service using AGI Toolkit."""
    
    # Supported languages with their codes
    SUPPORTED_LANGUAGES = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "dutch": "nl",
        "russian": "ru",
        "japanese": "ja",
        "chinese": "zh",
        "arabic": "ar",
        "korean": "ko",
        "hindi": "hi"
    }
    
    # Domain-specific translation options
    DOMAINS = [
        "general",
        "technical",
        "medical",
        "legal",
        "business",
        "academic"
    ]
    
    def __init__(self):
        """
        Initialize the translation service.
        """
        # Configure logging
        self.logger = logging.getLogger("TranslationService")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing Translation Service")
        
        # Initialize real ASI components
        initialize_asi_components()
        
        # Set environment variable to ensure interface uses real components
        os.environ['USE_REAL_ASI'] = 'true'
        
        # Initialize the AGI Toolkit API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        # Initialize translation memory
        self.translation_memory = {}
        self.load_translation_memory()
        
        self.logger.info("Translation Service initialized")
    
    def load_translation_memory(self):
        """Load translation memory from API memory if available."""
        try:
            memory_key = "translation_memory"
            memory_data = self.api.retrieve_data(memory_key)
            
            if memory_data:
                self.translation_memory = memory_data
                self.logger.info(f"Loaded {len(self.translation_memory)} entries from translation memory")
            else:
                self.logger.info("No translation memory found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading translation memory: {str(e)}")
    
    def save_translation_memory(self):
        """Save translation memory to API memory."""
        try:
            memory_key = "translation_memory"
            self.api.store_data(memory_key, self.translation_memory)
            self.logger.info(f"Saved {len(self.translation_memory)} entries to translation memory")
        except Exception as e:
            self.logger.error(f"Error saving translation memory: {str(e)}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            Language code (e.g., 'en', 'es', etc.)
        """
        self.logger.info(f"Detecting language for text: {text[:50]}...")
        
        # Try using ASI for language detection first
        if self.api.has_asi:
            try:
                # Import ASI helper for language detection
                from real_world_apps.asi_helper import process_with_asi
                
                result = process_with_asi(self.api, {
                    "task": "detect_language",
                    "text": text[:1000]  # Limit text size
                })
                
                if isinstance(result, dict) and result.get("success", False) and "result" in result:
                    lang_data = result["result"]
                    
                    # Extract language code based on different possible formats
                    if isinstance(lang_data, dict):
                        # Try different field names that might contain language info
                        for field in ["language", "language_code", "code", "lang"]:
                            if field in lang_data:
                                lang_code = lang_data[field].lower()
                                # Validate the code
                                for _, code in self.SUPPORTED_LANGUAGES.items():
                                    if code == lang_code:
                                        return code
                    elif isinstance(lang_data, str):
                        lang_code = lang_data.lower()
                        # Check if it's a language name or code
                        for lang, code in self.SUPPORTED_LANGUAGES.items():
                            if code == lang_code or lang.lower() == lang_code:
                                return code
            except Exception as e:
                self.logger.error(f"Error detecting language with ASI: {str(e)}")
        
        # Fall back to MOCK-LLM if available
        if self.api.has_mock_llm:
            # Use MOCK-LLM for language detection
            prompt = f"Detect the language of the following text. Respond with only the language code (e.g., 'en', 'es', 'fr', etc.):\n\n{text}"
            
            response = self.api.generate_text(prompt)
            
            # Extract the language code from the response
            language_code = response.strip().lower()
            
            # Validate the code is in our supported languages
            for lang, code in self.SUPPORTED_LANGUAGES.items():
                if code == language_code or lang.lower() == language_code:
                    return code
            
            # If no match, default to English
            return "en"
        else:
            # Fallback language detection using frequency analysis of common words
            # This is a simple approximation for demonstration purposes
            text = text.lower()
            
            language_scores = {}
            
            # Word frequency analysis for some common languages
            # English
            english_words = ["the", "and", "is", "in", "to", "of", "a", "for", "that", "it"]
            en_score = sum(1 for word in english_words if f" {word} " in f" {text} ")
            language_scores["en"] = en_score
            
            # Spanish
            spanish_words = ["el", "la", "los", "las", "un", "una", "es", "en", "por", "que"]
            es_score = sum(1 for word in spanish_words if f" {word} " in f" {text} ")
            language_scores["es"] = es_score
            
            # French
            french_words = ["le", "la", "les", "un", "une", "des", "est", "dans", "pour", "que"]
            fr_score = sum(1 for word in french_words if f" {word} " in f" {text} ")
            language_scores["fr"] = fr_score
            
            # German
            german_words = ["der", "die", "das", "ein", "eine", "ist", "für", "und", "mit", "von"]
            de_score = sum(1 for word in german_words if f" {word} " in f" {text} ")
            language_scores["de"] = de_score
            
            # Find the language with the highest score
            highest_score = 0
            detected_lang = "en"  # Default to English
            
            for lang, score in language_scores.items():
                if score > highest_score:
                    highest_score = score
                    detected_lang = lang
            
            self.logger.info(f"Detected language: {detected_lang}")
            return detected_lang
    
    def get_language_name(self, language_code: str) -> str:
        """Get the language name from its code."""
        for name, code in self.SUPPORTED_LANGUAGES.items():
            if code == language_code.lower():
                return name.capitalize()
        return "Unknown"
    
    def translate_text(self, 
                      text: str, 
                      source_lang: Optional[str] = None, 
                      target_lang: str = "en", 
                      domain: str = "general",
                      context: Optional[str] = None) -> Dict:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (auto-detect if None)
            target_lang: Target language code
            domain: Domain for specialized translation
            context: Additional context for translation
            
        Returns:
            Dictionary with translation results
        """
        # Validate inputs
        if not text.strip():
            return {
                "success": False,
                "error": "Empty text provided",
                "source_lang": source_lang,
                "target_lang": target_lang
            }
        
        # Normalize language codes
        target_lang = target_lang.lower()
        if source_lang:
            source_lang = source_lang.lower()
        
        # Validate target language
        if target_lang not in self.SUPPORTED_LANGUAGES.values():
            target_match = None
            for name, code in self.SUPPORTED_LANGUAGES.items():
                if name.lower() == target_lang:
                    target_match = code
                    break
            
            if target_match:
                target_lang = target_match
            else:
                return {
                    "success": False,
                    "error": f"Unsupported target language: {target_lang}",
                    "supported_languages": list(self.SUPPORTED_LANGUAGES.keys())
                }
        
        # Validate domain
        if domain not in self.DOMAINS:
            return {
                "success": False,
                "error": f"Unsupported domain: {domain}",
                "supported_domains": self.DOMAINS
            }
        
        # Detect source language if not provided
        if not source_lang:
            source_lang = self.detect_language(text)
        
        # Skip translation if source and target languages are the same
        if source_lang == target_lang:
            return {
                "success": True,
                "text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": domain,
                "source_confidence": 1.0
            }
        
        # Check translation memory for exact matches
        memory_key = f"{source_lang}|{target_lang}|{domain}|{text.strip()}"
        if memory_key in self.translation_memory:
            self.logger.info("Found exact match in translation memory")
            memory_entry = self.translation_memory[memory_key]
            return {
                "success": True,
                "text": text,
                "translated_text": memory_entry["translation"],
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": domain,
                "source_confidence": 1.0,
                "from_memory": True,
                "last_used": memory_entry.get("last_used", "unknown")
            }
        
        # Perform the translation
        self.logger.info(f"Translating from {source_lang} to {target_lang} (domain: {domain})")
        
        # Try using ASI for translation first
        if self.api.has_asi:
            try:
                # Use the translation helper function
                source_lang_name = self.get_language_name(source_lang)
                target_lang_name = self.get_language_name(target_lang)
                
                # Prepare context information including domain
                context_info = f"Domain: {domain}"
                if context:
                    context_info += f". Additional context: {context}"
                
                # Perform the translation using the ASI helper
                translated_text = translate_text(self.api, text, source_lang, target_lang)
                
                if translated_text and translated_text != text:  # Check that we got a valid translation
                    # Save to translation memory
                    timestamp = datetime.now().isoformat()
                    self.translation_memory[memory_key] = {
                        "translation": translated_text,
                        "created": timestamp,
                        "last_used": timestamp
                    }
                    self.save_translation_memory()
                    
                    return {
                        "success": True,
                        "text": text,
                        "translated_text": translated_text,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "domain": domain,
                        "source_confidence": 0.95,
                        "engine": "ASI"
                    }
            except Exception as e:
                self.logger.error(f"Error translating with ASI: {str(e)}")
        
        # Fall back to MOCK-LLM if available
        if self.api.has_mock_llm:
            # Create a prompt for the translation
            source_lang_name = self.get_language_name(source_lang)
            target_lang_name = self.get_language_name(target_lang)
            
            prompt = f"""Translate the following text from {source_lang_name} to {target_lang_name}.
            Domain: {domain}
            """
            
            if context:
                prompt += f"\nContext: {context}\n\n"
            
            prompt += f"Text to translate: {text}\n\nTranslation:"
            
            # Generate the translation
            translated_text = self.api.generate_text(prompt)
            
            # Clean up the response
            translated_text = translated_text.strip()
            
            # Save to translation memory
            timestamp = datetime.now().isoformat()
            self.translation_memory[memory_key] = {
                "translation": translated_text,
                "created": timestamp,
                "last_used": timestamp
            }
            self.save_translation_memory()
            
            return {
                "success": True,
                "text": text,
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": domain,
                "source_confidence": 0.9,
                "engine": "MOCK-LLM"
            }
        else:
            # Fallback translation for demonstration purposes
            source_lang_name = self.get_language_name(source_lang)
            target_lang_name = self.get_language_name(target_lang)
            
            # Simple dictionary-based replacements for demo
            demo_translations = {
                "en|es": {
                    "hello": "hola",
                    "goodbye": "adiós",
                    "thank you": "gracias",
                    "please": "por favor",
                    "yes": "sí",
                    "no": "no",
                    "how are you": "cómo estás",
                    "what is your name": "cómo te llamas",
                    "my name is": "me llamo",
                    "welcome": "bienvenido",
                },
                "en|fr": {
                    "hello": "bonjour",
                    "goodbye": "au revoir",
                    "thank you": "merci",
                    "please": "s'il vous plaît",
                    "yes": "oui",
                    "no": "non",
                    "how are you": "comment allez-vous",
                    "what is your name": "comment vous appelez-vous",
                    "my name is": "je m'appelle",
                    "welcome": "bienvenue",
                },
                "en|de": {
                    "hello": "hallo",
                    "goodbye": "auf wiedersehen",
                    "thank you": "danke",
                    "please": "bitte",
                    "yes": "ja",
                    "no": "nein",
                    "how are you": "wie geht es dir",
                    "what is your name": "wie heißt du",
                    "my name is": "ich heiße",
                    "welcome": "willkommen",
                }
            }
            
            # Prepare simulated translation
            trans_key = f"{source_lang}|{target_lang}"
            if trans_key in demo_translations:
                word_map = demo_translations[trans_key]
                translated_text = text
                
                # Apply word substitutions
                for en_word, trans_word in word_map.items():
                    pattern = r'\b' + re.escape(en_word) + r'\b'
                    translated_text = re.sub(pattern, trans_word, translated_text, flags=re.IGNORECASE)
                
                result = {
                    "success": True,
                    "text": text,
                    "translated_text": translated_text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "domain": domain,
                    "source_confidence": 0.7,
                    "note": "Using simplified fallback translation"
                }
            else:
                # For unsupported language pairs, just return a message
                result = {
                    "success": True,
                    "text": text,
                    "translated_text": f"[Translation from {source_lang_name} to {target_lang_name} not available in fallback mode]",
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "domain": domain,
                    "source_confidence": 0.5,
                    "note": "Using fallback translation mode, limited functionality"
                }
            
            return result
    
    def batch_translate(self, 
                        texts: List[str], 
                        source_lang: Optional[str] = None, 
                        target_lang: str = "en", 
                        domain: str = "general") -> List[Dict]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code (auto-detect if None)
            target_lang: Target language code
            domain: Domain for specialized translation
            
        Returns:
            List of translation results
        """
        self.logger.info(f"Batch translating {len(texts)} texts from {source_lang} to {target_lang}")
        
        results = []
        for text in texts:
            result = self.translate_text(text, source_lang, target_lang, domain)
            results.append(result)
        
        return results
    
    def get_supported_languages(self) -> List[Dict]:
        """Get list of supported languages with their codes."""
        return [
            {"name": name.capitalize(), "code": code}
            for name, code in self.SUPPORTED_LANGUAGES.items()
        ]
    
    def get_supported_domains(self) -> List[str]:
        """Get list of supported translation domains."""
        return self.DOMAINS


def display_translation_result(result: Dict):
    """Display a translation result in a user-friendly format."""
    print("\n" + "="*80)
    print("TRANSLATION RESULT".center(80))
    print("="*80 + "\n")
    
    if not result.get("success", False):
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        if "supported_languages" in result:
            print("\nSupported languages:")
            for lang in result["supported_languages"]:
                print(f"  - {lang.capitalize()}")
        if "supported_domains" in result:
            print("\nSupported domains:")
            for domain in result["supported_domains"]:
                print(f"  - {domain.capitalize()}")
        return
    
    source_lang = result.get("source_lang", "unknown")
    target_lang = result.get("target_lang", "unknown")
    
    source_lang_name = None
    target_lang_name = None
    
    for name, code in TranslationService.SUPPORTED_LANGUAGES.items():
        if code == source_lang:
            source_lang_name = name.capitalize()
        if code == target_lang:
            target_lang_name = name.capitalize()
    
    print(f"📝 Original Text ({source_lang_name}):")
    print(f"{result.get('text', '')}\n")
    
    print(f"🔤 Translated Text ({target_lang_name}):")
    print(f"{result.get('translated_text', '')}\n")
    
    print(f"Domain: {result.get('domain', 'general').capitalize()}")
    
    if "from_memory" in result and result["from_memory"]:
        print("\nℹ️ Retrieved from translation memory")
        if "last_used" in result:
            print(f"Last used: {result['last_used']}")
    
    if "note" in result:
        print(f"\nNote: {result['note']}")
    
    print("="*80)


def display_batch_results(results: List[Dict]):
    """Display batch translation results."""
    print("\n" + "="*80)
    print("BATCH TRANSLATION RESULTS".center(80))
    print("="*80 + "\n")
    
    for i, result in enumerate(results, 1):
        print(f"Translation #{i}:")
        
        if not result.get("success", False):
            print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
            continue
        
        source_lang = result.get("source_lang", "unknown")
        target_lang = result.get("target_lang", "unknown")
        
        print(f"  📝 Original ({source_lang}): {result.get('text', '')}")
        print(f"  🔤 Translated ({target_lang}): {result.get('translated_text', '')}")
        print("")
    
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Language Translation Service")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--source", type=str, help="Source language code (auto-detect if not provided)")
    parser.add_argument("--target", type=str, required=True, help="Target language code")
    parser.add_argument("--domain", type=str, default="general", help="Domain for specialized translation")
    parser.add_argument("--context", type=str, help="Additional context for translation")
    parser.add_argument("--detect", type=str, help="Detect language of this text")
    parser.add_argument("--list_languages", action="store_true", help="List supported languages and domains")
    parser.add_argument("--batch", type=str, help="Path to file with texts for batch translation")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the translation service
    translator = TranslationService()
    
    # List supported languages
    if args.list_languages:
        print("\nSupported Languages:")
        languages = translator.get_supported_languages()
        for lang in languages:
            print(f"  - {lang['name']} ({lang['code']})")
        
        print("\nSupported Domains:")
        domains = translator.get_supported_domains()
        for domain in domains:
            print(f"  - {domain.capitalize()}")
        
        return
    
    # Batch translation
    if args.batch:
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if not texts:
                print("Error: Batch file is empty or contains no valid text lines")
                return
            
            results = translator.batch_translate(
                texts=texts,
                source_lang=args.source,
                target_lang=args.target,
                domain=args.domain
            )
            
            display_batch_results(results)
            return
        except Exception as e:
            print(f"Error processing batch file: {str(e)}")
            return
    
    # Single translation
    if args.text:
        result = translator.translate_text(
            text=args.text,
            source_lang=args.source,
            target_lang=args.target,
            domain=args.domain,
            context=args.context
        )
        
        display_translation_result(result)
    else:
        # Use some example texts if no input is provided
        print("No text provided. Showing example translations:")
        
        example_texts = [
            "Hello, how are you today?",
            "Thank you for your help.",
            "I would like to book a hotel room, please.",
            "Where is the nearest train station?",
            "The weather is nice today."
        ]
        
        for text in example_texts:
            result = translator.translate_text(
                text=text,
                source_lang="en",
                target_lang="es",  # Spanish
                domain="general"
            )
            
            display_translation_result(result)


if __name__ == "__main__":
    main()
