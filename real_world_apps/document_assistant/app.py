#!/usr/bin/env python3
"""
Smart Document Assistant
-----------------------

A real-world application that demonstrates how to use the AGI Toolkit
to build a document assistant that can analyze, categorize, and extract 
information from documents.

Features:
- Document categorization
- Entity extraction (people, organizations, dates)
- Action item identification
- Document summary and insights
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Any, Optional
import re

# Add the parent directory to path so we can import the AGI Toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the ASI helper module and AGI Toolkit
from real_world_apps.asi_helper import initialize_asi_components, analyze_document, extract_key_points
from agi_toolkit import AGIAPI

class DocumentAssistant:
    """A smart assistant for document processing using AGI Toolkit."""
    
    def __init__(self):
        """Initialize the document assistant."""
        # Configure logging
        self.logger = logging.getLogger("DocumentAssistant")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing Document Assistant")
        
        # Initialize real ASI components
        initialize_asi_components()
        
        # Set environment variable to ensure interface uses real components
        os.environ['USE_REAL_ASI'] = 'true'
        
        # Initialize the AGI Toolkit API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        self.logger.info("Document Assistant initialized")
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """
        Process a document and extract useful information.
        
        Args:
            text: The document text content
            
        Returns:
            Dictionary containing extracted information and analysis
        """
        self.logger.info(f"Processing document ({len(text)} characters)")
        
        # Store document in memory
        self.api.store_data("current_document", {
            "content": text,
            "length": len(text)
        })
        
        # Get document category
        category = self._categorize_document(text)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Identify action items
        action_items = self._identify_action_items(text)
        
        # Generate summary
        summary = self._generate_summary(text)
        
        # Extract insights
        insights = self._extract_insights(text, category)
        
        # Create the result
        result = {
            "category": category,
            "entities": entities,
            "action_items": action_items,
            "summary": summary,
            "insights": insights,
            "stats": {
                "word_count": len(text.split()),
                "character_count": len(text)
            }
        }
        
        # Store the result in memory
        self.api.store_data("document_analysis_result", result)
        
        return result
    
    def _categorize_document(self, text: str) -> str:
        """Categorize the document based on its content."""
        self.logger.info("Categorizing document")
        
        # Use ASI for advanced categorization if available
        if self.api.has_asi:
            # Import ASI helper for document analysis
            from real_world_apps.asi_helper import process_with_asi
            
            # Process with real ASI engine
            result = process_with_asi(self.api, {
                "task": "categorize_document",
                "content": text[:5000]  # Limit content size
            })
            
            if isinstance(result, dict) and result.get("success", False) and "result" in result:
                category_data = result["result"]
                # Extract category from various result formats
                if isinstance(category_data, dict):
                    # Try different field names that might contain category info
                    for field in ["category", "document_type", "type", "classification"]:
                        if field in category_data:
                            return category_data[field]
                elif isinstance(category_data, str):
                    return category_data
        
        # Fallback categorization using keyword matching
        categories = {
            "business": ["company", "business", "market", "revenue", "profit", "financial"],
            "technical": ["technical", "technology", "software", "hardware", "system", "data"],
            "legal": ["legal", "law", "contract", "agreement", "clause", "party", "rights"],
            "academic": ["research", "study", "analysis", "hypothesis", "conclusion", "findings"],
            "email": ["email", "mail", "send", "receive", "attachment", "reply", "forward"],
            "report": ["report", "summary", "overview", "results", "findings", "recommendation"]
        }
        
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in categories.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            category_scores[category] = score
        
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        # If no clear category, default to "general"
        if best_category[1] == 0:
            return "general"
        
        return best_category[0]
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from the document."""
        self.logger.info("Extracting entities")
        
        # Try using real ASI for entity extraction first
        if self.api.has_asi:
            try:
                # Import ASI helper for entity extraction
                from real_world_apps.asi_helper import process_with_asi
                
                result = process_with_asi(self.api, {
                    "task": "extract_entities",
                    "content": text[:5000]  # Limit content size
                })
                
                if isinstance(result, dict) and result.get("success", False) and "result" in result:
                    entities_data = result["result"]
                    
                    # Handle different output formats
                    if isinstance(entities_data, dict):
                        # If result already has our expected structure
                        if all(key in entities_data for key in ["people", "organizations", "locations", "dates"]):
                            return {
                                "people": entities_data.get("people", []),
                                "organizations": entities_data.get("organizations", []),
                                "locations": entities_data.get("locations", []),
                                "dates": entities_data.get("dates", [])
                            }
                        # Try to extract entities from other formats
                        elif "entities" in entities_data:
                            entity_groups = {}
                            for entity in entities_data["entities"]:
                                if isinstance(entity, dict) and "type" in entity and "text" in entity:
                                    entity_type = entity["type"].lower()
                                    entity_text = entity["text"]
                                    
                                    # Map to our categories
                                    if entity_type in ["person", "people"]:
                                        if "people" not in entity_groups:
                                            entity_groups["people"] = []
                                        entity_groups["people"].append(entity_text)
                                    elif entity_type in ["organization", "org", "company"]:
                                        if "organizations" not in entity_groups:
                                            entity_groups["organizations"] = []
                                        entity_groups["organizations"].append(entity_text)
                                    elif entity_type in ["location", "place", "gpe"]:
                                        if "locations" not in entity_groups:
                                            entity_groups["locations"] = []
                                        entity_groups["locations"].append(entity_text)
                                    elif entity_type in ["date", "time"]:
                                        if "dates" not in entity_groups:
                                            entity_groups["dates"] = []
                                        entity_groups["dates"].append(entity_text)
                            
                            # Ensure all entity types exist
                            for key in ["people", "organizations", "locations", "dates"]:
                                if key not in entity_groups:
                                    entity_groups[key] = []
                            
                            return entity_groups
            except Exception as e:
                self.logger.error(f"Error extracting entities with ASI: {str(e)}")
        
        # Fallback to MOCK-LLM if available
        if self.api.has_mock_llm:
            prompt = f"""Extract all people, organizations, locations, and dates from the following text:

{text[:2000]}...

Return only the extracted entities in the following format:
People: [list of people]
Organizations: [list of organizations]
Locations: [list of locations]
Dates: [list of dates]
"""
            response = self.api.generate_text(prompt)
            return self._parse_entity_response(response)
        
        # Fallback entity extraction using regex patterns
        people = self._extract_people(text)
        organizations = self._extract_organizations(text)
        locations = self._extract_locations(text)
        dates = self._extract_dates(text)
        
        return {
            "people": people,
            "organizations": organizations,
            "locations": locations,
            "dates": dates
        }
    
    def _parse_entity_response(self, response: str) -> Dict[str, List[str]]:
        """Parse the entity extraction response from MOCK-LLM."""
        entities = {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": []
        }
        
        # Simple parsing of response
        current_entity = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.lower().startswith("people:"):
                current_entity = "people"
                items = line[7:].strip()
                if items:
                    entities[current_entity].extend([item.strip() for item in items.split(',')])
            elif line.lower().startswith("organizations:"):
                current_entity = "organizations"
                items = line[13:].strip()
                if items:
                    entities[current_entity].extend([item.strip() for item in items.split(',')])
            elif line.lower().startswith("locations:"):
                current_entity = "locations"
                items = line[10:].strip()
                if items:
                    entities[current_entity].extend([item.strip() for item in items.split(',')])
            elif line.lower().startswith("dates:"):
                current_entity = "dates"
                items = line[6:].strip()
                if items:
                    entities[current_entity].extend([item.strip() for item in items.split(',')])
            elif current_entity:
                entities[current_entity].extend([item.strip() for item in line.split(',')])
        
        # Clean up empty entries
        for entity_type in entities:
            entities[entity_type] = [e for e in entities[entity_type] if e and not e.startswith('[') and not e.endswith(']')]
        
        return entities
    
    def _extract_people(self, text: str) -> List[str]:
        """Extract people names using regex patterns."""
        # Simple regex for names (Mr./Ms./Dr. followed by capitalized words)
        name_patterns = [
            r'Mr\.\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Ms\.\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Mrs\.\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Dr\.\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)'  # Simple first last name pattern
        ]
        
        people = []
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            people.extend(matches)
        
        # Remove duplicates and sort
        return sorted(list(set(people)))
    
    def _extract_organizations(self, text: str) -> List[str]:
        """Extract organization names using regex patterns."""
        # Look for organization indicators (Corp, Inc, LLC, Ltd)
        org_patterns = [
            r'([A-Z][A-Za-z0-9\s]+\s+Corp\.?)',
            r'([A-Z][A-Za-z0-9\s]+\s+Inc\.?)',
            r'([A-Z][A-Za-z0-9\s]+\s+LLC)',
            r'([A-Z][A-Za-z0-9\s]+\s+Ltd\.?)',
            r'([A-Z][A-Za-z0-9\s]+\s+Company)'
        ]
        
        orgs = []
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            orgs.extend(matches)
        
        # Remove duplicates and sort
        return sorted(list(set(orgs)))
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract location names using common location words."""
        common_locations = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "United States", "Canada", "UK", "China", "Japan", "Germany",
            "France", "Italy", "Australia", "India", "Brazil"
        ]
        
        locations = []
        for location in common_locations:
            if location in text:
                locations.append(location)
        
        return locations
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates using regex patterns."""
        # Various date formats
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'[A-Z][a-z]+\s+\d{1,2},\s+\d{4}',  # Month DD, YYYY
            r'\d{1,2}\s+[A-Z][a-z]+\s+\d{4}'  # DD Month YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        # Remove duplicates and sort
        return sorted(list(set(dates)))
    
    def _identify_action_items(self, text: str) -> List[str]:
        """Identify action items in the document."""
        self.logger.info("Identifying action items")
        
        # Use ASI for action item identification if available
        if self.api.has_asi:
            # Import ASI helper for action item identification
            from real_world_apps.asi_helper import process_with_asi
            
            result = process_with_asi(self.api, {
                "task": "identify_action_items",
                "content": text[:5000]  # Limit content size
            })
            
            if isinstance(result, dict) and result.get("success", False) and "result" in result:
                actions_data = result["result"]
                # Handle different output formats
                if isinstance(actions_data, dict):
                    # Try various field names for actions
                    for field in ["actions", "action_items", "tasks", "todos"]:
                        if field in actions_data and isinstance(actions_data[field], list):
                            return actions_data[field]
                    
                    # If we have a points field (like key points), treat them as actions
                    if "points" in actions_data and isinstance(actions_data["points"], list):
                        return [p for p in actions_data["points"] if any(kw in p.lower() for kw in ["must", "should", "need", "require", "please", "by", "due", "deadline"])]                  
                elif isinstance(actions_data, list):
                    return actions_data
        
        # Fallback action item identification
        action_phrases = [
            "need to", "must", "should", "required to", "action required",
            "please", "by tomorrow", "by next week", "by friday",
            "deadline", "due date", "follow up", "get back"
        ]
        
        sentences = text.replace("\n", " ").split(".")
        action_items = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains action phrases
            if any(phrase in sentence.lower() for phrase in action_phrases):
                action_items.append(sentence + ".")
        
        return action_items[:10]  # Limit to top 10 action items
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the document."""
        self.logger.info("Generating document summary")
        
        # Use ASI for summary generation if available
        if self.api.has_asi:
            try:
                # Import ASI helper for summary generation
                from real_world_apps.asi_helper import generate_summary
                
                summary = generate_summary(self.api, text[:5000], "medium")
                if summary:
                    return summary
            except Exception as e:
                self.logger.error(f"Error generating summary with ASI: {str(e)}")
        
        # Fallback to MOCK-LLM if available
        if self.api.has_mock_llm:
            prompt = f"""Summarize the following document in 3-4 sentences:

{text[:2000]}...
"""
            return self.api.generate_text(prompt)
        
        # Fallback summary generation
        sentences = text.replace("\n", " ").split(".")
        summary_sentences = sentences[:3]  # Take first 3 sentences
        
        return ". ".join(summary_sentences).strip() + "."
    
    def _extract_insights(self, text: str, category: str) -> List[str]:
        """Extract insights from the document based on category."""
        self.logger.info("Extracting document insights")
        
        # Use ASI for insights extraction if available
        if self.api.has_asi:
            try:
                # First try using our helper function to extract key points
                insights = extract_key_points(self.api, text[:5000], max_points=5)
                if insights and len(insights) > 0:
                    return insights
                
                # Alternatively, try directly with process_with_asi
                from real_world_apps.asi_helper import process_with_asi
                
                result = process_with_asi(self.api, {
                    "task": "extract_insights",
                    "content": text[:5000],  # Limit content size
                    "category": category
                })
                
                if isinstance(result, dict) and result.get("success", False) and "result" in result:
                    insights_data = result["result"]
                    # Handle different output formats
                    if isinstance(insights_data, dict):
                        # Try various field names for insights
                        for field in ["insights", "key_points", "takeaways", "points", "patterns"]:
                            if field in insights_data and isinstance(insights_data[field], list):
                                return insights_data[field]
                        
                        # Try text fields that might contain insights
                        for field in ["insight", "text", "analysis"]:
                            if field in insights_data and isinstance(insights_data[field], str):
                                # Split text into sentences
                                sentences = insights_data[field].split(". ")
                                return [s.strip() + "." for s in sentences if len(s.strip()) > 10][:5]
                    elif isinstance(insights_data, list):
                        return insights_data
            except Exception as e:
                self.logger.error(f"Error extracting insights with ASI: {str(e)}")
        
        # Fallback insights extraction
        insights = []
        
        # Different insights based on document category
        if category == "business":
            if "revenue" in text.lower() or "profit" in text.lower():
                insights.append("Document contains financial information.")
            if "strategy" in text.lower() or "plan" in text.lower():
                insights.append("Document discusses business strategy or planning.")
            if "market" in text.lower():
                insights.append("Document contains market analysis or market-related information.")
        
        elif category == "technical":
            if "system" in text.lower() or "software" in text.lower():
                insights.append("Document contains technical system information.")
            if "data" in text.lower() or "analysis" in text.lower():
                insights.append("Document discusses data analysis or data handling.")
            if "implementation" in text.lower() or "deploy" in text.lower():
                insights.append("Document covers implementation or deployment details.")
        
        elif category == "legal":
            if "contract" in text.lower() or "agreement" in text.lower():
                insights.append("Document is a legal contract or agreement.")
            if "rights" in text.lower() or "obligations" in text.lower():
                insights.append("Document specifies legal rights and obligations.")
            if "confidential" in text.lower():
                insights.append("Document contains confidential information.")
        
        # General insights for any category
        sentences = text.replace("\n", " ").split(".")
        important_sentences = [s.strip() for s in sentences if "important" in s.lower() or "key" in s.lower()]
        
        if important_sentences:
            insights.append(f"Key point: {important_sentences[0]}.")
        
        # If we couldn't find specific insights
        if not insights:
            insights.append(f"This appears to be a {category} document.")
            insights.append("Consider a more detailed analysis for specific insights.")
        
        return insights


def display_analysis(result: Dict[str, Any]):
    """Display the document analysis in a user-friendly format."""
    print("\n" + "="*80)
    print("DOCUMENT ANALYSIS".center(80))
    print("="*80 + "\n")
    
    # Document category
    print(f"Category: {result['category'].upper()}")
    print(f"Word Count: {result['stats']['word_count']} | Character Count: {result['stats']['character_count']}")
    print("-" * 80)
    
    # Summary
    print("\nSUMMARY:")
    print(result['summary'])
    
    # Entities
    print("\nENTITIES:")
    entities = result['entities']
    if entities['people']:
        print(f"  People: {', '.join(entities['people'])}")
    if entities['organizations']:
        print(f"  Organizations: {', '.join(entities['organizations'])}")
    if entities['locations']:
        print(f"  Locations: {', '.join(entities['locations'])}")
    if entities['dates']:
        print(f"  Dates: {', '.join(entities['dates'])}")
    
    # Action Items
    if result['action_items']:
        print("\nACTION ITEMS:")
        for i, item in enumerate(result['action_items'], 1):
            print(f"  {i}. {item}")
    
    # Insights
    if result['insights']:
        print("\nINSIGHTS:")
        for insight in result['insights']:
            print(f"  • {insight}")
    
    print("\n" + "="*80)


def read_file(file_path: str) -> str:
    """Read content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smart Document Assistant")
    parser.add_argument("--file", type=str, help="Path to a document file to analyze")
    parser.add_argument("--text", type=str, help="Text content to analyze")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Ensure USE_REAL_ASI is set to true
    os.environ['USE_REAL_ASI'] = 'true'
    
    # Initialize the document assistant
    assistant = DocumentAssistant()
    
    # Get content to analyze
    content = ""
    if args.file:
        try:
            content = read_file(args.file)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return
    elif args.text:
        content = args.text
    else:
        # Sample document for demo
        content = """
QUARTERLY BUSINESS REVIEW
Q1 2025

Prepared by: John Smith
Date: April 15, 2025

Executive Summary:
XYZ Corporation achieved strong results in Q1 2025, with revenue growth of 15% year-over-year, 
reaching $25M. Profitability improved with EBITDA margins expanding to 22% (+3% YoY). Our new 
product line, launched in February, has exceeded expectations with over 1,000 units sold in the 
first month.

Key Highlights:
- Revenue: $25M (+15% YoY)
- EBITDA: $5.5M (+32% YoY)
- New customers acquired: 230 (+25% YoY)
- Customer retention rate: 93% (unchanged)

Regional Performance:
The North America region continues to be our strongest market, accounting for 65% of total revenue. 
Europe showed significant improvement with 22% growth, while Asia-Pacific growth was modest at 8%. 
Our new office in Chicago has been fully staffed and is now operational.

Product Performance:
Our enterprise solution suite remains the top revenue generator (55% of total), with the 
Professional Services segment showing the highest growth rate at 30% YoY. The new mobile 
application platform launched in February has been well-received, with adoption rates 
exceeding our initial projections by 40%.

Action Items:
1. Marketing team must finalize Q2 campaign strategy by April 30, 2025
2. Sales team should follow up with the top 20 prospects identified during the recent trade show
3. Product development needs to address the reported issues with the mobile app by next week
4. Finance team required to prepare updated forecast based on Q1 results by April 25
5. HR team please schedule mid-year review meetings before May 15

Outlook:
Based on strong Q1 performance, we are revising our annual revenue projection upward by 5%. 
However, we anticipate increased competition in the European market in Q2, which may require 
additional marketing investment. The executive team will review the updated annual plan 
during the May 10 strategy meeting.

For questions, please contact John Smith at john.smith@xyzcorp.com or call the Finance 
department at ext. 5567 by Friday.

CONFIDENTIAL - Internal Use Only
"""
    
    if not content:
        print("Error: No content provided to analyze.")
        return
    
    # Process the document
    result = assistant.process_document(content)
    
    # Display the analysis
    display_analysis(result)


if __name__ == "__main__":
    main()
