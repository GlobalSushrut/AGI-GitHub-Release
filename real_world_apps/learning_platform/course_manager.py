#!/usr/bin/env python3
"""
Course Manager for Educational Learning Platform
-----------------------------------------------

Handles course creation, management, and quiz generation.
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger("CourseManager")


class CourseManager:
    """Manages courses in the learning platform."""
    
    def __init__(self, api):
        """
        Initialize the course manager.
        
        Args:
            api: The AGI Toolkit API instance
        """
        self.api = api
        self.logger = logger
        
        # Load existing courses
        self.courses = {}
        self._load_courses()
    
    def _load_courses(self):
        """Load courses from API memory."""
        try:
            memory_key = "learning_platform_courses"
            courses_data = self.api.retrieve_data(memory_key)
            
            # Ensure courses_data is a dictionary
            if courses_data and isinstance(courses_data, dict):
                self.courses = courses_data
                self.logger.info(f"Loaded {len(self.courses)} courses from memory")
            else:
                self.courses = {}
                self.logger.info("No valid courses found in memory, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading courses: {str(e)}")
    
    def _save_courses(self):
        """Save courses to API memory."""
        try:
            memory_key = "learning_platform_courses"
            self.api.store_data(memory_key, self.courses)
            self.logger.info(f"Saved {len(self.courses)} courses to memory")
        except Exception as e:
            self.logger.error(f"Error saving courses: {str(e)}")
    
    def create_course(self, title: str, description: str, subjects: List[str], difficulty: str) -> Dict:
        """
        Create a new course with AI-generated content.
        
        Args:
            title: Course title
            description: Course description
            subjects: List of subject areas
            difficulty: Difficulty level (beginner, intermediate, advanced)
            
        Returns:
            Course data dictionary
        """
        self.logger.info(f"Creating course: {title}")
        
        # Generate a unique ID
        course_id = f"course_{uuid.uuid4().hex[:8]}"
        
        # Generate module content if MOCK-LLM is available
        modules = []
        
        if self.api.has_mock_llm:
            # Use the MOCK-LLM to generate course outline and content
            prompt = f"""
            Generate a course outline for "{title}".
            Description: {description}
            Subjects: {', '.join(subjects)}
            Difficulty: {difficulty}
            
            Create 3-5 modules with titles and a brief description for each.
            """
            
            outline_response = self.api.generate_text(prompt)
            
            # Parse the outline (simplified for demo)
            module_titles = []
            try:
                lines = outline_response.split('\n')
                for line in lines:
                    if line.strip().startswith(("Module", "Chapter", "Unit", "Section", "1.", "2.", "3.", "4.", "5.")):
                        title = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                        module_titles.append(title)
            except Exception as e:
                self.logger.error(f"Error parsing outline: {str(e)}")
                module_titles = [f"Module {i+1}" for i in range(3)]
            
            # Ensure we have at least some modules
            if not module_titles:
                module_titles = [f"Module {i+1}" for i in range(3)]
            
            # Generate content for each module
            for i, title in enumerate(module_titles[:5]):  # Limit to 5 modules
                module_id = f"{course_id}_module_{i+1}"
                
                # Generate module content
                content_prompt = f"""
                Generate detailed educational content for the module "{title}" 
                in the course "{title}".
                
                The content should be appropriate for {difficulty} level students.
                Focus on the following subjects: {', '.join(subjects)}
                
                Write approximately 500 words of educational content.
                """
                
                content = self.api.generate_text(content_prompt)
                
                modules.append({
                    "id": module_id,
                    "title": title,
                    "content": content,
                    "order": i + 1
                })
        else:
            # Generate sample modules for demonstration
            sample_content = (
                f"This is sample content for the course '{title}'. "
                f"In a real deployment with MOCK-LLM available, this would contain "
                f"AI-generated educational content appropriate for {difficulty} level students "
                f"focusing on {', '.join(subjects)}."
            )
            
            modules = [
                {
                    "id": f"{course_id}_module_1",
                    "title": "Introduction",
                    "content": sample_content,
                    "order": 1
                },
                {
                    "id": f"{course_id}_module_2",
                    "title": "Core Concepts",
                    "content": sample_content,
                    "order": 2
                },
                {
                    "id": f"{course_id}_module_3",
                    "title": "Advanced Topics",
                    "content": sample_content,
                    "order": 3
                }
            ]
        
        # Create the course object
        course = {
            "id": course_id,
            "title": title,
            "description": description,
            "subjects": subjects,
            "difficulty": difficulty,
            "modules": modules,
            "created_at": datetime.now().isoformat()
        }
        
        # Store the course
        self.courses[course_id] = course
        self._save_courses()
        
        return course
    
    def get_course(self, course_id: str) -> Optional[Dict]:
        """
        Retrieve a course by ID.
        
        Args:
            course_id: Course ID
            
        Returns:
            Course data or None if not found
        """
        return self.courses.get(course_id)
    
    def list_courses(self) -> List[Dict]:
        """
        List all available courses.
        
        Returns:
            List of course data dictionaries
        """
        return list(self.courses.values())
    
    def update_course(self, course_id: str, updates: Dict) -> Optional[Dict]:
        """
        Update a course.
        
        Args:
            course_id: Course ID
            updates: Dictionary of fields to update
            
        Returns:
            Updated course data or None if not found
        """
        if course_id not in self.courses:
            return None
        
        course = self.courses[course_id]
        
        # Update allowed fields
        for field in ["title", "description", "subjects", "difficulty"]:
            if field in updates:
                course[field] = updates[field]
        
        # Special handling for modules
        if "modules" in updates:
            # We don't replace modules, just update specific ones
            new_modules = updates["modules"]
            module_dict = {m["id"]: m for m in course["modules"]}
            
            for module in new_modules:
                if "id" in module and module["id"] in module_dict:
                    # Update existing module
                    for field in ["title", "content", "order"]:
                        if field in module:
                            module_dict[module["id"]][field] = module[field]
            
            # Recreate modules list, sorted by order
            course["modules"] = sorted(module_dict.values(), key=lambda m: m.get("order", 0))
        
        # Update timestamp
        course["updated_at"] = datetime.now().isoformat()
        
        # Save changes
        self.courses[course_id] = course
        self._save_courses()
        
        return course
    
    def delete_course(self, course_id: str) -> bool:
        """
        Delete a course.
        
        Args:
            course_id: Course ID
            
        Returns:
            Success status
        """
        if course_id not in self.courses:
            return False
        
        del self.courses[course_id]
        self._save_courses()
        
        return True
    
    def generate_quiz(self, course_id: str, module_id: str, 
                     difficulty: str = "medium", num_questions: int = 5) -> Dict:
        """
        Generate a quiz for a course module.
        
        Args:
            course_id: Course ID
            module_id: Module ID
            difficulty: Quiz difficulty (easy, medium, hard)
            num_questions: Number of questions to generate
            
        Returns:
            Quiz data dictionary
        """
        self.logger.info(f"Generating quiz for course {course_id}, module {module_id}")
        
        # Get the course and module
        course = self.get_course(course_id)
        if not course:
            return {"error": "Course not found"}
        
        module = None
        for m in course.get("modules", []):
            if m.get("id") == module_id:
                module = m
                break
        
        if not module:
            return {"error": "Module not found"}
        
        module_title = module.get("title", "Untitled Module")
        content = module.get("content", "")
        
        # Create a quiz ID
        quiz_id = f"quiz_{uuid.uuid4().hex[:8]}"
        
        # Generate quiz questions if MOCK-LLM is available
        questions = []
        
        if self.api.has_mock_llm and content:
            # Use the MOCK-LLM to generate quiz questions based on content
            prompt = f"""
            Generate {num_questions} multiple-choice quiz questions based on the following educational content:
            
            Title: {module_title}
            
            Content: {content[:2000]}  # Limit content to avoid token limits
            
            The quiz should be {difficulty} difficulty.
            
            For each question, provide:
            1. The question text
            2. Four possible answers (options)
            3. The index of the correct answer (0-3)
            
            Format each question like this:
            Question: [question text]
            Options: ["option1", "option2", "option3", "option4"]
            Answer: [index of correct answer, 0-3]
            """
            
            quiz_response = self.api.generate_text(prompt)
            
            # Parse the quiz questions (simplified for demo)
            try:
                current_question = {}
                for line in quiz_response.split('\n'):
                    line = line.strip()
                    if line.startswith("Question:"):
                        if current_question:
                            questions.append(current_question)
                        current_question = {"question": line.replace("Question:", "").strip()}
                    elif line.startswith("Options:"):
                        options_str = line.replace("Options:", "").strip()
                        try:
                            # Try to parse as list
                            import ast
                            options = ast.literal_eval(options_str)
                            current_question["options"] = options
                        except:
                            # Fallback: split by commas
                            options = [o.strip(' "\'') for o in options_str.split(",")]
                            current_question["options"] = options
                    elif line.startswith("Answer:"):
                        answer_str = line.replace("Answer:", "").strip()
                        try:
                            current_question["answer"] = int(answer_str)
                        except:
                            current_question["answer"] = answer_str
                
                # Add the last question
                if current_question and "question" in current_question:
                    questions.append(current_question)
            except Exception as e:
                self.logger.error(f"Error parsing quiz questions: {str(e)}")
        
        # If we don't have enough questions (or parsing failed), create placeholder questions
        if len(questions) < num_questions:
            for i in range(len(questions), num_questions):
                questions.append({
                    "question": f"Sample question #{i+1} for {module_title}?",
                    "options": [
                        "Sample answer 1",
                        "Sample answer 2",
                        "Sample answer 3",
                        "Sample answer 4"
                    ],
                    "answer": 0  # First option is correct in samples
                })
        
        # Create the quiz object
        quiz = {
            "id": quiz_id,
            "title": f"Quiz: {module_title}",
            "course_id": course_id,
            "module_id": module_id,
            "difficulty": difficulty,
            "questions": questions[:num_questions],  # Limit to requested number
            "created_at": datetime.now().isoformat()
        }
        
        return quiz
