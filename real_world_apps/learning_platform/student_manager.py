#!/usr/bin/env python3
"""
Student Manager for Educational Learning Platform
-----------------------------------------------

Handles student registration, enrollment, and progress tracking.
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger("StudentManager")


class StudentManager:
    """Manages students in the learning platform."""
    
    def __init__(self, api):
        """
        Initialize the student manager.
        
        Args:
            api: The AGI Toolkit API instance
        """
        self.api = api
        self.logger = logger
        
        # Load existing students and progress data
        self.students = {}
        self.progress = {}
        self._load_data()
    
    def _load_data(self):
        """Load student data from API memory."""
        try:
            # Load students
            students_key = "learning_platform_students"
            students_data = self.api.retrieve_data(students_key)
            
            if students_data and isinstance(students_data, dict):
                self.students = students_data
                self.logger.info(f"Loaded {len(self.students)} students from memory")
            else:
                self.students = {}
                self.logger.info("No valid students found in memory, starting fresh")
            
            # Load progress
            progress_key = "learning_platform_progress"
            progress_data = self.api.retrieve_data(progress_key)
            
            if progress_data and isinstance(progress_data, dict):
                self.progress = progress_data
                self.logger.info(f"Loaded progress data for {len(self.progress)} students")
            else:
                self.progress = {}
                self.logger.info("No valid progress data found in memory, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading student data: {str(e)}")
    
    def _save_data(self):
        """Save student data to API memory."""
        try:
            # Save students
            students_key = "learning_platform_students"
            self.api.store_data(students_key, self.students)
            
            # Save progress
            progress_key = "learning_platform_progress"
            self.api.store_data(progress_key, self.progress)
            
            self.logger.info(f"Saved data for {len(self.students)} students and {len(self.progress)} progress records")
        except Exception as e:
            self.logger.error(f"Error saving student data: {str(e)}")
    
    def register_student(self, name: str, email: str) -> Dict:
        """
        Register a new student.
        
        Args:
            name: Student name
            email: Student email
            
        Returns:
            Student data dictionary
        """
        self.logger.info(f"Registering student: {name}")
        
        # Check if email already exists
        for student_id, student in self.students.items():
            if student.get("email") == email:
                return student
        
        # Generate a unique ID
        student_id = f"student_{uuid.uuid4().hex[:8]}"
        
        # Create the student object
        student = {
            "id": student_id,
            "name": name,
            "email": email,
            "registered_at": datetime.now().isoformat(),
            "enrolled_courses": []
        }
        
        # Store the student
        self.students[student_id] = student
        self._save_data()
        
        return student
    
    def get_student(self, student_id: str) -> Optional[Dict]:
        """
        Retrieve a student by ID.
        
        Args:
            student_id: Student ID
            
        Returns:
            Student data or None if not found
        """
        return self.students.get(student_id)
    
    def list_students(self) -> List[Dict]:
        """
        List all registered students.
        
        Returns:
            List of student data dictionaries
        """
        return list(self.students.values())
    
    def update_student(self, student_id: str, updates: Dict) -> Optional[Dict]:
        """
        Update student information.
        
        Args:
            student_id: Student ID
            updates: Dictionary of fields to update
            
        Returns:
            Updated student data or None if not found
        """
        if student_id not in self.students:
            return None
        
        student = self.students[student_id]
        
        # Update allowed fields
        for field in ["name", "email"]:
            if field in updates:
                student[field] = updates[field]
        
        # Update timestamp
        student["updated_at"] = datetime.now().isoformat()
        
        # Save changes
        self.students[student_id] = student
        self._save_data()
        
        return student
    
    def delete_student(self, student_id: str) -> bool:
        """
        Delete a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            Success status
        """
        if student_id not in self.students:
            return False
        
        # Remove student
        del self.students[student_id]
        
        # Remove progress data
        if student_id in self.progress:
            del self.progress[student_id]
        
        self._save_data()
        
        return True
    
    def enroll_student(self, student_id: str, course_id: str) -> Dict:
        """
        Enroll a student in a course.
        
        Args:
            student_id: Student ID
            course_id: Course ID
            
        Returns:
            Enrollment result
        """
        self.logger.info(f"Enrolling student {student_id} in course {course_id}")
        
        # Check if student exists
        student = self.get_student(student_id)
        if not student:
            return {"error": "Student not found"}
        
        # Add course to enrolled courses if not already enrolled
        if course_id not in student.get("enrolled_courses", []):
            if "enrolled_courses" not in student:
                student["enrolled_courses"] = []
            
            student["enrolled_courses"].append(course_id)
            
            # Initialize progress for this course
            if student_id not in self.progress:
                self.progress[student_id] = {"courses": {}}
            
            if "courses" not in self.progress[student_id]:
                self.progress[student_id]["courses"] = {}
            
            self.progress[student_id]["courses"][course_id] = {
                "enrolled_at": datetime.now().isoformat(),
                "overall_completion": 0.0,
                "modules": {}
            }
            
            # Save changes
            self.students[student_id] = student
            self._save_data()
        
        return {
            "student_id": student_id,
            "course_id": course_id,
            "status": "enrolled",
            "enrolled_at": datetime.now().isoformat()
        }
    
    def track_progress(self, student_id: str, course_id: str, module_id: str, 
                      completion_pct: float, quiz_score: Optional[float] = None) -> Dict:
        """
        Update student progress in a course module.
        
        Args:
            student_id: Student ID
            course_id: Course ID
            module_id: Module ID
            completion_pct: Module completion percentage (0-100)
            quiz_score: Optional quiz score percentage (0-100)
            
        Returns:
            Updated progress data
        """
        self.logger.info(f"Updating progress for student {student_id}, course {course_id}, module {module_id}")
        
        # Check if student is enrolled in the course
        student = self.get_student(student_id)
        if not student:
            return {"error": "Student not found"}
        
        if course_id not in student.get("enrolled_courses", []):
            return {"error": "Student not enrolled in this course"}
        
        # Initialize progress structures if needed
        if student_id not in self.progress:
            self.progress[student_id] = {"courses": {}}
        
        if "courses" not in self.progress[student_id]:
            self.progress[student_id]["courses"] = {}
        
        if course_id not in self.progress[student_id]["courses"]:
            self.progress[student_id]["courses"][course_id] = {
                "enrolled_at": datetime.now().isoformat(),
                "overall_completion": 0.0,
                "modules": {}
            }
        
        course_progress = self.progress[student_id]["courses"][course_id]
        if "modules" not in course_progress:
            course_progress["modules"] = {}
        
        # Update module progress
        if module_id not in course_progress["modules"]:
            course_progress["modules"][module_id] = {
                "first_accessed": datetime.now().isoformat(),
                "completion": 0.0
            }
        
        module_progress = course_progress["modules"][module_id]
        module_progress["completion"] = max(min(completion_pct, 100.0), 0.0)  # Ensure within 0-100 range
        module_progress["last_accessed"] = datetime.now().isoformat()
        
        # Add quiz score if provided
        if quiz_score is not None:
            module_progress["quiz_score"] = max(min(quiz_score, 100.0), 0.0)  # Ensure within 0-100 range
            module_progress["quiz_completed_at"] = datetime.now().isoformat()
        
        # Recalculate overall course completion
        modules = course_progress["modules"]
        if modules:
            overall = sum(m.get("completion", 0.0) for m in modules.values()) / len(modules)
            course_progress["overall_completion"] = overall
        
        # Save changes
        self._save_data()
        
        return {
            "student_id": student_id,
            "course_id": course_id,
            "module_id": module_id,
            "completion": completion_pct,
            "quiz_score": quiz_score,
            "overall_completion": course_progress["overall_completion"]
        }
    
    def get_progress(self, student_id: str, course_id: Optional[str] = None) -> Dict:
        """
        Get student progress for one or all courses.
        
        Args:
            student_id: Student ID
            course_id: Optional course ID to filter by
            
        Returns:
            Progress data
        """
        # Check if student exists
        student = self.get_student(student_id)
        if not student:
            return {"error": "Student not found"}
        
        # Get progress data
        student_progress = self.progress.get(student_id, {"courses": {}})
        
        # Filter by course if specified
        if course_id:
            if course_id in student_progress.get("courses", {}):
                course_data = student_progress["courses"][course_id]
                return {
                    "student_id": student_id,
                    "student_name": student.get("name"),
                    "course_id": course_id,
                    "overall_completion": course_data.get("overall_completion", 0.0),
                    "modules": course_data.get("modules", {})
                }
            else:
                return {
                    "student_id": student_id,
                    "student_name": student.get("name"),
                    "course_id": course_id,
                    "error": "Student not enrolled in this course"
                }
        
        # Add course titles to the data if available
        courses_data = student_progress.get("courses", {})
        
        # Construct the complete progress report
        return {
            "student_id": student_id,
            "student_name": student.get("name"),
            "courses": courses_data
        }
