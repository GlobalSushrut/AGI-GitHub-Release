#!/usr/bin/env python3
"""
Educational Learning Platform
----------------------------

A real-world application that demonstrates how to use the AGI Toolkit
to build an adaptive educational learning platform.

Features:
- Course content generation
- Quiz generation based on content
- Student progress tracking
- Adaptive learning paths
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to path so we can import the AGI Toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the ASI helper module and AGI Toolkit
from real_world_apps.asi_helper import initialize_asi_components
from agi_toolkit import AGIAPI
from course_manager import CourseManager
from student_manager import StudentManager


class LearningPlatform:
    """Educational Learning Platform using AGI Toolkit."""
    
    def __init__(self):
        """Initialize the learning platform."""
        # Configure logging
        self.logger = logging.getLogger("LearningPlatform")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing Learning Platform")
        
        # Initialize real ASI components
        initialize_asi_components()
        
        # Set environment variable to ensure interface uses real components
        os.environ['USE_REAL_ASI'] = 'true'
        
        # Initialize the AGI Toolkit API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        # Initialize course and student managers
        self.course_manager = CourseManager(self.api)
        self.student_manager = StudentManager(self.api)
        
        self.logger.info("Learning Platform initialized")
    
    def create_course(self, title: str, description: str, subjects: List[str], difficulty: str) -> Dict:
        """Create a new course with generated content."""
        self.logger.info(f"Creating course: {title}")
        return self.course_manager.create_course(title, description, subjects, difficulty)
    
    def get_course(self, course_id: str) -> Dict:
        """Retrieve a course by ID."""
        return self.course_manager.get_course(course_id)
    
    def list_courses(self) -> List[Dict]:
        """List all available courses."""
        return self.course_manager.list_courses()
    
    def generate_quiz(self, course_id: str, module_id: str, difficulty: str = "medium", 
                     num_questions: int = 5) -> Dict:
        """Generate a quiz for a course module."""
        self.logger.info(f"Generating quiz for course {course_id}, module {module_id}")
        return self.course_manager.generate_quiz(course_id, module_id, difficulty, num_questions)
    
    def register_student(self, name: str, email: str) -> Dict:
        """Register a new student."""
        self.logger.info(f"Registering student: {name}")
        return self.student_manager.register_student(name, email)
    
    def enroll_student(self, student_id: str, course_id: str) -> Dict:
        """Enroll a student in a course."""
        self.logger.info(f"Enrolling student {student_id} in course {course_id}")
        return self.student_manager.enroll_student(student_id, course_id)
    
    def track_progress(self, student_id: str, course_id: str, module_id: str, 
                      completion_pct: float, quiz_score: Optional[float] = None) -> Dict:
        """Update student progress in a course module."""
        return self.student_manager.track_progress(
            student_id, course_id, module_id, completion_pct, quiz_score)
    
    def get_student_progress(self, student_id: str, course_id: Optional[str] = None) -> Dict:
        """Get student progress for one or all courses."""
        return self.student_manager.get_progress(student_id, course_id)
    
    def recommend_learning_path(self, student_id: str) -> List[Dict]:
        """Generate a personalized learning path for a student."""
        self.logger.info(f"Generating learning path for student {student_id}")
        
        # Get student details and progress
        student = self.student_manager.get_student(student_id)
        if not student:
            return {"error": "Student not found"}
        
        progress = self.student_manager.get_progress(student_id)
        all_courses = self.course_manager.list_courses()
        
        # Use ASI if available
        if self.api.has_asi:
            recommendation_data = {
                "student": student,
                "progress": progress,
                "available_courses": all_courses
            }
            
            result = self.api.process_with_asi({
                "task": "recommend_learning_path",
                "data": recommendation_data
            })
            
            if result.get("success", False):
                return result.get("recommendations", [])
        
        # Fallback recommendation logic
        enrolled_courses = progress.get("courses", {})
        
        # Find courses not yet completed
        recommendations = []
        
        # First, check for courses in progress
        for course_id, course_progress in enrolled_courses.items():
            course = self.course_manager.get_course(course_id)
            if not course:
                continue
                
            completion = course_progress.get("overall_completion", 0)
            if completion < 100:
                recommendations.append({
                    "course_id": course_id,
                    "title": course.get("title", "Unknown Course"),
                    "completion": completion,
                    "status": "in_progress",
                    "recommendation_type": "continue",
                    "reason": "You have already started this course"
                })
        
        # Then, recommend new courses based on categories of interest
        enrolled_subjects = set()
        for course_id in enrolled_courses:
            course = self.course_manager.get_course(course_id)
            if course and "subjects" in course:
                enrolled_subjects.update(course["subjects"])
        
        # Find new courses with matching subjects
        for course in all_courses:
            course_id = course.get("id")
            
            # Skip if already enrolled
            if course_id in enrolled_courses:
                continue
                
            course_subjects = set(course.get("subjects", []))
            matching_subjects = enrolled_subjects.intersection(course_subjects)
            
            if matching_subjects:
                recommendations.append({
                    "course_id": course_id,
                    "title": course.get("title", "Unknown Course"),
                    "completion": 0,
                    "status": "not_started",
                    "recommendation_type": "new_course",
                    "reason": f"Based on your interest in {', '.join(list(matching_subjects)[:2])}"
                })
        
        # Sort recommendations: in-progress first, then by match strength
        recommendations.sort(key=lambda x: (
            0 if x["status"] == "in_progress" else 1,
            -len(x.get("reason", "").split("interest in ")[1].split(",")) if "interest in " in x.get("reason", "") else 0
        ))
        
        return recommendations[:5]  # Limit to top 5 recommendations


def display_course(course: Dict):
    """Display course information in a user-friendly format."""
    print("\n" + "="*80)
    print(f"COURSE: {course.get('title', 'Untitled Course')}".center(80))
    print("="*80 + "\n")
    
    print(f"ID: {course.get('id', 'unknown')}")
    print(f"Description: {course.get('description', 'No description')}")
    print(f"Subjects: {', '.join(course.get('subjects', []))}")
    print(f"Difficulty: {course.get('difficulty', 'Not specified')}")
    print(f"Created: {course.get('created_at', 'Unknown date')}")
    
    print("\nModules:")
    for i, module in enumerate(course.get("modules", []), 1):
        print(f"  {i}. {module.get('title', 'Untitled module')}")
        print(f"     Content length: {len(module.get('content', '')):.0f} characters")
    
    print("="*80)


def display_quiz(quiz: Dict):
    """Display a quiz in a user-friendly format."""
    print("\n" + "="*80)
    print(f"QUIZ: {quiz.get('title', 'Untitled Quiz')}".center(80))
    print("="*80 + "\n")
    
    for i, question in enumerate(quiz.get("questions", []), 1):
        print(f"Question {i}: {question.get('question', 'No question')}")
        
        options = question.get("options", [])
        for j, option in enumerate(options, 1):
            print(f"  {j}. {option}")
        
        # Don't show the answer during a real quiz
        answer = question.get("answer", 0)
        if isinstance(answer, int):
            if 0 <= answer < len(options):
                correct = options[answer]
                print(f"  Correct answer: {correct}")
        else:
            print(f"  Correct answer: {answer}")
        
        print("")
    
    print("="*80)


def display_student_progress(progress: Dict):
    """Display student progress in a user-friendly format."""
    print("\n" + "="*80)
    print(f"STUDENT PROGRESS: {progress.get('student_name', 'Unknown Student')}".center(80))
    print("="*80 + "\n")
    
    courses = progress.get("courses", {})
    if not courses:
        print("No courses enrolled.")
        return
    
    for course_id, course_progress in courses.items():
        print(f"Course: {course_progress.get('course_title', course_id)}")
        print(f"Overall completion: {course_progress.get('overall_completion', 0):.1f}%")
        
        print("Modules:")
        modules = course_progress.get("modules", {})
        for module_id, module_data in modules.items():
            print(f"  - {module_data.get('title', module_id)}: {module_data.get('completion', 0):.1f}% complete")
            if "quiz_score" in module_data:
                print(f"    Quiz score: {module_data['quiz_score']:.1f}%")
        
        print("")
    
    print("="*80)


def display_recommendations(recommendations: List[Dict]):
    """Display course recommendations in a user-friendly format."""
    print("\n" + "="*80)
    print("COURSE RECOMMENDATIONS".center(80))
    print("="*80 + "\n")
    
    if not recommendations:
        print("No recommendations available at this time.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.get('title', 'Unknown Course')}")
        print(f"   Status: {rec.get('status', 'unknown').replace('_', ' ').title()}")
        
        if "completion" in rec:
            print(f"   Completion: {rec.get('completion', 0):.1f}%")
            
        print(f"   Recommendation: {rec.get('recommendation_type', 'unknown').replace('_', ' ').title()}")
        print(f"   Why: {rec.get('reason', 'No reason provided')}")
        print("")
    
    print("="*80)


def demo_platform():
    """Run a demonstration of the learning platform."""
    platform = LearningPlatform()
    
    # Create sample courses
    programming_course = platform.create_course(
        title="Introduction to Python Programming",
        description="Learn the basics of Python programming, suitable for beginners.",
        subjects=["Computer Science", "Programming", "Python"],
        difficulty="beginner"
    )
    
    math_course = platform.create_course(
        title="Advanced Mathematics",
        description="Calculus, linear algebra, and statistics for technical fields.",
        subjects=["Mathematics", "Calculus", "Statistics"],
        difficulty="advanced"
    )
    
    art_course = platform.create_course(
        title="Digital Art Fundamentals",
        description="Learn digital art concepts and techniques.",
        subjects=["Art", "Design", "Digital Media"],
        difficulty="intermediate"
    )
    
    # Register a student
    student = platform.register_student(
        name="John Doe",
        email="john.doe@example.com"
    )
    
    # Enroll in courses
    student_id = student.get("id")
    platform.enroll_student(student_id, programming_course.get("id"))
    platform.enroll_student(student_id, math_course.get("id"))
    
    # Track progress
    platform.track_progress(
        student_id=student_id,
        course_id=programming_course.get("id"),
        module_id=programming_course.get("modules", [{}])[0].get("id", "module1"),
        completion_pct=75.0,
        quiz_score=85.0
    )
    
    platform.track_progress(
        student_id=student_id,
        course_id=math_course.get("id"),
        module_id=math_course.get("modules", [{}])[0].get("id", "module1"),
        completion_pct=30.0
    )
    
    # Get student progress
    progress = platform.get_student_progress(student_id)
    display_student_progress(progress)
    
    # Generate a quiz
    quiz = platform.generate_quiz(
        course_id=programming_course.get("id"),
        module_id=programming_course.get("modules", [{}])[0].get("id", "module1"),
        difficulty="medium",
        num_questions=3
    )
    display_quiz(quiz)
    
    # Get recommendations
    recommendations = platform.recommend_learning_path(student_id)
    display_recommendations(recommendations)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Educational Learning Platform")
    parser.add_argument("--demo", action="store_true", help="Run demonstration mode")
    parser.add_argument("--student", type=str, help="Student ID for operations")
    parser.add_argument("--list-courses", action="store_true", help="List all courses")
    parser.add_argument("--course", type=str, help="Course ID for operations")
    parser.add_argument("--show-progress", action="store_true", help="Show student progress")
    parser.add_argument("--generate-quiz", action="store_true", help="Generate a quiz for the course")
    parser.add_argument("--recommendations", action="store_true", help="Get course recommendations")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the platform
    platform = LearningPlatform()
    
    # Run as demo if no arguments or demo flag is set
    if args.demo or len(sys.argv) == 1:
        demo_platform()
        return
    
    # List courses
    if args.list_courses:
        courses = platform.list_courses()
        for course in courses:
            display_course(course)
    
    # Show course details
    if args.course and not (args.show_progress or args.generate_quiz):
        course = platform.get_course(args.course)
        if course:
            display_course(course)
        else:
            print(f"Course not found: {args.course}")
    
    # Show student progress
    if args.student and args.show_progress:
        progress = platform.get_student_progress(args.student, args.course)
        display_student_progress(progress)
    
    # Generate quiz
    if args.course and args.generate_quiz:
        course = platform.get_course(args.course)
        if not course:
            print(f"Course not found: {args.course}")
            return
            
        # Use the first module if available
        module_id = course.get("modules", [{}])[0].get("id", "module1")
        quiz = platform.generate_quiz(args.course, module_id)
        display_quiz(quiz)
    
    # Get recommendations
    if args.student and args.recommendations:
        recommendations = platform.recommend_learning_path(args.student)
        display_recommendations(recommendations)


if __name__ == "__main__":
    main()
