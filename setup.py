#!/usr/bin/env python
"""
AGI Toolkit
-----------
A unified toolkit for ASI and MOCK-LLM integration.

This package provides tools for leveraging advanced AI capabilities
through a simplified, stable API without touching core implementation details.
"""

import os
from setuptools import setup, find_packages

# Read the contents of README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agi-toolkit",
    version="1.0.0",
    description="A unified toolkit for ASI and MOCK-LLM integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GlobalSushrut",
    author_email="hello@example.com",
    url="https://github.com/GlobalSushrut/AGI-GitHub-Release",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pyyaml>=5.1.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai, asi, llm, machine learning, artificial intelligence, transformer",
)
