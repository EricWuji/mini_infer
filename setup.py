"""
Setup script for MiniLLM
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mini-llm",
    version="1.0.0",
    author="EricWuji",
    author_email="your.email@example.com",
    description="A lightweight LLM implementation with Flash Attention and KV Cache",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EricWuji/mini_infer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "benchmarks": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
)
