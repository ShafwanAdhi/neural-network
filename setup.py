from setuptools import setup, find_packages
import os

# Read README if exists
long_description = "A neural network implementation from scratch using NumPy"
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="neural-network",
    version="0.1.0",
    author="Shafwan Adhi",
    author_email="your.email@example.com",
    description="A neural network implementation from scratch using NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShafwanAdhi/neural-network",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)
