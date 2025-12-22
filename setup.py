#!/usr/bin/env python3
"""
AVA - Afsah's Virtual Assistant
Setup configuration for local agentic AI optimized for NVIDIA RTX A2000 4GB VRAM
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() 
            and not line.startswith("#") 
            and not line.startswith("-")
        ]

# Development dependencies
dev_requirements = [
    "pytest>=7.4.0,<8.0.0",
    "pytest-asyncio>=0.23.0,<1.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "ruff>=0.1.15,<1.0.0",
    "black>=23.12.0,<25.0.0",
    "mypy>=1.8.0,<2.0.0",
    "pre-commit>=3.6.0,<4.0.0",
]

# Optional dependencies for different use cases
extras_require = {
    "dev": dev_requirements,
    "notebooks": [
        "jupyter>=1.0.0,<2.0.0",
        "ipykernel>=6.25.0,<7.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.13.0,<1.0.0",
    ],
    "analysis": [
        "scikit-learn>=1.3.0,<2.0.0",
        "scipy>=1.11.0,<2.0.0",
        "plotly>=5.17.0,<6.0.0",
    ],
    "all": dev_requirements + [
        "jupyter>=1.0.0,<2.0.0",
        "ipykernel>=6.25.0,<7.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.13.0,<1.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "scipy>=1.11.0,<2.0.0",
        "plotly>=5.17.0,<6.0.0",
    ],
}

setup(
    name="ava-agent",
    version="0.1.0",
    author="Muhammad Afsah Mumtaz",
    author_email="",  # Add email if desired
    description="Local agentic AI optimized for NVIDIA RTX A2000 4GB VRAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NAME0x0/AVA",
    project_urls={
        "Bug Reports": "https://github.com/NAME0x0/AVA/issues",
        "Source": "https://github.com/NAME0x0/AVA",
        "Documentation": "https://github.com/NAME0x0/AVA/tree/main/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.9,<3.12",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ava=ava.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ava_core": ["*.yaml", "*.json"],
        "config": ["*.yaml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "artificial-intelligence",
        "local-ai",
        "agentic-ai",
        "llm",
        "quantization",
        "nvidia",
        "gpu",
        "optimization",
        "transformers",
        "ollama",
    ],
    license="MIT",
)
