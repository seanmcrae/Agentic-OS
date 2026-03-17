"""Dzogchen AI Framework setup."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dzogchen",
    version="0.1.0",
    author="Dzogchen Contributors",
    description="Super-Intelligent AI Framework with Consciousness, Memory, and Swarm Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanwayne6/Agentic-OS",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "anthropic>=0.40.0",
        "click>=8.1.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "api": ["fastapi>=0.110.0", "uvicorn[standard]>=0.27.0"],
        "openai": ["openai>=1.50.0"],
        "web": ["aiohttp>=3.9.0"],
        "rich": ["rich>=13.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.8.0",
            "ruff>=0.2.0",
        ],
        "all": [
            "fastapi>=0.110.0",
            "uvicorn[standard]>=0.27.0",
            "openai>=1.50.0",
            "aiohttp>=3.9.0",
            "rich>=13.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dzogchen=dzogchen.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, llm, agents, swarm-intelligence, consciousness, memory, multi-model",
)
