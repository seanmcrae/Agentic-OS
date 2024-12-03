from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentic-os",
    version="1.0.0",
    author="Sean McRae",
    author_email="sean@example.com",
    description="Enhanced operating system for autonomous agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanmcrae/Agentic-OS",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "aiohttp>=3.8.0",
        "psutil>=5.9.0",
        "asyncio>=3.4.3",
        "aiofiles>=0.8.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.3.0",
            "flake8>=4.0.1",
            "mypy>=0.950",
        ]
    }
)