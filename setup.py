"""Setup script for ONA platform."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements for backward compatibility
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = []

setup(
    name="orgnet",
    version="0.1.2",
    author="Kyle Jones",
    author_email="",  # Add your email here
    description="Organizational Network Analysis Platform using ML and Graph Analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kylejones200/orgnet",
    project_urls={
        "Homepage": "https://github.com/kylejones200/orgnet",
        "Documentation": "https://github.com/kylejones200/orgnet#readme",
        "Repository": "https://github.com/kylejones200/orgnet",
        "Issues": "https://github.com/kylejones200/orgnet/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "orgnet-api=orgnet.api.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
