from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rxnkinetics",
    version="1.0.0",
    author="Emre Yilmaz",
    author_email="emreyilmazch@gmail.com",
    description="A Python CLI toolkit for chemical reaction kinetics analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emreyilmazcode/ReactionKinetics",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.4",
    ],
    entry_points={
        "console_scripts": [
            "rxnkinetics=rxnkinetics.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
    keywords="chemistry kinetics reaction ODE arrhenius half-life",
)
