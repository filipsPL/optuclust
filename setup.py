from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="optuclust",
    version="0.0.2",
    author="Filip S.",
    author_email="filip.ursynow@gmail.com",
    description="Hyperparameter optimization for multiple clustering algorithms using Optuna, with Scikit-learn API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filipsPL/optuclust",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.1",
        "hdbscan>=0.8.29",
        "optuna>=3.0",
        "kmedoids>=0.3.0",
        "matplotlib>=3.4",
        "pandas>=1.3",
        "sklearn-som",
    ],
)
