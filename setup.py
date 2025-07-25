from setuptools import setup, find_packages

setup(
    name="hyper-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "polygon-api-client>=1.12.3",
        "lxml>=4.9.0",
        "duckdb",
        "yfinance"
    ],
    python_requires=">=3.9",
    author="Yeon Lee",
    description="A financial data and modeling system",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
) 