"""
Setup script for the TradingAgents package.
"""

from setuptools import setup, find_packages

setup(
    name="tradingagents",
    version="0.1.0",
    description="Lightweight fundamentals and news review agents",
    author="TradingAgents Team",
    author_email="yijia.xiao@cs.ucla.edu",
    url="https://github.com/TauricResearch",
    packages=find_packages(),
    install_requires=[
    "pandas>=1.5",
    "yfinance>=0.2.37",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "openai>=1.40.0",
    "google-generativeai>=0.7.0",
    "vaderSentiment>=3.3.2",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "tradingagents=cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Trading Industry",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
