#!/usr/bin/env python3

import re
import sys
import glob
from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as fd:
    long_description = fd.read()

with open("electrolyze/__init__.py") as fd:
    lines = "\n".join(fd.readlines())

version = re.search('__version__ = "(.*)"', lines).group(1)
author = re.search('__author__ = "(.*)"', lines).group(1)
maintainer_email = re.search('__maintainer_email__ = "(.*)"', lines).group(1)
description = re.search('__description__ = "(.*)"', lines).group(1)


name = "electrolyze"

setup(
    name=name,
    version=version,
    python_requires='>=3.6',
    description=description,
    long_description=long_description,
    author=author,
    maintainer_email=maintainer_email,
 #   scripts=glob.glob("electrolyze/electrolyze_run"),
    platforms=["any"],
    install_requires=[
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "gpyopt",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            "electrolyze = electrolyze.cli:run"
            ]
        }
)
