#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

style_packages = [
    "black==22.3.0",
    # "flake8==5.0.4",
    "isort==5.10.1",
]

setup(

    author="RecList",
    author_email='',
    python_requires='>=3.6',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="RecList",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="reclist",
    name="reclist",
    packages=find_packages(include=["reclist", "reclist.*"]),
    test_suite="tests",
    tests_require=test_requirements,

    url='https://github.com/jacopotagliabue/reclist',
    version='2.1.0',
    zip_safe=False,
    extras_require={
        "dev": style_packages + ["pre-commit==2.20.0"],
    },
)
