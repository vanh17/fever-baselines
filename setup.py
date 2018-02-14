from setuptools import setup, find_packages
import sys


with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='fever',
    version='0.0.1',
    description='Fact Extraction and VERification baselines',
    long_description="readme",
    license=license,
    python_requires='>=3.5',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)