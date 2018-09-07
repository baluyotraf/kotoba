from setuptools import setup, find_packages

setup(
    name='kotoba',
    version='0.0.1',
    packages=find_packages(exclude=['tests.*', 'tests']),
    install_requires=[
        'numpy>=1.14.5'
    ],
    extras_require={
        'nltk': ['nltk>=3.3'],
    },
)
