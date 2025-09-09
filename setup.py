from setuptools import setup, find_packages

setup(
    name='imnf',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
)