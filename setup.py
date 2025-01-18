from setuptools import setup, find_packages

setup(
    name='localgradientimportance',
    version='0.1.0',
    description='Importances for heterogeneous data',
    author='Evan Mason',
    author_email='emaso007@ucr.edu',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow>=2.0.0"
    ]
)
