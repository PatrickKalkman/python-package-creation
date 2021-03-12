from setuptools import setup, find_packages

setup(
    author='Patrick Kalkman',
    description='helper functions for image augmentation and training data distribution',
    name='tensorflow_helpers',
    version='0.2.1',
    packages=find_packages(include=['tensorflow_helpers','tensorflow_helpers.*']),
    install_requires=[
         'pandas>=1.2.3',
         'numpy>=1.19.5',
         'tensorflow>=2.4.1',
    ], 
    python_requires='>=3.8.8'
)