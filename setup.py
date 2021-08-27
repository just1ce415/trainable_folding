"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
#from Cython.Build import cythonize
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='alphadock',  # Required
    version='1.0.0',  # Required
    description='Learning FFT docking with steerable SE3 CNNs',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/ignatovmg/alphadock.git',  # Optional
    author='Mikhail Ignatov',  # Optional
    author_email='ignatovmg@gmail.com',  # Optional
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),  # Required
    python_requires='>=3.6',
)
