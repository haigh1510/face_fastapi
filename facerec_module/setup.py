from os import path
from setuptools import find_packages, setup


REQUIRES_PYTHON = '>=3.6'
VERSION = '1.0.0'

here = path.abspath(path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        REQUIRED = f.read().split('\n')
except:
    REQUIRED = []

# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='facerec_module',
    version=VERSION,
    description='Package for face detection, encoding and verification',
    packages=find_packages(where=here, exclude=('tests', 'tests.*')),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    include_package_data=True,
    # long_description=long_description,
    long_description_content_type='text/markdown',
)
