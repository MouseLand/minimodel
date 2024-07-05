import setuptools
from setuptools import setup

install_deps = ['numpy>=1.20.0', 'scipy', 
                'torch>=1.6',
                'opencv-python-headless',
                ]

try:
    import torch
    a = torch.ones(2, 3)
    version = int(torch.__version__[2])
    if version >= 6:
        install_deps.remove('torch>=1.6')
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
setup(
    name="minimodel",
    license="BSD",
    author="Stringer lab",
    author_email="fengtongd@janelia.hhmi.org",
    description="predictive models for neural data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/minimodel",
    setup_requires=[
      'pytest-runner',
      'setuptools_scm',
    ],
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires = install_deps,
    tests_require=[
      'pytest'
    ],
    
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
)