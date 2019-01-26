from setuptools import find_packages, setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='obj-loc-utils',
      version='0.1',
      description='Object Localization Utilities',
      url='https://github.com/benjaminrwilson/obj-loc-utils',
      author='Benjamin Wilson',
      license='GPL-3.0',
      install_requires=requirements,
      packages=find_packages(),
      zip_safe=False)
