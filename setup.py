from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='fmri_encoder',
    version='0.0.1',
    packages=find_packages(include=['fmri_encoder', 'fmri_encoder.*']),
    install_requires=required,
)
