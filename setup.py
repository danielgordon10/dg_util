from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="dg_util",
    version="1.0.4",
    packages=find_packages(),
    install_requires=requirements,
    url="",
    license="",
    author="Daniel Gordon",
    author_email="",
    description="",
)
