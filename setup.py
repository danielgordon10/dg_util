from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="dg_util",
    version="1.0.4",
    packages=find_packages(),
    install_requires=requirements,
    url="https://github.com/danielgordon10/dg_util",
    license="Apache-2.0",
    author="Daniel Gordon",
    author_email="",
    description="",
)
