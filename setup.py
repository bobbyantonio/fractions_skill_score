from setuptools import setup

description = (
    "Functions to calculate Fractions Skill Score and the useful criteria"
)

setup(
    name="fractions_skill_score",
    version="0.1",
    description=description,
    long_description=description,
    author="Bobby Antonio",
    license="MIT",
    packages=["fractions_skill_score"],
    install_requires=[
        "scipy",
        "numpy"
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)