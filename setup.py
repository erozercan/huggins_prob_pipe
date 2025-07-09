from setuptools import setup, find_packages

setup(
    name="probpipe",
    version="0.1.0",
    description="Probpipe package",
    author="Yongho Lim, Can Erozer",
    author_email="ylim2@bu.edu, caner@bu.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
)