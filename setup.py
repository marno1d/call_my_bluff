"""Setup file for the call_my_bluff package."""
from setuptools import setup, find_packages

setup(
    name="call_my_bluff",
    version="0.1",
    description="Implementation of the Call My Bluff game",
    author="Matthew Arnold",
    author_email="10799696+marno1d@users.noreply.github.com",
    url="https://github.com/marno1d/call_my_bluff",
    packages=find_packages(),
    install_requires=["numpy", "tqdm"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
