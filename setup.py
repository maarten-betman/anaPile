from setuptools import setup

setup(
    name="anapile",
    version="0.1",
    description="pile foundations.",
    author="Ritchie Vink",
    author_email="ritchie46@gmail.com",
    url="https://ritchievink.com",
    download_url="https://github.com/ritchie46/anaPile",
    license="GPL-3.0",
    packages=["anapile", "anapile.geo", "anapile.pressure", "anapile.settlement"],
    install_requires=["matplotlib>=3.1", "numpy>=1.16", "scipy==1.3.0"],
)
