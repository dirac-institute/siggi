from setuptools import setup, find_packages

setup(
    name="siggi",
    version="0.1.4",
    author="Bryce Kalmbach",
    author_email="brycek@uw.edu",
    url="https://github.com/jbkalmbach/siggi",
    packages=find_packages(),
    description="Spectral Information Gain Optimization",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"],
                  "siggi": ["data/*",
                            "data/lsst_baseline_throughputs/*"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
)
