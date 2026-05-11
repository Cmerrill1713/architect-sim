from setuptools import setup, find_packages
setup(
    name="architect-sim",
    version="0.1.0",
    packages=find_packages(),
    entry_points={"console_scripts": ["architect-sim=architect_sim.cli:main"]},
    install_requires=["pyyaml>=5.0"],
    python_requires=">=3.8",
)
