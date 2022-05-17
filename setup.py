from setuptools import setup

setup(
    name='modular-coevolution',
    packages=[
        'AlternateGenotypes',
        'AlternateGenerators',
        'diversity',
        'Evolution',
        'GeneticProgramming',
    ],
    include_package_data=True,
    install_requires=["munkres"]
)
