from setuptools import setup

setup(
    name='modular-coevolution',
    packages=[
        'alternate_genotypes',
        'AlternateGenerators',
        'diversity',
        'Evolution',
        'GeneticProgramming',
    ],
    include_package_data=True,
    install_requires=["munkres"]
)
