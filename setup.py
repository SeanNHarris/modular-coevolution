from setuptools import setup

setup(
    name='modular-coevolution',
    packages=[
        'alternategenotypes',
        'alternategenerators',
        'diversity',
        'evolution',
        'geneticprogramming',
    ],
    include_package_data=True,
    install_requires=["munkres"]
)
