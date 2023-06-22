from setuptools import setup

setup(
    name='modular-coevolution',
    description='Evolutionary algorithm library focusing on competitive coevolution',
    author='Sean N. Harris',
    author_email='snh0037@auburn.edu',
    url='https://github.com/SeanNHarris/modular-coevolution',
    packages=[
        'alternategenotypes',
        'alternategenerators',
        'diversity',
        'evolution',
        'geneticprogramming',
    ],
    include_package_data=True,
    install_requires=[],
    extras_require= {
        "alpharank": ["open-spiel"],
        "numba": ["numba"]
    }
)
