from setuptools import setup, find_packages

setup(
    name='bitcoin_markets',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.21.1',
        'pandas==1.3.1',
        'scikit-learn==0.24.2',
        'tensorflow==2.5.0',
    ],
    entry_points={
        'console_scripts': [
            'bitcoin_markets=bitcoin_markets.main:main',
        ],
    },
)
