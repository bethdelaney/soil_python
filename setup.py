from setuptools import setup, find_packages

setup(
    name='soil-python',
    version='0.9',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'rasterio',
        'matplotlib'
    ],
    author='Beth Delaney',
    author_email='beth_delaney@outlook.com',
    description='Python implementation of various soil spectral applications',
    url='https://github.com/bethdelaney/soil_python.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
