import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-optimizer",
    version="0.0.3",
    author="Kazim SOYLU",
    author_email="kazimsoylu@gmail.com",
    description="Evolutionary Keras Hyperparameter Optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazimsoylu/KerasOptimizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)