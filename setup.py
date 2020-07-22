import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="greenotyper",
    version="0.7.0",
    scripts=["greenotyper"],
    author="Marni Tausen",
    author_email="marni.tausen@gmail.com",
    data_files=[('', ['icon/icon.png'])],
    description="Plant image-based phenotyping pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarniTausen/Greenotyper",
    packages=setuptools.find_packages(),
    python_requires='~=3.6',
    install_requires=[
        "tensorflow>=2",
        "PyQt5>=5.9",
        "numpy>=1.15",
        "scikit-image>=0.14",
        "pillow>=5.2",
        "tqdm>=4.40"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ],
    keywords="phenotyping detection "
)
