import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="greenotyper",
    version="0.6.0",
    scripts=["greenotyper"],
    author="Marni Tausen",
    author_email="Marni16ox@gmail.com",
    description="Plant image-based phenotyping pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarniTausen/Greenotyper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
