from PythonKnitr import Knitr

document = Knitr("Greenotyper", "Marni Tausen")

document.text("""
Greenotyper is a image analysis tool for large scale plant phenotyping experiments.

It uses google's object detection api ([github here](https://github.com/tensorflow/models/tree/master/research/object_detection)) to find the plants and thresholding to measure the size of the plants.

paper doi: link here
""")


document.text("""
There are precompiled graphical versions in the releases folder.

**Currently there is only a Mac OS X version.**
**There is no guarantee that the version works on versions less than 10.14.6**

Install the commandline package through conda:

conda install -c *channel_name* greenotyper

""", title="Installation")


document.text("""


""", title="General workflow guide")


document.text("""

""", title="GUI interface guide")

document.text("""

""", title="Command line interface guide")

document.text("""

""", title="Pipeline setup guide")

document.text("""

""", title="Neural net training")

document.compile("README.Rmd", output_type="github")
