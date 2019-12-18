Greenotyper (v0.6.0.dev3)
================
[![Build Status](https://api.travis-ci.com/MarniTausen/Greenotyper.svg?branch=master)](https://travis-ci.com/MarniTausen/Greenotyper)[![codecov](https://codecov.io/gh/MarniTausen/Greenotyper/branch/master/graph/badge.svg)](https://codecov.io/gh/MarniTausen/Greenotyper)

-   [Installation](#installation)
-   [General workflow guide](#general-workflow-guide)
-   [GUI interface guide](#gui-interface-guide)
-   [Command line interface guide](#command-line-interface-guide)
-   [Pipeline setup guide](#pipeline-setup-guide)
-   [Neural net training](#neural-net-training)

Greenotyper is a image analysis tool for large scale plant phenotyping experiments.

It uses google's object detection api ([github link](https://github.com/tensorflow/models/tree/master/research/object_detection)) to find the plants and thresholding to measure the size of the plants.

Requirements
------------

- python version 3.6 or 3.7
- tensorflow v2.0.0 or higher
- PyQt5 v5.9.2 or higher
- numpy v1.15.2 or higher
- pillow v5.2.0 or higher
- scikit-image v0.14.0 or higher

Installation
------------

There are precompiled graphical versions in the releases folder.

**Currently there is only a Mac OS X version.** **There is no guarantee that the version works on versions less than 10.14.6**

Install the commandline package through conda:

conda install -c *channel\_name* greenotyper

General workflow guide
----------------------

GUI interface guide
-------------------

Command line interface guide
----------------------------

Command usage help message:
```
=========== GREENOTYPER (v0.6.0.dev3) ===========
greenotyper -i image/directory -p settings.pipeline [options]

Options:
  -h, --help            show this help message and exit
  -i IMAGE, --in=IMAGE  Input image or directory of images for inference
                        (required)
  -n NETWORK, --network=NETWORK
                        Input neural network directory (required, if not
                        provided with pipeline file).
  -p PIPELINE, --pipeline=PIPELINE
                        Pipeline file containing all settings
  -t THREADS, --threads=THREADS
                        Number of threads available. Only used to run on
                        multiple images at a time. Default: 1. Settings less
                        than 0 use all available cores.
  -s SIZEDIRECTORY, --size_output=SIZEDIRECTORY
                        Output directory for the size measurements. Default is
                        no output.
  -g GREENNESSDIRECTORY, --greenness_output=GREENNESSDIRECTORY
                        Output directory for the greenness measurements.
                        Default is no output.
  -m MASKDIRECTORY, --mask_output=MASKDIRECTORY
                        Output directory for the produced masks. Default is no
                        output.
  -c CROPDIRECTORY, --crop_output=CROPDIRECTORY
                        Output directory for the cropped images. Default is no
                        output.
  --by_day              Subdividing the outputs based on per day. Recommended
                        to avoid file system overflow.
  --by_individual       Subdividing the outputs based on per day. Recommended
                        to avoid file system overflow.
  --GUI                 Open up the GREENOTYPER GUI.
  -o ORGANIZEOUTPUT, --organize=ORGANIZEOUTPUT
                        Organize and clean the output. Usage:
                        --organize=input_file output_file.   If included only
                        this action will be performed.
```

Pipeline setup guide
--------------------

Neural net training
-------------------
