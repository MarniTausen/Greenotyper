Greenotyper

[![Build Status](https://api.travis-ci.com/MarniTausen/Greenotyper.svg?branch=master)](https://travis-ci.com/MarniTausen/Greenotyper)
================
Marni Tausen
10/12/2019 - 13:44:25

-   [Installation](#installation)
-   [General workflow guide](#general-workflow-guide)
-   [GUI interface guide](#gui-interface-guide)
-   [Command line interface guide](#command-line-interface-guide)
-   [Pipeline setup guide](#pipeline-setup-guide)
-   [Neural net training](#neural-net-training)

Greenotyper is a image analysis tool for large scale plant phenotyping experiments.

It uses google's object detection api ([github here](https://github.com/tensorflow/models/tree/master/research/object_detection)) to find the plants and thresholding to measure the size of the plants.

paper doi: link here


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

Pipeline setup guide
--------------------

Neural net training
-------------------
