Greenotyper (v0.6.1)
================
[![Build Status](https://api.travis-ci.com/MarniTausen/Greenotyper.svg?branch=master)](https://travis-ci.com/MarniTausen/Greenotyper)[![codecov](https://codecov.io/gh/MarniTausen/Greenotyper/branch/master/graph/badge.svg)](https://codecov.io/gh/MarniTausen/Greenotyper)[![PyPI version](https://badge.fury.io/py/greenotyper.svg)](https://badge.fury.io/py/greenotyper)

-   [Installation](#installation)
-   [General workflow guide](#general-workflow-guide)
-   [GUI interface guide](#gui-interface-guide)
-   [Command line interface guide](#command-line-interface-guide)
-   [Pipeline setup guide](#pipeline-setup-guide)
-   [Neural net training](#neural-net-training)

Greenotyper is an image analysis tool for large scale plant phenotyping experiments.

It uses google's object detection api ([GitHub link](https://github.com/tensorflow/models/tree/master/research/object_detection)) to find the plants and thresholding to measure the size of the plants.

Requirements
------------

- python version 3.6 or 3.7
- tensorflow v2.0.0 or higher
- PyQt5 v5.9.2 or higher
- numpy v1.15.2 or higher
- pillow v5.2.0 or higher
- scikit-image v0.14.0 or higher
- Keras v2 or higher

Installation
------------

It is recommended to install the tool in a virtualenv or in an environment in conda. Example:

```bash
conda create -n greenotyper_env python=3.7

conda activate greenotyper_env

pip install greenotyper
```
Install the latest version of greenotyper through pip:

```bash
pip install greenotyper
```
If there are problems with pip you can try calling pip3 instead:

```bash
pip3 install greenotyper
```

Install greenotyper through conda:
```
not available yet
```


General workflow guide
----------------------

Starting a new workflow requires setting up and testing the pipeline. It starts by opening the pipeline planner. Either you open the Greenotyper app, or opening the GUI through the command line interface:

```bash
greenotyper --GUI
```

To open the pipeline planner, click the Pipeline planner button.

Testing the plant area detection, the network and pipeline settings are all done through the pipeline planner. For information on how use the interface go to the next section, and for general information on Pipeline setups click [here](#pipeline-setup-guide).

Running the pipeline is done either through the command line or through the GUI. The command line is more efficient and can more easily be deployed on computing clusters.

The pipeline can be run on individual images or directories of images. The results are a single "database" file, which uses file locking. (If your file system has blocked file locking, then there is no guarantee the results will be correctly written when run using multi processing.)

To organise the results into a table you can use the command line option:

```bash
greenotyper -p mypipeline.pipeline -o input_file.csv output_file.csv
```

GUI interface guide
-------------------

Open the app, or run the GUI from the terminal:
https://github.com/MarniTausen/Greenotyper

### Pipeline Planner

#### Basics
First open the pipeline planner from the initial window.
![](README_images/open_pipeline_planner.gif)

Open your image.

![](README_images/open_image.gif)

Opening a trained network.

![](README_images/open_network.gif)

After both an image and the network have been opened, you can run find plants feature. Clicking on Find plants will draw bounding boxes around the detected plants.

![](README_images/find_plants.gif)

To test the detection of the plant area you can use apply mask function.

![](README_images/apply_mask.gif)

#### Adjust mask settings

#### Adjust pipeline settings

### Pipeline Runner (Initial window)



Command line interface guide
----------------------------

Command line usage help message:

```
Usage: 
=========== GREENOTYPER (v0.6.1) ===========
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
  --by_individual       Subdividing the outputs based on per individual.
                        Recommended to avoid file system overflow.
  --GUI                 Open up the GREENOTYPER GUI.
  -o ORGANIZEOUTPUT, --organize=ORGANIZEOUTPUT
                        Organize and clean the output. Usage:
                        --organize=input_file output_file.   If included only
                        this action will be performed.
  --unet=UNET           Whether a UNET should be used to segment the plants.
                        Input: {unet.hdf5} {preprocess_dir}
  --unet-preprocess=UNETPRE
                        Preprocess crops for running in Unet. Provide a
                        directory to output the preprocessing information.
  --unet-postprocess=UNETPOST
                        Postprocess the masks from Unet, and output results.
                        Provide the {preprocess_dir}.
```

Pipeline setup guide
--------------------

Neural net training
-------------------

### Object detection
#### installation
The object detection is done using the tensorflow object detection api, found on [GitHub here](https://github.com/tensorflow/models/tree/master/research/object_detection).

This guide has been tested on commit up to: [8518d05](https://github.com/tensorflow/models/commit/8518d053936aaf30afb9ed0a4ea01baddca5bd17). Future versions might change and the following guide might not be relevant. To use the version that is known to work, you can open the commit, and click browse files and download the whole models repository from that commit.

The object detection api only works on tensorflow 1.x versions, and therefore should be trainined an enivorinment installed with the latest tensorflow 1.x version. It does not work with tensorflow 2+.

The whole install guide provided [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). If access to a GPU is available choose the tensorflow-gpu install over tensorflow. To be able use GPU, the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) must be installed. Depending on the version of tensorflow installed, it depends on [different versions](https://www.tensorflow.org/install/source#tested_build_configurations). Supported tensorflow versions, 1.12.0, 1.13.0, 1.14.0 use different versions of CUDA. Version 1.12.0, depends on version 9 of CUDA, and versions 1.13.0 and 1.14.0 depend on version 10 of CUDA.

Here is a version of installing that worked on a Mac OS X system:

```bash
conda create -n ObjectDetection python=3.6

conda activate ObjectDetection

pip install tensorflow==1.14
```

Versions 1.13 and 1.12 of tensorflow should also work. Install the tensorflow-gpu version if the intent is to train on a GPU.
```bash
pip install tensorflow-gpu==1.14
```

Pip install tensorflow gets nearly all of the dependencies listed on the guide. However the remaining dependencies were installed like this:

```bash
conda install protobuf

pip install --user pycocotools
```

Next was retreiving the object detection API, by downloading the whole models repository. The API is dependent on other research packages in the repository. So start by cloning the latest version, or download this [commit](https://github.com/tensorflow/models/commit/8518d053936aaf30afb9ed0a4ea01baddca5bd17).

```bash
git clone https://github.com/tensorflow/models.git
```

Next is to "compile" some of the code from the api using the following command:

```bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
```

Next make API callable, by exporting the directory to the python path:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Now you can test whether the API works by running the following command:

```bash
python object_detection/builders/model_builder_tf1_test.py
```

You should get OK on all of the tests at the end. If you use tensorflow 1.14 you will get a lot of warnings, due to the version preparing people to upgrade to version 2, but you can ignore these.

#### Preparing training and testing data
The training and testing data was created using the [labelImg tool](https://github.com/tzutalin/labelImg). The bounding boxes are manually drawn using labelImg, which outputs .xml files which describes the bounding boxes which have been drawn and the name of the class.

The data has to be processed into into tha different format so that the object detection api can read and use the training and testing data.

For this we created a simple script which converts the image + .xml files into .record files used by the object detection api. The scripts can be found [here](https://github.com/MarniTausen/Greenotyper/tree/master/training_data/object%20detection/create_tf_input.py).
Usage of the script is as follows:

```bash
python create_tf_input.py inputdirectory -r output.record -l label_map.pbtxt
```

To produce the training data, the images with the xml files must be stored in a directory:

```bash
python create_tf_input.py traindirectory -r train.record -l label_map.pbtxt
```

The same for the testing data:

```bash
python create_tf_input.py testdirectory -r test.record -l label_map.pbtxt
```

Finally the [pipeline.config](https://github.com/MarniTausen/Greenotyper/blob/master/training_data/object%20detection/pipeline.config) file must be updated. Depending on what is being training, setting what the number of classes are being trained is important, and the number of steps the network is trained on. The full file locations of the training and testing (evaluation) data must be updated.

#### Training and Testing

Training can now be run following the guide [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md). Training and evaluation (testing) are run with the same command.

To see the evaluation results you use tensorboard, which has been installed with tensorflow.

To export the network you can use the following export_inference_graph.py

```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path path/to/filename.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
```

This function outputs the frozen\_inference\_graph.pd. Adding this file together with the label_map.pbtxt into a network directory creates the network input used in Greenotyper.

### U-net

