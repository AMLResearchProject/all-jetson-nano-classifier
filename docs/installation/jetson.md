# Jetson Nano Installation

![ALL Jetson Nano Classifier](../img/project-banner.jpg)

# Introduction
This guide will guide you through the installation process for the **ALL Jetson Nano Classifier** on your Jetson Nano device.

&nbsp;

# Operating System
This project requires the [Jetson Nano Developer Kit SD Card Image](https://developer.nvidia.com/jetson-nano-sd-card-image). For information on how to set up your Jetson Nano prior to starting this tutorial visit [Getting Started with Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

&nbsp;

# Hardware
The following hardware is required for this project:

- [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)

&nbsp;

# Software
We have prepared a Docker image for this project to allow easy installation on the NVIDIA Jetson Nano. To download the image visit the [petermossleukemiaairesearch/all-jetson-nano](https://hub.docker.com/r/petermossleukemiaairesearch/all-jetson-nano) repository.

&nbsp;

# Prerequisites
You will need to ensure you have followed the provided guides below:

- [Ubuntu Installation Guide](../installation/ubuntu.md)
- [Python Usage Guide](../usage/python.md) or [Jupyter Notebooks Usage Guide](../usage/notebooks.md)

&nbsp;

# Setup

The first thing you need to do is transfer the trained model and data from your development/training machine to your home directory on the Jetson Nano. The remainder of this tutorial assumes you have copied the entire ALL-Jetson-Nano-Classifier directory to your home directory on the Jetson Nano.

In your home directory on your Jetson Nano, ensure that you have completed this step.

``` bash
ls
```

You should see the following output:

```
ALL-Jetson-Nano-Classifier
```

You should now ensure that you have the following:

- The test data is in the model/data/test directory
- The model/saved directory is not empty and contains the saved_model.pb file, assets and variables directories.
- The model/tfrt directory is not empty and contains the saved_model.pb file, assets and variables directories.
- The all_jetson_nano.json file is in the model directory
- The all_jetson_nano.h5 file is in the model directory
- The all_jetson_nano.onnx file is in the model directory

&nbsp;

# Docker Image

Now you are ready to use the ALL Jetson Nano Classifier docker image on your Jetson Nano.

Open terminal on your Jetson Nano or login using ssh and use the following command to download the image, start the container and mount the ALL-Jetson-Nano-Classifier directory as a volume in the container. Ensure you change YourUser with the user you are logged into the nano on.

``` bash
sudo docker run -it --name all-jetson-nano --rm --runtime nvidia --network host -v /home/YourUser/ALL-Jetson-Nano-Classifier:/ALL-Jetson-Nano-Classifier petermossleukemiaairesearch/all-jetson-nano:v1
```

If you want to create your own Docker image, steps have been provided in the [scripts/docker.txt](https://github.com/AMLResearchProject/ALL-Jetson-Nano-Classifier/blob/main/scripts/docker.txt) file.

&nbsp;

# Continue

Now you are ready to use the ALL Jetson Nano Classifier. Head over to the [Jetson Nano Usage Guide](../usage/jetson.md) for instructions on how to use your model with on the Jetson Nano.

&nbsp;

# Contributing
Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss encourages and welcomes code contributions, bug fixes and enhancements from the Github community.

Please read the [CONTRIBUTING](https://github.com/AMLResearchProject/Contributing-Guide/blob/main/CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find our code of conduct in the [Code of Conduct](https://github.com/AMLResearchProject/Contributing-Guide/blob/main/CODE-OF-CONDUCT.md) document.

## Contributors
- [Adam Milton-Barker](https://www.leukemiaairesearch.com/association/volunteers/adam-milton-barker "Adam Milton-Barker") - [Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss") President/Founder & Lead Developer, Sabadell, Spain

&nbsp;

# Versioning
We use [SemVer](https://semver.org/) for versioning.

&nbsp;

# License
This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/AMLResearchProject/ALL-Jetson-Nano-Classifier/blob/main/LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues
We use the [repo issues](https://github.com/AMLResearchProject/ALL-Jetson-Nano-Classifier/issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](https://github.com/AMLResearchProject/Contributing-Guide/blob/main/CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.