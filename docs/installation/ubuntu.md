# Ubuntu Installation

![ALL Jetson Nano Classifier](../img/project-banner.jpg)

# Introduction
This guide will take you through the installation process for the **ALL Jetson Nano Classifier** trainer.

&nbsp;

# Operating System
This project supports the following operating system(s), but may work as described on other OS.

- [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)

&nbsp;

# Software
This project uses the following libraries.

- Conda
- Intel® oneAPI AI Analytics Toolkit
- Jupyter Notebooks
- NBConda
- Mlxtend
- Pillow
- Opencv
- Scipy
- Scikit Image
- Scikit Learn

&nbsp;

# Clone the repository

Clone the [ALL Jetson Nano Classifier](https://github.com/AMLResearchProject/ALL-Jetson-Nano-Classifier " ALL Jetson Nano Classifier") repository from the [Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project](https://github.com/AMLResearchProject "Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project") Github Organization.

To clone the repository and install the project, make sure you have Git installed. Now navigate to the directory you would like to clone the project to and then use the following command.

``` bash
 git clone https://github.com/AMLResearchProject/ALL-Jetson-Nano-Classifier.git
```

This will clone the ALL Jetson Nano Classifier repository.

``` bash
 ls
```

Using the ls command in your home directory should show you the following.

``` bash
 ALL-Jetson-Nano-Classifier
```

Navigate to the **ALL-Jetson-Nano-Classifier** directory, this is your project root directory for this tutorial.

## Developer forks

Developers from the Github community that would like to contribute to the development of this project should first create a fork, and clone that repository. For detailed information please view the research project [CONTRIBUTING](https://github.com/AMLResearchProject/Contributing-Guide/blob/main/CONTRIBUTING.md "CONTRIBUTING") guide. You should pull the latest code from the development branch.

``` bash
 git clone -b "develop" https://github.com/AMLResearchProject/ALL-Jetson-Nano-Classifier.git
```

The **-b "develop"** parameter ensures you get the code from the latest develop branch.

&nbsp;

# Installation

First you will install the [Intel® oneAPI Basekit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html), [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) and the Intel oneAPI GPU drivers.

**WARNING:** This script assumes you have not already installed the oneAPI Basekit.

**WARNING:** This script assumes you have not already installed the oneAPI AI Analytics Toolkit.

**WARNING:** This script assumes you have an Intel GPU.

**WARNING:** This script assumes you have already installed the Intel GPU drivers.

**HINT:** If any of the above are not relevant to you, please comment out the relevant sections below before running this installation script.

From the project root run the following command in terminal:

``` bash
 sh scripts/oneapi.sh
```

You are now ready to install the ALL Jetson Nano Classifier trainer. All software requirements are included in **scripts/install.sh**. You can run this file on your machine from the project root in terminal.

Before you begin make sure you have activated the **all-jetson-nano** Conda environment.

``` bash
 conda activate all-jetson-nano
```

Then use the following command to install the required software:

``` bash
 sh scripts/install.sh
```

&nbsp;

# Continue

Choose one of the following usage guides to train your model:

- [Python Usage Guide](../usage/python.md)
- [Jupyter Notebook Usage Guide](../usage/notebooks.md)

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
We use the [repo issues](https://github.com/AMLResearchProject/ALL-Jetson-Nano/issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](https://github.com/AMLResearchProject/Contributing-Guide/blob/main/CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.