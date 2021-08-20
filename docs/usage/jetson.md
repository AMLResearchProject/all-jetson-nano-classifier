# Jetson Nano Usage

![ALL Jetson Nano Classifier](../img/project-banner.jpg)

# Introduction
This guide will take you through the using the **ALL Jetson Nano Classifier** to detect Acute Lymphoblastic Leukemia.

&nbsp;

# Installation
You will need to ensure you have followed the provided guides below:

- [Ubuntu Installation Guide](../installation/ubuntu.md)
- [NVIDIA Jetson Nano Installation Guide](../installation/jetson.md)

&nbsp;

# Training
Before you can start to use this tutorial you must have already trained your classifier, to do so use one of the following guides:

- [Python Usage Guide](../usage/python.md).
- [Jupyter Notebooks Usage Guide](../usage/notebooks.md)

&nbsp;

# Docker Container
During the [NVIDIA Jetson Nano Installation](../installation/jetson.md) you download the ALL Jetson Nano Classifier Docker Image and started it. You should make sure you are inside the Docker container before you continue this tutorial.

# Running The Tensorflow Model

Now you are ready to run the Tensorflow model. You will use the 20 images that were removed from the training data in the previous tutorials.

To run the classifier in use the following command:

```
python3 classifier.py classify
```

You should see the following which shows you the network architecture:

```
2021-08-19 21:21:53,645 - Classifier - INFO - Model loaded
Model: "AllJetsonNano"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
average_pooling2d (AveragePo (None, 50, 50, 3)         0
_________________________________________________________________
conv2d (Conv2D)              (None, 46, 46, 30)        2280
_________________________________________________________________
depthwise_conv2d (DepthwiseC (None, 17, 17, 30)        27030
_________________________________________________________________
flatten (Flatten)            (None, 8670)              0
_________________________________________________________________
dense (Dense)                (None, 2)                 17342
_________________________________________________________________
softmax (Activation)         (None, 2)                 0
=================================================================
Total params: 46,652
Trainable params: 46,652
Non-trainable params: 0
_________________________________________________________________
```

Finally the application will start processing the test images and the results will be displayed in the console. The first classification will take longer due to loading the required dynamic libraries.

```
2021-08-19 21:21:54,042 - Classifier - INFO - Loaded test image model/data/test/Im041_0.jpg
2021-08-19 21:21:54.781830: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2021-08-19 21:21:56.736191: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-08-19 21:22:04,620 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 10.574294567108154 seconds.
2021-08-19 21:22:05,944 - Classifier - INFO - Loaded test image model/data/test/Im028_1.jpg
2021-08-19 21:22:06,134 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.18299126625061035 seconds.
2021-08-19 21:22:07,205 - Classifier - INFO - Loaded test image model/data/test/Im053_1.jpg
2021-08-19 21:22:07,382 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.16392922401428223 seconds.
2021-08-19 21:22:07,852 - Classifier - INFO - Loaded test image model/data/test/Im047_0.jpg
2021-08-19 21:22:08,032 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.16694879531860352 seconds.
2021-08-19 21:22:08,107 - Classifier - INFO - Loaded test image model/data/test/Im026_1.jpg
2021-08-19 21:22:08,271 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.1616969108581543 seconds.
2021-08-19 21:22:08,807 - Classifier - INFO - Loaded test image model/data/test/Im106_0.jpg
2021-08-19 21:22:08,995 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.17544174194335938 seconds.
2021-08-19 21:22:09,192 - Classifier - INFO - Loaded test image model/data/test/Im088_0.jpg
2021-08-19 21:22:09,371 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.16634845733642578 seconds.
2021-08-19 21:22:09,806 - Classifier - INFO - Loaded test image model/data/test/Im074_0.jpg
2021-08-19 21:22:09,980 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.16352033615112305 seconds.
2021-08-19 21:22:10,259 - Classifier - INFO - Loaded test image model/data/test/Im057_1.jpg
2021-08-19 21:22:10,441 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.17132163047790527 seconds.
2021-08-19 21:22:10,650 - Classifier - INFO - Loaded test image model/data/test/Im101_0.jpg
2021-08-19 21:22:10,822 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.1612401008605957 seconds.
2021-08-19 21:22:10,886 - Classifier - INFO - Loaded test image model/data/test/Im024_1.jpg
2021-08-19 21:22:11,051 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.16300010681152344 seconds.
2021-08-19 21:22:11,115 - Classifier - INFO - Loaded test image model/data/test/Im006_1.jpg
2021-08-19 21:22:11,278 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.16106534004211426 seconds.
2021-08-19 21:22:11,500 - Classifier - INFO - Loaded test image model/data/test/Im035_0.jpg
2021-08-19 21:22:11,684 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.17350363731384277 seconds.
2021-08-19 21:22:11,748 - Classifier - INFO - Loaded test image model/data/test/Im031_1.jpg
2021-08-19 21:22:11,912 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.16186809539794922 seconds.
2021-08-19 21:22:12,151 - Classifier - INFO - Loaded test image model/data/test/Im069_0.jpg
2021-08-19 21:22:12,322 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.16080045700073242 seconds.
2021-08-19 21:22:12,839 - Classifier - INFO - Loaded test image model/data/test/Im063_1.jpg
2021-08-19 21:22:13,020 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.17079424858093262 seconds.
2021-08-19 21:22:13,434 - Classifier - INFO - Loaded test image model/data/test/Im060_1.jpg
2021-08-19 21:22:14,971 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 1.5258052349090576 seconds.
2021-08-19 21:22:15,034 - Classifier - INFO - Loaded test image model/data/test/Im020_1.jpg
2021-08-19 21:22:15,199 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.16217803955078125 seconds.
2021-08-19 21:22:16,319 - Classifier - INFO - Loaded test image model/data/test/Im095_0.jpg
2021-08-19 21:22:16,492 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.1624619960784912 seconds.
2021-08-19 21:22:18,036 - Classifier - INFO - Loaded test image model/data/test/Im099_0.jpg
2021-08-19 21:22:18,223 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.17438983917236328 seconds.
2021-08-19 21:22:18,224 - Classifier - INFO - Images Classified: 20
2021-08-19 21:22:18,224 - Classifier - INFO - True Positives: 10
2021-08-19 21:22:18,224 - Classifier - INFO - False Positives: 0
2021-08-19 21:22:18,225 - Classifier - INFO - True Negatives: 10
2021-08-19 21:22:18,225 - Classifier - INFO - False Negatives: 0
2021-08-19 21:22:18,225 - Classifier - INFO - Total Time Taken: 15.103600025177002
```

&nbsp;

# Running The TFRT Model

Now you will run the TFRT model:

``` bash
python3 classifier.py classify_tfrt
```

You will see the following output:

``` bash
2021-08-19 21:23:17.557074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1428] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 375 MB memory) -> physical GPU (device: 0, name: NVIDIA Tegra X1, pci bus id: 0000:00:00.0, compute capability: 5.3)
2021-08-19 21:23:22,277 - Classifier - INFO - Loaded test image model/data/test/Im041_0.jpg
2021-08-19 21:23:22.403990: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2021-08-19 21:23:24.297541: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-08-19 21:23:29,946 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 7.66000771522522 seconds.
2021-08-19 21:23:31,574 - Classifier - INFO - Loaded test image model/data/test/Im028_1.jpg
2021-08-19 21:23:31,589 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.007614850997924805 seconds.
2021-08-19 21:23:33,488 - Classifier - INFO - Loaded test image model/data/test/Im053_1.jpg
2021-08-19 21:23:33,506 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006653547286987305 seconds.
2021-08-19 21:23:33,825 - Classifier - INFO - Loaded test image model/data/test/Im047_0.jpg
2021-08-19 21:23:33,844 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006745576858520508 seconds.
2021-08-19 21:23:33,915 - Classifier - INFO - Loaded test image model/data/test/Im026_1.jpg
2021-08-19 21:23:33,924 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.0067136287689208984 seconds.
2021-08-19 21:23:34,398 - Classifier - INFO - Loaded test image model/data/test/Im106_0.jpg
2021-08-19 21:23:34,417 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006788492202758789 seconds.
2021-08-19 21:23:34,708 - Classifier - INFO - Loaded test image model/data/test/Im088_0.jpg
2021-08-19 21:23:34,726 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006501197814941406 seconds.
2021-08-19 21:23:34,928 - Classifier - INFO - Loaded test image model/data/test/Im074_0.jpg
2021-08-19 21:23:34,946 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006418704986572266 seconds.
2021-08-19 21:23:35,234 - Classifier - INFO - Loaded test image model/data/test/Im057_1.jpg
2021-08-19 21:23:35,251 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006302595138549805 seconds.
2021-08-19 21:23:35,519 - Classifier - INFO - Loaded test image model/data/test/Im101_0.jpg
2021-08-19 21:23:35,535 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006386280059814453 seconds.
2021-08-19 21:23:35,599 - Classifier - INFO - Loaded test image model/data/test/Im024_1.jpg
2021-08-19 21:23:35,607 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.0062999725341796875 seconds.
2021-08-19 21:23:35,669 - Classifier - INFO - Loaded test image model/data/test/Im006_1.jpg
2021-08-19 21:23:35,677 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.00655674934387207 seconds.
2021-08-19 21:23:35,983 - Classifier - INFO - Loaded test image model/data/test/Im035_0.jpg
2021-08-19 21:23:36,000 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006490468978881836 seconds.
2021-08-19 21:23:36,066 - Classifier - INFO - Loaded test image model/data/test/Im031_1.jpg
2021-08-19 21:23:36,075 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006487846374511719 seconds.
2021-08-19 21:23:36,384 - Classifier - INFO - Loaded test image model/data/test/Im069_0.jpg
2021-08-19 21:23:36,401 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006212949752807617 seconds.
2021-08-19 21:23:36,629 - Classifier - INFO - Loaded test image model/data/test/Im063_1.jpg
2021-08-19 21:23:36,647 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006224155426025391 seconds.
2021-08-19 21:23:36,874 - Classifier - INFO - Loaded test image model/data/test/Im060_1.jpg
2021-08-19 21:23:36,892 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006279706954956055 seconds.
2021-08-19 21:23:36,954 - Classifier - INFO - Loaded test image model/data/test/Im020_1.jpg
2021-08-19 21:23:36,962 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006190299987792969 seconds.
2021-08-19 21:23:37,256 - Classifier - INFO - Loaded test image model/data/test/Im095_0.jpg
2021-08-19 21:23:37,276 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006523609161376953 seconds.
2021-08-19 21:23:37,482 - Classifier - INFO - Loaded test image model/data/test/Im099_0.jpg
2021-08-19 21:23:37,499 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.00617218017578125 seconds.
2021-08-19 21:23:37,500 - Classifier - INFO - Images Classified: 20
2021-08-19 21:23:37,500 - Classifier - INFO - True Positives: 10
2021-08-19 21:23:37,500 - Classifier - INFO - False Positives: 0
2021-08-19 21:23:37,501 - Classifier - INFO - True Negatives: 10
2021-08-19 21:23:37,501 - Classifier - INFO - False Negatives: 0
2021-08-19 21:23:37,501 - Classifier - INFO - Total Time Taken: 7.7835705280303955
```

&nbsp;

# Running The TensorRT Model

Now you will run the TensorRT model:

``` bash
python3 classifier.py classify_tensorrt
```

You will see the following output:

``` bash
2021-08-19 21:25:47,794 - Classifier - INFO - Loaded test image model/data/test/Im041_0.jpg
2021-08-19 21:25:57,528 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 9.695297002792358 seconds.
2021-08-19 21:25:58,770 - Classifier - INFO - Loaded test image model/data/test/Im028_1.jpg
2021-08-19 21:25:58,891 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.05570673942565918 seconds.
2021-08-19 21:25:59,799 - Classifier - INFO - Loaded test image model/data/test/Im053_1.jpg
2021-08-19 21:25:59,825 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.01838541030883789 seconds.
2021-08-19 21:26:00,039 - Classifier - INFO - Loaded test image model/data/test/Im047_0.jpg
2021-08-19 21:26:00,064 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.017393112182617188 seconds.
2021-08-19 21:26:00,174 - Classifier - INFO - Loaded test image model/data/test/Im026_1.jpg
2021-08-19 21:26:00,192 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.016171693801879883 seconds.
2021-08-19 21:26:06,638 - Classifier - INFO - Loaded test image model/data/test/Im106_0.jpg
2021-08-19 21:26:06,665 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.017331838607788086 seconds.
2021-08-19 21:26:07,826 - Classifier - INFO - Loaded test image model/data/test/Im088_0.jpg
2021-08-19 21:26:07,854 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.018476009368896484 seconds.
2021-08-19 21:26:16,488 - Classifier - INFO - Loaded test image model/data/test/Im074_0.jpg
2021-08-19 21:26:16,516 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.017465829849243164 seconds.
2021-08-19 21:26:24,054 - Classifier - INFO - Loaded test image model/data/test/Im057_1.jpg
2021-08-19 21:26:24,082 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.01739645004272461 seconds.
2021-08-19 21:26:26,183 - Classifier - INFO - Loaded test image model/data/test/Im101_0.jpg
2021-08-19 21:26:26,208 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.014073848724365234 seconds.
2021-08-19 21:26:26,272 - Classifier - INFO - Loaded test image model/data/test/Im024_1.jpg
2021-08-19 21:26:26,292 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.018465757369995117 seconds.
2021-08-19 21:26:26,355 - Classifier - INFO - Loaded test image model/data/test/Im006_1.jpg
2021-08-19 21:26:26,375 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.018265962600708008 seconds.
2021-08-19 21:26:28,109 - Classifier - INFO - Loaded test image model/data/test/Im035_0.jpg
2021-08-19 21:26:28,137 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.017232894897460938 seconds.
2021-08-19 21:26:28,208 - Classifier - INFO - Loaded test image model/data/test/Im031_1.jpg
2021-08-19 21:26:28,223 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.013829946517944336 seconds.
2021-08-19 21:26:29,144 - Classifier - INFO - Loaded test image model/data/test/Im069_0.jpg
2021-08-19 21:26:29,172 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.01685953140258789 seconds.
2021-08-19 21:26:30,215 - Classifier - INFO - Loaded test image model/data/test/Im063_1.jpg
2021-08-19 21:26:30,243 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.018213987350463867 seconds.
2021-08-19 21:26:31,108 - Classifier - INFO - Loaded test image model/data/test/Im060_1.jpg
2021-08-19 21:26:31,136 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.016917705535888672 seconds.
2021-08-19 21:26:31,199 - Classifier - INFO - Loaded test image model/data/test/Im020_1.jpg
2021-08-19 21:26:31,216 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.015204191207885742 seconds.
2021-08-19 21:26:33,055 - Classifier - INFO - Loaded test image model/data/test/Im095_0.jpg
2021-08-19 21:26:33,083 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.01775836944580078 seconds.
2021-08-19 21:26:34,294 - Classifier - INFO - Loaded test image model/data/test/Im099_0.jpg
2021-08-19 21:26:34,322 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.017148733139038086 seconds.
2021-08-19 21:26:34,322 - Classifier - INFO - Images Classified: 20
2021-08-19 21:26:34,322 - Classifier - INFO - True Positives: 10
2021-08-19 21:26:34,323 - Classifier - INFO - False Positives: 0
2021-08-19 21:26:34,323 - Classifier - INFO - True Negatives: 10
2021-08-19 21:26:34,323 - Classifier - INFO - False Negatives: 0
2021-08-19 21:26:34,324 - Classifier - INFO - Total Time Taken: 10.057595014572144
```

&nbsp;

# Conclusion

Here we can see that optimizing our Tensorflow model using TFRT improves our classification by around 8 seconds which is a considerable improvement. When comparing the performance of the TFRT model with the TensorRT model however, we see an increase in classification time of around 3 seconds, this is something that should be investigated.

This completes our tutorial for now. You have created a Tensorflow model for Acute Lymphoblastic Leukemia classification, converted it to TFRT and ONNX formats, and converted the ONNX model to TensorRT format, then ran each model on your desktop/laptop device and the NVIDIA Jetson Nano.

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