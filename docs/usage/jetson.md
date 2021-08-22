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
2021-08-22 18:01:15,260 - Classifier - INFO - Loaded test image model/data/test/Im041_0.jpg
2021-08-22 18:01:16.234253: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2021-08-22 18:01:17.260057: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-08-22 18:01:26,193 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 10.915842771530151 seconds.
2021-08-22 18:01:28,123 - Classifier - INFO - Loaded test image model/data/test/Im028_1.jpg
2021-08-22 18:01:28,798 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.6637485027313232 seconds.
2021-08-22 18:01:31,314 - Classifier - INFO - Loaded test image model/data/test/Im053_1.jpg
2021-08-22 18:01:31,490 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.16409754753112793 seconds.
2021-08-22 18:01:32,955 - Classifier - INFO - Loaded test image model/data/test/Im047_0.jpg
2021-08-22 18:01:33,130 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.16338133811950684 seconds.
2021-08-22 18:01:33,202 - Classifier - INFO - Loaded test image model/data/test/Im026_1.jpg
2021-08-22 18:01:33,367 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.16285347938537598 seconds.
2021-08-22 18:01:35,027 - Classifier - INFO - Loaded test image model/data/test/Im106_0.jpg
2021-08-22 18:01:35,211 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.17293715476989746 seconds.
2021-08-22 18:01:35,485 - Classifier - INFO - Loaded test image model/data/test/Im088_0.jpg
2021-08-22 18:01:35,659 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.16274356842041016 seconds.
2021-08-22 18:01:36,228 - Classifier - INFO - Loaded test image model/data/test/Im074_0.jpg
2021-08-22 18:01:36,401 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.16106271743774414 seconds.
2021-08-22 18:01:36,629 - Classifier - INFO - Loaded test image model/data/test/Im057_1.jpg
2021-08-22 18:01:36,818 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.1783151626586914 seconds.
2021-08-22 18:01:37,055 - Classifier - INFO - Loaded test image model/data/test/Im101_0.jpg
2021-08-22 18:01:37,236 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.170271635055542 seconds.
2021-08-22 18:01:37,300 - Classifier - INFO - Loaded test image model/data/test/Im024_1.jpg
2021-08-22 18:01:37,464 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.16158628463745117 seconds.
2021-08-22 18:01:37,531 - Classifier - INFO - Loaded test image model/data/test/Im006_1.jpg
2021-08-22 18:01:37,706 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.17259979248046875 seconds.
2021-08-22 18:01:38,329 - Classifier - INFO - Loaded test image model/data/test/Im035_0.jpg
2021-08-22 18:01:38,527 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.1852095127105713 seconds.
2021-08-22 18:01:38,590 - Classifier - INFO - Loaded test image model/data/test/Im031_1.jpg
2021-08-22 18:01:38,753 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.1616823673248291 seconds.
2021-08-22 18:01:38,971 - Classifier - INFO - Loaded test image model/data/test/Im069_0.jpg
2021-08-22 18:01:39,144 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.16157913208007812 seconds.
2021-08-22 18:01:39,495 - Classifier - INFO - Loaded test image model/data/test/Im063_1.jpg
2021-08-22 18:01:39,677 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.1709306240081787 seconds.
2021-08-22 18:01:39,900 - Classifier - INFO - Loaded test image model/data/test/Im060_1.jpg
2021-08-22 18:01:41,829 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 1.917875051498413 seconds.
2021-08-22 18:01:41,892 - Classifier - INFO - Loaded test image model/data/test/Im020_1.jpg
2021-08-22 18:01:42,057 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.1628572940826416 seconds.
2021-08-22 18:01:44,639 - Classifier - INFO - Loaded test image model/data/test/Im095_0.jpg
2021-08-22 18:01:44,828 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.17487740516662598 seconds.
2021-08-22 18:01:45,801 - Classifier - INFO - Loaded test image model/data/test/Im099_0.jpg
2021-08-22 18:01:45,987 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.17336750030517578 seconds.
2021-08-22 18:01:45,987 - Classifier - INFO - Images Classified: 20
2021-08-22 18:01:45,988 - Classifier - INFO - True Positives: 10
2021-08-22 18:01:45,988 - Classifier - INFO - False Positives: 0
2021-08-22 18:01:45,988 - Classifier - INFO - True Negatives: 10
2021-08-22 18:01:45,989 - Classifier - INFO - False Negatives: 0
2021-08-22 18:01:45,989 - Classifier - INFO - Total Time Taken: 16.357818841934204
```

&nbsp;

# Running The TFRT Model

Now you will run the TFRT model:

``` bash
python3 classifier.py classify_tfrt
```

You will see the following output:

``` bash
2021-08-22 18:02:50,101 - Classifier - INFO - Loaded test image model/data/test/Im041_0.jpg
2021-08-22 18:02:50.198938: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2021-08-22 18:02:51.947017: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-08-22 18:02:58,320 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 8.210758924484253 seconds.
2021-08-22 18:02:58,938 - Classifier - INFO - Loaded test image model/data/test/Im028_1.jpg
2021-08-22 18:02:58,951 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.008095026016235352 seconds.
2021-08-22 18:03:04,524 - Classifier - INFO - Loaded test image model/data/test/Im053_1.jpg
2021-08-22 18:03:04,545 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.007901668548583984 seconds.
2021-08-22 18:03:04,830 - Classifier - INFO - Loaded test image model/data/test/Im047_0.jpg
2021-08-22 18:03:04,849 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006689786911010742 seconds.
2021-08-22 18:03:04,920 - Classifier - INFO - Loaded test image model/data/test/Im026_1.jpg
2021-08-22 18:03:04,929 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.007235288619995117 seconds.
2021-08-22 18:03:05,379 - Classifier - INFO - Loaded test image model/data/test/Im106_0.jpg
2021-08-22 18:03:05,398 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.007020711898803711 seconds.
2021-08-22 18:03:05,880 - Classifier - INFO - Loaded test image model/data/test/Im088_0.jpg
2021-08-22 18:03:05,897 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006475210189819336 seconds.
2021-08-22 18:03:06,688 - Classifier - INFO - Loaded test image model/data/test/Im074_0.jpg
2021-08-22 18:03:06,705 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006563425064086914 seconds.
2021-08-22 18:03:07,274 - Classifier - INFO - Loaded test image model/data/test/Im057_1.jpg
2021-08-22 18:03:07,292 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006519794464111328 seconds.
2021-08-22 18:03:07,509 - Classifier - INFO - Loaded test image model/data/test/Im101_0.jpg
2021-08-22 18:03:07,527 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006318807601928711 seconds.
2021-08-22 18:03:07,590 - Classifier - INFO - Loaded test image model/data/test/Im024_1.jpg
2021-08-22 18:03:07,598 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006488323211669922 seconds.
2021-08-22 18:03:07,660 - Classifier - INFO - Loaded test image model/data/test/Im006_1.jpg
2021-08-22 18:03:07,668 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006326436996459961 seconds.
2021-08-22 18:03:07,966 - Classifier - INFO - Loaded test image model/data/test/Im035_0.jpg
2021-08-22 18:03:07,983 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006209611892700195 seconds.
2021-08-22 18:03:08,046 - Classifier - INFO - Loaded test image model/data/test/Im031_1.jpg
2021-08-22 18:03:08,054 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006255626678466797 seconds.
2021-08-22 18:03:08,268 - Classifier - INFO - Loaded test image model/data/test/Im069_0.jpg
2021-08-22 18:03:08,290 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006395816802978516 seconds.
2021-08-22 18:03:08,586 - Classifier - INFO - Loaded test image model/data/test/Im063_1.jpg
2021-08-22 18:03:08,604 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006394624710083008 seconds.
2021-08-22 18:03:08,799 - Classifier - INFO - Loaded test image model/data/test/Im060_1.jpg
2021-08-22 18:03:08,817 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006200551986694336 seconds.
2021-08-22 18:03:08,879 - Classifier - INFO - Loaded test image model/data/test/Im020_1.jpg
2021-08-22 18:03:08,888 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.006281137466430664 seconds.
2021-08-22 18:03:09,188 - Classifier - INFO - Loaded test image model/data/test/Im095_0.jpg
2021-08-22 18:03:09,205 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.006270170211791992 seconds.
2021-08-22 18:03:09,413 - Classifier - INFO - Loaded test image model/data/test/Im099_0.jpg
2021-08-22 18:03:09,430 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.0063779354095458984 seconds.
2021-08-22 18:03:09,430 - Classifier - INFO - Images Classified: 20
2021-08-22 18:03:09,431 - Classifier - INFO - True Positives: 10
2021-08-22 18:03:09,431 - Classifier - INFO - False Positives: 0
2021-08-22 18:03:09,431 - Classifier - INFO - True Negatives: 10
2021-08-22 18:03:09,431 - Classifier - INFO - False Negatives: 0
2021-08-22 18:03:09,432 - Classifier - INFO - Total Time Taken: 8.33677887916565

```

&nbsp;

# Running The TensorRT Model

Now you will run the TensorRT model:

``` bash
python3 classifier.py classify_tensorrt
```

You will see the following output:

``` bash
2021-08-22 18:04:13,896 - Classifier - INFO - Loaded test image model/data/test/Im041_0.jpg
2021-08-22 18:04:13,941 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.03255295753479004 seconds.
2021-08-22 18:04:14,115 - Classifier - INFO - Loaded test image model/data/test/Im028_1.jpg
2021-08-22 18:04:14,123 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.002234220504760742 seconds.
2021-08-22 18:04:14,829 - Classifier - INFO - Loaded test image model/data/test/Im053_1.jpg
2021-08-22 18:04:14,842 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.0021958351135253906 seconds.
2021-08-22 18:04:15,070 - Classifier - INFO - Loaded test image model/data/test/Im047_0.jpg
2021-08-22 18:04:15,083 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.0021576881408691406 seconds.
2021-08-22 18:04:15,157 - Classifier - INFO - Loaded test image model/data/test/Im026_1.jpg
2021-08-22 18:04:15,161 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.0022268295288085938 seconds.
2021-08-22 18:04:15,739 - Classifier - INFO - Loaded test image model/data/test/Im106_0.jpg
2021-08-22 18:04:15,752 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.0021431446075439453 seconds.
2021-08-22 18:04:16,051 - Classifier - INFO - Loaded test image model/data/test/Im088_0.jpg
2021-08-22 18:04:16,064 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.002117633819580078 seconds.
2021-08-22 18:04:16,453 - Classifier - INFO - Loaded test image model/data/test/Im074_0.jpg
2021-08-22 18:04:16,466 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.002123594284057617 seconds.
2021-08-22 18:04:16,924 - Classifier - INFO - Loaded test image model/data/test/Im057_1.jpg
2021-08-22 18:04:16,940 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.0021796226501464844 seconds.
2021-08-22 18:04:17,393 - Classifier - INFO - Loaded test image model/data/test/Im101_0.jpg
2021-08-22 18:04:17,406 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.0023069381713867188 seconds.
2021-08-22 18:04:17,473 - Classifier - INFO - Loaded test image model/data/test/Im024_1.jpg
2021-08-22 18:04:17,477 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.0021584033966064453 seconds.
2021-08-22 18:04:17,539 - Classifier - INFO - Loaded test image model/data/test/Im006_1.jpg
2021-08-22 18:04:17,543 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.0021212100982666016 seconds.
2021-08-22 18:04:17,806 - Classifier - INFO - Loaded test image model/data/test/Im035_0.jpg
2021-08-22 18:04:17,818 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.0021283626556396484 seconds.
2021-08-22 18:04:17,889 - Classifier - INFO - Loaded test image model/data/test/Im031_1.jpg
2021-08-22 18:04:17,893 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.0021491050720214844 seconds.
2021-08-22 18:04:18,190 - Classifier - INFO - Loaded test image model/data/test/Im069_0.jpg
2021-08-22 18:04:18,203 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.002172708511352539 seconds.
2021-08-22 18:04:18,922 - Classifier - INFO - Loaded test image model/data/test/Im063_1.jpg
2021-08-22 18:04:18,935 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.002173900604248047 seconds.
2021-08-22 18:04:19,488 - Classifier - INFO - Loaded test image model/data/test/Im060_1.jpg
2021-08-22 18:04:19,501 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.002492666244506836 seconds.
2021-08-22 18:04:19,565 - Classifier - INFO - Loaded test image model/data/test/Im020_1.jpg
2021-08-22 18:04:19,569 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.002119779586791992 seconds.
2021-08-22 18:04:20,562 - Classifier - INFO - Loaded test image model/data/test/Im095_0.jpg
2021-08-22 18:04:20,575 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.0021810531616210938 seconds.
2021-08-22 18:04:21,296 - Classifier - INFO - Loaded test image model/data/test/Im099_0.jpg
2021-08-22 18:04:21,308 - Classifier - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.0022246837615966797 seconds.
2021-08-22 18:04:21,308 - Classifier - INFO - Images Classified: 20
2021-08-22 18:04:21,309 - Classifier - INFO - True Positives: 10
2021-08-22 18:04:21,309 - Classifier - INFO - False Positives: 0
2021-08-22 18:04:21,309 - Classifier - INFO - True Negatives: 10
2021-08-22 18:04:21,310 - Classifier - INFO - False Negatives: 0
2021-08-22 18:04:21,310 - Classifier - INFO - Total Time Taken: 0.07416033744812012
```

&nbsp;

# Conclusion

Here we can see that optimizing our Tensorflow model using TFRT improves our classification by around 8 seconds which is a considerable improvement. When comparing the performance of the TFRT model with the TensorRT model we see an improvement of more than 8 seconds, demonstrating the pure power of Tensorrt and the possibilities it brings to AI on the edge.

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