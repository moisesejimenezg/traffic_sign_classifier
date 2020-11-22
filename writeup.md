# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[histogram]: ./images/histogram.png "Histogram"
[signs]: ./images/signs.png "Signs"
[grayscale]: ./images/grayscale.png "Grayscale"
[normalize]: ./images/normalize.png "Normalized"
[internet0]: ./extra_traffic_signs/00000.jpg "Extra Traffic Sign 0"
[internet1]: ./extra_traffic_signs/00001.jpg "Extra Traffic Sign 1"
[internet2]: ./extra_traffic_signs/00002.jpg "Extra Traffic Sign 2"
[internet3]: ./extra_traffic_signs/00003.jpg "Extra Traffic Sign 3"
[internet4]: ./extra_traffic_signs/00004.jpg "Extra Traffic Sign 4"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/moisesejimenezg/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the
test, validation and trainin datasets.

![alt text][histogram]

Furthermore an example of each one of the different classes is also displayed below.

![alt text][signs]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the information on the signs is mainly conveyed
through shapes and not color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscale]

As a last step, I normalized the image data because it improves numerical stability and also improves convergence. Below
is the distribution of pixel values of a grayscale image and the distribution for a normalized image. Clearly the bins
are all within 0 and 1 (actual min-max normalization used was for 0.1 - 0.9).

![alt text][normalize]

Note: Because I used tensorflow functions for both grayscale conversion and normalization the writeup_generator.py
script was necessary to generate this images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer | Description |
|:---------------------:|:---------------------------------------------:|
| Input | 32x32 normalized grayscale image |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU ||
| Dropout | 70% keep probability |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU ||
| Max Pooling 2x2| 1x1 stride, valid padding, outputs 5x5x16|
| Convolution 5x5 | 1x1 stride, valid padding, output 8x8x16 |
| flatten | Output 26368 |
| Dropout | 70% keep probability |
| Fully connected | Output 512 |
| RELU ||
| Dropout | 50% keep probability |
| Fully connected | Output 86 |
| RELU ||
| Dropout | 50% keep probability |
| Fully connected | Output 43 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Some different learning rates like 0.001, 0.0001, 0.0005 and 0.01; while 0.01 never converged and 0.0001 took longer,
0.001 and 0.0005 showed similar convergence time. Then using 100 epochs the convergence behavior for 0.001 was analyzed.
Whereas the 0.95 mark was reached before the 10th epoch, there was consistently no improvement past the 17th epoch. As
such 20 epochs were the chosen value. As for the weights in the convolution and fully connected layers, a mean of 0 and
standard deviation of 0.1 was used while the bias was initialized to 0.

The adam optimizer was chosen over regular gradient descent since it's easier to tune and converges faster. Softmax
cross entropy calculation was used as the loss function.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 95.9%
* test set accuracy of 96.6%

An iterative approach was chosen:
##### What was the first architecture that was tried and why was it chosen?

The LeNet5 from the course was used as a basis since it was mentioned as a good starting point.

##### What were some problems with the initial architecture?

The initial architecture used only pooling as a regularization technique which is considered inferior to the dropout
technique.

##### How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The addition of one convolutional layer and four regularization droupout layers with two different dropout thresholds (a higher one used for the first two dropouts and a lower one used for the last two) already achieved the target 93% accuracy. However, the removal of the max pooling layer significantly increased the computation time while keeping it didn't have negative implications for the accuracy.

##### Which parameters were tuned? How were they adjusted and why?

* The learning rate was adjusted to guarantee convergence.
* The batch size eased computational requirements while also affecting accuracy.
* As mentioned above, the epochs where used to improve the accuracy for each attempt.
* The number of dropout layers _and_ their rates was used to prevent over/underfitting.
* The dimensions of the convolutional layers were tweaked as they improve accuracy while
preventing over/underfitting. Same applies for the fully connected layers.

##### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolutional layers help with problems where the solution should be position-invariant; as is the case in classifying traffic signs. The dropout layers helped with under/overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][internet0] ![alt text][internet1] ![alt text][internet2] ![alt text][internet3] ![alt text][internet4]

The first and third images are severely under represented in the training set while being presumably different enough to be successfully classified. This is already a foreshadowing of the results that will be presented in the coming sections. The fourth image was unproperly cropped; which could affect its labeling.

Furthermore the second and fifth images are not parallel to the image plane which means their shape is also distorted.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image | Prediction |
|:---------------------:|:---------------------------------------------:|
| Slippery road | Slippery road |
| Priority road | Priority road |
| Bumpy road | Bumpy road |
| Right of way at next intersection | Priority road |
| General caution | General caution |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. While this is slightly below the validation accuracy, with 135 additional images downloaded from the web the accuracy reaches 94.2%; a much better result.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Images 1-3 and 5 were correctly predicted with at least 95% probability. The fourth image was mislabeled with a very
high likelihood (99.81%) while the second highest probability was the correct prediction with a non-negligible 0.18%.

![alt text][internet0]
| n | id | name | probability |
|:-:|:-:|:-:|:-:|
| 0 | 23 | Slippery road | 95.5289900302887 |
| 1 | 19 | Dangerous curve to the left | 4.167603328824043 |
| 2 | 29 | Bicycles crossing | 0.19897620659321547 |
| 3 | 30 | Beware of ice/snow | 0.04287060291972011 |
| 4 | 20 | Dangerous curve to the right | 0.03440383297856897 |

![alt text][internet1]
| n | id | name | probability |
|:-:|:-:|:-:|:-:|
| 0 | 12 | Priority road | 100.0 |
| 1 | 40 | Roundabout mandatory | 2.9199130092971703e-10 |
| 2 | 15 | No vehicles | 8.569333393923517e-12 |
| 3 | 2 | Speed limit (50km/h) | 1.4733165088390396e-13 |
| 4 | 13 | Yield | 6.617590484212031e-15 |

![alt text][internet2]
| n | id | name | probability |
|:-:|:-:|:-:|:-:|
| 0 | 22 | Bumpy road | 100.0 |
| 1 | 29 | Bicycles crossing | 2.6676545803513703e-16 |
| 2 | 25 | Road work | 3.4101047433904215e-18 |
| 3 | 24 | Road narrows on the right | 3.007628061007955e-18 |
| 4 | 26 | Traffic signals | 1.1772605730090882e-18 |

![alt text][internet3]
| n | id | name | probability |
|:-:|:-:|:-:|:-:|
| 0 | 12 | Priority road | 99.81569647789001 |
| 1 | 11 | Right-of-way at the next intersection | 0.18348684534430504 |
| 2 | 25 | Road work | 0.00046538170863641426 |
| 3 | 10 | No passing for vehicles over 3.5 metric tons | 0.0001663604962232057 |
| 4 | 30 | Beware of ice/snow | 0.00015407151749968762 |

![alt text][internet4]
| n | id | name | probability |
|:-:|:-:|:-:|:-:|
| 0 | 18 | General caution | 99.99997615814209 |
| 1 | 25 | Road work | 1.3436017809453915e-05 |
| 2 | 24 | Road narrows on the right | 4.346501469854047e-06 |
| 3 | 26 | Traffic signals | 1.3973732393424143e-06 |
| 4 | 31 | Wild animals crossing | 2.745428537287431e-09 |
