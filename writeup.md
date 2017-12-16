# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 presenting the result of the test of the model on simulator

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the model which NVIDIA guys proposed in https://arxiv.org/abs/1604.07316, (code line 94-112)

The model structure is as following:

![image8](https://github.com/FredericLiu/Behavioral-Cloning-P3/blob/master/witeup_img/model_structure.png)


I didn't change any one parameter of this model, so the input data need to be resized to fit this model.

Before feed the data into model, the data was normalized, then cutted up 70 and low 25 pixels, then resized to shape [66, 200], to fit into the model:

```sh
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Lambda(resize))
```

#### 2. Attempts to reduce overfittinog in the model

To reduce over fitting, I added dropout layers with keep_prob= 0.5 after every full_connected layers. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 114-161). 
training data and validation data are splitted from same datasets with 8:2 propotion.(code line 76-77)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

training epochs was set to 5, to reduce the training loss. larger epoch might get better result, but the training time is really long, since I use CPU to train the model.

#### 4. Appropriate training data

Training data wasr chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the mature model with CNN, better has been validated on real vehicle, with and a E2E learning structure.

My first step was to use a convolution neural network model similar to the  https://arxiv.org/abs/1604.07316. I thought this model might be appropriate because it has been working on the test vehicle by NVIDIA researchers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model, added dropout layers after each full_connected layers.

Then I tried to augmented the data by flipping each image and steering angle, so that the data number could be 2 times larger.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.

To improve the driving behavior in these cases, I supplement the data set by collect more recovery record.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 94-112) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Convolution 5x5     	| 2x2 stride, 24 filters 						|
| Convolution 5x5  		| 2x2 stride, 36 filters 						|
| Convolution 5x5      	| 2x2 stride, 48 filters   						|
| Convolution 3x3	    | 1x1 stride, 64 filters				  	    |
| Convolution 3x3		| 1x1 stride, 64 filters						|
| Fully connected		| output 1164, with dropout      				|
| Fully connected		| output 200, with dropout   					|
| Fully connected		| output 50, with dropout					    |
| Fully connected		| output 10, with dropout   					|
| Fully connected		| output 1, with dropout					    |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps: one lap clockwise, and one lap anti-clockwise, attampting to keep the car at the center of the road.

![alt text](https://github.com/FredericLiu/Behavioral-Cloning-P3/blob/master/witeup_imgcenter.jpg)

When training with this dataset, there are a few cornors the car would go off the road, so I supplement the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to road when almost going off road.
These images show what a recovery looks like from left side back to center:

![alt text](https://github.com/FredericLiu/Behavioral-Cloning-P3/blob/master/witeup_img/recover1.jpg)
![alt text](https://github.com/FredericLiu/Behavioral-Cloning-P3/blob/master/witeup_img/recover2.jpg)
![alt text](https://github.com/FredericLiu/Behavioral-Cloning-P3/blob/master/witeup_img/recover3.jpg)

To augment the data sat, I also flipped images and angles thinking that this would extend the dataset.

Also, I utilized the left camera and right camera, correct the angle with correction retio 0.2(code line 20-28). then dataset would be 3 times larger,

After the collection process, I had 24804 number of data points. comparing 4134 original data.

I then preprocessed this data by normalizing, cropping and resize to fit model (code line 83-92), all reprocessing are using keras model, which means the data are preprocessed only when it would be feed into the model, other than reprocessed all in once.


I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by a relatvely low loss on validation set.

I also used the visualizion to show the loss value with the epochs. (code line 118-129)

But what strange is that during whole epochs, the validation loss is always lower than training loss, which should not be possible in real world. as following:

![alt text](https://github.com/FredericLiu/Behavioral-Cloning-P3/blob/master/witeup_img/training_result.png)

I didn't find the problem yet. I would be appreciate if anyone could figure out what is wrong in my code.
