# Vehicle Detection Project

The design steps executed have been provided with reference to the project rubric. 

## Section A : Using Histogram of Oriented Gradients (HOG)

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in section A1 of the Jupyter notebook Vehicle Detection – Part 1. Post reading images, I explored different color spaces and skimage.hog() parameters. The images and hog features have been displayed below in a grid (columns containing 4 car and 2 non-car images) and rows having the output using different color spaces.


### 2. Explain how you settled on your final choice of HOG parameters.

HOG parameters of orientations=9, pixels_per_cell = (8, 8) and cells_per_block = (4, 4) gave the best results across color spaces.  HSV, YCrCb and YUV gave good results – but the choice of colorspace was deferred to the Machine Learning iterations phase. 

### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Section A2 of the Jupyter notebook – Vehicle Detection-Part 1 describes the Machine Learning iterations performed to choose the right classifier. 

* Iteration 1: Run SVC, Decision Tree and Random Forest classifiers on top 5 colorspaces.  Choose 1 classifier and  1 or 2 colorspaces.
  
  SVC outperforms Decision Tree and Random Forest classifiers across colorspaces and is retained for subsequent iterations. 
  HSV and YCrCb color spaces were shortlisted. 

* Iteration 2: Hog Channel selection using SVC classifier and Colorspaces HSV and YCrCb. 

  For both colorspaces,  ‘ALL’ channels were chosen instead of choosing only 1 of the 3 available channels. 
  
* Iteration 3: Add Histogram of Color as features

  Histogram of Color is added as a feature.  The accuracy improves only slightly.  The decision to retain this will be made when applied   to an image.

* Iteration 4: Evaluate impact of tuning the hyperparameter C in SVC.  C values evaluated were [ 0.003, 0.01, 0.03, 0.1, 0.3, 1]

  Choice of C did not impact classifier result when using a hold out validation data set. The reason was due to multiple very similar images for each car / non-car existing in the data set. When shuffled, similar images exist in both the training data set and validation data set causing very high accuracies. C will be chosen in a later iteration. 
  
* Iteration 5: Models created using both shuffled and unshuffled data for combinations of [HSV, YCrCb], [cells_per_block of 2, 4], [ SVC  , Random Forest] , [ C=0.1, 0.3, 1] for SVC classifier 

  Using both shuffled and unshuffled data, YCrCb ,  cells_per_block of 4 was better than 2, SVC classifier gave the best results and
  were chosen.  C =1 and 0.3 gave good results and the outcome deteriorated using a lower C.  C= 1 was retained eventually. 
 
 * Iteration 6: Make a final decision regarding use of Histogram of Color
 
   Using Histogram of Color features gave terrible results when applied to images – this is most likely due to the fact that the target    cars are black and white in color.  The road is mostly dark grey and white in some patches.  In test image 1, it can be clearly seen    that the white road patch is confused with the white car. As trees, signposts, roads can all have similar colors as some of the cars    seen on the roads, it cannot be used as a reliable feature and is hence discarded. 
   
## Section A : Using Histogram of Oriented Gradients (HOG)
  


# Original README
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You can submit your writeup in markdown or use another method and submit a pdf instead.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

