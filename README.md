# Vehicle Detection Project

The steps used for executing the project have been described against the backdrop of the project rubric.

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
   
## Section B : Sliding Window Search
  
### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

Sliding window search is implemented in the Jupyter notebook: “Vehicle Detection – Part 2” in section B1.
An image is read and converted to the colorspace YCrCb. HOG features are extracted and sliding windows are used between a scale of 0.9 to 1.2. The chosen SVM classifier is used to flag cars vs. non-cars in each sliding window.  A cells per step of 2 is used i.e. 75% overlap. 

### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier? Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Sliding windows were generated. I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. Finally, a video was generated using moviepy. 

False positives were eliminated using the following techniques – using better representative data, leveraging heat maps, removing the top half of the image to ignore any cars at a distance away and objects like the sky, creating a region of interest to eliminate oncoming cars identified on the other side of the road, discarding any boxes smaller than 70 pixels x 40 pixels as they are too small to be cars and averaging heatmaps across frames when making the video. This was performed for multiple iterations to identify the best possible output. 



* Iteration 1: Use all images provided

  Too many non-car portions were identified as cars. On visual examination of the images, the issue was identified as the use of GTI
  images. ‘Non-cars’ are everything in the world that are not cars – it is too diverse a set of data elements and it is important to use
  non-car images that are relevant to the track that is being targeted for vehicle identification. A video was also created – the video
  was terrible – car boxes were created everywhere including on roads, sides of roads and on trees. Hence, GTI images are discarded in
  subsequent iterations.

* Iteration 2: Discarding all GTI images when training the classifier.

  The output is much better although there are still a few windows well outside the car region. When a video was created, it was much 
  better than Iteration 1, though a few freeway right side barriers were classified as cars.

* Iteration 3: Adding some extra images of trees, freeway barriers to reduce misclassifications.  Also, adding black and white cars from
  BMW, Audi and Honda websites to better represent the cars used in the test track. 

  The output is improved over Iteration 3 although a few right side barriers are still misclassified as cars.  Additionally, a region of
  interest polygon is used to exclude identification of oncoming cars travelling on the other side of the freeway.  The cars are well
  separated in the images below. 

* Iteration 4: Adding many more images of right side barriers, trees etc. to reduce misclassifications. 

  The output deteriorates over iteration 3.  Adding more images for some reason results in road surfaces being classified as cars. 

* Iteration 5: Smoothing the video across frames 

  The iteration 3 SVC model is chosen based on sliding windows images, windows identified, heat map generated and the video generated.
  The heat map generated is stored in a ‘Car’ class and is smoothed across multiple frames while generating the video.  There is a
  tradeoff between getting low-wobble boxes and speed of flagging of new cars entering the video frame.  A smoothing across 25 frames
  generates fairly low-wobble boxes. This has been attached as part of the submission.   Smoothing across 15 frames has more wobbling
  but identifies the new black car entering the frame faster. 

## Section C: Discussion

### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The following are the issues faced during the implementation of the project: 

* Correctly identifying non-cars:   The definition of non-cars is very fuzzy – it can be a mountain, trees, mud banks, the road,
  barricades and even trucks, buses, motorcycles.  It needs a massive data set and maybe a neural network to improve accuracy here. 

* Shiny cars with smooth surfaces:  The black car is extremely shiny and in many sections of the road reflects the road next to the car.
  This is extremely tricky to handle - when I included images of the black car with these reflections, it led to the road itself being
  wrongly classified.  Reflecting surfaces are a problem.   Also, smooth surfaces have no gradients and the classifier is better at
  identifying the backs of cars rather than the sides. 

* Occlusion:  If a car overtakes another one, it is difficult to separate one car from the other using the heat map model.  It is also
  difficult to track a car accurately across the video due to this reason. 

* Shadows and lighting changes:  The model makes mistakes when there are shadows of trees on the road or the road surface is irregular.

* Rain and snow:  Existence of rain and snow will alter the HOG features that are identified and impact the output. 


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

