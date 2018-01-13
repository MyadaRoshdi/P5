## Writeup 


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset_exploration.png
[image2]: ./output_images/hog_exploration.png
[image3]: ./output_images/hog_exploration_1.png
[image4]: ./output_images/hog_exploration_2.png

[image5]: ./output_images/region_of_interest.png
[image6]: ./output_images/sliding_window_size1.png
[image7]: ./output_images/sliding_window_size2.png
[image8]: ./output_images/sliding_window_size3.png
[image9]: ./output_images/sliding_window_RGB.png
[image10]: ./output_images/sliding_window_YUV.png
[image11]: ./output_images/sliding_window_YCrCb.png
[image12]: ./output_images/slidingwindows_7_regions_merged.png
[image17]: ./output_images/best_sliding_window.png
[image18]: ./output_images/pipeline.png

[image13]: ./output_images/heat_map_1.png
[image14]: ./output_images/heat_map_2.png
[image15]: ./output_images/heat_map_3.png
[image19]: ./output_images/heat_map_4.png
[image20]: ./output_images/heat_map_5.png
[image21]: ./output_images/heat_map_6.png

[image16]: ./output_images/bounding_boxes.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.    

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 6th, 9th and 10th In-cells in the  IPython notebook 'Vehicle_Detection.ipynb' (or in lines #185 through #267 of the file called `Vehicle_Detection.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orient = 9`, `pix_per_cell = 8`, `cell_per_block = 2`, `hog_channel = 2`:


![alt text][image2]

![alt text][image3]

![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the best I got is using :
`YCrCb` color space and HOG parameters of `orient = 9`, `pix_per_cell = 8`, `cell_per_block = 2`, `hog_channel = 'ALL'`. This part will be described in details in the next section (3)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using using 80% of Data for training, and 20% for testing after normalizing and randomizing data, I only used the [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) datasets. I used Hog-features to train the classifier. I tried some experiments till at the end I used the best as a matter of testing accuracy. Here are the experiments I did: 



| Experiment no. | Parameters      					| Testing accuracy                    | Time to train SVM|
|:-------------------------:|:-------------------------:|:------------------------------------:|:-------------:| 
| 1        		| Cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'| 0.965  | 23.49sec |
| 2          | Cspace='HLS', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'| 0.9834  | 21.96sec  |
| 3            |Cspace='YUV', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'| 0.98  | 19.39sec    |
| 4           |Cspace='YCrCb', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'| 0.979 | 16.49sec   	|
|						|										                                                         		|       |            |
|						|    |       |            |


The code for this step is contained in the 13th In-cells in the  IPython notebook 'Vehicle_Detection.ipynb' (or in lines #344 through #401 of the file called `Vehicle_Detection.py`).  

I then final run was the one I chose to train the classifier, as will be discused in the sliding winsow section, experiments showed the using 'YCrCB' is the best for Hog-features extraction and best features I got for prediction.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided In the begining to search in a region of interest and use a fixed size window, The code for this step is contained in the 18th In-cells in the  IPython notebook 'Vehicle_Detection.ipynb' (or in lines #568 through #644 of the file called `Vehicle_Detection.py`).  
, so only a part of the image is used to scan with sliding windows to not waste time scanning all parts of the image, as shown below:
used parameters:

ystart = 400
ystop = 656
scale = 1.5
colorspace =  YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
window= 64
The code for this expetiment is contained in the 19th In-cells in the  IPython notebook 'Vehicle_Detection.ipynb' (or in lines #653 through #711 of the file called `Vehicle_Detection.py`).

![alt text][image5]

Then I started testing the fixed sliding window algorithm on different color spaces for HOG-features extraction to see the best to use, and here's what I got for RGB, YUV and YCrCb sliding windows results respectively:

![alt text][image9]

![alt text][image10]

![alt text][image11]

Then I chose the best which was 'YCrCb'.

After, I started runing my so far Algorithm on the 6-test images, using fixed size sliding window, The result was very good as shown in the below image, but when I ran on the project Video,  bounding boxes sizes were not changing from frame to anothe in a nice way :

![alt text][image17]

So, I decided to go for multiple size window., as shown in the following images:

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image12]

Then, the best of my trials could be found in the modified_sliding_windows(), The code for it is contained in in lines #846 through #874 of the file called `Vehicle_Detection.py`.

**NOTE:** In the code, you will notice I also trimmed the image from the left side, I didn this step to decrease the number of sliding windows, as my pipeline was taking almost 2-hrs to run. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code for the final pipeline is contained in in lines #1168 through #1163 of the file called `Vehicle_Detection.py`.

Ultimately I searched on two scales using YCrCb 3-channel HOG featuresin the feature vector, Cut only the region of interest from the image(which made feature extraction and prediction faster) and decreasng # of sliding windows/frame, this part is done in th function call of modified_sliding_window() step.

Then, From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions, The code for this part is done by calling the add_heat() and apply_threshold(). is contained in in lines #941 through #964 of the file called `Vehicle_Detection.py`..  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.

### Here are six test images  and their corresponding heatmaps:

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image19]

![alt text][image20]

![alt text][image21]

### Here the resulting bounding boxes are drawn for the same heat maps detected above:

![alt text][image16]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Look at step 2. in the above section.
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most challenging part in my project that in the 1st run, it took around 12 sec to run 1-frame, so for the project video (1261-frames), so it took more than 4-hrs to run pipeline on the frame using fixed size window. then I started to minimize the number of used sliding windows, by cuting only the region of interest out of the image, using multiple sizes windows, using only HOG-features for training the classifer and predicting later /frame. 

My pipeline will likely fail if captured at night, weather conditions(rain, snow, fog, ...etc), Also the car tested on is on the left lane of the road, so it might fail if it is in the middle of in the right lane of the road.

Suggested Improvements:
1) Use date from last no of video frames for predicting current frame boxes
2) Train the classifier with more data .
3) As data used in training is already frames of video, so successive frames are almost similar, so may be a better randomization for data to make sure most of the images situations are found in both training and testing.
4) If I have a faster GPU, I'd increase the # of windows used for scanning the region of interest in the image.
5) Different classifier Could be experimented
6) Adding some Computer vision techniques as pre-processing for the image (e.g. Camera Caliberation)

