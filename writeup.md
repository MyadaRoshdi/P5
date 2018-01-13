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
[image9]: ./output_images/sliding_window_size_RGB.png
[image10]: ./output_images/sliding_window_size3_YUV.png
[image11]: ./output_images/sliding_window_size3_YCrCb.png
[image12]: ./output_images/sliding_windows_7.png
[image17]: ./output_images/best_sliding_window.png

[image13]: ./output_images/heat_map_1.png
[image14]: ./output_images/heat_map_2.png
[image15]: ./output_images/heat_map_3.png

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
`YCrCb` color space and HOG parameters of `orient = 9`, `pix_per_cell = 8`, `cell_per_block = 2`, `hog_channel = 'ALL'`

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

After, I started runing my so far Algorithm on the 6-test images, using fixed size sliding window, but the results had some false-positives as shown below:

![alt text][image5]




#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

