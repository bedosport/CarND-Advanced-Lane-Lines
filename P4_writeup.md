## Writeup Template

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[img1]: ./output_images/undistort_sample.png "undistort"
[img2]: ./output_images/pp_transform.png "pptransform"
[img3]: ./output_images/Thresh_3.png "thresh"
[img4]: ./output_images/pp_curve1.png "curve1"
[img5]: ./output_images/pp_curve2.png "curve2"
[img6]: ./output_images/line_detection.png "lines"
[img7]: ./output_images/test1_out.png "test1"
[img8]: ./output_images/test2_out.png "test2"
[img9]: ./output_images/test3_out.png "test3"
[img10]: ./output_images/test4_out.png "test4"
[img11]: ./output_images/test5_out.png "test5"
[img12]: ./output_images/test6_out.png "test6"
[img13]: ./output_images/final.png "final"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

To reduce the error introduced by camera distortion, images taken for a standard chessboard is used to correct the distortion (calibration images are in the "camera_cal" folder). The camera calibration script is "Camera_calib.py". In the images, the 9 by 6 intersection points are detected. Then the actual coordinates of these intersection points are matched up with the image.

After camera calibration, the camera calibration matrix is saved into the pickle file "camera_cal/calibration_undistort.p". 


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the calibration data from the pickle file, we can recover the undistorted image as shown below.
![undistort][img1]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To extract the lane information from the image in a reliable way, we need to find some universal features in the image. After experimenting with different color spaces, the following three features are used to threshold the original image.

* The L channel of HLS color space. This lightness channel helps to distingush the lane line from the rest of the environment.
* Gradient in x direction of the L channel in HLS color space. Since the lane line is approximately vertical from the camera's perspective, the gradient in the x direction can make the lane line standout significantly.
* The B channel of LAB color space. This "blue–yellow" color channel greatly helps picking up the yellow lane line. Adding this channel helps us picking up the yellow lane lines in a reliable way.

The image thresholding based on these three features is visualized in Green, Red, and Blue colors, respectfuly. This combination is very effective for detecting the lane lines in different lighting and road conditions. An interesting observation is that the B channel detects left yellow lane very well (as expected, B means "blue–yellow"), the L channel detects the right white lane very well (white lane is bright!), the x gradient detects all the important pixels with lots of other stuff. 
![thresh][img3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The lane lines detected directly from the camera's perspective is different from what they really are. A perspective transformation is performed to get a bird-eye view of the lane lines. To figure out the transformation between the camera's perspective and the bird-eye view, we take a image of the straight lane line as a reference.

In the straight lane image, the lanes lines should be parallel to each other and straight. With this intuition, we select four corner points in the image and map them to a rectangle. It is discovered that the further away the corner points are, the more sensitive the mapping accuracy becomes. With some experimenting, we find the following source and destination points that will give us moderate look ahead horizon and good accuracy.

    # line2, moderate horizon
    src = np.float32([[272., 673.],[593., 450.],[691., 450.],[1052., 673.]])
    # set up 4 target points (assume flat ground, 1280, 720)
    dst = np.float32([[300., 720.],[300, 0],[980, 0],[980., 720.]])

After applying perspective transformation based on these points, we can verify that the straight lanes are indeed straight and parallel in the bird-eye view.

![pptrans][img2]


This perspective trasformation is also applied on some curved roads. It can be observed that the tranformed lane lines are also approximately parallel to each other. This validates that our perspective transformation is correct.

![curve1][img4]
![curve2][img5]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

fit polynomial to lines
![lines][img6]




#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

radius, center
f(y) = Ay2 + By +C

R=(1+(2Ay+B)^2)^(3/2)/abs(2A)

convet from pixel to meters,     ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Calculate radius of curvature and shift from center
![test1][img7]
![test2][img8]
![test3][img9]
![test4][img10]
![test5][img11]
![test6][img12]




---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Final pipeline video (click on the image to view the video from youtube)
[![final][img13]](https://www.youtube.com/watch?v=X8QN-qY7uIo)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

Lighting conditions, broken lines, repainted lines, cause broken line

save lane info into lane object, use continuity to save processing time

Combine with sensor fusion, use high precisioin maps, know approximate value from map then combine with lane detection algorithm.  
