# lane detection pipeline video
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from pp_transform import corners_unwarp

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return np.uint8(color_binary)
    

if __name__ == "__main__":
    # load camera calibration data
    dist_pickle = pickle.load( open( "camera_cal/calibration_undistort.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # load image
    image = mpimg.imread('test_images/test1.jpg')
    # threshold image
    img_thresh = pipeline(image)
    # unwarp image (TODO: change to load M from file)
    top_down, perspective_M = corners_unwarp(img_thresh, mtx, dist)

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=20)

    ax2.imshow(img_thresh)
    #plt.save_figure(result, "result.jpg")
    ax2.set_title('Threshold Result', fontsize=20)

    ax3.imshow(top_down)
    #plt.save_figure(result, "result.jpg")
    ax3.set_title('Pipeline Result', fontsize=20)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


    binary_warped = np.zeros((top_down.shape[0], top_down.shape[1]))
    binary_warped[(top_down[:,:,1]==1) | (top_down[:,:,2]==1)] = 1




    fig = plt.figure(2)
    # Read in a thresholded image
    warped = binary_warped#mpimg.imread('warped_example.jpg')
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(image, window_width, window_height, margin):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows    
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(np.uint8(warpage), 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()





    # # Assuming you have created a warped binary image called "binary_warped"
    # # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # # Find the peak of the left and right halves of the histogram
    # # These will be the starting point for the left and right lines
    # midpoint = np.int(histogram.shape[0]/2)
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # # Choose the number of sliding windows
    # nwindows = 9
    # # Set height of windows
    # window_height = np.int(binary_warped.shape[0]/nwindows)
    # # Identify the x and y positions of all nonzero pixels in the image
    # nonzero = binary_warped.nonzero()
    # nonzeroy = np.array(nonzero[0])
    # nonzerox = np.array(nonzero[1])
    # # Current positions to be updated for each window
    # leftx_current = leftx_base
    # rightx_current = rightx_base
    # # Set the width of the windows +/- margin
    # margin = 100
    # # Set minimum number of pixels found to recenter window
    # minpix = 50
    # # Create empty lists to receive left and right lane pixel indices
    # left_lane_inds = []
    # right_lane_inds = []

    # # Step through the windows one by one
    # for window in range(nwindows):
    #     # Identify window boundaries in x and y (and right and left)
    #     win_y_low = binary_warped.shape[0] - (window+1)*window_height
    #     win_y_high = binary_warped.shape[0] - window*window_height
    #     win_xleft_low = leftx_current - margin
    #     win_xleft_high = leftx_current + margin
    #     win_xright_low = rightx_current - margin
    #     win_xright_high = rightx_current + margin
    #     # Draw the windows on the visualization image
    #     cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    #     (0,255,0), 2) 
    #     cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    #     (0,255,0), 2) 
    #     # Identify the nonzero pixels in x and y within the window
    #     good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    #     (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    #     good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    #     (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    #     # Append these indices to the lists
    #     left_lane_inds.append(good_left_inds)
    #     right_lane_inds.append(good_right_inds)
    #     # If you found > minpix pixels, recenter next window on their mean position
    #     if len(good_left_inds) > minpix:
    #         leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    #     if len(good_right_inds) > minpix:        
    #         rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # # Concatenate the arrays of indices
    # left_lane_inds = np.concatenate(left_lane_inds)
    # right_lane_inds = np.concatenate(right_lane_inds)

    # # Extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds] 
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds] 

    # # Fit a second order polynomial to each
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)


    # # Generate x and y values for plotting
    # ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # fig = plt.figure(2)
    # print(binary_warped.shape)
    # plt.imshow(top_down)
    # #plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)




    # TODO:
    # 1. create a video pipeline to detect lanes frame by frame
    # 2. create a line class to keep track of the lane
    # 3. calculate curvature and car center dist from lane


    RGB 
    HLS  S thresh=(125, 255)

    # HLS L-channel Threshold (using default parameters)
    img_LThresh = hls_lthresh(img_unwarp)
    thresh=(220, 255)

    # Lab B-channel Threshold (using default parameters)
    img_BThresh = lab_bthresh(img_unwarp)
    thresh=(190,255)

    # sobel mag + direction
    # https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines/blob/master/project.ipynb


    plt.show()
    print("finish plotting")
