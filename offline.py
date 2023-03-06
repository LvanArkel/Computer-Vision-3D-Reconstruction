# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import scipy.misc

import matplotlib.image
from numpy import asarray

from PIL import Image


##FIRST GET GAUSSIAN MODELS

def get_background_model(namespace):

    hue_stack = []
    saturation_stack = []
    value_stack = []
    
    #background = cv2.VideoCapture('data/cam2/background.avi')
    background = cv2.VideoCapture(namespace)
    total_frames = int(background.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
 #get values for every 20th frame 
    for i in range(0, total_frames-2, 20):
        background.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = background.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_stack.append(hsv_frame[:,:,0])
        saturation_stack.append(hsv_frame[:,:,1])
        value_stack.append(hsv_frame[:,:,2])
    
    
    hue_stack = np.array(hue_stack)
    saturation_stack = np.array(saturation_stack)
    value_stack = np.array(value_stack)
    
    
    hue_mean = np.mean(hue_stack, axis = 0)
    hue_std = np.std(hue_stack, axis = 0)

    saturation_mean = np.mean(saturation_stack, axis = 0)
    saturation_std = np.std(saturation_stack, axis = 0)

    value_mean = np.mean(value_stack, axis = 0)
    value_std = np.std(value_stack, axis = 0)

    gaussian_model = np.stack([hue_mean, hue_std, saturation_mean, saturation_std, value_mean, value_std], axis = 2)
    
    #gaussian_model has shape (imgsize x, imgsize y, 6)
    return gaussian_model

#COMPARE IMAGE WITH BACKGROUND MODEL


def find_camera_foreground_automatic(dirname):

    vid = cv2.VideoCapture('data/background/' + dirname)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = vid.read()
    

    gaussian_model = get_background_model('data/background/' + dirname)
    video = cv2.VideoCapture('data/video/' + dirname)
    video.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, frame = video.read()
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame = np.array(hsv_frame)
    
    #compute the difference with the background model 
    hue_diff = np.absolute(np.subtract(gaussian_model[:,:,0], hsv_frame[:,:,0]))
    sat_diff = np.absolute(np.subtract(gaussian_model[:,:,2], hsv_frame[:,:,1]))
    val_diff = np.absolute(np.subtract(gaussian_model[:,:,4], hsv_frame[:,:,2]))
    
    #open ground truth 
    ground_truth = np.array(Image.open(dirname + '/cam.png'))
    ground_truth = cv2.cvtColor(ground_truth[:,:,:3], cv2.COLOR_BGR2GRAY)
    #find optimal threshold for hue_diff, sat_diff, and val_diff
    ground_truth = (ground_truth > 250).astype('uint8')

    t_hue = 0
    t_sat = 0
    t_val = 0
    best_error_h = ground_truth.size
    best_error_s = ground_truth.size
    best_error_v = ground_truth.size
    
    #parse through all combinations of errors
    for h in range(256):
        hue = (hue_diff > h).astype('uint8')
        intersect = np.logical_xor(hue,ground_truth )
        error = np.sum(intersect)
        if error < best_error_h:
            t_hue = h
            best_error_h = error
            
    for s in range(256):
        sat = (sat_diff > s).astype('uint8')
        intersect = np.logical_xor(sat,ground_truth )
        error = np.sum(intersect)
        if error < best_error_s:
            t_sat = s
            best_error_s = error
            
    for v in range(256):
        val = (val_diff > v).astype('uint8')
        intersect = np.logical_xor(val,ground_truth )
        error = np.sum(intersect)
        if error < best_error_v:
            t_val = v
            best_error_v = error    
            
    print(t_hue)
    print(t_sat)
    print(t_val)
    
    hue = (hue_diff > t_hue).astype('uint8')
    sat = (sat_diff > t_sat).astype('uint8')
    val = (val_diff > t_val).astype('uint8')
    

    s = hue+sat+val
    s = (s >= 1).astype('uint8')

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(s, connectivity=4)
    second = np.argsort(-stats[:,4])[1]
    output = output == second


    return output



def find_camera_foreground(dirname):
    vid = cv2.VideoCapture('data/background/' + dirname)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = vid.read()
    

    gaussian_model = get_background_model('data/background/' + dirname)
    video = cv2.VideoCapture('data/video/' + dirname)
    video.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = video.read()
    
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame = np.array(hsv_frame)
    
    hue_diff = np.absolute(np.subtract(gaussian_model[:,:,0], hsv_frame[:,:,0]))
    sat_diff = np.absolute(np.subtract(gaussian_model[:,:,2], hsv_frame[:,:,1]))
    val_diff = np.absolute(np.subtract(gaussian_model[:,:,4], hsv_frame[:,:,2]))
    
    
    threshold = filters.threshold_otsu(hue_diff)
    hue = (hue_diff > threshold).astype("float32")
    threshold = filters.threshold_otsu(sat_diff)
    sat = (sat_diff > threshold).astype("float32")
    threshold = filters.threshold_otsu(val_diff)
    val = (val_diff > threshold).astype("float32")
 
    res = val+hue+sat
    res = (res > 0 ).astype("uint8")
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    return opening

if __name__ == "__main__":

    #get initial segmentation
    res = find_camera_foreground('cam1.avi').astype("uint8")
    plt.figure()
    plt.imshow(res, cmap = 'gray')
    plt.show()
    

    image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1,3)) 
    pixel_vals = np.float32(pixel_vals)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 4
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 


    centers = np.uint8(centers) # convert data into 8-bit values 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    segmented_image = segmented_data.reshape((image.shape)) # reshape data into the original image dimensions
    plt.imshow(segmented_image)