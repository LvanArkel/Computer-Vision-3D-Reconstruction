# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    ret, og_frame = video.read()
    cv2.imwrite('cam1_og.jpg', og_frame)
    
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
    return opening, og_frame


def get_hist(ground_truth, og):
    #each person is one color in the ground truth
    red = (ground_truth[:,:,0] >= 235).astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=4)
    second = np.argsort(-stats[:,4])[1]
    red = output == second
    
    green_g = (ground_truth[:,:,1] >= 250).astype('uint8')
    green_b = (ground_truth[:,:,2] <= 10).astype('uint8')
    green = np.logical_and(green_g, green_b).astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(green, connectivity=4)
    second = np.argsort(-stats[:,4])[1]
    green = output == second

    
    blue_b = (ground_truth[:,:,2] >= 230).astype('uint8')
    blue_g = (ground_truth[:,:,1] <= 10).astype('uint8')
    blue = np.logical_and(blue_b, blue_g).astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(blue, connectivity=4)
    second = np.argsort(-stats[:,4])[1]
    blue = output == second
    
    cyan_b = (ground_truth[:,:,2] >= 245).astype('uint8')
    cyan_g = (ground_truth[:,:,1] >= 245).astype('uint8')
    cyan = np.logical_and(cyan_b, cyan_g).astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cyan, connectivity=4)
    second = np.argsort(-stats[:,4])[1]
    cyan = output == second

    #segmentation of all four is the union
    four = np.logical_or(red, np.logical_or(green, np.logical_or(blue, cyan))).astype('uint8')
    
    #multiply by original image to get the colored shilouette
    red = og*np.dstack((red, red,red))
    green = og*np.dstack((green, green,green))
    blue = og*np.dstack((blue, blue,blue))
    cyan = og*np.dstack((cyan, cyan,cyan))

    
    people = [red, green, blue, cyan]
    histograms = []
    for person in people:
        plt.figure()
        plt.imshow(person)
        plt.show()
    #compute histogram for each channel
        histogram_r, bin_edges_r = np.histogram(person[:,:,0], bins=256, range=(0, 256))
        histogram_g, bin_edges_g = np.histogram(person[:,:,1], bins=256, range=(0, 256))
        histogram_b, bin_edges_b = np.histogram(person[:,:,2], bins=256, range=(0, 256))
        
        #stack in one array
        histogram = np.dstack((histogram_r, histogram_g, histogram_b))
        histograms.append(histogram)
        #value in 0 is too much so we don't show it to visualize the rest
        plt.figure()
        plt.plot(bin_edges_r[2:-1], histogram_r[1:-1], color='red')
        plt.plot(bin_edges_g[2:-1], histogram_g[1:-1], color='green')
        plt.plot(bin_edges_b[2:-1], histogram_b[1:-1], color='blue')
        plt.show()
    
    
    return histograms
    #print(numpy.corrcoef(a,b))
    #from scipy.stats.stats import pearsonr   
    #print(pearsonr(a,b))
    
if __name__ == "__main__":


    res, og_frame = find_camera_foreground('cam1.avi')
    ground_truth = np.array(Image.open('cam1.jpg'))
    og = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
    histograms1 = get_hist(ground_truth, og)
    
    
    res, og_frame = find_camera_foreground('cam2.avi')
    ground_truth = np.array(Image.open('cam2gt.jpg'))
    og = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
    histograms2 = get_hist(ground_truth, og)
    res1 = []
    for j in range(4):
        res2 = []
        for i in range(4):      
            #compute correlation for each channel
            r = np.corrcoef(histograms1[j][:,1:,0],histograms2[i][:,1:,0])[0]
            g = np.corrcoef(histograms1[j][:,1:,1],histograms2[i][:,1:,1])[0]
            b = np.corrcoef(histograms1[j][:,1:,2],histograms2[i][:,1:,2])[0]
            res2.append(np.mean((r,g,b)))
        res1.append(res2.index(max(res2)))
    #print the index of the second histogram to which they correspond
    #since they are in order ideal result should be 0, 1, 2, 3
    print(res1)