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

def get_background_model(background):

    hue_stack = []
    saturation_stack = []
    value_stack = []
    
    #background = cv2.VideoCapture('data/cam2/background.avi')
    #background = cv2.VideoCapture(namespace)
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

def find_camera_foreground(gaussian_model, og_frame):
 
    #gaussian_model = get_background_model('data/background/' + dirname)

    #cv2.imwrite('cam4og.jpg', og_frame)
    
    hsv_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2HSV)
    hsv_frame = np.array(hsv_frame)
    
    hue_diff = np.absolute(np.subtract(gaussian_model[:,:,0], hsv_frame[:,:,0]))
    sat_diff = np.absolute(np.subtract(gaussian_model[:,:,2], hsv_frame[:,:,1]))
    val_diff = np.absolute(np.subtract(gaussian_model[:,:,4], hsv_frame[:,:,2]))
    
    
    threshold = filters.threshold_otsu(hue_diff)
    hue = (hue_diff > threshold).astype("float32")
    threshold = filters.threshold_otsu(sat_diff)
    sat = (sat_diff > threshold-5).astype("float32")
    threshold = filters.threshold_otsu(val_diff)
    val = (val_diff > threshold).astype("float32")
    

    res = (val+hue+sat)/3

    
    res = (res > 0 ).astype("uint8")
    # plt.figure()
    # plt.imshow(res, cmap = 'gray')
    # plt.show()
    
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    
    dilated = cv2.dilate(res, kernel, iterations=1)
    kernel = np.ones((2,2),np.uint8)
    eroded = cv2.erode(dilated, kernel, iterations=1)
 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=4)
    first = np.argsort(-stats[:,4])[1]
    output1 = output == first
    
    second = np.argsort(-stats[:,4])[2]
    output2 = output == second
    
    third = np.argsort(-stats[:,4])[3]
    output3 = output == third
    
    fourth = np.argsort(-stats[:,4])[4]
    output4 = output == fourth
    
# =============================================================================
#     fifth = np.argsort(-stats[:,4])[5]
#     output5 = output == fifth
# =============================================================================
    output = np.logical_or(output1, np.logical_or(output2, np.logical_or(output3,output4))).astype('uint8')
    # plt.figure()
    # plt.imshow(output)
    # plt.show()
# =============================================================================
#     img = np.zeros((486,644,3))
#     contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(img, contours, -1, (0,255,255), thickness=cv2.FILLED)
#     print("Number of Contours found = " + str(len(contours)))
#     plt.figure()
#     plt.imshow(img, cmap ='gray')
#     plt.show()
# =============================================================================
    return output


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
    red = og*np.dstack((red, red,red))[:]
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
    res = find_camera_foreground('cam3.avi', 250)
    plt.figure()
    plt.imshow(res)
    plt.show()
    
# =============================================================================
#     res, og_frame = find_camera_foreground('cam4.avi')
#     ground_truth = np.array(Image.open('cam3gt.jpg'))
#     og = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
#     histograms2 = get_hist(ground_truth, og)
#     res = np.zeros((4,4))
#     res1 = []
#     for j in range(4):
#         res2 = []
#         for i in range(4):
#             #compute correlation for each channel
#             r = np.corrcoef(histograms1[j][:,1:,0],histograms2[i][:,1:,0])[0]
#             g = np.corrcoef(histograms1[j][:,1:,1],histograms2[i][:,1:,1])[0]
#             b = np.corrcoef(histograms1[j][:,1:,2],histograms2[i][:,1:,2])[0]
#             res2.append(np.mean((r,g,b)))
#             res[j,i] = np.mean((r,g,b))
#         res1.append(res2.index(max(res2)))
#     #print the index of the second histogram to which they correspond
#     #since they are in order ideal result should be 0, 1, 2, 3
#     for k in range(4):
#         print("Histogram 1, person ", k, " corresponds to histogram 2 person", res1[k])
#     #check for clashes
# =============================================================================
