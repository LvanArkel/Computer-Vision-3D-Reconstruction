# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import filters
import scipy.misc
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
import matplotlib.image
from numpy import asarray

from PIL import Image


##FIRST GET GAUSSIAN MODELS
def get_background_model(background):

    hue_stack = []
    saturation_stack = []
    value_stack = []

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
def find_camera_foreground(gaussian_model, frame):
    
    hsv_frame = np.array(frame)
    
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

    kernel = np.ones((5,5),np.uint8)
    
    dilated = cv2.dilate(res, kernel, iterations=1)
    kernel = np.ones((2,2),np.uint8)
    eroded = cv2.erode(dilated, kernel, iterations=1)
 
    #get 4 biggest components and combine
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=4)
    first = np.argsort(-stats[:,4])[1]
    output1 = output == first
    
    second = np.argsort(-stats[:,4])[2]
    output2 = output == second
    
    third = np.argsort(-stats[:,4])[3]
    output3 = output == third
    
    fourth = np.argsort(-stats[:,4])[4]
    output4 = output == fourth
  
    output = np.logical_or(output1, np.logical_or(output2, np.logical_or(output3,output4))).astype('uint8')

    return output

def hist(active_colors):
    colors = np.array(active_colors)

    b, bin_edges = np.histogram(colors[:,0], bins = 16, density = True)
    g, bin_edges = np.histogram(colors[:,1], bins = 16, density = True)
    r, bin_edges = np.histogram(colors[:,2], bins = 16, density = True)
    histograms = [b,g,r]
    
# =============================================================================
#     plt.figure()
#     plt.plot(bin_edges[2:-1], r[1:-1], color='red')
#     plt.plot(bin_edges[2:-1], g[1:-1], color='green')
#     plt.plot(bin_edges[2:-1], b[1:-1], color='blue')
#     plt.show()
# =============================================================================
    
    return histograms

def gaussian(active_colors):
    colors = np.array(active_colors)
    colors = colors[:,:2]
    gm = GaussianMixture(n_components=2).fit(colors)
    return gm.means_

def compare_gaussian(og_means, means):
    corr = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            corr[i,j] = np.sum(np.abs(og_means[i] - means[j]))
    row_ind, col_ind = linear_sum_assignment(corr)
    ind = np.argmax(corr, axis = 1)
    return col_ind

def compare(og_hists, hists):
    corr = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            b = np.corrcoef(og_hists[i][0],hists[j][0])[0][1]
            g = np.corrcoef(og_hists[i][1],hists[j][1])[0][1]
            r = np.corrcoef(og_hists[i][2],hists[j][2])[0][1]
            corr[i,j] = np.mean([g,b])
            #corr[i,j] = b
    row_ind, col_ind = linear_sum_assignment(corr, maximize = True)
    ind = np.argmax(corr, axis = 1)
    return col_ind

def compare2(og_hists, hists):
    corr = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            h = cv2.compareHist(og_hists[i][0].astype('float32'), hists[j][0].astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
            s = cv2.compareHist(og_hists[i][1][1:-1].astype('float32'), hists[j][1][1:-1].astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
            v = cv2.compareHist(og_hists[i][2][1:-1].astype('float32'), hists[j][2][1:-1].astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
            corr[i,j] = np.mean([h,s])
            corr[i,j] = h
    row_ind, col_ind = linear_sum_assignment(corr)
    ind = np.argmin(corr, axis = 0)
    return col_ind




