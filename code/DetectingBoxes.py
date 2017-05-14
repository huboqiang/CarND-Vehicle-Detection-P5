import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix, auc, roc_auc_score
import matplotlib.pyplot as plt
import sklearn
from skimage.feature import hog
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
import time
from scipy.ndimage.measurements import label
import numpy as np
import functools

sns.set_style("ticks")

np.random.seed(0)

# Here, [0:1] for GRAY channel only, ignoring other channels.
l_colorSpace = [cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HLS,
                cv2.COLOR_BGR2HLS, cv2.COLOR_BGR2LAB, cv2.COLOR_BGR2LUV,
                cv2.COLOR_BGR2YCrCb, cv2.COLOR_BGR2YUV][0:1]
l_names = ["GRAY", "RGB", "HLS", "HLS", "LAB", "LUV", "YCrCb", "YUV"][0:1]
l_len   = [1,       3,     3,     3,       3,    3,    3,       3   ][0:1]


def get_hog_features(img, orient, pix_per_cell=8, cell_per_block=2,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def get_features(img, pix_per_cell=8,cell_per_block=2,orient=9, getImage=False, inputFile=True, feature_vec=True):
    l_imgLayers = []
    for cs in l_colorSpace:
        if inputFile:
            l_imgLayers.append(cv2.cvtColor(cv2.imread(img), cs))
        else:
            l_imgLayers.append(cv2.cvtColor(img, cs))

    l_hog_features = []
    l_images = []
    for feature_image in l_imgLayers:
        hog_features = []
        n_channel = 1
        if len(feature_image.shape) > 2:
            n_channel = feature_image.shape[2]
        for channel in range(n_channel):
            featureImg = feature_image
            if n_channel > 2:
                featureImg = feature_image[:,:,channel]

            vout,img = get_hog_features(featureImg,
                                orient, pix_per_cell, cell_per_block,
                                vis=True, feature_vec=feature_vec)
            if getImage:
                l_images.append(img)
            #print(featureImg.shape, vout.shape)
            hog_features.append(vout)

        l_hog_features.append(list(hog_features) )

    if getImage:
        return l_images
    else:
        return functools.reduce(lambda x,y: x+y, l_hog_features)

def parseBoxesInImageLess(img, svc, X_scaler, pix_per_cell=8,cell_per_block=2,orient=9,
                          ystart=400,ystop=656,scale=1.5,RGB2BGR=True):
    if RGB2BGR:
        imgRGB = img
        img    = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tosearch = img[ystart:ystop,:,:]
    imshape = img_tosearch.shape
    if scale != 1:

        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    l_features = get_features(img_tosearch, pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
                                  orient=orient,getImage=False,inputFile=False,feature_vec=False)
    #print(len(l_features), l_features[0].shape)
    nxblocks = (imshape[1] // pix_per_cell)-1
    nyblocks = (imshape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    l_boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            #print(xpos,ypos,pix_per_cell)
            l_hog_feats =  np.array(list(map(lambda feature: feature[ypos:7+ypos,xpos:7+xpos,:,:,:].ravel(),
                                             l_features))) \
                            .ravel().reshape(1,-1)

            #print(l_hog_feats.shape)
            if l_hog_feats.shape[1] != 1764:
                continue

            #print(len(l_features), l_hog_feats.shape)
            hog_features = l_hog_feats[0:1,0:1764]#np.hstack(l_hog_feats)
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            #print(hog_features.shape)
            test_features = X_scaler.transform(hog_features)
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                l_boxes.append( ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)) )

    return l_boxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
