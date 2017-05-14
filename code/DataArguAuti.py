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

def isIntersect(l1, l2):
    isInter = False
    if ((l1[0] > l2[0]) and (l1[0] < l2[1])):
        isInter = True
    if ((l1[1] > l2[0]) and (l1[1] < l2[1])):
        isInter = True
    if ((l2[0] > l1[0]) and (l2[0] < l1[1])):
        isInter = True
    if ((l2[1] > l1[0]) and (l2[1] < l1[1])):
        isInter = True

    return isInter


def fitGamma(data):
    mean = data.mean()
    variance = data.var()
    return mean*mean/variance, variance/mean

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, thick)

    return imcopy

def generator(pd_Extra, root_dir="./dataset/object-dataset", batch_size=32):
    num_samples = pd_Extra.shape[0]
    while 1:
        pd_ExtraShuffle = sklearn.utils.shuffle(pd_Extra)
        for offset in range(0, num_samples, batch_size):
            batch_samples = pd_ExtraShuffle.iloc[offset:offset+batch_size]
            l_image = []
            l_tags  = []
            for idx in range(batch_samples.shape[0]):
                if idx  % 100 == 0:
                    print("%d out of %d" % (int(idx / 100), int(batch_size/100)))
                try:
                    name = "%s/%s" % (root_dir, batch_samples.iloc[idx]['name'])
                    image = cv2.cvtColor( cv2.imread(name), cv2.COLOR_BGR2RGB )
                    tag   = batch_samples.iloc[idx]['tags']
                    np_box = batch_samples.iloc[idx].values[0:4].astype(np.int)
                    image2 = cv2.resize(image[np_box[1]:np_box[3], np_box[0]:np_box[2]], (64, 64))
                    l_image.append(image2)
                    l_tags.append(tag)
                except:
                    pass

            X_train = np.array(l_image)
            y_train = np.array(l_tags)
            yield sklearn.utils.shuffle(X_train, y_train)


def plotArgumentationInRaw(infile, pd_Extra, root_dir="./dataset/object-dataset"):
    pd_ExtraCars   = pd_Extra[pd_Extra['tags']==1]
    pd_ExtraNoCars = pd_Extra[pd_Extra['tags']==0]
    l_boxVal       = list(pd_ExtraCars[pd_ExtraCars['name']==infile].values[:,0:4])
    l_boxVal2      = list(pd_ExtraNoCars[pd_ExtraNoCars['name']==infile].values[:,0:4])
    matRGB = cv2.cvtColor(cv2.imread("%s/%s" % (root_dir,infile)), cv2.COLOR_BGR2RGB)
    return draw_boxes(draw_boxes(matRGB, list(l_boxVal2)), l_boxVal, color=(255,0,0))


class dataArgumentForAutti(object):
    def __init__(self, annoFile, root_dir, ratio_nonCars=5):
        self.annoFile = annoFile
        self.root_dir = root_dir
        self.ratio_nonCars = ratio_nonCars
        self.pd_extra1 = pd.read_csv(annoFile, header=None, sep="\s+")
        self.pd_extra1 = self.pd_extra1.where((self.pd_extra1[6]=="car") |\
                                              (self.pd_extra1[6]=="truck")
                        ).dropna()


    def _getHeightWidth(self):
        pd_WH = pd.DataFrame(list(zip(self.pd_extra1[3] - self.pd_extra1[1],
                                      self.pd_extra1[4] - self.pd_extra1[2])))
        self.g_alpha_W,self.g_beta_W = fitGamma(pd_WH[0])
        self.g_alpha_H,self.g_beta_H = fitGamma(pd_WH[1])

    def _generateNoCar_fromFile(self, infile):
        img = cv2.imread("%s/%s" % (self.root_dir, infile))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_boxCars = self.pd_extra1[self.pd_extra1[0]==infile].\
                                values[:,1:5].astype(np.int)
        while 1:
            while 1:
                init_y = int(np.random.uniform(0, img.shape[0]))
                init_x = int(np.random.uniform(0, img.shape[1]))
                width  = int(np.random.gamma(g_alpha_W, g_beta_W))
                height = int(np.random.gamma(g_alpha_H, g_beta_H))
                if (width > 1.2*height) or (height > 1.2*width):
                    continue

                box_nonCar = [init_x,init_y, init_x+width, init_y+height]
                if (box_nonCar[2] > img.shape[1]) or (box_nonCar[3] > img.shape[0]):
                    continue

                notPass = 0
                for boxCars in np_boxCars:
                    #print(notPass)
                    l_carx  = [boxCars[0], boxCars[2]]
                    l_ncarx = [box_nonCar[0], box_nonCar[2]]
                    l_cary  = [boxCars[1], boxCars[3]]
                    l_ncary = [box_nonCar[1], box_nonCar[3]]
                    if isIntersect(l_carx, l_ncarx) and isIntersect(l_cary, l_ncary):
                        notPass = 1
                        break

                if notPass:
                    continue
                else:
                    yield box_nonCar


    def _parseBoxes(self, infile):
        img = cv2.imread("%s/%s" % (root_dir,infile))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_boxCars = self.pd_extra1[self.pd_extra1[0]==infile].\
                            values[:,1:5].astype(np.int)
        n_ncars = np_boxCars.shape[0] * self.ratio_nonCars
        np_boxNoCars = np.array([next(self._generateNoCar_fromFile(pd_extra1, infile))
                                     for i in range(n_ncars)])

        return np_boxCars, np_boxNoCars

    def generateExtraFile(self, arguResults='./dataset/object-dataset_argument.csv'):
        self._getHeightWidth()
        l_boxCars   = []
        l_boxNoCars = []
        l_namesCar = []
        l_namesNoCar = []
        l_files = sorted(set(self.pd_extra1[0]))
        if not os.path.isfile(arguResults):
            for infile in tqdm(l_files):
                np_boxCars,np_boxNoCars = self._parseBoxes(infile)
                l_boxCars.append(np_boxCars)
                l_boxNoCars.append(np_boxNoCars)

            for idx,name in enumerate(l_files):
                for num in range(l_boxCars[idx].shape[0]):
                    l_namesCar.append(name)
                for num in range(l_boxCars[idx].shape[0]*ratio_nonCars):
                    l_namesNoCar.append(name)

            pd_ExtraCars = pd.DataFrame(np.vstack(l_boxCars))
            pd_ExtraCars['name'] = l_namesCar
            pd_ExtraCars['tags'] = 1
            pd_ExtraNoCars = pd.DataFrame(np.vstack(l_boxNoCars))
            pd_ExtraNoCars['name'] = l_namesNoCar
            pd_ExtraNoCars['tags'] = 0
            pd_Extra = pd_ExtraCars.append(pd_ExtraNoCars)
            pd_Extra.to_csv(arguResults)
        else:
            pd_Extra = pd.read_csv(arguResults, index_col=[0])

        return pd_Extra
