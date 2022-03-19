# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 22:54:01 2022

@author: HU
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from os import walk
import time
import pickle

saveDir = './Digits/'
width = 7
height = 11
yLoc = 557
xLocs = (877,884,895,902)



for (dirpath, dirnames, filenames) in walk('./Digits'):
    pass
    


label = []
img = np.zeros((11,7,3,0))

for fname in filenames:
    if(fname[-4:] == '.bmp'):
        arr = plt.imread('./Digits/'+fname)
        plt.imshow(arr)
        plt.show()
        time.sleep(0.01)
        idx = int(fname[:-4])
        try:
            lb = int(input('num: '))
        except ValueError:
            lb = 10
        label+=[lb]
        arr.resize(11,7,3,1)
        img = np.concatenate((img,arr),axis = 3)
        

f = open('digitLabel.pickle',"wb")
pickle.dump(label,f)
f = open('digitData.pickle',"wb")
pickle.dump(img,f)

for i in range(len(label)):
    digit = img[:,:,:,i]/255
    plt.imsave('./DigitCheck/'+str(label[i])+'/'+str(i)+'.bmp',digit)

