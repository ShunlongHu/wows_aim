# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 22:54:01 2022

@author: HU
"""

import matplotlib.pyplot as plt
from os import walk

saveDir = './Digits/'
width = 7
height = 11
yLoc = 557
xLocs = (877,884,895,902)



for (dirpath, dirnames, filenames) in walk('./ScreenShots'):
    pass
    
digit_count = 0
for fname in filenames:
    if(fname[-4:] == '.jpg'):
        arr = plt.imread('./ScreenShots/'+fname)
        for x in xLocs:
            digit = arr[yLoc:yLoc+height, x:x+width, :]
            plt.imsave(saveDir+str(digit_count)+'.bmp',digit)
            digit_count+=1
