#import acnet
'''

Code to demonstrate Gaussian fiting of the projections
PP 06/28/2015: built from online code on clx1 quadscan_hist.py

'''

import numpy as np
import time
import os
import scipy.optimize 
import math 
import matplotlib.pyplot as plt
import ImageTool as imgtl

# you have to change this line to the directory where the data are located in
UpperDir = "./data_samples/X121_20150601/"
# one of the data set 
SubDir   = "X121_20150601_160711"
rootname = "X121"

# number of point in the data set (here reduced for test purpose)
NPoints  = 2

# scan over the number of data point

for i in range (NPoints):
    filenameX= UpperDir+"/"+SubDir+"/"+rootname+"_"+str(i)+"_x"
    filenameY= UpperDir+"/"+SubDir+"/"+rootname+"_"+str(i)+"_y"

# load the save histograms
    histX=np.loadtxt(filenameX)
    histY=np.loadtxt(filenameY)
    
# create the horizontal coordinate    
    axisX=np.arange(len(histX))
    axisY=np.arange(len(histY))
    
# center the histogram and select a region of interest (ROI) around the peak 
    window=1000.   
    histXc, axisXc = imgtl.center_and_ROI(histX, axisX, window)
    histYc, axisYc = imgtl.center_and_ROI(histY, axisY, window)
    
# remove background (there are many way of doing this)   
    window = 10
    histXc, axisXc = imgtl.removebackground(histXc, axisXc, window)
    histYc, axisYc = imgtl.removebackground(histYc, axisYc, window)
    
    
    
    p2X= imgtl.FitProfile(histXc, axisXc)
    p2Y= imgtl.FitProfile(histYc, axisYc)
    
    print("fitX: ", p2X)
    print("fitY: ", p2Y)

    plt.figure()
    plt.plot(axisXc, histXc,'ob',alpha=0.1)
    plt.plot(axisXc, imgtl.dg(axisXc,p2X),'--b',linewidth=3)
    plt.plot(axisYc, histYc,'or',alpha=0.1)
    plt.plot(axisYc, imgtl.dg(axisYc,p2Y),'--r',linewidth=3)
    plt.title("point number: "+str(i))
    
    
plt.show()
