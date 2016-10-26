#import acnet
'''

Code to test the image analysis with a robust etraction of statistical
properties of the beam
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
UpperDir = "/Users/piot/ASTA_commissioning/laser_MLA/2015-11-18_COB_Divergence_measurements"
# one of the data set 
SubDir   = "/"
rootname = "0"

upperfile="./"
#/Users/piot/ASTA_commissioning/quadscan/X121_20150601//tight_focus/"
Filename  ="nml-2015-11-20-2203-05-791.png"

#nml-2015-06-01-2205-23-13076.png"


FTsize=8

test = 1  
# parameters
bbox = 200
cal  = 1
fudge = 0.3
threshold = 0.
# scan over the number of data point

#    fileonly = rootname+"-"+str(1+i)+".png"
#    filename = UpperDir+"/"+SubDir+"/"+fileonly
filename=upperfile+Filename
    
# load the image 

if test==0: 
   IMG=imgtl.Load(filename)

if test==1:
# this makes a test image 
# a gaussian curve
#       IMG = 10.+np.random.rand((480,640))
   IMG = np.zeros((480,640))
   s   = np.shape(IMG)

   v   = np.linspace(0, s[0], s[0])
   h   = np.linspace(0, s[1], s[1])
   sv  = 10.
   sh  = 30.
   for i in range(s[1]):
      IMG[:,i]= 0.0+0.1*2.*(0.5-np.random.rand(s[0]))+np.exp (- (v-np.mean(v))**2/(2.*sv**2))*np.exp (- (h[i]-np.mean(h))**2/(2.*sh**2))
      
# display raw image    
plt.figure()
plt.subplot(2,2,1)
imgtl.DisplayImage(IMG)
plt.title('raw data::'+Filename,fontsize=FTsize)
plt.axis('off')
# crop image 
#    plt.figure()
plt.subplot(2,2,2)
IMGc=imgtl.AutoCrop(IMG, bbox)
imgtl.DisplayCalibratedProj(IMGc, cal, fudge)
plt.title('cropped'+Filename,fontsize=FTsize)
plt.axis('off')
# threshold image 
#    plt.figure()
plt.subplot(2,2,3)
IMGt=IMGc
# imgtl.Threshold(IMGc, threshold)
imgtl.DisplayCalibratedProj(IMGt, cal, fudge)
plt.title('cropped::'+Filename,fontsize=FTsize)
plt.axis('off')
# compute profiles
histx, histy, x, y = imgtl.GetImageProjection(IMGt,cal)   
#    plt.figure()
plt.subplot(2,2,4)
plt.plot (x,histx,'o--')
plt.plot (y,histy,'s--')
plt.xlabel ('distance')
plt.ylabel ('population')
plt.title('profiles::'+Filename,fontsize=FTsize)
plt.show()
# RMS calculations: 
plt.figure()
imgtl.stats1d(x, histx)
imgtl.stats1d(y, histy)
imgtl.stats2d(x,y,IMGt)
norm0, meanx0, meany0, meanI0, stdx0, stdy0, stdI0, Wx0, Wy0, averImage0 = imgtl.window_scan2d(IMGt, 1., 100, 0.00)
plt.subplot (2,2,1)
plt.plot (Wx0, averImage0,'o')
plt.subplot (2,2,2)
plt.plot (Wx0, stdI0,'ro')
plt.subplot (2,2,3)
plt.plot (Wx0, meanI0,'ro')
plt.show()


ThresholdAve = np.mean(averImage0[len(averImage0)-10:len(averImage0)-1])
ThresholdStd = np.sqrt(np.std(averImage0[len(averImage0)-10:len(averImage0)-1]))
plt.figure()
#    threshold=np.linspace(0, 0.02, 11)
threshold=ThresholdAve*(1.+4.*ThresholdStd*np.linspace(-1.,1., 11))
print threshold
for i in range(len(threshold)):
   norm, meanx, meany, meanI, stdx, stdy, stdI, Wx, Wy, averImage = imgtl.window_scan2d(IMGt, 1., 100, threshold[i])
   plt.subplot(2,2,1)
   plt.plot (Wx, stdx,'o')
   plt.plot (Wx, stdy,'o')
   plt.subplot(2,2,2)
   plt.plot (Wx, norm,'o')
#       plt.plot (Wx, norm/sum(norm),'o')
             
print "ThreholdAve=", ThresholdAve 
print "ThreholdStd=", ThresholdStd 
print threshold

plt.show()


