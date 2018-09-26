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
from scipy import ndimage
from cosmetics import *

# you have to change this line to the directory where the data are located in
FileDir = "./data_samples/"
# one of the data set 
SubDir   = "/"
rootname = "0"

#/Users/piot/ASTA_commissioning/quadscan/X121_20150601//tight_focus/"
FilenameBkgd  ="X101_bkg_0.png"
FilenameBeam  ="X101_img_0.png"


FTsize=8

test = 0  
# parameters
bbox = 200   # bounding box 
cal  = 0.05  # calibration 
fudge = 0.3  #
threshold = 0. # threshold (remove anything below this minimum pixel)
interlacing = 1 

filenameBeam=FileDir+FilenameBeam
filenameBkgd=FileDir+FilenameBkgd
    
# load the image 

if filenameBkgd!=None: 
   IMGbeam=imgtl.Load(filenameBeam)
   IMGbkgd=imgtl.Load(filenameBkgd)
   IMG=1.*IMGbeam-IMGbkgd
else:
   IMG=IMGbeam  
  
if interlacing==1:
   IMG=ndimage.gaussian_filter(IMG, 1) 
   
   
# display raw image    
plt.figure()
plt.subplot(2,2,1)
imgtl.DisplayImage(IMG)
plt.title('raw data::'+FilenameBeam,fontsize=FTsize)
plt.axis('off')
print('size raw:', np.shape(IMG))
# crop image 
#    plt.figure()
plt.subplot(2,2,2)
IMGc=imgtl.RemoveEdge(IMG, 100)
print('size removed:', np.shape(IMGc))
imgtl.DisplayCalibratedProj(IMGc, cal, fudge)
plt.title('cropped'+FilenameBeam,fontsize=FTsize)
plt.axis('off')
# threshold image 
#    plt.figure()
plt.subplot(2,2,3)
IMGt=ndimage.gaussian_filter(IMGc, 0) 
# imgtl.Threshold(IMGc, threshold)
imgtl.DisplayCalibratedProj(IMGt, cal, fudge)
plt.title('cropped::'+FilenameBeam,fontsize=FTsize)
plt.axis('off')
# compute profiles
histx, histy, x, y = imgtl.GetImageProjection(IMGt,cal)   
#    plt.figure()
plt.subplot(2,2,4)
x=x-x[np.argmax(histx)]
y=y-y[np.argmax(histy)]
plt.plot (x,histx,'-')
plt.plot (y,histy,'-')
plt.xlabel ('distance')
plt.ylabel ('population')
plt.title('profiles::'+FilenameBeam,fontsize=FTsize)
plt.show()
# RMS calculations: 

imgtl.stats1d(x, histx)
imgtl.stats1d(y, histy)
imgtl.stats2d(x,y,IMGt)

# copmute statistics using old algorithm 
norm0, meanx0, meany0, meanI0, stdx0, stdy0, stdI0, correl0, Wx0, Wy0, averImage0, IMGf = \
                 imgtl.window_scan2dthreshold(IMGt, cal, 50, 0.00)

plt.figure()
plt.subplot (2,2,1)
plt.plot (Wx0, averImage0,'o')
plt.subplot (2,2,2)
plt.plot (Wx0, stdx0,'ro')
plt.plot (Wx0, stdy0,'go')
plt.subplot (2,2,3)
plt.plot (Wx0, correl0,'ro')
plt.subplot (2,2,4)
plt.plot (Wx0, norm0,'ro')


print(np.shape(IMGf))

histx, histy, x, y = imgtl.GetImageProjection(IMGf,cal)  

x=x-x[np.argmax(histx)]
y=y-y[np.argmax(histy)]
p2X= imgtl.FitProfile(histx, x)
p2Y= imgtl.FitProfile(histy, y)

print('RMS_x=',np.mean(stdx0[len(stdx0)-2:len(stdx0)+2]))
print('RMS_y=',np.mean(stdy0[len(stdy0)-2:len(stdy0)+2]))
print("fitX: ", p2X)
print("fitY: ", p2Y)

plt.figure()
plt.subplot (2,2,1)
plt.imshow(ndimage.gaussian_filter(IMGf, 5), extent=[min(x), max(x), min(y), max(y)], \
          aspect='auto', origin='lower',  cmap=beam_map)
plt.subplot (2,2,3)
plt.plot (x, histx)
plt.plot (x, imgtl.dg(x,p2X),'--',linewidth=3)
plt.xlim(min(x), max(x))
plt.subplot (2,2,2)
plt.plot (histy,y)
plt.plot (imgtl.dg(y,p2Y),y,'--',linewidth=3)
plt.ylim(min(y), max(y))
plt.subplot (2,2,4)
plt.ylim(0,10)
plt.xlim(0,10)
text1='rms_x = '+str('{0:.2f}'.format((np.mean(stdx0[len(stdx0)-2:len(stdx0)+2])))) \
                      +'+/-'+str('{0:.2f}'.format(np.std(stdx0[len(stdx0)-2:len(stdx0)+2])))
text2='rms_y = '+str('{0:.2f}'.format((np.mean(stdy0[len(stdy0)-2:len(stdy0)+2])))) \
                      +'+/-'+str('{0:.2f}'.format(np.std(stdy0[len(stdy0)-2:len(stdy0)+2])))
text3='<xy>/(rms_x*rms_y) = '+str('{0:.2f}'.format((np.mean(correl0[len(correl0)-2:len(correl0)+2])))) \
                      +'+/-'+str('{0:.2f}'.format(np.std(correl0[len(correl0)-2:len(correl0)+2])))

text4='sig_x = '+str('{0:.2f}'.format((np.mean(p2X[2])))) 
text5='sig_y = '+str('{0:.2f}'.format((np.mean(p2Y[2])))) 
		      
plt.text (2,9,'statistical analysis')
plt.text (0.5,8,text1)
plt.text (0.5,7,text2)
plt.text (0.5,6,text3)

plt.text (2,4,'Gaussian fit')
plt.text (0.5,3,text4)
plt.text (0.5,2,text5)

plt.text (0.5,0.1,'units given by calibration coeff.', color='red')


plt.axis('off')
plt.show()



