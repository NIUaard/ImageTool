import subprocess
import string 
import numpy as np
import time
import tarfile as tar
import ImageTool as imgtl
import matplotlib.pyplot as plt
import sys
from scipy import ndimage
import pylab as pyl

cmdargs = str(sys.argv)

# defaults values
Num = 1
srv = 1 
cal = 1.0


if len(sys.argv)<2:
   print("Error: usage is python ImageAnalysis.py $cal $filename [$background] ...")
   print("      where [$background] is optioinal but highly recommended...")
   sys.exit()
   
if len(sys.argv)==2:
   filenameIm = sys.argv[1]
   print("filename/rootnames: "+filenameIm)
   X = pyl.imread(filenameIm)
   
if len(sys.argv)==3:
   filenameIm = sys.argv[1]
   print("image file: "+filenameIm)
   filenameBk = sys.argv[2]
   print("background file: "+filenameBk)
   Xi = pyl.imread(filenameIm)
   Xb = pyl.imread(filenameBk)
   X=Xi-Xb










Xc=imgtl.RemoveEdge(X, 0) 
plt.imshow(Xc)
norm0, meanx0, meany0, meanI0, stdx0, stdy0, stdI0, correl0, Wx0, Wy0, averImage0, IMGf = \
	      imgtl.window_scan2dthreshold(Xc, 1., 50, 0.00)



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

plt.figure()
plt.imshow(Xc)

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
disp=filenameIm.split('/')
print(disp)
print(len(disp))


plt.subplot (2,2,1)
plt.title (disp[len(disp)-1])		      
plt.imshow(ndimage.gaussian_filter(IMGf, 10), extent=[min(x), max(x), min(y), max(y)], \
          aspect='auto',  cmap='coolwarm')
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

plt.text (0.5,1,'units are pixels')


plt.axis('off')
plt.show()
