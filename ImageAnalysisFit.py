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
   print("Error: usage is python ImageAnalysis.py $filename [$background] ...")
   print("      where [$background] is optional but highly recommended...")
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

IMGf=imgtl.AutoCrop(Xc, 2000)

histx, histy, x, y = imgtl.GetImageProjection(IMGf,cal)  

x=x-x[np.argmax(histx)]
y=y-y[np.argmax(histy)]
p2X= imgtl.FitProfile(histx, x)
p2Y= imgtl.FitProfile(histy, y)

print("fitX: ", p2X)
print("fitY: ", p2Y)

plt.figure()
disp=filenameIm.split('/')
print(disp)
print(len(disp))


plt.subplot (2,2,1)
plt.title (disp[len(disp)-1])		      
plt.imshow(ndimage.gaussian_filter(IMGf, 10), extent=[min(x), max(x), min(y), max(y)], \
          aspect='auto', origin='lower',  cmap='coolwarm')
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

text4='sig_x = '+str('{0:.2f}'.format((np.mean(p2X[2])))) 
text5='sig_y = '+str('{0:.2f}'.format((np.mean(p2Y[2])))) 

plt.text (2,4,'Gaussian fit')
plt.text (0.5,3,text4)
plt.text (0.5,2,text5)

plt.text (0.5,1,'units are pixels')


plt.axis('off')
plt.show()
