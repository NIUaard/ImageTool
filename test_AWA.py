import numpy as np
import time
import os
import scipy.optimize 
import math 
import matplotlib.pyplot as plt
import ImageTool as imtl
import AWA_tool as awa

Filename  ="./data_samples/vc.dat"
cal = 0.039 # mm per pixel

X, dx, dy, Nframes=imtl.LoadAWA(Filename)
print ("Dx,Dy,NFrames= ",dx,dy,Nframes)

# sum m all the frame
X_all_frame = imtl.DesInterlace(np.sum(X, axis=2))

TT=imtl.MouseCrop(X_all_frame)
centers= imtl.ImageCenter(TT)
print ('center=', centers)
 

plt.figure()
#imtl.DisplayImage(X_all_frame)
#plt.subplot (2,2,1)
#imtl.DisplayCalibratedProj(X_all_frame, cal, 0.3)
#
#plt.subplot (2,2,2)
#imtl.DisplayCalibratedProj(imtl.Crop(X_all_frame,centers, [200,200]), cal, 0.3)


#plt.subplot (2,2,3)
imtl.DisplayCalibratedProj(TT, cal, 0.3)



plt.show()
