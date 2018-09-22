import numpy as np
import time
import os
import scipy.optimize 
import math 
import matplotlib.pyplot as plt
import ImageTool as imtl
import AWA_tool as awa
Filename  ="./data_samples/vc.dat"
cal = 0.050 # mm per pixel

X, dx, dy, Nframes=imtl.LoadAWA(Filename)
print ("Dx,Dy,NFrames= ",dx,dy,Nframes)

# sum m all the frame
X_all_frame = np.sum(X, axis=2)

centers= imtl.ImageCenter(X_all_frame)
print (np.shape(X_all_frame))
print ('center=', centers)


imtl.DisplayImage(X_all_frame)
#imtl.DisplayCalibratedProj(X_all_frame, cal, 0.3)

plt.show()
