import ImageTool as imtl
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
import pydefaults 

# directory
upperfile="./data_samples/X121_20150601//tight_focus/"
filename  ="nml-2015-06-01-2205-23-13076.png"
# in um/pixel
cal   = 9. 
thres = 0.02

A=  imtl.Load(upperfile+filename)
imtl.DisplayImage(A)
plt.title ("raw: "+filename, fontsize=24)
B = imtl.AutoCrop(A, 100)
C = imtl.Threshold(B, thres)

plt.figure()
imtl.DisplayCalibratedProj(B, cal, 0.3)
plt.xlabel ('x ($\mu$m)',fontsize=24)
plt.ylabel ('y ($\mu$m)',fontsize=24)
plt.title ("cal: "+filename, fontsize=24)
plt.tight_layout()
plt.show()
