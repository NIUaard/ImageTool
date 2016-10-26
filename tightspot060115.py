import ImageTool as imtl
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
import pydefaults 

# directory
upperfile="/Users/piot/ASTA_commissioning/quadscan/X121_20150601//tight_focus/"
filename  ="nml-2015-06-01-2205-23-13076.png"
# in um/pixel
cal   = 9. 
thres = 0.02

A=  imtl.Load(upperfile+filename)
B = imtl.AutoCrop(A, 100)
C = imtl.Threshold(B, thres)
#imtl.DisplayCalibrated(B, cal)
imtl.DisplayCalibratedProj(B, cal, 0.3)
plt.xlabel ('x ($\mu$m)',fontsize=24)
plt.ylabel ('y ($\mu$m)',fontsize=24)
plt.title (filename, fontsize=24)
plt.tight_layout()
plt.show()
