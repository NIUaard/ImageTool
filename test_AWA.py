import numpy as np
import time
import os
import scipy.optimize 
import math 
import matplotlib.pyplot as plt
import ImageTool as imgtl
import AWA_tool as awa

Filename  ="./data_samples/TSP1700.dat"

X=awa.ReadFrameGrabberDat(Filename)


plt.imshow(X[:,:,10], cmap='viridis')
plt.show()
