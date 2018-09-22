'''
  AWA tools used to analyze image taken at AWA
  created: PP 10/29/2016
'''

import struct
import numpy as np
import matplotlib.pyplot as plt 


#
# obsolete should not be imported anymore -- will be removed in future versions. 
#
def fread_m(fid, nelements, dtype):

     if dtype is np.str:
         dt = np.uint8 
     else:
         dt = dtype

     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array


def ReadFrameGrabberDat(imageFilename):
     '''
     open the imageFilename from the matlab framegrabber
     returns a 3D array with all the frames
     '''
     fid = open(imageFilename, 'rb');

     ImSize=fread_m(fid, 2, np.int16)
     Dx=int(ImSize[0])
     Dy=int(ImSize[1])
     print('imagesize=',Dx, Dy)

     NFrame=fread_m(fid, 1, np.int32)
     print('number of frames=',int(NFrame))

     Frame=np.zeros((Dy,Dx,int(NFrame)))

     for i in range(NFrame):
         Offset=i*Dx*Dy+2*2+4;
         fid.seek(Offset);
         Image1D=fread_m(fid,Dx*Dy,np.uint8);
         Image2D=np.reshape(Image1D,(Dy,Dx))
         Frame[:,:,i]=Image2D
	 
     return(Frame) 
#    print np.shape (Frame)
    
    
#    plt.imshow(Frame[:,:,10], cmap='viridis')
#    plt.show()

