'''
  ImageTool.py 
  description: set of functions to analyze picture from OTR/YAG viewers
  w. computation of statistical correlation matrix on the x,y beam distribution, 
  Gaussian fit, etc...
  originated: P. Piot (PP), June 2015 
  changes:

  - AH, 03/15/2016: added peak normalization, and MonteCarlo functions (AH= A. Halavanau)
  - PP, 11/18/2015: merged different version + added comments on all functions
  - PP, 11/21/2015: added Kyle Capobianco-Hogan's rms calculations
   
'''
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
#import pydefaults 
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import scipy.optimize 
import math 
import scipy.stats
from scipy import ndimage
from cosmetics import * 
from scipy import ndimage
from skimage import feature 
from skimage import measure

#from skimage.filters import sobel


'''
  define a beam-density friendly color map 
'''

global debug

debug=1

def Load(filename):
     '''
       load an image stored in filename are return as a numpy object
     '''
     return(pyl.imread(filename))


def LoadAWA(filename):
     '''
       load an image stored in filename are return as a numpy object
       courtesy from N. Neuveu IIT/ANL
       open the imageFilename from the matlab framegrabber
       returns a 3D array with all the frames
     '''
     images  = np.fromfile(filename, dtype=np.uint16, count=-1,sep='')
     # header info vert/horiz pixels and number of frames
     dx      = int(images[0])
     dy      = int(images[1])
     Nframes = int(images[2])
     hold    = images[6:] # skipping header info
     #==========================================================
     #Reading images into 3D array 
     # X by Y by Frame Number
     
     print('---------LoadAWA()----------')
     print('[dx,dy]=', dx, dy)
     print("NFrames:", Nframes)
     print('-----------------------------')
     imagesArray = np.reshape(hold,(dx, dy, -1), order='F')

     return(imagesArray, dx, dy, Nframes)    
    

def ImageCenter (MyImage):
     '''
       find the barycenter of an image byt copmuting the projection and looking at 
       position averaged on the projections
     '''    
     MyImage=ndimage.gaussian_filter(MyImage, 12)  # smooth the image for max calc only
     indexHmax=np.argmax(np.sum(MyImage,1))
     indexVmax=np.argmax(np.sum(MyImage,0))
     return([indexHmax, indexVmax])
    
    
def AutoCrop(MyImage, hbbox):
     '''
       do a square crop (if possible) around the image center (defined as the area with 
       maximum intensity)
     '''    
     indexVmax=np.argmax(np.sum(MyImage,1))
     indexHmax=np.argmax(np.sum(MyImage,0))
     print('---------AutoCrop()----------')
     print(indexHmax, indexVmax)
     print('-----------------------------')
     return(Crop(MyImage, [indexHmax, indexVmax], [hbbox, hbbox]))
     
     
def MouseCrop(MyImage):
     '''
       displays image and wait for mouse action to select center, upper left and bottom right
     '''    
     # template TODO
     return()


def CannyCrop(MyImage):  # !!!!!! NOT WORKING !!!!!!
     '''
       displays image and wait for mouse action to select center, upper left and bottom right
     '''    
     SmoothImg=ndimage.gaussian_filter(MyImage, 1)
     
#     contours = measure.find_contours(r, 0.500)
     
     edges = feature.canny(MyImage)     
     pts = np.argwhere(edges>0)
#     y1,x1 = pts.min(axis=0)
#     y2,x2 = pts.max(axis=0)
#     Img = MyImage[y1:y2, x1:x2]
     Img=MyImage
     return(Img, edges)


def DesInterlace(MyImage):
     '''
       remove intelacing problem on AWA analog camera
       do this using a simple Gaussian filter
     '''
     return(ndimage.gaussian_filter(MyImage, 1))
     
     
def RemoveEdge(MyImage, edgesize):
     '''
       do a square (if possible) crop around the image center (defined as the area with 
       maximum intensity)
     '''    
     MyShape=np.shape(MyImage)
     return(MyImage[edgesize:MyShape[1]-edgesize,edgesize:MyShape[0]-edgesize ])
     
     
def Crop(MyImage, center, hbbox):
     '''
       do a crop around the point with coordinates center[0], center[1] with box size 
       hbbox[0]*2 hbbox[1]*2
       
     '''    
     shapec=np.shape(MyImage)
     minx=max(center[1]-hbbox[1],0)
     maxx=min(center[1]+hbbox[1],shapec[1])
     miny=max(center[0]-hbbox[0],0)
     maxy=min(center[0]+hbbox[0],shapec[0])
     
     if debug==1:
        print('-------------Crop()----------')
        print(center)
        print(hbbox)
        print(minx)
        print(maxx)
        print(miny)
        print(maxy)
        print('-----------------------------')
     
#     return(MyImage[center[1]-hbbox[1]:center[1]+hbbox[1], \
#                    center[0]-hbbox[0]:center[0]+hbbox[0]])
     return(MyImage[int(minx):int(maxx),int(miny):int(maxy)])
     
     
def DisplayImage(MyImage):
     '''
       do a crop around the image center (defined as the area with 
       maximum intensity)
     '''    
     plt.imshow(MyImage, aspect='auto', cmap=beam_map,origin='lower')
     plt.colorbar()
     return()
     
     
def Threshold(MyImage, thres):
     '''
       set to zero value of the image below the valye thres
     '''    
     index =np.where(MyImage<thres)
     MyImage[index]=0.0
     return(MyImage)


  
def center_and_ROI(projection, axiscoord, window):
     '''
       Locate the max of a projection and select a ROI around this max
     '''    
     MaxLoc = np.argmax(projection)
     Hist   = projection[MaxLoc-round(window/2.):MaxLoc+round(window/2.)]
     Coord  = axiscoord [MaxLoc-round(window/2.):MaxLoc+round(window/2.)]-axiscoord[MaxLoc]
     return(Hist, Coord)  
  

def removebackground(projection, axiscoord, window):
     '''
       Subtract background of an histogram by removing an average 
       of the background level measured on the left side of the profile
     '''    
     Bkgrd = np.mean(projection[0:window])
     Hist  = projection-Bkgrd*np.ones((len(projection)))
     return(Hist, axiscoord)
       
       
def DisplayCalibrated(MyImage, cal):
     '''
       Display image with calibrated axis
       cal is in um/pixel and assumed to be the same in both directions
     '''    
     indexXmax=np.argmax(np.sum(MyImage,0))
     indexYmax=np.argmax(np.sum(MyImage,1))
     ImShape=np.shape(MyImage)
     calx=cal
     caly=cal
     
     xmin=calx*(0.-indexXmax)
     xmax=calx*(ImShape[0]-indexXmax)
     ymin=caly*(0.-indexYmax)
     ymax=caly*(ImShape[1]-indexYmax)
     
#     print ImShape
#     print indexXmax, indexYmax
#     print xmin, xmax, ymin, ymax
     plt.imshow(MyImage, aspect='auto', cmap=beam_map,origin='lower',extent=[xmin, xmax, ymin, ymax])
     plt.colorbar()
 
 
def GetImageProjection(MyImage, cal):
     '''
       Return the horizontal and vertical projection
       cal is in um/pixel and assumed to be the same in both directions
     '''    
     
     if debug==1:
        print('GetImageProjection shape', np.shape(MyImage))
     indexXmax=np.argmax(np.sum(MyImage,0))
     indexYmax=np.argmax(np.sum(MyImage,1))
     
     ImShape=np.shape(MyImage)

     calx=cal
     caly=cal

     
     xhist = np.sum(MyImage,0)
     yhist = np.sum(MyImage,1)

     xcoord = np.linspace(0,len(xhist),len(xhist))*calx
     ycoord = np.linspace(0,len(yhist),len(yhist))*caly

     return(xhist,yhist,xcoord,ycoord)
     
     
     
def GetImageProjectionCal(MyImage, cal):
     '''
       Return the horizontal and vertical projection
       cal is in um/pixel and assumed to be the same in both directions
     '''    
     
     print('GetImageProjection shape', np.shape(MyImage))
     indexXmax=np.argmax(np.sum(MyImage,0))
     indexYmax=np.argmax(np.sum(MyImage,1))
     
     ImShape=np.shape(MyImage)

     calx=cal
     caly=cal

     xmin=calx*(0.-indexXmax)
     xmax=calx*(ImShape[0]-indexXmax)
     ymin=caly*(0.-indexYmax)
     ymax=caly*(ImShape[1]-indexYmax)
     
     xhist = np.sum(MyImage,0)/np.sum(np.sum(MyImage,0))
     yhist = np.sum(MyImage,1)/np.sum(np.sum(MyImage,1))

     xcoord = xmin+np.linspace(0,1,len(xhist))*(xmax-xmin)
     ycoord = ymin+np.linspace(0,1,len(yhist))*(ymax-ymin)

     return(xhist,yhist,xcoord,ycoord)
     
def DisplayCalibratedProj(MyImage, cal, fudge):
     '''
       Display a picture with superimposed histogram of the image
     '''    
     
     print('image size:', np.shape(MyImage))
     indexXmax=np.argmax(np.sum(MyImage,0))
     indexYmax=np.argmax(np.sum(MyImage,1))
     ImShape=np.shape(MyImage)
     calx=cal
     caly=cal
     
     xmin=calx*(0.-indexXmax)
     xmax=calx*(ImShape[0]-indexXmax)
     ymin=caly*(0.-indexYmax)
     ymax=caly*(ImShape[1]-indexYmax)
     
     xhist = np.sum(MyImage,0)/np.max(np.sum(MyImage,0))
     yhist = np.sum(MyImage,1)/np.max(np.sum(MyImage,1))
     
     xcoord = xmin+np.linspace(0,1,len(xhist))*(xmax-xmin)
     xhist  = ymin+ fudge*(ymax-ymin)*xhist
     ycoord = ymin+np.linspace(0,1,len(yhist))*(ymax-ymin)
     yhist  = xmin+ fudge*(xmax-xmin)*yhist
     
#     print ImShape
#     print indexXmax, indexYmax
     print(xmin, xmax, ymin, ymax)

     plt.imshow(MyImage, aspect='auto', cmap=beam_map,origin='lower',extent=[xmin, xmax, ymin, ymax])
     plt.plot(xcoord,xhist,color='r',linewidth=3) 
     plt.plot(yhist, ycoord,color='r', linewidth=3) 
     plt.ylim(ymin, ymax)
     plt.xlim(xmin, xmax)
     plt.colorbar()
     
     
     
def Normalize(MyImage):
     '''
        Renormalize Image according to its maximum value
     '''        
     maxv=np.amax(MyImage)
     MyImage=MyImage/maxv
     return(MyImage)



def MonteCarloXY(MyImage,N,cal):
     '''
       generate a bunch of N points with density distribution
       given by the image MyImage
       cal is the calibration assumed to be the same for the two
       directions
     '''
     x,y = np.shape(MyImage)
     print(x,y)
     dist=np.zeros((N,2))
     i=0
     while i<N:
        rand_x=np.random.random_integers(x-2)+np.random.uniform(-1, 1)
        rand_y=np.random.random_integers(y-2)+np.random.uniform(-1, 1)
        value=np.random.rand()
        if value<MyImage[int(round(rand_x)),int(round(rand_y))]:
#add randomized dx,dy
            dist[i,0]=rand_x
            dist[i,1]=rand_y
            i=i+1
     meanx,meany = dist.mean(axis=0)
     dist[:,0]=dist[:,0]-meanx
     dist[:,1]=dist[:,1]-meany
     dist=dist*cal*1.0e-6

     xrms=sqrt(mean(square(dist[:,0])))
     yrms=sqrt(mean(square(dist[:,1])))

     print("RMS values:")
     print(xrms*1000.0, yrms*1000.0)

     return(dist)
     
     

def FitProfile(projection, axiscoord):
     '''
       Fit the projection to a function dg()
     '''    
     xhist = projection
     xaxis = axiscoord
     indexXmax=xaxis[np.argmax(xhist)]
     bkg = np.mean(xhist[0:10])
     Xmax = np.max(xhist)
     p0x  = [indexXmax,Xmax, 1.,bkg]
     print(Xmax, indexXmax, bkg)
     ErrorFunc = lambda p0x,xaxis,xhist: dg(xaxis,p0x)-xhist
     p2,success = scipy.optimize.leastsq(ErrorFunc, p0x[:], args=(xaxis,xhist))
     
     return(p2)
 
 
     
def dg(x,p0):
     '''
       Gaussian distribution used in the fitprofile function
     '''    
     rv=np.zeros(len(x))
     for i in range(len(x)):
         rv[i]=p0[3]+p0[1]*math.exp(-(x[i]-p0[0])*(x[i]-p0[0])/2/p0[2]/p0[2])
     return rv



def stats1d(x, f):
    ''' 
       define moments of x associated to f(x)
       modified (vectorized + simplified) from Kyle's 
    ''' 
    F0 = sum(f[:])
    F1 = sum(x[:] * f[:])
    F2 = sum(x[:]**2 * f[:])
    F3 = sum(x[:]**3 * f[:])
    F4 = sum(x[:]**4 * f[:])
    mean = F1/F0
    mu_2_r = F2/F0
    var = mu_2_r - mean**2
    std = np.sqrt(var)
    mu_3_r = F3/F0
    skew = (mu_3_r - 3.*mu_2_r*mean + 2.*mean**3) / (var*std)
    mu_4_r = F4/F0
    kurt = ((mu_4_r - 4.*mu_3_r*mean + 6.*mu_2_r*mean**2 - 3.*mean**4)
        / (var**2))
    if debug==1:
       print('mean:\t' + str(mean))
       print('std:\t' + str(std))
       print('skew:\t' + str(skew))
       print('kurt:\t' + str(kurt) + '\t(Fisher: ' + str(kurt-3.) + ')')
    
    return (mean, std, skew, kurt)



def stats2d(x, y, f):
    ''' 
       define 1st and 2nd order moments of x and y associated 
       to the distributionfunction f(x,y)
    ''' 
    F0   = np.sum(np.sum(f))
    F1x  = np.sum(x * np.sum(f,0))
    F1y  = np.sum(y * np.sum(f,1))
    F2x  = np.sum(x**2 * np.sum(f,0))
    F2y  = np.sum(y**2 * np.sum(f,1))
    

    norm    = F0
    
    meanx   = F1x/F0
    mu_2_rx = F2x/F0
    varx    = mu_2_rx - meanx**2
    stdx    = np.sqrt(varx)
    
    meany   = F1y/F0
    mu_2_ry = F2y/F0
    vary    = mu_2_ry - meany**2
    stdy    = np.sqrt(vary)

    F2xy = np.sum((y-meany) * np.sum((x-meanx)*f,1))
    
    
    correl  = F2xy/norm/(stdx*stdy)
    
    meanI   = np.mean(np.mean(f))
    mu_2_rI = np.var(f)
    stdI    = mu_2_rI
    
    if debug==1:
       print('--------stats2d()-----------------')
       print('norm   :\t' + str(norm))
       print('meanx  :\t' + str(meanx))
       print('stdx   :\t' + str(stdx))
       print('meany  :\t' + str(meany))
       print('stdy   :\t' + str(stdy))
       print('correl :\t' + str(correl))
       print('----------------------------------')
#    print 'skew:\t' + str(skew)
#    print 'kurt:\t' + str(kurt) + '\t(Fisher: ' + str(kurt-3.) + ')'
    return (norm, meanx, meany, meanI, stdx, stdy, correl, stdI)


    
def window_scan2dthreshold (IMG, cal, Npts, threshold=0):
    '''
       compute statistics on an image with varying window as function 
       of image intensity. This assumes one already took care of centering 
       the image (i.e. the peak intensity is in the center of the image)
       
       IMG:        the image to analyse
       cal:        the pixel to mm calibration coefficient
       Npt:        number of windows
    '''

    indexXmax=np.argmax(np.sum(IMG,1))
    indexYmax=np.argmax(np.sum(IMG,0))
        
    histx, histy, xx, yy = GetImageProjection(IMG,cal)

    Meanx_0, dumm, dumm, dumm = stats1d(xx, histx)
    Meany_0, dumm, dumm, dumm = stats1d(yy, histy)
    
# start with a 5 pixel ROI       
    wx = 15
    wy = 15
    
        
    norm     = np.zeros((Npts))
    meanx    = np.zeros((Npts))
    meany    = np.zeros((Npts))
    meanI    = np.zeros((Npts))
    stdx     = np.zeros((Npts))
    stdy     = np.zeros((Npts))
    stdI     = np.zeros((Npts))
    Aver_Im  = np.zeros((Npts))
    meanxOut = np.zeros((Npts))
    meanyOut = np.zeros((Npts))
    stdxOut  = np.zeros((Npts))
    stdyOut  = np.zeros((Npts))
    Wx       = np.zeros((Npts))
    Wy       = np.zeros((Npts))
    correl   = np.zeros((Npts))
    
    imSize   = np.shape(IMG)

# first need to get an idea of the background? 
    auto=1
    
    epsilon=10
       
    i=0
#    for i in range(Npts):
    while (i<(Npts-5) and (epsilon>1.)):
        Cropped_Image=np.copy(IMG)
        if auto==1:
           if i==0:
              Wx[i] = wy 
              Wy[i] = wx 

           if i>0:
              Wy[i] = 4.*stdx[i-1]
              Wx[i] = Wy[i]*stdy[i-1]/stdx[i-1]
        
        if auto==0:
           if i==0:
              Wx[i] = wx 
              Wy[i] = wy 

           if i>0:
              Wy[i] = i*wx
              Wx[i] = i*wy
        
        histx, histy, x, y = GetImageProjection(Cropped_Image,1.)  # using pixel (force cal=1) within loop
        
        if debug==1:
           print("windowing ......")

        lx, ly = Cropped_Image.shape
        X, Y = np.ogrid[0:lx, 0:ly]
        mask  = (np.abs(indexXmax-X)>=round(Wx[i]))+(np.abs(indexYmax-Y)>=round(Wy[i]))
        
        Aver_Im [i] = np.mean(np.mean(Cropped_Image[mask]))
                
                        
        if threshold==0:
           if i==0:
              Cropped_Image = Cropped_Image-Aver_Im [i]
           if i>0:
              Cropped_Image = Cropped_Image-Aver_Im [i-1]

           Cropped_Image[mask]=0.0


        shapec=np.shape(Cropped_Image)
        minxc=max(round(indexXmax-Wx[i]),0)
        maxxc=min(round(indexXmax+Wx[i]),shapec[0]-1)
        minyc=max(round(indexYmax-Wy[i]),0)
        maxyc=min(round(indexYmax+Wy[i]),shapec[1]-1)
        
        
        if debug==1:
           print("shape and cropped area")
           print(shapec)
           print(minxc)
           print(maxxc)
           print(minyc)
           print(maxyc)
           print(i)
           print(epsilon) 
        
#        IMGf= Cropped_Image[round(indexXmax-Wx[i]):round(indexXmax+Wx[i]),round(indexYmax-Wy[i]):round(indexYmax+Wy[i])]
#        IMGf= Cropped_Image[minxc:maxxc, minyc:maxyc]
        IMGf= Crop(Cropped_Image, [indexYmax, indexXmax], [round(Wy[i]), round(Wx[i])])
	# use calibrated x,y when getting the startistics
        norm[i], meanx[i], meany[i], meanI[i], stdx[i], stdy[i], correl[i], stdI[i] = stats2d (x, y, Cropped_Image)

        if auto==1:
           if i>0:
              epsilon = np.sqrt((stdx[i]-stdx[i-1])**2+(stdy[i]-stdy[i-1])**2)
              if debug==1:
                 print("epsilon", epsilon)
        i=i+1
        
    return(norm[0:i-1], cal*meanx[0:i-1], cal*meany[0:i-1], meanI[0:i-1], cal*stdx[0:i-1], \
           cal*stdy[0:i-1], stdI[0:i-1], correl[0:i-1], Wx[0:i-1], Wy[0:i-1], Aver_Im[0:i-1], IMGf)
    
    
    
    
def window_scan1d(x, histx, w_n):
    ''' 
       define moments of x associated to f(x)
       modified (vectorized + simplified) from Kyle's 
    ''' 

    # Background threshold.
    w_t = 0.05

    # Window dimensions:
    #   Window offset.
    w_off = 80
    #   Window overlap.
    w_lap = 10
    #   Window length.
    w_len = w_off + w_lap
    

    # Find location of max.
    x_max = np.argmax(histx)
    x_len = len(histx)
    print(x_max)
    # Initalize window analysis arrays.
    w_a    = np.zeros(w_n, dtype=int)
    w_b    = np.zeros(w_n, dtype=int)
    w_mean = np.zeros(w_n, dtype=float)
    w_std  = np.zeros(w_n, dtype=float)
    w_skew = np.zeros(w_n, dtype=float)
    w_kurt = np.zeros(w_n, dtype=float)

    # Window after max.
    print('\nwindowing after max:')
    #print 'i\tw_a[i]\tw_b[i]\tw_mean[i]\tw_std[i]\tw_skew[i]\tw_kurt[i]'
    w_check = False # Controls whether or not to check for end of signal.
    sig_stop = x_len
    for i in range(w_n):
        w_a[i] = x_max + i*w_off
        w_b[i] = w_a[i] + w_len
        w_mean[i] = np.mean(histx[w_a[i]:w_b[i]])
        w_std[i] = np.std(histx[w_a[i]:w_b[i]], ddof=1)
        w_skew[i] = scipy.stats.skew(histx[w_a[i]:w_b[i]], bias=False)
        w_skew[i] = scipy.stats.kurtosis(histx[w_a[i]:w_b[i]], fisher=False,
            bias=False)
        print((str(i) + '\t' + str(w_a[i]) + '\t' + str(w_b[i]) + '\t'
            + str(w_mean[i]) + '\t' + str(w_std[i]) + '\t' + str(w_skew[i]) + '\t'
            + str(w_kurt[i])))
        if (w_check and (w_std[i] <= w_t) and (w_std[i-1] <= w_t)):
            sig_stop = w_a[i]
            print('sig_stop:\t' + str(sig_stop))
            w_check = False
        if (i == 1):
            w_check = True

    # Window before max.
    print('\nwindowing before max:')
    #print 'i\tw_a[i]\tw_b[i]\tw_mean[i]\tw_std[i]\tw_skew[i]\tw_kurt[i]'
    w_check = False # Controls whether or not to check for end of signal.
    sig_start = 0
    for i in range(w_n):
        w_a[i] = x_max - i*w_off
        w_b[i] = w_a[i] - w_len
        w_mean[i] = np.mean(histx[w_b[i]:w_a[i]])
        w_std[i] = np.std(histx[w_b[i]:w_a[i]], ddof=1)
        w_skew[i] = scipy.stats.skew(histx[w_b[i]:w_a[i]], bias=False)
        w_skew[i] = scipy.stats.kurtosis(histx[w_b[i]:w_a[i]], fisher=False,
            bias=False)
        #print (str(i) + '\t' + str(w_a[i]) + '\t' + str(w_b[i]) + '\t'
        #    + str(w_mean[i]) + '\t' + str(w_std[i]) + '\t' + str(w_skew[i]) + '\t'
        #    + str(w_kurt[i]))
        if (w_check and (w_std[i] <= w_t) and (w_std[i-1] <= w_t)):
            sig_start = w_a[i] + 1
            print('sig_start:\t' + str(sig_start))
            w_check = False
        if (i == 1):
            w_check = True

    sig_len = sig_stop - sig_start

    '''
    ========================================================================
    Calculate statistics on truncated region.
    ========================================================================
    '''

    print('\ntruncated dataset statistics:')
    trnk_mean, trnk_std, z_dum1, z_dum2 = stats1d(x[sig_start:sig_stop], histx[sig_start:sig_stop])

    '''
    ========================================================================
    Calculate and zero the mean background level. Then attampt statistical
    calculations on signal region.
    ========================================================================
    '''

    print('\nbackground level statistics:')

    A1 = 0.
    Q = 0.
    k = 0.
    for i in range(0, sig_start):
        k += 1.
        A2 = A1
        A1 += (histx[i] - A2) / k
        Q += (histx[i] - A2)*(histx[i] - A1)
    for i in range(sig_stop, x_len):
        k += 1.
        A2 = A1
        A1 += (histx[i] - A2) / k
        Q += (histx[i] - A2)*(histx[i] - A1)
    bkg_mean = A1
    bkg_std = np.sqrt(Q / (k-1.))

    print('mean:\t' + str(bkg_mean))
    print('std:\t' + str(bkg_std))

    # Zero mean background level.
    histx_z = histx - bkg_mean

    print('\ntruncated dataset (with background reduction) statistics:')
    z_mean, z_std, z_dum1, z_dum2 = stats1d(x[sig_start:sig_stop], histx_z[sig_start:sig_stop])

    #print '\nfull dataset (with background reduction) statistics:'
    #stats(x[:], histx_z[:], x_len)


