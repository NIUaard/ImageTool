import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import ImageTool as imtl
from cosmetics import *

def onselect_orign(eclick, erelease):
    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
    ax.set_ylim(erelease.ydata,eclick.ydata)
    ax.set_xlim(eclick.xdata,erelease.xdata)
    fig.canvas.draw()
    return()

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def onselect_getroi(eclick, erelease):
    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
    print (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)	
#    fig.canvas.draw()
    bbox = np.array ([int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)])
    return (bbox)

Filename  ="./data_samples/vc.dat"
X, dx, dy, Nframes=imtl.LoadAWA(Filename)
print ("Dx,Dy,NFrames= ",dx,dy,Nframes)

# sum m all the frame
arr = imtl.DesInterlace(np.sum(X, axis=2))




fig = plt.figure()
ax = fig.add_subplot(111)

print ("select bounding box for signal and press Q")
plt_image=plt.imshow(arr)
toggle_selector.RS =widgets.RectangleSelector(
    ax, onselect_getroi, drawtype='box',
    rectprops = dict(facecolor='red', edgecolor = 'red', alpha=0.5, fill=True), interactive=True)

print (toggle_selector.RS)
plt.connect('key_press_event', toggle_selector)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

print ("select bounding box for background and press Q")
plt_image=plt.imshow(arr)
toggle_selector.RS2 =widgets.RectangleSelector(
    ax, onselect_getroi, drawtype='box',
    rectprops = dict(facecolor='blue', edgecolor = 'blue', alpha=0.5, fill=True), interactive=True)

plt.connect('key_press_event', toggle_selector)
    
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

plt_image=plt.imshow(arr, cmap=beam_map)

plt.connect('key_press_event', toggle_selector)
    
plt.show()
