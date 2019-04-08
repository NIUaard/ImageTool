import sys

import ImageTool as imgtl
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction, QFileDialog, QLineEdit
from PyQt5.QtWidgets import QRadioButton, QCheckBox, QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import random
from cosmetics import * 

# define global varaiables accessible anywhere
# variables associated with data
ImageDisplayed=np.zeros((10,10))
Image=np.zeros((10,10))
Background=np.zeros((10,10))
projx = np.zeros((10))
projy = np.zeros((10))

# variables associated with settings
calibration = 1.0
FLAG_background = 1
FLAG_regofinter = 0
regofint_Value  = np.zeros((4), dtype=int)
FLAG_fit = 0



class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'ImageTool Offline GUI '
        self.width = 740
        self.height = 410
        self.initUI()
 
 # Add button widget
        pybutton = QPushButton('Pyqt', self)
        pybutton.clicked.connect(self.clickMethod)
        pybutton.resize(100,32)
        pybutton.move(130, 30)        
        pybutton.setToolTip('This is a tooltip message.')  

        # Create new action
        loadimAction = QAction(QIcon('new.png'), ' &Load Image', self)        
        loadimAction.setShortcut('Ctrl+I')
        loadimAction.setStatusTip('New document')
        loadimAction.setMenuRole(QAction.NoRole)
        loadimAction.triggered.connect(self.LoadImage)

        # Create new action
        loadbkgAction = QAction(QIcon('open.png'), ' &Load Background', self)        
        loadbkgAction.setShortcut('Ctrl+B')
        loadbkgAction.setMenuRole(QAction.NoRole)
        loadbkgAction.setStatusTip('Open document')
        loadbkgAction.triggered.connect(self.LoadBackground)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), ' &Exit', self)        
        exitAction.setMenuRole(QAction.NoRole)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        self.statusBar()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
	
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu(' &File')
        fileMenu.addAction(exitAction)
	
        imagMenu = menuBar.addMenu(' &Image File')
        imagMenu.addAction(loadimAction)
        imagMenu.addAction(loadbkgAction)
	
        ccdmenu = menuBar.addMenu(' &CCD')
#        ccdmenu.addAction(setcalibration)
	
        procMenu = menuBar.addMenu(' &Image Processing')
#        procMenu.addAction(loadimAction)
#        procMenu.addAction(loadbkgAction)
#        procMenu.addAction(exitAction)    
	
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
	
        m = MyDynamicMplCanvas(self, width=5, height=4)
        m.move(0,0)

# now doe everything in a loop (not efficient) 
#        updater = QPushButton('update display + data', self)
#        updater.setToolTip('This s an example button')
#        updater.move(500,20)
#        updater.resize(220,40)
 
        background = QCheckBox('subtract background', self)
        background.move(520,60)
        background.resize(220,40)
        background.toggle()
        background.stateChanged.connect(self.apply_background)
 
        threshold = QCheckBox('enable threshold', self)
        threshold.move(520,80)
        threshold.resize(220,40)
#        threshold.stateChanged.connect(self.apply_threshold)

        calibration = QCheckBox('apply calibration', self)
        calibration.move(520,100)
        calibration.resize(220,40)
#        threshold.stateChanged.connect(self.apply_calibration)

# enable ROI and take bounding-box corner (TODO need to make simpler with crop feature)
        regofint = QCheckBox('enable ROI', self)
        regofint.move(520,120)
        regofint.resize(220,40)
        regofint.stateChanged.connect(self.apply_regofinter)
	
        regofint_ul_h_l = QLabel("ROI ULX:", self)
        regofint_ul_h_l.move(510,150)
        self.regofint_ul_h = QLineEdit(self)
        self.regofint_ul_h.resize(40,20)
        self.regofint_ul_h.move(570,155)
#        tmp0 = regofint_ul_h.text()

        regofint_ul_v_l = QLabel("ROI ULY:", self)
        regofint_ul_v_l.move(620,150)
        self.regofint_ul_v = QLineEdit(self)
        self.regofint_ul_v.resize(40,20)
        self.regofint_ul_v.move(680,155)
#        tmp1 = regofint_ul_v.text()
	
        regofint_lr_h_l = QLabel("ROI LRX:", self)
        regofint_lr_h_l.move(510,170)
        self.regofint_lr_h = QLineEdit(self)
        self.regofint_lr_h.resize(40,20)
        self.regofint_lr_h.move(570,175)
#        tmp2 = regofint_lr_h.text()

        regofint_lr_v_l = QLabel("ROI LRY:", self)
        regofint_lr_v_l.move(620,170)
        self.regofint_lr_v = QLineEdit(self)
        self.regofint_lr_v.resize(40,20)
        self.regofint_lr_v.move(680,175)
#        tmp3 = regofint_lr_v.text()
   
        fit_check = QCheckBox('turn on fit', self)
        fit_check.move(520,190)
        fit_check.resize(220,40)
        fit_check.stateChanged.connect(self.apply_fit)
 

        self.show()

    def LoadBackground(self):
        global Background
        print('background')
        fnametmp = QFileDialog.getOpenFileName(self, 'Open file', './')
        fname = fnametmp[0]
        print ("filename",fname)
        Background=imgtl.Load(fname)
        Refresh()

    def LoadImage(self):
        global Image
        print('loading image...')
        fnametmp = QFileDialog.getOpenFileName(self, 'Open file', './')  
        fname = fnametmp[0]
        print ("filename",fname)
        Image=imgtl.Load(fname)
        Refresh()

    def exitCall(self):
        print('Exit app')
        exit()

    def clickMethod(self):
        print('PyQt')

    def apply_background(self,state):
        global FLAG_background
        if state>0:
           FLAG_background=1
        else:
           FLAG_background=0	   
        print (state,FLAG_background)
        Refresh()
 
    def apply_fit(self,state):
        global FLAG_fit
        if state>0:
           FLAG_fit=1
        else:
           FLAG_fit=0	   
        print (state,FLAG_fit)
        Refresh()
 
    def apply_regofinter(self,state):
        global FLAG_regofinter
        global regofint_Value
	
        tmp0 = self.regofint_ul_h.text()
        tmp1 = self.regofint_ul_v.text()
        tmp2 = self.regofint_lr_h.text()
        tmp3 = self.regofint_lr_v.text()
	
        if state>0:
           FLAG_regofinter=1
           if len(tmp0)>0 and len(tmp1)>0 and len(tmp2)>0 and len(tmp3)>0:
              regofint_Value[0]= int(tmp0)
              regofint_Value[1]= int(tmp1)
              regofint_Value[2]= int(tmp2)
              regofint_Value[3]= int(tmp3)
        else:
           FLAG_regofinter=0
	
        print (state,FLAG_regofinter)
        Refresh()
 
class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=20, height=15, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
#        self.axes = fig.add_subplot(111)
#        self.axes = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(10, 12)
        self.ax0 = fig.add_subplot(gs[2:10, 0:10])
        self.axx = fig.add_subplot(gs[0:2, 0:10])
        self.axy = fig.add_subplot(gs[2:10,10:12])


        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

 
class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.ax0.imshow(ImageDisplayed)

    def update_figure(self):
        global ImageDisplayed 

        self.ax0.cla()
        self.axx.cla()
        self.axy.cla()
        self.ax0.imshow(ImageDisplayed, aspect='auto', cmap=beam_map,origin='lower')
        projx, projy = ImageDisplayed.sum(axis=0), ImageDisplayed.sum(axis=1)
        x=np.arange(len(projx))
        y=np.arange(len(projy))
        if FLAG_fit>0:
           p2X= imgtl.FitProfile(projx, x)
           p2Y= imgtl.FitProfile(projy, y)
#        # top plot
        self.axx.plot(projx)
        if FLAG_fit>0:
           self.axx.plot (x, imgtl.dg(x,p2X),'r--',linewidth=3)
        self.axx.set_xlim(self.ax0.get_xlim())
#        # Right plot
        self.axy.plot(projy, range(len(projy)))
        if FLAG_fit>0:
           self.axy.plot (imgtl.dg(y,p2Y),y,'r--',linewidth=3)
        self.axy.set_ylim(self.ax0.get_ylim())
        # Remove tick labels
        nullfmt = ticker.NullFormatter()
        self.axx.xaxis.set_major_formatter(nullfmt)
        self.axx.yaxis.set_major_formatter(nullfmt)
        self.axy.xaxis.set_major_formatter(nullfmt)
        self.axy.yaxis.set_major_formatter(nullfmt)

        self.draw()

def Refresh(im=None, bk=None):
    global ImageDisplayed
    global Background
    global Image
    global FLAG_background
    global FLAG_regofinter
    global regofint_Value
    if Image.any()!=0:
       ImageDisplayed=Image
       if (Background.any()!=0) and (FLAG_background>0):
          print('**FLAG"', FLAG_background)
          ImageDisplayed=Image-Background   
       if (Background.any()!=0) and (FLAG_background==0):
          print('**FLAG"', FLAG_background)
          ImageDisplayed=Image  
       if (FLAG_regofinter>0):
          ImageDisplayed=ImageDisplayed[regofint_Value[1]:regofint_Value[3],regofint_Value[0]:regofint_Value[2]]
       
    print ("REFRESH")
    print (regofint_Value )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
