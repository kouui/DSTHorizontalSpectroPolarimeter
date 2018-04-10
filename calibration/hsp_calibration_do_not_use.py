"""
author: kouui
date: 2018/03/28

python version of ../dsthssp_calib.pro
"""
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import sys, glob, os, gc, time, datetime
from scipy.io import readsav
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QWidget, qApp
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QSizePolicy
from PyQt5.QtWidgets import QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from DSTPolarimeterLib import *
from lmfit import minimize, Parameters

################################################################################
#
#
################################################################################


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.title = "DST/HS/SP CALIBRATION"
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 400

        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #--- add statusbar
        self.statusBar().showMessage("message from statusbar!")

        #--- add menuBar
        self.addMenuBar()


        #--- layout
        self.mainWidget = MainWidget(self)
        self.setCentralWidget(self.mainWidget)

        self.show()

    def addMenuBar(self):

        mainMenu = self.menuBar()

        """
        if is macOS then turn off the native menubar
        otherwise it will be covered by the systems menubar
        """
        if sys.platform=="darwin":
            mainMenu.setNativeMenuBar(False)

        #--- add menu into menubar
        fileMenu = mainMenu.addMenu("File")

        #--- add action into menu "File"
        exitAction = QAction("Exit", self)
        exitAction.setStatusTip("Exit Application")
        for action in [exitAction]:
            fileMenu.addAction(action)

        #--- define functionality of each action
        exitAction.triggered.connect(self.close)


class MainWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.parent = parent

        mainLayout = QVBoxLayout()

        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # sizePolicy.setHorizontalStretch(7)
        sizePolicy.setWidthForHeight(True)

        #--- part1
        mainLayout1 = QVBoxLayout()
        layout1 = QGridLayout()

        self.lineEdit_calibrationData = QLineEdit("/Users/liu/Desktop/dstsp-master/idl/PythonVer/20171128/camera01/cal*.sav")
        self.lineEdit_firstCameraData = QLineEdit("/Users/liu/Desktop/dstsp-master/idl/PythonVer/20171128/camera01/cal*.sav")
        self.lineEdit_badData = QLineEdit()
        self.btn_startCalibration = QPushButton("start Calibration")
        self.btn_startCalibration.setSizePolicy(sizePolicy)

        layout1.addWidget(QLabel("Data folder :"), 0, 0)
        layout1.addWidget(self.lineEdit_calibrationData, 0, 1)
        layout1.addWidget(QLabel("First Camera :"), 1, 0)
        layout1.addWidget(self.lineEdit_firstCameraData, 1, 1)
        layout1.addWidget(QLabel("Bad data :"))
        layout1.addWidget(self.lineEdit_badData)

        mainLayout1.addLayout(layout1)
        mainLayout1.addWidget(self.btn_startCalibration)
        mainLayout.addLayout(mainLayout1)

        #---part2
        mainLayout2 = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout22 = QHBoxLayout()

        self.btn_png = QPushButton("save .png")
        self.btn_eps = QPushButton("save .eps")
        for widget in [self.btn_png, self.btn_eps]:
            layout22.addWidget(widget)


        self.lineEdit_loadData = QLineEdit()
        layout2.addWidget(QLabel("Load data"))
        layout2.addWidget(self.lineEdit_loadData)

        mainLayout2.addLayout(layout2)
        mainLayout2.addLayout(layout22)

        mainLayout.addLayout(mainLayout2)

        #---
        self.setLayout(mainLayout)

        #--- button functionality
        self.btn_startCalibration.clicked.connect(self.clicked_StartCalibration)

    def clicked_StartCalibration(self):


        #--- check whether folders exist
        """
        if os.path.isdir(self.lineEdit_calibrationData.text()) is False:

            QMessageBox.about(self, "Error", "data folder doesn't exist.")

            return None

        elif os.path.isdir(self.lineEdit_firstCamera.Data.text()) is False:

            QMessageBox.about(self, "Error", "first camera folder doesn't exist.")

            return None

        else:

            pass
        """

        #--- read filenames
        files = sorted( glob.glob( self.lineEdit_calibrationData.text() ) )
        reffiles = sorted( glob.glob( self.lineEdit_firstCameraData.text() ) )

        if len(files)==0 :
            QMessageBox.about(self, "Error", "data files doesn't exist.")
        elif len(reffiles)==0 :
            QMessageBox.about(self, "Error", "first camera files doesn't exist.")
        else :
            pass
        # bad = self.lineEdit_badData.text()

        #--- calculate ref_index and ref_time
        ref_index, ref_time = self.readReferenceIndex(reffiles)
        print("ref_index ref_time ok")
        """
        saveFilesHds = []
        for reffile in reffiles:
            saveFilesHds.append(readsav(reffile, verbose=False)["hds"])
        ref_index, ref_time = self.readReferenceIndex(reffiles, saveFilesHds=saveFilesHds)
        saveFilesHds.clear()
        del saveFilesHds
        """

        #--- decide threshold by clicking histogram
        self._running = False

        filesDict = dict()
        for i, file in enumerate(files):
            filesDict["{}".format(i)] = readsav(file, verbose=False)

        threshold = self.decideThreshold(saveFile=filesDict["0"])

        print("threshold ok!")


        #--- take average and standard deviation over the full image
        for i in range(len(filesDict)):
            if i == 0:
                iquv = filesDict["{}".format(i)]["iquv"]
                hd = filesDict["{}".format(i)]["hds"]
            else :
                iquv = np.append(iquv,filesDict["{}".format(i)]["iquv"], axis=0)
                hd = np.append(hd, filesDict["{}".format(i)]["hds"])
                del filesDict["{}".format(i-1)]
            print("{} is ok!".format(i))
        del filesDict["{}".format(i)]

        for k in range(1,4):
            iquv[:,k,:,:] /= iquv[:,0,:,:]

        print("divisoin ok!")

            #--- create mask
        mask = self.createMaskByThreshold(threshold,iquv)

        print("mask ok")
        print("iquv : {} GB".format(iquv.nbytes/1024/1024/1024))

        iquv_mean = np.ma.MaskedArray(iquv,mask=mask).mean(axis=(2,3)).data
        # iquv_std  = np.ma.MaskedArray(iquv,mask=mask).std(axis=(2,3)).data

        print("iquv mean ok!")


        #--- check synchronization
        assert ref_index.ndim==1
        order = self.createOrderByComparingIndex(hd["date_end"].astype(np.datetime64), ref_time)

        #--- manipulate bad Data
        bad=np.array([105,104,135,324,343,344,323,322,26,11],dtype=np.int16)
        bad = np.append( bad, np.argwhere( hd["polstate"].astype(np.str) == '45' ).reshape(-1) )

        order_no_bad = np.array([],dtype=np.int16)
        for o in order:
            if o in bad:
                continue
            order_no_bad = np.append(order_no_bad, o)

        print("order_no_bad ok!")

        #--- lmfit
            #--- dtor * deg = rad
        dtor = 0.0174533

        # --- position parameter
        imgrot = hd["imgrot"]/3600.*dtor
        azimuth = hd["az"]/3600.*dtor


        waves = hd["wave"].astype(np.float32) * 0.1
        mm45_flat, pos_wl = parameterMMSP2Mirror(waves[0])
        matImageRotatorSeries = angleSeriesToImageRotatorMuellerMatrixSeries(imgrot, pos_wl)
        print("mm45_flat, pos_wl, matImageRotatorSeries ok!")

        #--- parameter generation

        par=parameterDST(hd[0],th_mmsp2_hsp=(88.2 - 131.2)*dtor)
        print("Parameters generation ok!")
        par["xn"].set(value=-0.0387)
        par["tn"].set(value=-16.8*dtor)
        par["xc"].set(value=-0.0321)
        par["tc"].set(value=12.5*dtor)
        par["sc"].set(value=0.0521)

            #--- mmsp2 part
        # [0:16] : image rotator
        # [16:32] : mirror

        assert (mm45_flat.ndim==1 and len(mm45_flat)==16)
        for i in range(mm45_flat.shape[0]):
            if i==0 :
                par["par_mmsp2_{}".format(i+16)].set(value=mm45_flat[i],vary=False)
            else :
                par["par_mmsp2_{}".format(i+16)].set(value=mm45_flat[i],vary=True)


        # --- angle between the sun and DST
        par.add("th_calunit", value=0., vary=True)

        print("par ok!")
        #--- lmfit fitting
        result = lmfitHSPCalibration(iquv_mean[order_no_bad,1:], hd[order_no_bad], par, imgrot[order_no_bad], azimuth[order_no_bad], matImageRotatorSeries)

        print("No Problem!")

    def readReferenceIndex(self, reffiles, saveFilesHds=None):

        if not saveFilesHds :
            temp = readsav(reffiles[0], verbose=False)["hds"]
            ref_index = temp.copy()
            del temp
        else :
            ref_index = saveFilesHds[0]

        for i, reffile in enumerate( reffiles[1:] ):
            if not saveFilesHds:
                temp = readsav(reffile, verbose=False)["hds"]
                ref_index = np.append(ref_index, temp)
                del temp
            else :
                ref_index = np.append(ref_index, saveFilesHds[i+1])

        temp = ref_index["date_end"].astype(np.datetime64) + np.timedelta64(9,"h")
        ref_time = temp - temp[0].astype("datetime64[D]")
        del temp

        return ref_index, ref_time

    def decideThreshold(self, saveFile):

        self.clickCount = 0

        hmin = 0
        hmax = saveFile["iquv"].max()
        xx = np.linspace(hmin, hmax, 100)

        self.fig, ax = plt.subplots(1,1, figsize=(8,5), dpi=70)
        ax.hist(saveFile["iquv"][:,0,:,:].reshape(-1), bins=xx)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

        while not self._running :
            qApp.processEvents()
            time.sleep(0.5)

        # ax.axvline(x=self.threshold, linestyle='--',color='r')
        # self.fig.canvas.draw()

        return self.threshold

    def on_click(self, event):

        if self.clickCount == 0:

            self.threshold = event.xdata
            print(self.threshold)
            self.clickCount += 1
            plt.close(self.fig)
            self._running = True

        elif self.clickCount == 1:

            plt.close(self.fig)
            self.clickCount += 1

        else:

            pass

    def createMaskByThreshold(self,threshold,iquv):

        mask = np.zeros(iquv.shape, dtype=bool)

        mask_temp = iquv[:,0,:,:] < threshold
        for i in range(1,4):
            mask[:,i,:,:] = mask_temp

        return mask

    def createOrderByComparingIndex(self, date_end, ref_time):

        order = np.zeros(date_end.shape, dtype=np.int8)

        assert order.ndim==1
        date_end_sec = date_end + np.timedelta64(9,"h")
        date_end_sec2 = date_end_sec- date_end_sec[0].astype("datetime64[D]")

        for i in range(order.shape[0]):
            order[i] = np.argmin( abs(date_end_sec2-ref_time[i]) )

        return order

if __name__=="__main__":
    #gc.disable()
    app = QApplication(sys.argv)
    m = MainWindow()
    sys.exit(app.exec_())
