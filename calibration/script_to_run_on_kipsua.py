"""
author: kouui
date: 2018/04/01

Purpose:
    this is a python script to run calibration (multi)processing
    of DST/HSSP on kipsuajupyter
"""

import sys, glob, os, gc, time, datetime
from scipy.io import readsav
import numpy as np

from DSTPolarimeterLib import *
from lmfit import minimize, Parameters
from multiprocessing import Pool

class Main:

    def __init__(self, isDebug=False):

        self._isDebug = isDebug


    def run(self, pathData, pathCamera01, threshold=None, nCPU=1, isMap=False, isSigma=False):

        iquv, hd, imgrot, azimuth, mm45_flat, matImageRotatorSeries = self.prepareFittingProcess(pathData, pathCamera01)

        self.result,self.result_Mean, self.sigma, self.key = self.startFittingProcess(iquv, hd, imgrot, azimuth, mm45_flat, matImageRotatorSeries, threshold=threshold, nCPU=nCPU, isMap=isMap, isSigma=isSigma)

    def prepareFittingProcess(self, pathData, pathCamera01):
        
        print("--- Preparing fitting process ---")

        #--- read filenames
        files = sorted( glob.glob( pathData+"*.sav" ) )
        reffiles = sorted( glob.glob( pathCamera01+"*.sav" ) )

        assert len(files)>0, "calibration files not found."
        assert len(reffiles)>0, "camera01 files not found."

        #--- create reference index and reference time
        ref_index, ref_time = self.readReferenceIndex(reffiles)
        if self._isDebug:
            print("ref_index ref_time ok.")

        #--- read calibration data .sav files to a dictionary
        filesDict = dict()
        for i, file in enumerate(files):
            filesDict["{}".format(i)] = readsav(file, verbose=False)

        if self._isDebug:
            print("calibration data dictionary ok.")

        #--- append all iquv arrays into a single array
        for i in range(len(filesDict)):
            if i == 0:
                iquv = filesDict["{}".format(i)]["iquv"]
                hd = filesDict["{}".format(i)]["hds"]
            else :
                iquv = np.append(iquv,filesDict["{}".format(i)]["iquv"], axis=0)
                hd = np.append(hd, filesDict["{}".format(i)]["hds"])
                del filesDict["{}".format(i-1)]
        del filesDict["{}".format(i)]

        for k in range(1,4):
            iquv[:,k,:,:] /= iquv[:,0,:,:]

        if self._isDebug:
            print("Array iquv and hd ok.")
            print("iquv : {} GB".format(iquv.nbytes/1024/1024/1024))

        #--- check synchronization
        assert ref_index.ndim==1
        order = self.createOrderByComparingIndex(hd["date_end"].astype(np.datetime64), ref_time)
        
        #--- copy polstate
        if pathData != pathCamera01:
            for i, o in enumerate(order):
                hd["polstate"][o] = ref_index["polstate"][i]

        #--- manipulate bad data
        #bad=np.array([105,104,135,324,343,344,323,322,26,11],dtype=np.int16)
        #bad = np.append( bad, np.argwhere( hd["polstate"].astype(np.str) == '45' ).reshape(-1) )
        bad = []

        if len(bad)>0:
            order_no_bad = np.array([],dtype=np.int16)
            for o in order:
                if o in bad:
                    continue
                order_no_bad = np.append(order_no_bad, o)
        else:
            order_no_bad = order

        if self._isDebug:
            print("order_no_bad ok.")

        #--- imgrot and azimuth
        dtor = 0.0174533 # dtor * deg = rad
        imgrot = hd["imgrot"]/3600.*dtor
        azimuth = hd["az"]/3600.*dtor

        #--- read mm45_flat and matImageRotatorSeries
        waves = hd["wave"].astype(np.float32) * 0.1 # unit:[nm]
        mm45_flat, pos_wl = parameterMMSP2Mirror(waves[0])
        matImageRotatorSeries = angleSeriesToImageRotatorMuellerMatrixSeries(imgrot, pos_wl)
        
        if self._isDebug:
            print("mm45_flat, matImageRotatorSeries ok.")

        return iquv[order_no_bad].copy(), hd[order_no_bad].copy(), imgrot[order_no_bad].copy(), azimuth[order_no_bad].copy(), mm45_flat, matImageRotatorSeries



    def startFittingProcess(self, iquv, hd, imgrot, azimuth, mm45_flat, matImageRotatorSeries,threshold=None, nCPU=1, isMap=False, isSigma=False):
        
        assert isinstance(nCPU, int) and nCPU > 0, "nCPU must be a positive integer."

        dtor = 0.0174533 # dtor * deg = rad
        self.iquv = iquv
        self.hd = hd

        print("--- Start fitting process ---")

        #--- mask iquv
        if threshold==None:
            iquv_mean = iquv.mean(axis=(2,3))
        else:
            mask = self.createMaskByThreshold(threshold,iquv)
            iquv_mean = np.ma.MaskedArray(iquv,mask=mask).mean(axis=(2,3)).data
        self.iquv_mean = iquv_mean

        result_mean, weight, key, self.responseMatrixMean = self.fitOneTimeSeries(iquv_mean, hd, imgrot, azimuth, mm45_flat, matImageRotatorSeries, isSigma)
        print("Fitting iquv_mean complete.")
        
        assert isinstance(isMap,bool), "keyword isMap must be a bool variable"
        if isMap==True:
            
            assert nCPU<5, "currently do not use more than 4 processor"

            batchSize = 100
            y_start, x_start = 100,200
            y_end, x_end =  2000, 900
            nx = int( (x_end-x_start)/batchSize  )
            ny = int( (y_end-y_start)/batchSize  )
            
            if nCPU > 1:
                """
                doesn't work
                """

                arguments = []
                for i in range(ny):
                    for j in range(nx):
                        profile = iquv[:,:,(y_start+i*batchSize):(y_start+(i+1)*batchSize),(x_start+j*batchSize):(x_start+(j+1)*batchSize)].mean(axis=(2,3))
                        tempTuple = (profile, hd, imgrot, azimuth, mm45_flat, matImageRotatorSeries)
                        arguments.append( tempTuple )

                print("--- start fitting 2d Map with multi-processing ---")
                pool = Pool(nCPU)
                result = pool.map(self.fitOneTimeSeriesWrapper, arguments)
                pool.close()
                # result,_,_ = self.fitOneTimeSeriesWrapper( arguments[0] )
                print("fitting 2d Map with multi-processing completed.")
                
            else :
                
                result = []
                nTime = iquv.shape[0]
                responseMatrixMap = np.zeros((nTime,4,4,ny,nx))
                
                for i in range(ny):
                    for j in range(nx):
                        profile = iquv[:,:,(y_start+i*batchSize):(y_start+(i+1)*batchSize),(x_start+j*batchSize):(x_start+(j+1)*batchSize)].mean(axis=(2,3))
                        result_temp,wa,ka, responseMatrix = self.fitOneTimeSeries(profile, hd, imgrot, azimuth, mm45_flat, matImageRotatorSeries, isSigma)
                        result.append( result_temp )
                        responseMatrixMap[:,:,:,i,j] = responseMatrix[:,:,:]
                
                self.responseMatrixMap = responseMatrixMap
            
        
        elif isMap==False:
            
            result = []
            
            
        return result, result_mean, weight, key
    
    def fitOneTimeSeriesWrapper(self, args):
        
        return self.fitOneTimeSeries(*args)
    
        
    def fitOneTimeSeries(self, iquv_one, hd, imgrot, azimuth, mm45_flat, matImageRotatorSeries, isSigma):
        
        #--- parameter generation
        par=parameterDST(hd[0],th_mmsp2_hsp=(88.2 - 131.2)*dtor)
        
        par["xn"].set(value=-0.0387)
        par["tn"].set(value=-16.8*dtor)
        par["xc"].set(value=-0.0321)
        par["tc"].set(value=12.5*dtor)
        par["sc"].set(value=0.0521)
        # --- angle between the sun and DST
        par.add("th_calunit", value=0., vary=True)

        #--- mmsp2 part
        # [0:16] : image rotator
        # [16:32] : mirror
        assert (mm45_flat.ndim==1 and len(mm45_flat)==16)
        for i in range(mm45_flat.shape[0]):
            if i==0 :
                par["par_mmsp2_{}".format(i+16)].set(value=mm45_flat[i],vary=False)
            else :
                par["par_mmsp2_{}".format(i+16)].set(value=mm45_flat[i],vary=True)
        
        #--- start fitting
        result, weight, key, responseMatrix = lmfitHSPCalibration(iquv_one[:,1:], hd, par, imgrot, azimuth, matImageRotatorSeries, isSigma)
        
        return result, weight, key, responseMatrix
        

    def saveResult(self):

        pass

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

if __name__ == "__main__":

    # pathData     = "/nwork/kouui/dstsp/data/calibration/20171128/camera01/cal*.sav"
    # pathCamera01 = "/nwork/kouui/dstsp/data/calibration/20171128/camera01/cal*.sav"
    print(sys.argv)
    assert len(sys.argv)==3, "need 2 arguments: pathData, pathCamera01"
    pathData = sys.argv[1]
    pathCamera01 = sys.argv[2]
    
    print("pathData     : ", pathData)
    print("pathCamera01 : ", pathCamera01)
    
    main = Main(isDebug=True)
    # main.run(pathData, pathCamera01, threshold=129626, nCPU=1)
    main.run(pathData, pathCamera01, threshold=129626, nCPU=1, isMap=True, isSigma=False)
