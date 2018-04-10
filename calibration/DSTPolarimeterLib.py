"""
author: kouui
date: 2018/03/30

DST spectropolarimeter code library
"""

import numpy as np
from lmfit import minimize, Parameters
from scipy.io import readsav
import glob, sys, datetime
# import pdb   ; pdb.set_trace()

#--- global parameters
    #--- dtor * deg = rad
# dtor = 0.0174533
dtor = 0.017453292519943295

pwd = "/nwork/kouui/dstsp/data/calibration/save/"
DebugPrint = False

toleranceDict = {
    0 : 0.0003,
    1 : 0.0003,
    2 : 0.0003,
    3 : 0.050,
    4 : 0.007,
    5 : 0.007,
    6 : 0.050,
    7 : 0.007,
    8 : 0.007,
    9 : 0.007,
    10: 0.050,
    11: 0.007,
    12: 0.007,
    13: 0.050,
    14: 0.007
}

#--- functions

def parameterDST(hd, telpos=None, th_mmsp2_hsp=None):

    if DebugPrint==True:
        print("------------> inside parameterDST")

    #--- dtor * deg = rad
    # dtor = 0.0174533

    wave = float( hd["wave"] )
    if not telpos:
        if hd["zd"] >= 0 :
            telpos="west"
        else :
            telpos="east"
    if not th_mmsp2_hsp:
        th_mmsp2_hsp=(172.-132.2)*dtor


    ref = np.array([
        [0,10830.,-0.0236, -9.547*dtor, 0.0219,-5.561 *dtor,0.0000],
        [0,10050.,-0.0275, -9.713*dtor, 0.0270, 2.268 *dtor,0.0045],
        [0, 9016.,-0.0417, -8.646*dtor, 0.0246,16.335 *dtor,0.0499],
        [0, 8662.,-0.0477, -9.878*dtor, 0.0153,21.319 *dtor,0.0035],
        [0, 8542.,-0.0512,-11.038*dtor, 0.0113,23.180 *dtor,0.0200],
        [0, 8498.,-0.0505,-10.610*dtor, 0.0097,23.454 *dtor,0.0078],
        [0, 8392.,-0.0505,-10.262*dtor, 0.0062,24.779 *dtor,0.0005],
        [0, 6563.,-0.0419,-17.057*dtor,-0.0399,29.294 *dtor,0.0096],
        [1, 6563.,-0.0417,-15.707*dtor,-0.0392,30.443 *dtor,0.0084],
        [0, 6303.,-0.0406,-17.693*dtor,-0.0385,26.178 *dtor,0.0146],
        [1, 6303.,-0.0407,-16.301*dtor,-0.0378,27.453 *dtor,0.0041],
        [0, 5890.,-0.0371,-19.969*dtor,-0.0368,20.876 *dtor,0.0227],
        [1, 5890.,-0.0390,-17.711*dtor,-0.0340,21.373 *dtor,0.0000],
        [0, 5100.,-0.0369,-21.639*dtor,-0.0271, 7.647 *dtor,0.0177],
        [1, 5100.,-0.0374,-20.579*dtor,-0.0276, 9.103 *dtor,0.0000],
        [0, 4861.,-0.0372,-22.886*dtor,-0.0252, 1.075 *dtor,0.0405],
        [1, 4861.,-0.0365,-21.858*dtor,-0.0251, 2.737 *dtor,0.0000],
        [0, 4340.,-0.0394,-24.283*dtor,-0.0212,-9.910 *dtor,0.0548],
        [1, 4340.,-0.0400,-22.434*dtor,-0.0265,-7.655 *dtor,0.0000],
        [0, 4101.,-0.0440,-25.292*dtor,-0.0211,-14.340*dtor,0.1317]
    ],dtype=np.float64)

    if telpos == "west" :
        pos = np.argwhere(ref[:,0] == 0).reshape(-1)
    else :
        pos = np.argwhere(ref[:,0] == 1).reshape(-1)

    params = Parameters()
    # diattenuation of Newton mirror
    params.add( "xn", value=np.interp(wave, ref[pos,1], ref[pos,2]), min=-1., max=1. )
    # retardation of Newton mirror
    params.add( "tn", value=np.interp(wave, ref[pos,1], ref[pos,3]) )
    # diattenuation of Coude mirror
    params.add( "xc", value=np.interp(wave, ref[pos,1], ref[pos,4]), min=-1., max=1. )
    # retardation of Coude mirror
    params.add( "tc", value=np.interp(wave, ref[pos,1], ref[pos,5]) )
    # stray light
    params.add( "sc", value=np.interp(wave, ref[pos,1], ref[pos,6]), min=0., max=1. )
    # retardance of entrance window
    params.add( "t_en", value=0., vary=False )
    # angle of the axis of entrance window
    params.add( "dlen", value=0., vary=False )
    # retardance of exit window
    params.add( "t_ex", value=0., vary=False )
    # angle of the axis of exit window
    params.add( "dlex", value=0., vary=False )
    # angle DST - MMSP2
    params.add( "th_dst_mmsp2", value=0. , vary=True, min=-np.pi, max=+np.pi)
    # angle MMSP2 - Analyzer
    params.add( "th_mmsp2_hsp", value=th_mmsp2_hsp , vary=True,  min=-np.pi, max=+np.pi)

    for i in range(34):
        """
        [0:16] : image rotator
        [16:32] : mirror
        [32:] : th1, th2
        """
        params.add( "par_mmsp2_{}".format(i), value=0., vary=False )
    for i in range(32):
        params["par_mmsp2_{}".format(i)].set(min=-1., max=1.)


    return params

def parameterMMSP2Mirror(wave):
    
    assert wave<1200 

    if DebugPrint==True:
        print("------------> inside parameterMMSP2Mirror")

    mmsp = readsav(pwd+"mm45/m20170103_17511_MM.sav", verbose=False)
    pos = np.argmin( abs(mmsp["wl"]-wave) )
    mm45 = mmsp["mm"][pos,:,:]
    assert mm45.shape==(4,4)
    mm45[3,:] = -mm45[3,:]
    mm45[:,3] = -mm45[:,3]
    mm45_flat = mm45.reshape(-1)

    return mm45_flat, pos #/mm45_flat[0], pos

def lmfitHSPCalibration(quv, hd, par, imgrot, azimuth, matImageRotatorSeries, isSigma):

    if DebugPrint==True:
        print("------------> inside lmfitHSPCalibration")

    nd = hd.shape[0]
    ha,zd,r,p,incli = hd2angle(hd)

    hds = append3times(hd)
    zd  = append3times(zd)
    ha  = append3times(ha)
    incli = append3times(incli)
    ir = append3times(imgrot)
    az = append3times(azimuth)

    key = np.zeros(nd*3)

    for kk in range(5):

        polstate = hd["polstate"].astype(np.str)
        if kk==0:
            pp = np.argwhere( polstate=="" ).reshape(-1)
        elif kk==1:
            pp = np.argwhere( np.logical_or(polstate=="0",polstate=="180") ).reshape(-1)
        elif kk==2:
            pp = np.argwhere( np.logical_or(polstate=="90",polstate=="270") ).reshape(-1)
        elif kk==3:
            pp = np.argwhere( np.logical_or(polstate=="45",polstate=="225") ).reshape(-1)
        elif kk==4:
            pp = np.argwhere( np.logical_or(polstate=="135",polstate=="315") ).reshape(-1)
        else:
            sys.exit("bad kk value!")

        key[pp] = 3*kk+0
        key[pp+nd] = 3*kk+1
        key[pp+2*nd] = 3*kk+2

    assert quv.ndim==2 and quv.shape[1]==3
    yy = np.append(np.append(quv[:,0].reshape(-1),quv[:,1].reshape(-1)),quv[:,2].reshape(-1))

    pos = np.argwhere( np.logical_and(np.logical_and(zd != 0, ha != 0), abs(yy) <= 1) ).reshape(-1)
    if len(pos) >= 1:
        zd_pos = zd[pos]
        ha_pos = ha[pos]
        incli_pos = incli[pos]
        yy_pos = yy[pos]
        key_pos = key[pos]
        ir_pos = ir[pos]
        az_pos = az[pos]

    # sy = 0.001  # this is the allowed fitting error
    xx = np.stack( [ha,zd,ir,az,incli,key], axis=0 )
    if isSigma == True:
        sigma = key2sigma(key)
    else :
        sigma = np.ones(key.shape, dtype=np.int16)

    startTime = datetime.datetime.now()
    result = minimize(residual, par, args=(xx,matImageRotatorSeries, yy, sigma), method='leastsq')#, ftol=1E-7, xtol=1E-7)
    endTime = datetime.datetime.now()
    print("time used in fitting: ",(endTime-startTime).total_seconds())
    
    #--- calculate response matrix
    _, responseMatrix = model(result.params, xx, matImageRotatorSeries)

    return result, sigma, key, responseMatrix

def residual(par, x, matSeries, data, sigma):

    if DebugPrint==True:
        print("------------> inside residual")

    y_model, _ = model(par, x, matSeries)

    return (y_model-data)/sigma

def model(par, x, matSeries):

    if DebugPrint==True:
        print("------------> inside model")

    nd = x.shape[1]
    nd_origin = int(nd/3)

    ha = x[0,:]
    zd = x[1,:]
    ir = x[2,:]
    az = x[3,:]
    incli = x[4,:]
    key = x[5,:]

    responseMatrix = np.zeros((nd_origin,4,4))
    ymod = np.zeros(nd)

    for i, k in enumerate(key):

        if k == 0:
            # incident stokes vector
            stk0 = np.array([1.,0.,0.,0.]).reshape(-1,1)
            # the stokes signal being processed now is (1:Q), (2:U), (3:V)
            istks = 1
        elif k == 1:
            stk0 = np.array([1.,0.,0.,0.]).reshape(-1,1)
            istks = 2
        elif k == 2:
            stk0 = np.array([1.,0.,0.,0.]).reshape(-1,1)
            istks = 3

        elif k == 3:
            stk0 = np.array([1.,1.,0.,0.]).reshape(-1,1)
            istks = 1
        elif k == 4:
            stk0 = np.array([1.,1.,0.,0.]).reshape(-1,1)
            istks = 2
        elif k == 5:
            stk0 = np.array([1.,1.,0.,0.]).reshape(-1,1)
            istks = 3

        elif k == 6:
            stk0 = np.array([1.,-1.,0.,0.]).reshape(-1,1)
            istks = 1
        elif k == 7:
            stk0 = np.array([1.,-1.,0.,0.]).reshape(-1,1)
            istks = 2
        elif k == 8:
            stk0 = np.array([1.,-1.,0.,0.]).reshape(-1,1)
            istks = 3

        elif k == 9:
            stk0 = np.array([1.,0.,1.,0.]).reshape(-1,1)
            istks = 1
        elif k == 10:
            stk0 = np.array([1.,0.,1.,0.]).reshape(-1,1)
            istks = 2
        elif k == 11:
            stk0 = np.array([1.,0.,1.,0.]).reshape(-1,1)
            istks = 3

        elif k == 12:
            stk0 = np.array([1.,0.,-1.,0.]).reshape(-1,1)
            istks = 1
        elif k == 13:
            stk0 = np.array([1.,0.,-1.,0.]).reshape(-1,1)
            istks = 2
        elif k == 14:
            stk0 = np.array([1.,0.,-1.,0.]).reshape(-1,1)
            istks = 3

        # !! select mueller matrix of ratoted image rotator from matSeries here
        # !! and set them into par["par_mmsp2_[0:16]"] as initial value
        # mm_ir = matSeries[i%nd_origin,:,:]
        # mm = MuellerMatrixDST(par,zd[i],ha[i],az[i],mm_ir, incli[i],hsp=True, newton=True)
        # responseMatrix[i,:,:] = mm
        #--- the following modification avoids redundant calculation of mm
        if i < nd_origin:
            mm_ir = matSeries[i,:,:]
            mm = MuellerMatrixDST(par,zd[i],ha[i],az[i],mm_ir, incli[i],hsp=True, newton=True)
            responseMatrix[i,:,:] = mm
        else :
            mm = responseMatrix[i%nd_origin,:,:]

        stk1 = mm @ MuellerMatrixRotation(par["th_calunit"]) @ stk0

        ymod[i] = stk1[istks]/stk1[0]

    return ymod, responseMatrix

def MuellerMatrixDST(par, zd, ha, az, mm_ir, incli, hsp=True, newton=True):
    """
    this function works inside series loop.
    for example if there is 99 data inside time series then loop over 99*3 (3 comes from q,u,v)

    OUTPUT :
        ro_N	ro of Newton mirror
        tau_N	tau of Newton mirror
        ro_C	ro of Coude mirror
        tau_N	tau of Coude mirror
        ha	hour angle
        zd	zenith distance

    INPUT:
        mm_ir   image rotator's mueller matrix (4x4)
        imgrot  angle of image rotator [rad]
        hsp     horizontal spectropolarimeter
        phin    unknown usage, deleted.
    """

    lat = 36.252 * dtor		# Hidaten

    if zd >= 0 :
        telpos = "west"
    else :
        telpos = "east"

    if hsp :
        """
        horizontal mueller matrix
        """
        xn = par["xn"].value
        tn = par["tn"].value
        xc = par["xc"].value
        tc = par["tc"].value
        sc = par["sc"].value
        dlen = par["dlen"].value
        ten = par["t_en"].value
        dlex = par["dlex"].value
        tex = par["t_ex"].value
        th_dst_mmsp2 = par["th_dst_mmsp2"].value
        th_mmsp2_hsp = par["th_mmsp2_hsp"].value

        mm_45 = np.zeros(16)
        for i in range(16):
            mm_45[i] = par["par_mmsp2_{}".format(i+16)].value
        mm_45 = mm_45.reshape(4,4)

        zd = abs(zd)  # ?
        za = np.arcsin(np.cos(lat)*np.sin(ha)/np.sin(zd))
        phi_N = za

        phi_v = (-az+np.pi)
        if telpos == "west" :
            phi_C = -zd
            # phi_v = +zd -za + incli # VS
        else :
            phi_C = +zd
            # phi_v = -zd -za + incli # VS

        M_S = np.array([
            [1.+sc, 0., 0., 0.],
            [0.   , 1., 0., 0.],
            [0.   , 0., 1., 0.],
            [0.   , 0., 0., 1.]
        ])
        M_P = np.array([
            [1., 0., 0. , 0. ],
            [0., 1., 0. , 0. ],
            [0., 0., -1., 0. ],
            [0., 0., 0. , -1.]
        ])
        M_G = M_P
        M_N  = MuellerMatrixMirror(tn, xn, gen=True)
        M_C  = MuellerMatrixMirror(tc, xc, gen=True)
        D_en = MuellerMatrixWaveplate(dlen, ten)
        D_ex = MuellerMatrixWaveplate(dlex, tex)
        R_N  = MuellerMatrixRotation(phi_N)
        R_C  = MuellerMatrixRotation(phi_C)
        R_pl = MuellerMatrixRotation(phi_v)

        if newton==True : # don't need za ?
            mat = R_pl @ D_ex @ M_C @ M_G @ R_C @ M_N @ M_P @ D_en
        elif newton==False :
            mat = R_pl @ D_ex @ M_C @ M_G @ R_C @ M_N @ M_P @ D_en @ R_N
        else :
            sys.exit("keyword newton must be bool value.")

        #--- from now on mmsp2
        R_mmsp2 = MuellerMatrixRotation(th_dst_mmsp2)
        R_rw    = MuellerMatrixRotation(th_mmsp2_hsp)
        mmsp2   = mm_ir @ mm_45

        mat = R_rw @ mmsp2 @ R_mmsp2 @ mat
        mat = M_S @ mat

    else :
        """
        vertical mueller matrix
        """
        sys.exit("Vertical spectropolarimeter not yet!")


    return mat

def MuellerMatrixMirror(tau, ro, gen=True):
    """
    return (normalized?) Mueller matrix for a mirror reflection.
    positive Q-direction is in the plane of incidence

    > Stenflo "Solar Magnetic Field", p320.
    Input:
        tau: ...
        ro : ...
        gen: ...
    """

    tau = float(tau)
    ro  = float(ro)

    if gen == True :
        mat = np.array([
            [1., ro, 0., 0.],
            [ro, 1., 0., 0.],
            [0., 0., -np.sqrt(1.-ro*ro)*np.cos(tau), -np.sqrt(1.-ro*ro)*np.sin(tau)],
            [0., 0., +np.sqrt(1.-ro*ro)*np.sin(tau), -np.sqrt(1.-ro*ro)*np.cos(tau)]
        ]) / (1.+abs(ro))

    elif gen == False :
        mat = np.array([
            [ro*ro+1, ro*ro-1, 0, 0],
            [ro*ro-1, ro*ro+1, 0, 0],
            [0, 0, -2.*ro*np.cos(tau), +2.*ro*np.sin(tau)],
            [0, 0, -2.*ro*np.sin(tau), -2.*ro*np.cos(tau)]
        ]) * 0.5

    else :
        sys.exit("keyword gen must be a bool value.")

    return mat

def MuellerMatrixWaveplate(delta, phai, jones=False, ref=None, thick=None, wv=None):
    """
    return Mueller matrix of linear retarder

    Input:
        delta   : retardance (rad)
        phai  : angle of the axis (rad, couter clockwise)
        jones : whether to return Jones vector
        ref   : reflective index
        thick : thickness of waveplate (mm)
        wv    : wavelength (nm)
    """

    #--- to return Mueller matrix
    if jones == False:

        c2 = np.cos(2.*phai)
        s2 = np.sin(2.*phai)
        cd = np.cos(delta)
        sd = np.sin(delta)

        if ref == None: # without reflection

            mat = np.array([
                [1., 0.            , 0.            , 0.      ],
                [0., c2*c2+s2*s2*cd, s2*c2*(1.-cd) , -s2*sd  ],
                [0., s2*c2*(1.-cd) , s2*s2+c2*c2*cd, c2*sd   ],
                [0., s2*sd         , -c2*sd        , cd	     ]
            ])
        else : # with reflection

            if (thick==None) or (wv==None) :
                sys.exit('you must input two keywords "thick [mm]" and "wv [nm]"')
            else :
                ref = float(ref)
                thick = thick * 1E-3 # [m]
                wv = wv * 1E-9       # [m]
                rr  = 2.*((1.-ref)/(1.+ref))**2
                clm = np.cos(4.*np.pi*thick*ref/wv)
                slm = np.sin(4.*np.pi*thick*ref/wv)
                c2d = np.cos(2.*delta)
                s2d = np.sin(2.*delta)
                f11 = rr*clm*cd+1.
                f12 = -1.*rr*slm*sd
                f33 = rr*clm*c2d+cd
                f43 = -rr*clm*s2d-sd

                mat = np.array([
                    [f11   ,	f12*c2             ,	f12*s2             ,	0.	    ],
                    [f12*c2,	f11*c2*c2+f33*s2*s2,	s2*c2*(f11-f33)    ,	f43*s2  ],
                    [f12*s2,	s2*c2*(f11-f33)    ,	f11*s2*s2+f33*c2*c2,	-f43*c2	],
                    [0.    ,	-f43*s2            ,	f43*c2             ,	f33	    ]
                ]) * (1.-(1.-ref)**2/(1.+ref)**2)**2

    #--- to return Jones Vector
    elif jones == True:

        c1 = np.cos(phai)
        s1 = np.sin(phai)
        i  = 0 + 1j

        if ref==None : # without reflection

            edel = np.exp(-i*delta) # sign ok?
            m11=c1*c1+edel*s1*s1
            m22=s1*s1+edel*c1*c1
            m12=(1.-edel)*c1*s1

            mat = np.array([
                [m11,m12],
                [m12,m22]
            ])

        else : # with reflection

            if (thick==None) or (wv==None) :
                sys.exit('you must input two keywords "thick [mm]" and "wv [nm]"')
            else :
                ref = float(ref)
                thick = thick * 1E-3 # [m]
                wv = wv * 1E-9       # [m]
                lx   = (4.*np.pi*thick*ref/wv+delta)*0.5    # x is fast axis
                ly   = (4.*np.pi*thick*ref/wv-delta)*0.5
                rr   = (1.-ref)**2/(1.+ref)**2
                fx   = rr*np.exp(3.*i*lx)+np.exp(i*lx)
                fy   = rr*np.exp(3.*i*ly)+np.exp(i*ly)
                m11  = fx*c1*c1+fy*s1*s1
                m22  = fx*s1*s1+fy*c1*c1
                m12  = fx*s1*c1-fy*c1*s1

                mat = np.array([
                    [m11,m12],
                    [m12,m22]
                ]) * (1-(1.-ref)**2/(1.+ref)**2)


    else:
        sys.exit("keyword jones must be a bool value.")

    return mat

def MuellerMatrixRotation(phai):
    """
    return Mueller matrix for axis rotation

    Input:
        phai : angle of axis rotation
                (rad., counterclockwise when we view towards the sun)
    """

    c2 = np.cos(2.*phai)
    s2 = np.sin(2.*phai)

    mat = np.array([
        [1., 0. , 0., 0.],
        [0., c2 , s2, 0.],
        [0., -s2, c2, 0.],
        [0., 0. , 0., 1.]
    ])

    return mat

def readImageRotatorTable(path):

    if DebugPrint==True:
        print("------------> inside readImageRotatorTable")

    filenameList = []
    angleList    = []
    with open(path) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i in [52, 67, 68]:
                # skip an extra 0 deg and -30, -31.99 deg
                continue
            temp = line[:-2].split()
            filenameList.append(temp[0][:-5]+"_MM.sav")
            angleList.append( float(temp[1]) )

    return filenameList, np.array(angleList).astype(np.float32)

def angleSeriesToImageRotatorMuellerMatrixSeries(angleSeries_rad, pos_wl):
    """
    image rotator angle series during observation --> Mueller Matrix data cube

    Input :
        angleSeries_rad : image rotator angle series during observation (numpy.ndarray, rad, before tripled)
        pos_wl          : position of wavelength found in `mm45_flat, pos_wl = parameterMMSP2Mirror(waves[0])`
    """
    if DebugPrint==True:
        print("------------> inside angleSeriesToImageRotatorMuellerMatrixSeries")

    filenameList, angleArray = readImageRotatorTable(pwd+"memo.txt")

    angleSeries_deg = angleSeries_rad / dtor
    matSeries = np.zeros((angleSeries_rad.shape[0],4,4), dtype=np.float32)

    for i, angle in enumerate(angleSeries_deg):

        pos_angle = np.argmin( abs(angleArray-angle) )
        dangle_deg = angle - angleArray[pos_angle]
        dangle_rad = dangle_deg * dtor

        saveFile = readsav(pwd+filenameList[pos_angle])
        mm_ir = saveFile["mm"][pos_wl,:,:]
        mm_ir = mm_ir #/ mm_ir[0,0]

        matSeries[i,:,:] = MuellerMatrixRotation(dangle_rad) @ mm_ir @ MuellerMatrixRotation(-dangle_rad) @ MuellerMatrixRotation((88.2 - 358.2)*dtor)

    return matSeries

def hd2angle(hd):

    if DebugPrint==True:
        print("------------> inside hd2angle")

    zd = hd["zd"]/3600.*dtor
    ha = hd["ha"]/3600.*15.*dtor
    pos=np.argwhere(ha >= np.pi/2.).reshape(-1)
    assert pos.ndim==1
    if len(pos) > 0 :
        ha[pos] = ha[pos] - 2.*np.pi

    r = hd["r"]/3600.*dtor
    p = hd["p"]*dtor
    i = hd["i"]*dtor

    return ha, zd, r, p, i

def key2sigma(key):
    
    eps= np.ones(key.shape)
    
    for i,k in enumerate(key):
        eps[i] = toleranceDict[int(k)]
    
    return eps

def append3times(a):

    return np.append( np.append(a,a), a )

def resultToResponseMatrix(result, hd):
    """
    this is a function to calculate DSTHSSP's response matrix from fitted result.
    if result isinstance of list, then this would be a 2dmap result.
    if result is instane of MinimizeResult, then this would be a mean fitting result.
    """
    
    
    
    return None

def paramterToResponceMatrix(param, hd):
    """
    this is a function to calculate DSTHSSP's response matrix from fitted MinimizeParameter.
    """
    
    
    return None
