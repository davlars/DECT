#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:47:32 2017

@author: davlars
"""

import odl
import numpy as np
import sys
import pickle

PY3 = (sys.version_info > (3, 0))

#recoFile = '/home/davlars/DECT/WMGM_cube_reco_space.p' #AD_GPUMCI/WMGM_cube_phantom/WMGM_cube_reco_space.p'
file = 'WMGM_cube_phantom_1_thin_bone_140kV'
geomFile = file + '_geometry.p' #'/home/davlars/DECT/WMGM_cube_phantom_2_TEST_geometry.p' #WMGM_cube_geometry.p' #AD_GPUMCI/WMGM_cube_phantom/WMGM_cube_geometry.p'
dataFile = file + '.npy' #'/home/davlars/DECT/WMGM_cube_phantom_2_TEST.npy' #WMGM_cube.npy'

'''
with open(recoFile, 'rb') as f:
    if PY3:
        reco_space = pickle.load(f, encoding='latin1')
    else:
        reco_space = pickle.load(f)
'''     


phantomPixelSize = 230./512
volumeSize = np.array([150, 150])*phantomPixelSize
volumeOrigin = -volumeSize/2
nVoxels = [1000,1000]

#Reco space
reco_space = odl.uniform_discr(volumeOrigin,
                               volumeOrigin + volumeSize,
                               nVoxels, dtype='float32')

with open(geomFile, 'rb') as f:
    if PY3:
        geom = pickle.load(f, encoding='latin1')
    else:
        geom = pickle.load(f)

ray_trafo = odl.tomo.RayTransform(reco_space, geom)

data = np.load(dataFile)
data = data[...,5]
data /= np.max(data)
data = -np.log(data)

rhs = ray_trafo.range.element(data)

if False:
    fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', padding='True', frequency_scaling=0.8)
    
    fbp_reco = fbp(rhs)
    fbp_reco.show() #clim=(0.019,0.022)
else:
    
    callbackShowReco = (odl.solvers.CallbackPrintIteration() &
                        odl.solvers.CallbackShow()) #clim=[0.018, 0.022]))
    x = ray_trafo.domain.zero()
    niter = 50
    odl.solvers.conjugate_gradient_normal(ray_trafo, x, rhs, niter=niter, callback=callbackShowReco)
