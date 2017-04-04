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


energy = '140'
recoFile = '/home/davlars/DECT/K_in_C/K_in_C_' + energy + 'kV_reco_space.p'
geomFile = '/home/davlars/DECT/K_in_C/K_in_C_' + energy + 'kV_geometry.p'
dataFile = '/home/davlars/DECT/K_in_C/K_in_C_' + energy + 'kV_normalized_Fan.npy'


with open(recoFile, 'rb') as f:
    if PY3:
        reco_space = pickle.load(f, encoding='latin1')
    else:
        reco_space = pickle.load(f)

   
'''     
phantomPixelSize = 230./512
volumeSize = np.array([400, 400])*phantomPixelSize
volumeOrigin = -volumeSize/2

# Discretization parameters
nVoxels = np.array([250,250])
nPixels = [250]
pixelSize = np.array([2.4])   
sourceAxisDistance = 542.8
detectorAxisDistance = 542.8
pitch_mm = 0

reco_space = odl.uniform_discr(volumeOrigin,
                               volumeOrigin + volumeSize,
                               nVoxels, dtype='float32')
'''        
        
with open(geomFile, 'rb') as f:
    if PY3:
        geom = pickle.load(f, encoding='latin1')
    else:
        geom = pickle.load(f)

ray_trafo = odl.tomo.RayTransform(reco_space, geom)

data = np.load(dataFile)

rhs = ray_trafo.range.element(data)

fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

fbp_reco = fbp(rhs)
fbp_reco.show()