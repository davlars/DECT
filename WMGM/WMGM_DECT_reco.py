#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:00:58 2017

@author: davlars
"""

import odl
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from DECT_data import *
from lmfit import minimize, Parameters

PY3 = (sys.version_info > (3, 0))
data_path = '/mnt/datahd/davlars/DECT/WMGM/'
file = 'WMGM_cube_phantom_1_thin_bone'

def load_data(filename):
    dataFile = os.path.join(data_path, filename)
    projections = np.load(dataFile).astype('float32')
    return projections

def get_spectrum(filename, nbins=134):
    data = np.loadtxt(os.path.join(data_path,filename))
    energies = data[:, 0] / 1000.0
    spectrum = data[:, 1]

    indices = np.linspace(0, energies.size - 1, nbins, dtype=int)
    energies = energies[indices]
    spectrum = spectrum[indices]
    spectrum /= spectrum.sum()
    return energies, spectrum

#Discretization
volumeSize = np.array([230.0, 230.0])
volumeOrigin = np.array([-115.0, -115.0])

# Discretization parameters
nVoxels = np.array([512, 512])

# Discrete reconstruction space
reco_space = odl.uniform_discr(volumeOrigin,
                               volumeOrigin + volumeSize,
                               nVoxels, dtype='float32')

print("Loading geometry")

load_pickle = False

if load_pickle is True:
    #Load through pickle
    geomFile = os.path.join(data_path, file + '_140kV_geometry.p')
    
    with open(geomFile, 'rb') as f:
        if PY3:
            geom = pickle.load(f, encoding='latin1')
        else:
            geom = pickle.load(f)
else:
    #Define geom manually
    pixelSize = np.array([2.4, 2.4])
    sourceAxisDistance = 542.8
    detectorAxisDistance = 542.8
    n_turns = 1 
    turn_number = 16
    nPixels = [100, 10]
    nProjection = 400 * n_turns
        
    # Scale factors
    detectorSize = pixelSize * nPixels
    detectorOrigin = -detectorSize/2
        
    angle_partition = odl.uniform_partition(2 * np.pi* turn_number, 
                                                2 * np.pi* (turn_number + n_turns), 
                                                nProjection)
    
    detector_partition = odl.uniform_partition(detectorOrigin, detectorOrigin+detectorSize, nPixels)
    
    geom = odl.tomo.FanFlatGeometry(angle_partition,
                                    odl.uniform_partition(detectorOrigin[0], detectorOrigin[0]+detectorSize[0], nPixels[0]),
                                    src_radius=sourceAxisDistance,
                                    det_radius=detectorAxisDistance)

#
phantomPixelSize = 230./512
volumeSize = np.array([150, 150])*phantomPixelSize
volumeOrigin = -volumeSize/2
nVoxels = [1000,1000]

#Reco space
reco_space = odl.uniform_discr(volumeOrigin,
                               volumeOrigin + volumeSize,
                               nVoxels, dtype='float32')

A = odl.tomo.RayTransform(reco_space, geom, impl='astra_cuda')

print("Loading high dose data")
projections1 = load_data(os.path.join(data_path, file + '_140kV.npy'))
projections1 = projections1[::,...,5]
#projections1 /= 13900

projections2 = load_data(os.path.join(data_path, file + '_80kV.npy'))
projections2 = projections2[::,...,5]
#projections2 /= 6730

#Spectrum high
spectrumHigh = 'spectrumHighVoltage.txt'
energies1, spectrum1 = get_spectrum(spectrumHigh)

#Spectum low
spectrumLow = 'spectrumLowVoltage.txt'
energies2, spectrum2 = get_spectrum(spectrumLow)

'''
plt.plot(energies1, spectrum1)
plt.plot(energies2, spectrum2)
'''

f1 = np.asarray(WM_tot)/10.0
f2 = np.asarray(GM_tot)/10.0

#projections1 = projections1[2000:2002, 250:251]
#projections2 = projections2[2000:2002, 250:251]
proj_shape = projections1.shape

A1 = 0.5*np.ones(proj_shape)
A2 = 0.5*np.ones(proj_shape)
nRuns = 60
photonsRun = 2000
N = photonsRun*nRuns #2000 ph/pix/sim, 100 sim

# Start from zero guess
A1b = A1[..., None]
A2b = A2[..., None]
eps = 1e-3

# Start at previous iteration
'''
B = np.load('/home/davlars/DECT/WMGM_reco/WM_4999iterations.npy')
A1b[..., 0] = B
C = np.load('/home/davlars/DECT/WMGM_reco/GM_4999iterations.npy')
A2b[..., 0] = C
'''
       
counter = 0

for iter in range(100):
    step1 = 1.0*np.ones(proj_shape)
    step2 = 1.0*np.ones(proj_shape)
    counter += 1
    e1 = N*np.sum(spectrum1*energies1*np.exp(-(A1b*f1+A2b*f2)), axis=-1)
    e2 = N*np.sum(spectrum2*energies2*np.exp(-(A1b*f1+A2b*f2)), axis=-1)
    '''
    if np.mod(iter,10) == 0:
        eps /= 2
    '''
    J = np.zeros([proj_shape[0], proj_shape[1], 2, 2])
    J[..., 0, 0] = N*np.sum(f1*spectrum1*energies1*np.exp(-(A1b*f1+A2b*f2))+eps, axis=-1)
    J[..., 0, 1] = N*np.sum(f2*spectrum1*energies1*np.exp(-(A1b*f1+A2b*f2)), axis=-1)
    J[..., 1, 0] = N*np.sum(f1*spectrum2*energies2*np.exp(-(A1b*f1+A2b*f2)), axis=-1)
    J[..., 1, 1] = N*np.sum(f2*spectrum2*energies2*np.exp(-(A1b*f1+A2b*f2))+eps, axis=-1)

    Jinv = np.linalg.inv(J)

    dA1 = (Jinv[..., 0, 0]*(projections1 - e1) +
           Jinv[..., 1, 0]*(projections2 - e2))
    dA2 = (Jinv[..., 0, 1]*(projections1 - e1) +
           Jinv[..., 1, 1]*(projections2 - e2))

    
    while (A1b[...,0] - step1*dA1).min() < 0 or (A2b[...,0] - step2*dA2).min() < 0:
        step1[A1b[...,0] - step1*dA1<0] /= 2
        step2[A2b[...,0] - step2*dA2<0] /= 2
        #step /= 2 
    print("Step1_max: %f, step2_max: %f " % (step1.max(), step2.max()))    
    print("Step1_min: %f, step2_min: %f " % (step1.min(), step2.min()))    
    A1b -= step1[..., None]*dA1[..., None]
    A2b -= step2[..., None]*dA2[..., None]

    norm = np.linalg.norm(projections1 - e1) + np.linalg.norm(projections2 - e2)
    if norm < 1e-2:
        break
    print("Iter: %i, norm: %f" % (iter, norm))
    if counter == 500:
        WM = np.squeeze(A1b[...,0])
        GM = np.squeeze(A2b[...,0])
        WMname = ('/mnt/datahd/davlars/DECT/WMGM/WMGM_reco/WM_phantom1_thin_bone_4000_proj_{}'.format(iter) + 'iterations.npy')
        np.save(WMname, WM)
        GMname = ('/mnt/datahd/davlars/DECT/WMGM/WMGM_reco/GM_phantom1_thin_bone_4000_proj_{}'.format(iter) + 'iterations.npy')
        np.save(GMname, GM)
        counter = 0
        
'''
raise Exception('stop')
B = np.squeeze(A2b[...,0])
rhs = A.range.element(B)

fbp = odl.tomo.fbp_op(A, filter_type='Hann', frequency_scaling=0.8)
fbp_reco = fbp(rhs)

x = A.domain.zero()

# Reconstruct CGLS
title = 'my reco'
lamb = 0.01

callbackShowReco = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow(title))

callbackPrintIter = (odl.solvers.CallbackPrintIteration())


niter = 20
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=niter, callback = callbackShowReco)

'''
