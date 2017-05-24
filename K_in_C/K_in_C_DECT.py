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
import matplotlib.cm as cm
from DECT_data import *


PY3 = (sys.version_info > (3, 0))
data_path = '.'

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

energy = '140'
recoFile = 'K_in_C_' + energy + 'kV_reco_space.p'
geomFile = 'K_in_C_' + energy + 'kV_geometry.p'
  
#Discretization
with open(recoFile, 'rb') as f:
    if PY3:
        reco_space = pickle.load(f, encoding='latin1')
    else:
        reco_space = pickle.load(f)
        
print("Loading geometry")
     
with open(geomFile, 'rb') as f:
    if PY3:
        geom = pickle.load(f, encoding='latin1')
    else:
        geom = pickle.load(f)

A = odl.tomo.RayTransform(reco_space, geom, impl='astra_cuda')

print("Loading data")
file1 = '/home/davlars/DECT/K_in_C/K_in_C_140kV.npy'
projections1 = load_data(file1)
projections1 = projections1[:,:,5]
#projections1 /= np.max(projections1)

file2 = '/home/davlars/DECT/K_in_C/K_in_C_80kV.npy'
projections2 = load_data(file2)
projections2 = projections2[:,:,5]
#projections2 /= np.max(projections2)

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

f1 = np.asarray(C_tot)/10.0
f2 = np.asarray(K_tot)/10.0


#projections1 = projections1[2000:2002, 250:251]
#projections2 = projections2[2000:2002, 250:251]
proj_shape = projections1.shape

A1 = 1*np.ones(proj_shape)
A2 = 0*np.ones(proj_shape)
N = 2000
step = 1.0

A1b = A1[..., None]
A2b = A2[..., None]
eps = 1e-5

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

    #Without adaptive step
    A1b -= step*dA1[..., None]
    A2b -= step*dA2[..., None]
    
    '''
    #With adaptive step
    while (A1b[...,0] - step1*dA1).min() < 0 or (A2b[...,0] - step2*dA2).min() < 0:
        step1[A1b[...,0] - step1*dA1<0] /= 2
        step2[A2b[...,0] - step2*dA2<0] /= 2
        #step /= 2 
    print("Step1_max: %f, step2_max: %f " % (step1.max(), step2.max()))    
    print("Step1_min: %f, step2_min: %f " % (step1.min(), step2.min()))    
    A1b -= step1[..., None]*dA1[..., None]
    A2b -= step2[..., None]*dA2[..., None]
    '''
    
    norm = np.linalg.norm(projections1 - e1) + np.linalg.norm(projections2 - e2)
    if norm < 1e-2:
        break
    print("Iter: %i, norm: %f" % (iter, norm))

#raise Exception('stop')
B = np.squeeze(A2b[...,0])
rhs = A.range.element(B)

x = A.domain.zero()

#Reconstruct FBP
fbp = odl.tomo.fbp_op(A, filter_type='Hann', frequency_scaling=0.8)
fbp_reco = fbp(rhs)
fbp_reco.show()

raise Exception('stop')
# Reconstruct CGLS
title = 'my reco'
lamb = 0.01

callbackShowReco = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow(title))

callbackPrintIter = (odl.solvers.CallbackPrintIteration())


niter = 20
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=niter, callback = callbackShowReco)


'''
for theta in range(np.shape(e1)[0]):
    print('Calculcating theta number: %i' % theta)
    for t in range(np.shape(e1)[1]):
        J = np.matrix([[J11[theta,t], J12[theta,t]],[J21[theta,t], J22[theta,t]]])
        Jinv = np.linalg.inv(J)
        A1[theta,t] += Jinv[0,0]*(projections1[theta,t] - e1[theta,t]) + Jinv[1,0]*(projections2[theta,t] - e2[theta,t])
        A2[theta,t] += Jinv[0,1]*(projections1[theta,t] - e1[theta,t]) + Jinv[1,1]*(projections2[theta,t] - e2[theta,t])
'''












'''
plt.loglog()
plt.plot(energiesBasis, C_tot,label='C')
plt.plot(energiesBasis, K_tot,label='K')
plt.legend()
'''













'''
rhs = A.range.element(logdata)

x = A.domain.zero()

# Reconstruct
title = 'my reco'
lamb = 0.01

callbackShowReco = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow(title))

callbackPrintIter = (odl.solvers.CallbackPrintIteration())


niter = 50
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=niter, callback = callbackShowReco)
'''












