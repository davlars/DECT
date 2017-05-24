# DECT_git

## K_in_C
One folder entails a simple phantom of a K-insert in a C-matrix. 

Run ```./K_in_C/K_in_C_DECT.py``` for DECT material decomposition with Gauss-Newton optimization.

A direct reconstruction of the simualted data can also be performed using ```./K_in_C/K_in_C_reco2D.py```

## WMGM
Another folder entails a simple phantom of a WM/GM-phantom, inside a CSF-matrix surrounded by 1 mm bone. 

Run ```./WMGM/WMGM_DECT_reco.py``` for DECT material decomposition with Gauss-Newton optimization.

A direct reconstruction of the simualted data can also be performed using ```./WMGM/WMGM_cube_phantom_2Dreco.py```. To switch between the different phantoms, simply change ```phantom_1``` to ```phantom_2``` in the file definition.
