import gwsurrogate as gws
#import gwsurrogate.gwtools as gwtools
import numpy as np
import matplotlib.pyplot as plt
import sys

'''
Paul Lasky
script for generating gravitational wave surrogate models

call from command line: 
python surrogate_waveform.py 

theta = inclination angle
phi = polarisation angle (often called \psi) -- two waveforms are output, one with phi, one with phi = phi + pi/2
z_rot = a proxy for the phase -- therefore leave fixed

'''

q = 1
M = 60.
dist = 410.
theta = 140.  # theta = iota (i.e., inclination angle)
phi = 0.   # polarisation angle
z_rot = 0.  # hardcode the phase, Phi = 0.

path_to_surrogate = '/home/ethan/LIGO/'
surrogate_file = path_to_surrogate + 'SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5'

spec = gws.EvaluateSurrogate(surrogate_file, ell_m = [(2,2)])

##  Calculate the plus and cross components for the phi = 0 case
time = np.linspace(-0.4,0.03,10**5)

_, hp_zero, hc_zero = spec(q=q, M=M, dist=dist, theta=theta, phi=phi,
                               z_rot=z_rot,
                               mode_sum=True, fake_neg_modes=True, samples=time,samples_units='mks')


out_dir = '/home/ethan/LIGO/surrogatemodel/'

out_file_name = 'cbc_q%.2f_M%i_d%i_t%.2f_p%.2f.dat' % (q, round(M), round(dist), round(100 * theta) / 100., round(100 * phi) / 100.)
file_name = out_dir + out_file_name

np.savetxt(file_name, np.c_[time, hp_zero, hc_zero], fmt='%.6e') #Removed hp_90, hc_90 from data as not required as yet

plt.plot(time, hp_zero)
plt.show()
