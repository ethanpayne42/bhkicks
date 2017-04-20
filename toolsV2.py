

'''
Paul Lasky
Handy tools
'''
from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata


def h_tot(hp, Fp, hx, Fx):
    # calculate h_total from plus and cross polarizations and antenna pattern
    return Fp * hp + Fx * hx


def fft(ht, Fs):
    '''
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real
    
    ht = time series
    Fs = sampling frequency

    returns
    hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f = frequencies associated with hf
    '''
    # add one zero padding if time series does not have even number of sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    LL = len(ht)
    # frequency range
    ff = Fs / 2 * np.linspace(0, 1, LL/2+1)

    # calculate FFT
    hf = np.fft.fft(ht)
    # only keep first half
    hf = hf[:LL/2+1]

    # normalise to units of strain / sqrt(Hz)
    hf = hf / Fs
    
    return ff, hf
    
def infft(hf, Fs):
    '''
    inverse FFT for use in conjunction with nfft
    eric.thrane@ligo.org
    input:
    hf = single-side FFT calculated by fft_eht
    Fs = sampling frequency
    output:
    h = time series
    '''
    # use irfft to work with positive frequencies only
    h = np.fft.irfft(hf)
    # undo LAL/Lasky normalisation
    h = h*Fs

    return h


def inner_product(aa, bb, freq, PSD):
    '''
    Calculate the inner product defined in the matched filter statistic
    '''
    # interpolate the PSD to the freq grid
    PSD_interp_func = interp1d(PSD[:, 0], PSD[:, 1]**2, bounds_error=False, fill_value=np.inf) # Notice **2 change (15/12/16)
    PSD_interp = PSD_interp_func(freq)

    # caluclate the inner product
    integrand = np.conj(aa) * bb / PSD_interp

    df = freq[1] - freq[0]
    integral = np.sum(integrand) * df

    product = 4. * np.real(integral) #Added absolute (??) 20/3/17

    return product


def snr_exp(aa, freq, PSD):
    '''
    Calculates the expectation value for the optimal matched filter SNR

    snr_exp(aa, freq, PSD)
    '''
    return np.sqrt(inner_product(aa, aa, freq, PSD))
    
    
def snr_act(aa, bb, freq, PSD):
    '''
    Calculates the matched fliter SNR value for the comparison of waveforms. 
    '''    
    return np.sqrt(inner_product(aa, bb, freq, PSD))
    

def overlap_calc(aa, bb, freq, PSD): 
    ''' 
    Calculates the overlap of any two functions in the freq domain.
    '''
    return inner_product(aa, bb, freq, PSD)/np.sqrt(inner_product(aa, aa,freq, PSD)*inner_product(bb, bb,freq, PSD))
    
    
def fft_plot(f, hf,title = "Waveform's FFT"):
    '''
    Plots the fft of a single waveform for convenience this is located here.
    '''
    
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    plt1.plot(list(f),list(abs(hf)))
    
    axes = plt.gca()
    axes.set_ylim([1e-27,1e-22])
    axes.set_xlim([10,1000])
    axes.set_yscale('log')
    axes.set_xscale('log')
    plt1.set_title(title)
    plt1.set_xlabel('Frequency (Hz)')
    plt1.set_ylabel('Amplitude (units)')
    plt.show()
    
    return None

def wave_plot(time,ht1,time2,ht2,title='Time domain waveform'):
    '''
    Plots a single waveform    
    '''
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    plt1.plot(list(time), list(ht1),'black')
    plt1.plot(list(time2),list(ht2),'g')
    plt1.set_xlabel('time (ms)')
    plt1.set_ylabel('strain')
    axes = plt.gca()
    axes.set_xlim([-0.04,0.02])
    plt.show()

def plot(time,ht1,title='Time domain waveform'):
    '''
    Plots a single waveform    
    '''
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    plt1.plot(list(time), list(ht1))
    plt1.set_title(title)
    axes = plt.gca()
    axes.set_xlim([-0.02,0.02])
    plt.show()
    
def oval_plot(velolist, sigtenlist, sigtwenlist):
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    plt1.plot(velolist, sigtenlist,'r', velolist, sigtwenlist,'b--')
    plt1.set_title('Kick Mismatch (See Figure 2 Gerosa and Moore)\n Still needs maximisation over the overlap')
    axes = plt.gca()
    axes.set_xlim([-0.017,0.017])
    plt.show()
    
def tval_plot(tclist,ovallist, title='Plot of overlap vs time'):
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    plt1.plot(tclist, ovallist)
    plt1.set_title(title)
    axes = plt.gca()
    axes.set_xlim([-0.001,0.0011])
    plt.show()
    
def pval_plot(plist,ovallist, title='Plot of overlap vs phase'):
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    plt1.plot(plist, ovallist)
    plt1.set_title(title)
    axes = plt.gca()
    axes.set_xlim([-3.5,3.5])
    plt.show()
    
def contour_plot(tlist,plist,olist):
    #define grid
    xi = np.linspace(min(tlist),max(tlist),100)
    yi = np.linspace(min(plist),max(plist),100)
    
    levels = np.linspace(0.9, 1, 30)
    
    #grid the data
    zi = griddata(tlist,plist,olist,xi,yi,interp='linear')
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    CS = plt1.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    CS = plt1.contourf(xi, yi, zi, levels=levels)
    fig.colorbar(CS,format="%.2f")  # draw colorbar
    plt1.set_xlabel('time shift (s)')
    plt1.set_ylabel('Phase shift (radians)')
    plt1.set_title('Contour plot')
    
    #Add data point for max overlap
    ind = olist.index(max(olist))
    x = [tlist[ind]]
    y = [plist[ind]]    
    
    plt1.scatter(x, y, marker='o', s=10, zorder=10)
    print(x,y)
    print(olist[ind])
    print((1-olist[ind])*10**5)
    
    plt.show()    
    
