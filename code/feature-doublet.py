
from makeplots import *

def rectangle_integrate(signal,dx):
    totalsum = 0
    for i in range(len(signal)):
        area = dx*np.abs(signal[i])
        totalsum+=area
    return totalsum

if __name__=='__main__':

    # load two sollocs (ref and comparison)
    homeloc = '/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Tritons_high_kperp/'
    solref = homeloc+'run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(0.2)
    solcom = homeloc+'run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(0.6)

    # load growth v freq
    dataref=read_all_data(loc=solref)
    w0,k0,_wref,_dwref,_,_ = dataref
    _wcom = read_pkl(solcom+'frequency')
    _dwcom= read_pkl(solcom+'growthrates')
    wref, dwref = make1D(_wref,_dwref,norm=(w0,w0),maxnormx=12) # default bins
    wcom, dwcom = make1D(_wcom,_dwcom,norm=(w0,w0),maxnormx=12)
    DeltaW = (wref[-1]-wref[0])/len(wref)
    # plt.plot(wref/w0,dwref/w0,color='k')
    # plt.plot(wcom/w0,dwcom/w0,color='r')
    # plt.show()

    # set box limits of repeating feature on ref
    wftr = [9.67*w0, 10.47*w0]
    thresh = (wref > wftr[0]) & (wref < wftr[1])
    boxlen = len(wref[thresh])
    wref_ftr = wref[thresh]; dwref_ftr = dwref[thresh]
    # plt.plot(wref/w0,dwref/w0,color='k')
    # plt.plot(wref_ftr/w0,dwref_ftr/w0,color='r')
    # plt.show()

    # loop through comparison spectra with box size = boxlen, subtracting feature spectra
    woff = np.zeros(len(wcom)-int(boxlen))
    dwintegral = np.zeros(len(woff))
    for i in range(0,len(dwcom)-boxlen):
        woff[i] = wcom[i]
        wbox = wcom[i:i+boxlen]/w0
        # take difference from start of box
        dwbox = dwcom[i:i+boxlen] - dwcom[i]
        # compare to feature (change in) and integrate
        dwintegral[i] = rectangle_integrate((dwbox - (dwref_ftr-dwref_ftr[0])),DeltaW)
        # plt.plot(wbox/w0,(dwref_ftr-dwref_ftr[0])/w0,color='k')
        # plt.plot(wbox/w0,dwbox/w0,color='g')
        # plt.plot(wbox/w0,(dwbox - (dwref_ftr-dwref_ftr[0]))/w0,color='r')
        # plt.show()
    # plt.plot(woff/w0,dwintegral/(DeltaW*w0))
    x = np.argmin(dwintegral)
    plt.plot(wcom/w0,dwcom/w0,color='k')
    plt.plot(wcom[x:x+boxlen]/w0,dwref_ftr/w0,color='r')
    plt.show()
