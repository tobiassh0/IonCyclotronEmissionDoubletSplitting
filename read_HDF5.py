
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
from makeplots import *

if __name__=='__main__':
    loc = '/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/run_2.07_0.11_-0.646_0.01_0.01_15.0_3.5__1.0_4.0_1.7e19_0.00015_1024/'
    # 11% run
    data=read_all_data(loc=loc)
    w0,k0,w,dw,kpara,kperp = data
    # create array of k for given angle

    angles = np.linspace(85*np.pi/180,np.pi/2,11)
    fig, axs = plt.subplots(nrows=6,ncols=2,sharex=True,sharey=True)
    ax=axs.ravel()
    j=0
    for ang in angles:
        try:
            DW = read_pkl(loc+'DW_{}'.format(ang))
            k = read_pkl(loc+'kang_{}'.format(ang))    
            tw = read_pkl(loc+'wang1_{}'.format(ang))
            extents = read_pkl(loc+'extang_{}'.format(ang))
        except:
            k = np.zeros(len(kpara))
            tkperp = np.zeros(len(kperp))
            tkpara = np.zeros(len(kpara))
            tw = np.zeros(len(w))
            tdw = np.zeros(len(dw))
            for i in range(len(kpara)): # provide error bars between as grid is not 100% accurate to angles
                if np.arctan(kperp[i]/kpara[i]) < ang*1.0025 and np.arctan(kperp[i]/kpara[i]) > ang*0.9975:
                    tkpara[i] = kpara[i] ; tkperp[i] = kperp[i]
                    k[i] = np.sqrt(kpara[i]**2 + kperp[i]**2)
                    tw[i] = w[i]
                    tdw[i] = dw[i]
            DW, extents = make2D(tw,k,tdw,limits=True,rowlim=(0,np.max(w)),collim=(0,np.max(k)),bins=(1000,1000))
            k = np.linspace(np.min(k),np.max(k),1000)
            tw = np.linspace(np.min(tw),np.max(tw),1000)
            dumpfiles(DW,loc+'DW_{}'.format(ang))
            dumpfiles(k,loc+'kang_{}'.format(ang))
            dumpfiles(tw,loc+'wang_{}'.format(ang))
            dumpfiles(extents,loc+'extang_{}'.format(ang))

        # plot FAW
        VA = getVa(w0,k0)
        K2 = (np.linspace(0,15*k0))**2
        KPARA = np.sqrt(K2)*np.cos(ang)
        wFAW = np.sqrt(((VA**2)/2)*(K2 + KPARA**2 + (K2*KPARA**2)*((VA**2)/(w0**2)) + \
                ((K2 + KPARA**2 + (K2*KPARA**2)*(VA**2)/(w0**2))**2 - 4*K2*KPARA**2)**0.5))
        ax[j].plot(np.sqrt(K2)/k0,wFAW/w0,color='w',linestyle='--',alpha=0.5)

        # # plot harmonics
        # for i in range(0,16):
        #     plt.axhline(i,color='w',linestyle='--',alpha=0.5)
        #     plt.axvline(i,color='w',linestyle='--',alpha=0.5)

        # plot heatmap
        print(k.shape,tw.shape)
        print(np.max(w)/w0,np.max(k)/k0)
        ax[j].imshow((np.abs(DW)/w0),cmap='jet',aspect='auto',origin='lower',extent=[extents[0]/k0,extents[1]/k0,extents[2]/w0,extents[3]/w0])#[0,np.max(k)/k0,0,np.max(w)/w0])
        ax[j].set_xlim(0,15) # k limit
        ax[j].set_ylim(0,15) # freq limit
        ax[j].annotate(ang*180/np.pi,xy=(0.1,0.75),xycoords='axes fraction',color='w')
        j+=1
    plt.show()

