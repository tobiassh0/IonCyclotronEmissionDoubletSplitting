
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
from makeplots import *


def FT_equiv(data,ax=None,angle=89,err_angle=0.0025,floor=0,plot=True,loc=''):
    angle *= np.pi/180
    w0,k0,w,dw,kpara,kperp = data
    try:
        DW = read_pkl(loc+'DW_{}'.format(angle))
        k = read_pkl(loc+'kang_{}'.format(angle))    
        tw = read_pkl(loc+'wang_{}'.format(angle))
        extents = read_pkl(loc+'extang_{}'.format(angle))
    except:
        k = np.zeros(len(kpara))
        tkperp = np.zeros(len(kperp))
        tkpara = np.zeros(len(kpara))
        tw = np.zeros(len(w))
        tdw = np.zeros(len(dw))
        for i in range(len(kpara)): # provide error bars between as grid is not 100% accurate to angles
            if np.arctan(kperp[i]/kpara[i]) < angle*(1+err_angle) and np.arctan(kperp[i]/kpara[i]) > angle*(1-err_angle): # angle+-(err_angle*np.pi/180)
                tkpara[i] = kpara[i] ; tkperp[i] = kperp[i]
                k[i] = np.sqrt(kpara[i]**2 + kperp[i]**2)
                tw[i] = w[i]
                tdw[i] = dw[i]
        DW, extents = make2D(tw,k,tdw,limits=True,rowlim=(0,np.max(w)),collim=(0,np.max(k)),bins=(1000,1000))
        k = np.linspace(np.min(k),np.max(k),1000)
        tw = np.linspace(np.min(tw),np.max(tw),1000)
        dumpfiles(DW,loc+'DW_{}'.format(angle))
        dumpfiles(k,loc+'kang_{}'.format(angle))
        dumpfiles(tw,loc+'wang_{}'.format(angle))
        dumpfiles(extents,loc+'extang_{}'.format(angle))

    if plot:
        # plot FAW
        VA = getVa(w0,k0)
        K2 = (np.linspace(0,15*k0))**2
        KPARA = np.sqrt(K2)*np.cos(angle)
        wFAW = np.sqrt(((VA**2)/2)*(K2 + KPARA**2 + (K2*KPARA**2)*((VA**2)/(w0**2)) + \
                ((K2 + KPARA**2 + (K2*KPARA**2)*(VA**2)/(w0**2))**2 - 4*K2*KPARA**2)**0.5))
        ax.plot(np.sqrt(K2)/k0,wFAW/w0,color='w',linestyle=':',alpha=0.5)
        # vA line
        ax.plot([0,15],[0,15],color='w',alpha=0.5,linestyle='--')
        # # plot harmonics
        # for i in range(0,16):
        #     plt.axhline(i,color='w',linestyle='--',alpha=0.5)
        #     plt.axvline(i,color='w',linestyle='--',alpha=0.5)

    # plot heatmap
    print(k.shape,tw.shape)
    print(np.max(w)/w0,np.max(k)/k0)
    
    # floor = 1e-8 # non-zero for when taking the log
    DW = DW + floor
    return ax, DW, extents

def loop_over_angles(data,angles,loc=''):
    w0,k0,w,dw,kpara,kperp = data

    # loop over angles
    fig, axs = plt.subplots(figsize=(15,5),nrows=2,ncols=6,sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.1,wspace=0.075)
    ax=axs.ravel()
    j=0
    for angle in angles:
        ax[j], DW, extents = FT_equiv(data,ax[j],angle,err_angle=0.005,loc=loc)
        im = ax[j].imshow(((DW)/w0),cmap='jet',aspect='auto',origin='lower',extent=[extents[0]/k0,extents[1]/k0,extents[2]/w0,extents[3]/w0],clim=(1e-3,1e-1))#[0,np.max(k)/k0,0,np.max(w)/w0])
        ax[j].annotate("{:.1f}".format(angle),xy=(0.1,0.75),xycoords='axes fraction',color='w')
        j+=1

    ax[0].set_xlim(0,15) # k limit
    ax[0].set_ylim(0,15) # freq limit

    # colorbar     
    p0 = ax[0].get_position().get_points().flatten()
    p1 = ax[-1].get_position().get_points().flatten()
    cbar = fig.add_axes([p0[0], p0[3]+0.07, p1[0], 0.02]) # [left bottom width height]
    plt.colorbar(im, cax=cbar, orientation='horizontal',ticks=np.linspace(1e-3,1e-1,5))#,labels=[1e-3,1e-1])

    fig.supylabel('Frequency' + '  ' + r'$[\Omega_i]$',**tnrfont)
    fig.supxlabel('Wavenumber' + '  ' + r'$[\Omega_i/V_A]$',**tnrfont)
    plt.show()
    # fig.savefig(loc+'/FT2d_equivalent_angles.png',bbox_inches='tight')
    return None

def loop_over_energies(locs,angle,labels):
    # # limit locs to just 4
    # if len(locs)>4:
    #     locs = locs[:4]
    # loop over sollocs
    fig, axs = plt.subplots(figsize=(10,8),nrows=5,ncols=len(locs)//5,sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.1,wspace=0.075)
    ax=axs.ravel()
    j=0
    for loc in locs:
        print(loc)
        data=read_all_data(loc=loc)
        w0,k0,w,dw,kpara,kperp = data
        ax[j], DW, extents = FT_equiv(data,ax[j],angle,floor=1e-8,loc=loc)
        im = ax[j].imshow(np.log10((DW)/w0),cmap='jet',aspect='auto',origin='lower',extent=[extents[0]/k0,extents[1]/k0,extents[2]/w0,extents[3]/w0])#,clim=(1e-3,1e-1))#[0,np.max(k)/k0,0,np.max(w)/w0])
        ax[j].annotate("{:.1f}".format(labels[j]),xy=(0.1,0.75),xycoords='axes fraction',color='w')
        j+=1
    # set axes limits
    ax[0].set_xlim(0,15) # k limit
    ax[0].set_ylim(0,15) # freq limit

    # colorbar     
    p0 = ax[0].get_position().get_points().flatten()
    p1 = ax[-1].get_position().get_points().flatten()
    cbar = fig.add_axes([p0[0], p0[3]+0.07, p1[0], 0.02]) # [left bottom width height]
    plt.colorbar(im, cax=cbar, orientation='horizontal',ticks=np.linspace(1e-3,1e-1,5))#,labels=[1e-3,1e-1])

    fig.supylabel('Frequency' + '  ' + r'$[\Omega_i]$',**tnrfont)
    fig.supxlabel('Wavenumber' + '  ' + r'$[\Omega_i/V_A]$',**tnrfont)
    # plt.show()
    fig.savefig(loc+'/FT2d_equivalent_energy.png',bbox_inches='tight')

    return None

def power_spectra_equiv(data,ax,DW=None,extents=None,floor=0):
    w0,k0,w,dw,kpara,kperp=data
    # get 2d FFT data
    if not DW.any():
        _, DW, extents = FT_equiv(data,ax,angle,err_angle=0.005,loc=loc,plot=False,floor=floor)
    # summate logged DW over freq axes
    power = np.zeros(DW.shape[0])+floor
    freq = np.zeros(DW.shape[0])+floor
    for j in range(DW.shape[0]):
        power[j]=np.sum((DW[j,:]/w0)**2)
        freq[j]=j
    # plot 
    plt.plot(freq,power)
    plt.show()
    return None

if __name__=='__main__':
    # # loop one sim over angles
    # # 11% run
    # sollocs = ['/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/run_2.07_0.11_-0.646_0.01_0.01_15.0_3.5__1.0_4.0_1.7e19_0.00015_1024/']
    # angles = np.linspace(85,90,11)
    # for loc in sollocs:
    #     data=read_all_data(loc=loc)
    #     w0,k0,w,dw,kpara,kperp = data
    #     loop_over_angles(data,angles,loc=loc)

    # loop one angle over multiple sims
    angle = 89.0
    home = '/home/space/phrmsf/Documents/ICE_DS/JET26148/D_T_energy_scan/'
    # get sollocs
    sollocs = [i for i in os.listdir(home) if 'run' in i]
    pitches = [] ; Energies = []
    for st in sollocs:
        stsplit = st.split('_')
        pitches.append(stsplit[3]) ; Energies.append(stsplit[7])
    # sort names & nearr (sort by increasing name (xi2))
    arr = np.array([(y,x) for y,x in sorted(zip(pitches,Energies))])
    Energies = np.array(arr[:,1],dtype=float)
    pitches = np.array(arr[:,0],dtype=float)
    sollocs = np.array([home+'run_2.07_0.11_{}_0.01_0.01_15.0_{}__1.0_4.0_1.7e19_0.00015_1024/'.format(pitches[i],Energies[i]) for i in range(len(Energies))])
    loop_over_energies(locs=sollocs,angle=angle,labels=Energies)
    sys.exit()
