
def PlotGrowthSubtraction(XI2=[],home_T='',home_noT='',wnorm=1.,isofreq=False,figname='growth_subtraction',saveloc=None,\
                          rowlim=(None,None),collim=(None,None),growth_diff=True):
    
    # save location of image
    if not saveloc:
        saveloc = os.getcwd()
    # loop through location and find runs
    os.chdir(home_noT)
    runs = np.sort([i for i in os.listdir() if 'run' in i])
    nearr=[]
    names=[]
    # separate to find parameters
    for run in runs:
        params = run.split("_")
        # if name not given, this will return ValueError
        _, B0, xiT, pitch, vthperp, vthpara, kperpmax, EminMeV, name, tempkeV, kparamax, ne, ximin, ngridpoints = params
        nearr.append(ne)
        names.append(name)
    # sort names & nearr (sort by increasing name (xi2))
    nearr = [x for _,x in sorted(zip(names,nearr))]
    names = np.sort(names)
    # list of solution files
    sollocs_noT = [home_noT+'run_2.07_0.0_-0.646_0.01_0.01_25.0_3.5_{}_1.0_4.0_{}_0.00015_2048/'.format(names[i],nearr[i]) for i in range(len(names))]
    sollocs_T = [home_T+'run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(xi2) for xi2 in XI2]
    # fig setup
    fig, axs = plt.subplots(nrows=8,ncols=5,figsize=(8,11),sharex=False,sharey=True)
    axshape = axs.shape
    axs=axs.ravel()
    fig.subplots_adjust(wspace=0.0,hspace=0.0)
    # loop through XI2 concentrations and compare
    for i in range(len(XI2)):
        print(XI2[i])
        sols = [sollocs_T[i],sollocs_noT[i]]
        growths=[]
        for sol in sols:
            os.chdir(sol)
            data=read_all_data(loc=sol)
            w0,k0,w,dw,kpara,kperp = data
            if growth_diff:
                Z,extents=make2D(kpara,kperp,dw,rowlim=np.array(rowlim)*k0,collim=np.array(collim)*k0,\
                                bins=(1000,1000),limits=True,dump=False,name='k2d_growth') # y, x, val
            else:
                Z,extents=make2D(kpara,kperp,w,rowlim=np.array(rowlim)*k0,collim=np.array(collim)*k0,\
                                bins=(1000,1000),limits=True,dump=True,name='k2d_freq') # y, x, val
        #     x,z  = make1D(w,dw,norm=(wnorm,wnorm),maxnormx=15,bins=800)
        #     growths.append(z)
        # for j in range(0,16):
        #     axs[i].axvline(j,color='darkgray',linestyle='--',alpha=0.5)
        # axs[i].plot(x/wnorm,(growths[0]-growths[1])/wnorm,color='k')
        # axs[i].set_xlim(0,15)
        # axs[i].set_ylim(-0.07,0.07)

        # read both growth rate files for T% and no T%
        if growth_diff:
            growth_T=read_pkl(sollocs_T[i]+'k2d_growth')/wnorm
            growth_noT=read_pkl(sollocs_noT[i]+'k2d_growth')/wnorm
        else:
            freq_T=read_pkl(sollocs_T[i]+'k2d_freq')
            freq_noT=read_pkl(sollocs_noT[i]+'k2d_freq')
        if isofreq: # plot cyclotron contours
            fig,axs[i]=plotCycContours(fig,axs[i],norm=[k0,k0],maxnormf=collim[-1],rowlim=np.array(rowlim)*k0,collim=np.array(rowlim)*k0,\
                                        alpha=0.05,color='k')
        # # heatmap of normalised difference of growth rates
        extent = np.append(collim,rowlim)
        if growth_diff:
            im = axs[i].imshow((growth_T-growth_noT),aspect='auto',origin='lower',cmap='bwr',clim=(-0.15,0.15),extent=extent)
        else:
            im = axs[i].imshow(np.log10(freq_T/freq_noT),aspect='auto',origin='lower',cmap='bwr',extent=[0,25,-4,4],clim=(-0.07,0.07))

        axs[i].locator_params(axis='y',nbins=4)
        axs[i].annotate(XI2[i],xy=(0.05,0.85),xycoords='axes fraction')
        # formatting axis
        if i in range((axshape[0]-1)*5,axshape[0]*axshape[1]-1):
            axs[i].set_xticks([0,5,10,15,20])
            axs[i].set_xticklabels(['0','5','10','15','20'])
        elif axs[i] == axs[-1]:
            axs[i].set_xticks([0,5,10,15,20,25])
            axs[i].set_xticklabels(['0','5','10','15','20','25'])
        else:
            axs[i].set_xticks([0,5,10,15,20,25])
            axs[i].set_xticklabels(['','','','','',''])

    fig.supxlabel(r'Perpendicular Wavenumber '+' '+r'$[\Omega_i/V_A]$',**tnrfont)
    fig.supylabel(r'Parallel Wavenumber '+' '+r'$[\Omega_i/V_A]$',**tnrfont)
    # formatting & colorbar
    axs[-1].set_ylim(-1.5,1.5)
    p0 = axs[0].get_position().get_points().flatten()
    plast = axs[-1].get_position().get_points().flatten()
    ax0_cbar = fig.add_axes([0.92, plast[1], 0.01, p0[3]-plast[1]]) # [left bottom width height]
    cbar = plt.colorbar(im, cax=ax0_cbar, orientation='vertical')
    if growth_diff:
        cbar.ax.set_ylabel(r'$(\gamma_{DT}-\gamma_{D})/\Omega_i$',**tnrfont,rotation=90.,labelpad=0)
    else:
        cbar.ax.set_ylabel(r'$\log_{10}(\omega_{DT}/\omega_{D})$',**tnrfont,rotation=90.,labelpad=0)
    # saving fig
    if isofreq:
        figname += '_isofreq'
    fig.savefig(saveloc+figname+'.pdf',bbox_inches='tight')
    # plt.show()
    return None

if __name__=='__main__':
    from makeplots import *
    import os,sys
    import numpy as np
    plt.style.use('classic')
    plt.tight_layout()
    plt.rcParams['axes.formatter.useoffset'] = False
    
    # constants
    qe = 1.6021766E-19 # C
    me = 9.1093829E-31 # kg
    malpha = 7294.3*me
    B0 = 2.07 # T
    # normalisation freq
    walpha = 2*qe*B0/malpha
    # xi_T parameter space
    XI2 = [i/200 for i in range(0,200,5)]
    print(XI2)
    # locations of runs to compare (and loop through)
    home_T = "/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration_high_kperp/"
    home_noT = "/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons_high_kperp/"
    
    PlotGrowthSubtraction(XI2=XI2,home_T=home_T,home_noT=home_noT,wnorm=walpha,isofreq=False,\
                        saveloc=os.getcwd()+'/JET26148/',figname='growth_subtraction_high_kperp',rowlim=(-4,4),collim=(0,25))
