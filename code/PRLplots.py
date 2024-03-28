import os,sys

def xoutside_ticks(lax):
	for ax in lax:
		ax.tick_params(axis='x',direction='out',top=False,right=False,left=False,bottom=True)

def Fig1_freqgrowth_freqkpara(w,dw,kpara,maxnormf=15,norm=[None,None],clims=(-1.5,1.5),cmap='hot',bins=(1000,1000),name=''):
    thresh = (w/norm[0] < maxnormf) & (dw/norm[0] > 0)

    # setup figure & plot
    fig,axs=plt.subplots(figsize=(9,7),nrows=2,ncols=1,sharex=True)
    fig.subplots_adjust(hspace=0.1)
    # plot freq v growth with kpara cmap
    sc = axs[0].scatter(w[thresh]/norm[0],dw[thresh]/norm[0],c=kpara[thresh]/norm[1],edgecolor='none',vmin=clims[0],vmax=clims[1],cmap=cmap)
    cbar = plt.colorbar(sc,ax=axs[0])
    cbar.ax.set_ylabel(r'$k_\parallel V_A/\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
    # plot freq v kpara with growth rate heatmap
    Z,extents=make2D(kpara[thresh],w[thresh],dw[thresh],rowlim=(-2*norm[1],2*norm[1]),\
                    collim=(0,maxnormf*norm[0]),bins=bins,limits=True)
    wmin,wmax,kmin,kmax = np.array(extents)
    im = axs[1].imshow(Z/norm[0],aspect='auto',origin='lower',extent=[wmin/norm[0],wmax/norm[0],kmin/norm[1],kmax/norm[1]],cmap=cmap)
    cbar = plt.colorbar(im,ax=axs[1])
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)

    # harmonics
    for i in range(0,maxnormf+1):
        axs[1].axvline(i,color='w',linestyle='--',alpha=0.5)
    # labelling and formatting
    axs[0].set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont)
    axs[1].set_xlabel('Frequency '+' '+r'$[\Omega_i]$',**tnrfont)
    axs[1].set_ylabel(r'$k_\parallel V_A/\Omega_i$',**tnrfont)
    axs[0].set_xlim(0,maxnormf)
    axs[0].set_ylim(0,0.15)
    xoutside_ticks(axs)
    fig.savefig('PRL_Fig1_freq_kpara_'+name+'.png',bbox_inches='tight')
    return None

def Fig2_peakfreq_freqextraction(sollocs,zoomed_sollocs,xiT,maxnormf=15,name=''):
    """
        In:
            sollocs : list of runs to loop over (should include all sollocs for main plot)
            zoomed_sollocs : a specific set of solution files to make "zoom" in plots of (len 2) 
            xiT : 1d array of xiT parameters, same len as sollocs
    """

    # fig setup
    fig = plt.figure(figsize=(10,5))
    # peaks of freqs vs xiT
    gs1 = GridSpec(4, 10, bottom=0.535, top=0.98, left=0.1, right=0.95, wspace=0.0, hspace=0.05)# bottom spacing creates space for gs2 
    ax1 = fig.add_subplot(gs1[:, :])
    ax1.set_facecolor('#0b0000') # first color in hot cmap (black)
    # specific xiT
    gs2 = GridSpec(8, 10, bottom=0.075, top=0.49, left=0.1, right=0.95, wspace=0.15, hspace=1.5) # nrows, ncols, l, r, wsp, hsp
    ax11 = fig.add_subplot(gs2[:4, :5],sharex=ax1)
    ax12 = fig.add_subplot(gs2[:4, 5:],sharey=ax11,sharex=ax11)
    ax21 = fig.add_subplot(gs2[4:, :5],sharey=ax11,sharex=ax11)
    ax22 = fig.add_subplot(gs2[4:, 5:],sharey=ax11,sharex=ax11)
    ax_zoomed = [ax11,ax12,ax21,ax22]

    x=[];y=[];z=[]
    # loop over concentrations
    c=0
    for i in range(len(sollocs)):
        data=read_all_data(loc=sollocs[i])
        w0,k0,w,dw,kpara,kperp = data

        xarr, zarr = make1D(w,dw,norm=(w0,w0),maxnormx=maxnormf)
        # 2D plot
        peaks = extractPeaks(zarr,Nperw=8,plateau_size=0.5)

        # # testing
        # peaks=[0,1]
        # xarr = w ; zarr = dw

        x.append(xarr[peaks]/w0)
        y.append([xiT[i]]*len(peaks)) # xiT concentration constant
        z.append(zarr[peaks]/w0)
        scpeak = ax1.scatter(xarr[peaks]/w0,[xiT[i]]*len(peaks),c=zarr[peaks]/w0,marker='s',s=25,vmin=0,vmax=0.15,cmap='hot',edgecolor='none')
        if sollocs[i] in zoomed_sollocs:
            # plot line on main plot
            ax1.axhline(xiT[i],color='w',linewidth=10,alpha=0.5)
            ax1.annotate(xiT[i],xy=(0.4,xiT[i]+0.035),xycoords='data',color='w',fontsize=14)
            ax_zoomed[c].annotate(r'$\xi_T=$'+str(xiT[i]),xy=(1,0.1),xycoords='data',color='k',fontsize=14,ha='left',va='bottom')
            # plot in separate axes
            thresh = (w/w0 < maxnormf) & (dw/w0 > 0)
            sc = ax_zoomed[c].scatter(w[thresh]/w0,dw[thresh]/w0,c=kpara[thresh]/k0,edgecolor='none',cmap='hot',clim=(-2*k0,2*k0))
            # # testing
            # sc = ax_zoomed[c].scatter([0,1],[0,1],c=[-2*k0,2*k0],cmap='hot',clim=(-2*k0,2*k0))
            c+=1

    # remove ticks on non-edge axes (ax12)
    ax11.tick_params(labelleft=True,labelbottom=False)
    ax12.tick_params(labelleft=False,labelbottom=False)
    ax22.tick_params(labelleft=False,labelbottom=True)
    # cyc harmonics
    for i in range(0,maxnormf+1):
        ax1.axvline(i,color='w',linestyle='--',alpha=0.5)
    # formatting and labels
    ax11.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',y=0,**tnrfont)
    fig.supxlabel('Frequency '+' '+r'$[\Omega_i]$',y=-0.05,**tnrfont)
    ax1.set_ylabel(r'$\xi_T$',**tnrfont)
    ax1.set_xlim(0,maxnormf)
    ax1.set_ylim(0,1) # xiT
    ax11.set_ylim(0,0.15)
    # ticklabels
    ax11.locator_params(axis='y',nbins=4)
    ax21.locator_params(axis='y',nbins=4)
    # colorbars
        # peaks
    p1 = ax1.get_position().get_points().flatten()
    cbar = fig.add_axes([0.97, p1[1], 0.01, p1[-1]-p1[1]]) # [left bottom width height]
    plt.colorbar(sc, cax=cbar, orientation='vertical')
    cbar.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
        # growth rates
    p11 = ax_zoomed[0].get_position().get_points().flatten()
    p22 = ax_zoomed[-1].get_position().get_points().flatten()
    cbar = fig.add_axes([0.97, p22[1], 0.01, p11[-1]-p22[1]]) # [left bottom width height]
    plt.colorbar(sc, cax=cbar, orientation='vertical')
    cbar.set_ylabel(r'$k_\parallel V_A\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
    # savefig
    fig.savefig('PRL_Fig2_freq_peaks_'+name+'.pdf',bbox_inches='tight')
    fig.savefig('PRL_Fig2_freq_peaks_'+name+'.png',bbox_inches='tight')
    return None

def Fig3_diffgrowthrates_D_DT_contour(sollocs_T,sollocs_noT,xi2_T=[],xi2_noT=[],smooth_cont=False,name=''):

    # setup plot
    fig,axs = plt.subplots(figsize=(8,4),nrows=2,ncols=2,sharey=True,sharex=True)
    fig.subplots_adjust(hspace=0,wspace=0)
    axs = axs.ravel()

    # loop through XI2 concentrations and compare
    home = os.getcwd()
    for i in range(len(sollocs_T)):
        try:
            growth_T=read_pkl(sollocs_T[i]+'k2d_growth')
            growth_noT=read_pkl(sollocs_noT[i]+'k2d_growth')
            freq_T=read_pkl(sollocs_T[i]+'k2d_freq')
            freq_noT=read_pkl(sollocs_noT[i]+'k2d_freq')
            data_T=read_all_data(loc=sollocs_T[i])
            w0,k0,w,dw,kpara,kperp = data_T
        except:
            # load with T
            os.chdir(sollocs_T[i])
            data_T=read_all_data(loc=sollocs_T[i])
            w0,k0,w,dw,kpara,kperp = data_T
            growth_T,extents=make2D(kpara,kperp,dw,rowlim=[-4*k0,4*k0],collim=[0,15*k0],\
                            bins=(1000,1000),limits=True,dump=False,name='k2d_growth') # y, x, val
            freq_T,extents=make2D(kpara,kperp,w,rowlim=[-k0,k0],collim=[0,15*k0],\
                            bins=(1000,1000),limits=True,dump=True,name='k2d_freq') # y, x, val
            # load with no T
            os.chdir(sollocs_noT[i])
            data_noT=read_all_data(loc=sollocs_noT[i])
            w0,k0,w,dw,kpara,kperp = data_noT
            growth_noT,extents=make2D(kpara,kperp,dw,rowlim=[-4*k0,4*k0],collim=[0,15*k0],\
                            bins=(1000,1000),limits=True,dump=False,name='k2d_growth') # y, x, val
            freq_noT,extents=make2D(kpara,kperp,w,rowlim=[-k0,k0],collim=[0,15*k0],\
                            bins=(1000,1000),limits=True,dump=True,name='k2d_freq') # y, x, val
            os.chdir(home)
            # imshow and contour diff
        extents = [0,15,-4,4]
        im = axs[i].imshow((growth_T-growth_noT)/w0,aspect='auto',origin='lower',cmap='bwr',clim=(-0.15,0.15),extent=extents)
        if smooth_cont:
            # smooth data for contour
            freqdiff = scipy.ndimage.filters.gaussian_filter((freq_T-freq_noT)/w0,5)
        else:
            freqdiff = (freq_T-freq_noT)/w0
        cont = axs[i].contour(freqdiff,levels=0,origin='lower',colors='k',alpha=0.5,extent=extents)
        axs[i].annotate(r'$\xi_T=$'+str(xi2_T[i]),xy=(0.1,0.8),xycoords='axes fraction',ha='left',va='bottom')
        axs[i].set_ylim(-1,1) ; axs[i].set_xlim(0,15)

    # colorbar
    p0 = axs[0].get_position().get_points().flatten() # [left bottom right top]
    p3 = axs[-1].get_position().get_points().flatten()
    cbar = fig.add_axes([p3[2]+0.02, p3[1], 0.01, p0[-1]-p3[1]]) # [left bottom width height]
    plt.colorbar(im, cax=cbar, orientation='vertical')
    cbar.set_ylabel(r'$(\gamma_{DT}-\gamma_D)/\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
    # labels
    fig.supylabel(r'$k_\parallel V_A/\Omega_i$',**tnrfont)
    fig.supxlabel(r'$k_\perp V_A/\Omega_i$',**tnrfont,y=-0.045)
    # savefigs
    fig.savefig('PRL_Fig3_growthratediff_'+name+'.png',bbox_inches='tight')
    fig.savefig('PRL_Fig3_growthratediff_'+name+'.pdf',bbox_inches='tight')

    plt.show()
    return None

def GETLOCSANDXI2():
    homes = {
        'lowkperp_T':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/",
        'highkperp_T':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration_high_kperp/",
        'lowkperp_noT':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons/",
        'highkperp_noT':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons_high_kperp/",        
    }

    home_noT = homes.get('highkperp_noT')
    home_T = homes.get('highkperp_T')
    sollocs_noT = getsollocs(home_noT)
    sollocs_T = getsollocs(home_T)
    # sort based on xi2 so know order of sollocs
    # extract unsorted xi2 and corresponding electron number density
    xi2_noT = np.array([sol.split('_')[8] for sol in np.sort(os.listdir(home_noT)) if 'run' in sol],dtype=float)
    ne_noT = np.array([sol.split('_')[11] for sol in np.sort(os.listdir(home_noT)) if 'run' in sol],dtype=str)
    # sort xi2 and ne with respect to increasing xi2
    arr = np.array([(y,x) for y,x in sorted(zip(xi2_noT,ne_noT))])
    xi2_noT = arr[:,0] ; ne_noT = arr[:,1]
    
    # sort sollocs based on increasing xi2
    tsollocs_noT = []
    locs = [loc for loc in os.listdir(home_noT) if 'run' in loc]
    for i in range(len(sollocs_noT)):
        lst = locs[i].split('_')
        # replace list items with sorted xi2 and electron number density
        lst[8] = str(xi2_noT[i])
        lst[11] = str(ne_noT[i])
        tsollocs_noT.append(home_noT+'_'.join(lst)+'/') # append '/' on end of dir
    sollocs_noT = np.array(tsollocs_noT)
    del tsollocs_noT

    # get xi2 parameter from solutions
    xi2_T = np.array([sol.split('_')[2] for sol in np.sort(os.listdir(home_T)) if 'run' in sol],dtype=float)
    # sort xi2 and sollocs with respect to increasing xi2
    arr = np.array([(y,x) for y,x in sorted(zip(xi2_T,sollocs_T))])
    # separate sorted array
    xi2_T = np.array(arr[:,0],dtype=float) ; sollocs_T = arr[:,1]

    return np.array(xi2_noT,dtype=float), sollocs_noT, xi2_T, sollocs_T

if __name__=='__main__':
    from makeplots import *
    import matplotlib as mpl
    import scipy.ndimage

    xi2_noT, sollocs_noT, xi2_T, sollocs_T = GETLOCSANDXI2()
    sollocs = [sollocs_T,sollocs_noT]
    xi2 = [xi2_T,xi2_noT]
    zoomed_sollocs_T = [sollocs_T[4],sollocs_T[12],sollocs_T[20],sollocs_T[28]]# 10, 30, 50, 70
    zoomed_sollocs_noT = [sollocs_noT[4],sollocs_noT[12],sollocs_noT[20],sollocs_noT[28]]# 10, 30, 50, 70
    zoomed_sollocs = [zoomed_sollocs_T,zoomed_sollocs_noT]

    # # Fig1
    # index = 0 # index of sollocs (and hence xi2) to choose
    # data=read_all_data(loc=sollocs_T[index])
    # w0,k0,w,dw,kpara,kperp = data
    # Fig1_freqgrowth_freqkpara(w,dw,kpara,norm=[w0,k0],name='xiT_{}'.format(xi2_T[index]))

    # # Fig2
    # name = ['withT','noT']
    # for i in range(0,2):
    #     Fig2_peakfreq_freqextraction(sollocs[i],zoomed_sollocs[i],xi2[i],name=name[i])

    # Fig3
    Fig3_diffgrowthrates_D_DT_contour(zoomed_sollocs[0],zoomed_sollocs[1],xi2_T=[0.1,0.3,0.5,0.7],\
                                        smooth_cont=True,name='smooth')