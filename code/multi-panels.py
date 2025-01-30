
from makeplots import *
import time

# load JET data
def load_JETdata(name='/home/space/phrmsf/Documents/ICE_DS/code/JET26148_ICE_POWER.txt',wnorm=2*3.141*17e6):
    data = np.loadtxt(name,delimiter=',')
    freq_MHz, power_dB = data[:,0], data[:,1] # 2 columns, N rows
    freq_rads = freq_MHz*1e6*(2*np.pi)
    # freq = (freq_MHz*1e6)*(2*const.PI/wnorm) # convert MHz to wcnorm
    return freq_rads, power_dB

# plot kperp vs. kpara with growth rate heatmap for a range (loop) of concentrations
def plot_k2d_growth_multipanel(sollocs=[''],loop=[],cmap='summer',clims=(0,0.15),rowlim=(-4,4),collim=(0,15),_ylim=(-0.5,0.5),_xlim=(13,15)):
    """
        _xlim & _ylim   : (ranges) normalised x and y limits on imshow plots
        clims           : (range) normalised growth rate limits (0,0.15)
    """
    rowlim = np.array(rowlim)
    collim = np.array(collim)

    fig = plt.figure(figsize=(12,5))#,layout='constrained')
    
    # loop (zoomed figs)
    gs2 = GridSpec(8, 10, bottom=0.075, top=0.98, left=0.1, right=0.95, wspace=0.35, hspace=0.35) # nrows, ncols, l, r, wsp, hsp
    ax11 = fig.add_subplot(gs2[:4, :2])               # 1
    ax12 = fig.add_subplot(gs2[:4, 2:4],sharey=ax11)  # 2
    ax13 = fig.add_subplot(gs2[:4, 4:6],sharey=ax11)  # 3
    ax14 = fig.add_subplot(gs2[:4, 6:8],sharey=ax11)  # 4
    ax15 = fig.add_subplot(gs2[:4, 8:10],sharey=ax11) # 5
    ax21 = fig.add_subplot(gs2[4:, :2],sharey=ax11)   # 6
    ax22 = fig.add_subplot(gs2[4:, 2:4],sharey=ax11)  # 7
    ax23 = fig.add_subplot(gs2[4:, 4:6],sharey=ax11)  # 8 
    ax24 = fig.add_subplot(gs2[4:, 6:8],sharey=ax11)  # 9 
    ax25 = fig.add_subplot(gs2[4:, 8:10],sharey=ax11) # 10

    allax = fig.axes
    ignorex(allax)
    ignorey(allax)
    outside_ticks(fig)

    # plotting data
    i=0
    for ax in fig.axes:
        print(loop[i])
        os.chdir(sollocs[i])
        data=read_all_data(loc=sollocs[i])
        w0,k0,w,dw,kpara,kperp = data
        norm = [w0,k0]
        try:
            Z = read_pkl('k2d_growth')
            extents = read_pkl('k2d_growth_ext')
        except:
            Z,extents=make2D(kpara,kperp,dw,rowlim=rowlim*norm[1],collim=collim*norm[1],\
                                bins=(1000,1000),limits=True,dump=True,name='k2d_growth') # y, x, val
        im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=np.array(extents)/norm[1],cmap=cmap,clim=clims)
        fig,ax=plotCycContours(fig,ax,norm=norm,rowlim=rowlim*norm[1],collim=collim*norm[1],levels=np.arange(_xlim[0],_xlim[1],1))
        # ax.plot([0,15],[-1,1])
        if i==0:
            ax.annotate(r"$\xi_T=$"+"{:.0f}%".format(100*loop[i]),xy=(0.0125,0.975),xycoords='axes fraction',**tnrfont,\
                        va='top',ha='left',color='w')
        else:
            ax.annotate("{:.0f}%".format(100*loop[i]),xy=(0.0125,0.975),xycoords='axes fraction',**tnrfont,\
                        va='top',ha='left',color='w')
        # im = ax.imshow(np.zeros((100,100)),aspect='auto',origin='lower')
        # ax.annotate(i,xy=(0.5,0.5),va='center',ha='center')
        i+=1

    # ticks and labels
    xticks = np.arange(_xlim[0],_xlim[1]+1,1)
    yticks = np.arange(_ylim[0],_ylim[1]+0.5,0.5)
    xlabels = [str(i) for i in xticks]
    ylabels = [str(i) for i in yticks]

    # formatting    
    for i in range(len(allax)):
        allax[i].set_xticks(xticks)
        allax[i].set_yticks(yticks)
        if i==0:
            allax[i].tick_params(labelleft=True)
        elif i==(len(loop))//2:
            allax[i].tick_params(labelleft=True,labelbottom=True)
        elif i >= (len(loop))//2:
            allax[i].tick_params(labelbottom=True)
        else:
            None
        allax[i].set_xlim(_xlim)
        allax[i].set_ylim(_ylim)

    # colorbar
    p0 = allax[0].get_position().get_points().flatten()
    p1 = allax[-1].get_position().get_points().flatten()
    # im = ax1.imshow([[0,1],[0,1]],aspect='auto',origin='lower')
    cbar = fig.add_axes([p1[2]+0.02, p1[1], 0.01, p0[3]-p1[1]]) # [left bottom width height]
                      # [0.97,0.05,0.01,0.93]
    plt.colorbar(im, cax=cbar, orientation='vertical')
    cbar.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    fig.supylabel('Parallel Wavenumber '+'  '+r'$[\Omega_i/V_A]$',**tnrfont)
    fig.supxlabel('Perpendicular Wavenumber '+'  '+r'$[\Omega_i/V_A]$',**tnrfont,y=-0.075)
    fig.savefig('../kperp_kpara_growthrates_{}_{}.png'.format(_xlim,_ylim),bbox_inches='tight')
    plt.show()
    return None

# plot k2d again for zero-percent tritium
def k2d_zeropercent(solloc,_xlim=(0,15),_ylim=(-4,4),clims=(0,0.15)):
    os.chdir(solloc)
    # load data and assign
    data=read_all_data(loc=solloc)
    w0,k0,w,dw,kpara,kperp = data
    norm = [w0,k0]
    # assume can read, otherwise make2D
    Z = read_pkl('k2d_growth')
    extents = read_pkl('k2d_growth_ext')
    collim = np.array([extents[0],extents[1]])
    rowlim = np.array([extents[2],extents[3]])
    # setup figure
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    outside_ticks(fig)
    im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=np.array(extents)/norm[1],cmap='summer',clim=clims)
    fig,ax=plotCycContours(fig,ax,norm=norm,rowlim=rowlim*norm[1],collim=collim*norm[1],levels=np.arange(_xlim[0],_xlim[1],1))
    # figure formatting & colorbar
    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)
    ax.set_ylabel('Parallel Wavenumber '+'  '+r'$[\Omega_i/V_A]$',**tnrfont)
    ax.set_xlabel('Perpendicular Wavenumber '+'  '+r'$[\Omega_i/V_A]$',**tnrfont)
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    fig.savefig('../kperp_kpara_growthrates_0%.png',bbox_inches='tight')
    return None

# plot growth rates (and maxima) for multiple concentrations
def growth_combined(sollocs,labels,_xlim=(0,15),_ylim=(0,0.15),clims=(-1.5,1.5)):
    colors = plt.cm.rainbow(np.linspace(0,1,len(sollocs)))
    fig,ax=plt.subplots(figsize=(8,10),sharex=True,sharey=True,layout='constrained',nrows=5,ncols=2)
    ax=ax.ravel()
    c=0
    # loop through each solloc
    for i in range(len(sollocs)):
        os.chdir(sollocs[i])
        data=read_all_data(loc=sollocs[i])
        w0,k0,w,dw,kpara,kperp = data
        norm = [w0,k0]
        
        # make growth rates 1D 
        thresh = (w/norm[0] < _xlim[1]) & (dw/norm[0] > _ylim[0])
        # sc = ax.scatter(w[thresh]/norm[0],dw[thresh]/norm[0],c=kpara[thresh]/norm[1],edgecolor='none',vmin=clims[0],vmax=clims[1])
        # plot black line (made 1D)
        sw, sdw = make1D(w[thresh]/norm[0],dw[thresh]/norm[0],norm=(norm[0],norm[0]),maxnormx=_xlim[1],bins=500) # very small No. bins
        # # colorbar
        # cbar = plt.colorbar(sc)
        # cbar.ax.set_ylabel('Parallel Wavenumber' + r'$[\Omega_i/V_A]$',**tnrfont,rotation=90.,labelpad=20)

        # # single panel
        # ax.plot(sw,sdw,color=colors[i])
        # multi-panel
        ax[c].plot(sw,sdw,color='k')#colors[i])
        ax[c].annotate('{:.0f}%'.format(100*labels[i]),xy=(0.025,0.975),xycoords='axes fraction',ha='left',va='top',**tnrfont)
        c+=1

    # formatting
    ax=np.array(ax)
    ax[0].set_xlim(_xlim)
    ax[0].set_ylim(_ylim)
    # plt.legend(['{:.0f}%'.format(100*i) for i in np.array(labels)],loc='best')
    fig.supxlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
    fig.supylabel('Growth Rate '+r'$[\Omega_i]$',**tnrfont)
    fig.savefig('../freq_growth-multipanel_{}.png'.format(_xlim),bbox_inches='tight')
    # plt.show()
    return None

# plot the growth rates from LMV against the JET [dB] power
def plot_growth_vs_JET_power(sollocs,labels,_xlim=(0,15),_ylim=(0,0.15)):
    # load JET data
    JETfreq,JETpower=load_JETdata()
    fig,ax=plt.subplots(figsize=(12,6),sharex=True,sharey=True,layout='constrained',nrows=2,ncols=3)
    ax=ax.ravel()
    c=0
    # loop through each solution 
    for i in range(len(sollocs)):
        print(labels[i])
        os.chdir(sollocs[i])
        data=read_all_data(loc=sollocs[i])
        w0,k0,w,dw,kpara,kperp = data
        norm = [w0,k0]
        # make growth rates 1D 
        thresh = (w/norm[0] < _xlim[1]) & (dw/norm[0] > _ylim[0])
        # plot black line (made 1D)
        sw, sdw = make1D(w[thresh]/norm[0],dw[thresh]/norm[0],norm=(norm[0],norm[0]),maxnormx=_xlim[1],bins=500) # very small No. bins
        ax[c].plot(sw,sdw,color='b')
        ax[c].annotate('{:.0f}%'.format(100*labels[c]),xy=(0.05,0.95),xycoords='axes fraction',ha='left',va='top',**tnrfont)
        ax[c].plot(JETfreq/norm[0],JETpower*(0.08/np.max(JETpower)),color='k',linestyle=':')
        # axJET = ax[c].twinx()
        # axJET.plot(JETfreq/norm[0],JETpower,color='b')
        # if c in [2,5]:
        #     axJET.set_ylabel('dB',**tnrfont)
        c+=1
    # formatting
    ax=np.array(ax)
    ax[0].set_xlim(_xlim)
    ax[0].set_ylim(_ylim)
    # plt.legend(['{:.0f}%'.format(100*i) for i in np.array(labels)],loc='best')
    fig.supxlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
    fig.supylabel('Growth Rate '+r'$[\Omega_i]$',**tnrfont)
    # plt.show()
    fig.savefig('../freq_growth_JET_{}.png'.format(_xlim),bbox_inches='tight')
    return None

# plot 2d or 3d over XI2 over y (xi2) in x (freq) and z (growth rate) space
def trends_get_peak_frqs(home,sollocs=[''],XI2=[],fbins=800,plateau_size=0.5,Nperw=10,\
                    maxnormf=18,_xlimA=(5,11),_xlimB=(9,11)):
    # 2d colormap
    fig = plt.figure(figsize=(8,2),layout='constrained')    
    # loop (zoomed figs)
    gs = GridSpec(1, 5, bottom=0.075, top=0.98, left=0.1, right=0.95, wspace=0.35, hspace=0.35, figure=fig) # nrows, ncols, l, r, wsp, hsp
    ax1 = fig.add_subplot(gs[:, :3])  # 1
    ax2 = fig.add_subplot(gs[:, 3:])  # 2

    allax = fig.axes
    for ax in allax: 
        ax.set_facecolor('#008066') # first color in summer heatmap
    # ax.set_facecolor('#FFFFFF') # white

    x=[];y=[];z=[]
    try:
        x=read_pkl(home+'freqpeaks_{}_{}'.format(Nperw,fbins))
        y=read_pkl(home+'xi2peaks_{}_{}'.format(Nperw,fbins))
        z=read_pkl(home+'growthpeaks_{}_{}'.format(Nperw,fbins))
    except:
        print('couldnt load files {} {}'.format(Nperw,fbins))
        sys.exit()
        # loop over concentrations XI2
        for i in range(len(XI2)):
            print(XI2[i])
            os.chdir(home+sollocs[i])
            data=read_all_data(loc=home+sollocs[i])
            w0,k0,w,dw,kpara,kperp = data
            xarr, zarr = make1D(w,dw,norm=(w0,w0),maxnormx=maxnormf,bins=fbins)
            # 2D plot
            peaks = extractPeaks(zarr,Nperw=Nperw,plateau_size=plateau_size)
            x.append(xarr[peaks]/w0)
            y.append([XI2[i]]*len(peaks))# xi2 concentration constant
            z.append(zarr[peaks]/w0)
        dumpfiles(x,'freqpeaks_{}_{}'.format(Nperw,fbins))
        dumpfiles(y,'xi2peaks_{}_{}'.format(Nperw,fbins))
        dumpfiles(z,'growthpeaks_{}_{}'.format(Nperw,fbins))
    # plot integer deuteron harmonics
    for ax in allax:
        for i in range(0,maxnormf+1,1):
            ax.axvline(i,color='darkgrey',linestyle='--')
        # plot freqs vs. xi2 and gamma as color
        for i in range(len(z)):
            im = ax.scatter(x[i],y[i],c=z[i],marker='s',s=25,vmin=0,vmax=0.15,cmap='summer',edgecolor='none')

    # plot two trend lines
    # A
    allax[0].plot([4.5,10],[1.05,0],linestyle='--',color='w',alpha=0.75)
    allax[0].annotate('A',xy=(6.5,0.6),xycoords='data',va='top',ha='right',color='w')
    # B
    allax[1].plot([9.5,10],[1,0],linestyle='--',color='w',alpha=0.75)
    allax[1].annotate('B',xy=(9.75,0.8),xycoords='data',ha='left',color='w')
    allax[1].set_yticklabels([])
    # colorbar, labelling and other formatting
    cbar2d = fig.colorbar(im)
    cbar2d.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    fig.supylabel(r'$\xi_T$',**tnrfont)
    fig.supxlabel('Frequency'+' '+r'$[\Omega_i]$',**tnrfont,y=-0.25)
    allax[0].set_ylim(0,np.max(XI2))
    allax[1].set_ylim(0,np.max(XI2))
    allax[0].set_xlim(_xlimA)
    allax[1].set_xlim(_xlimB)
    # plt.show()
    fig.savefig(home+'freq_xiT_growth_peaks_Nperw_{}_AB_highlight.png'.format(Nperw),bbox_inches='tight')
    
    return None

# plot isoangle lines for selected tritium %
def get_growth_isoangle(sollocs=[''],XI2=[],rowlim=(None,None),collim=(None,None),angle=89.0,wmax=15):
    # set perp and para limits
    if not rowlim[0]:
        rowlim = np.array([-4,4])
        collim = np.array([0,25])
    # setup fig
    fig,axes=plt.subplots(figsize=(12,6),sharex=True,sharey=True,layout='constrained',nrows=2,ncols=3)
    ax=axes.ravel()
    # loop through concentrations
    for i in range(len(XI2)):
        print(XI2[i])
        os.chdir(sollocs[i])
        # load data
        data=read_all_data(loc=sollocs[i])
        w0,k0,w,dw,kpara,kperp = data
        # make 2d growth (kperp,kpara)
        Z, extents = make2D(kpara,kperp,dw,rowlim=rowlim,collim=collim,limits=True,dump=False,name='k2d_growth')
        # real units
        collim=np.array([extents[0],extents[1]])
        rowlim=np.array([extents[2],extents[3]])
        # make 2d frequency TODO; make this work in get_frq_growth_angles()
        Zfreq = make2D(kpara,kperp,w,rowlim=rowlim,collim=collim,dump=False,name='k2d_freq')
        # ax[0].imshow(Z/w0,**imkwargs,cmap='summer',vmin=0.0,vmax=0.15,extent=[0,25,-4,4])
        # ax[0].contour(Zfreq/w0,levels=range(0,25),clim=(0,25),colors='k',extent=[0,25,-4,4])
        # get growth rate along angle
        freqs,growthrates,_=get_frq_growth_angles(Z=Z,Zfreq=Zfreq,wmax=wmax,rowlim=rowlim,collim=collim,norm=[w0,k0],angles=[angle])
        # plot
        ax[i].plot(freqs[0]/w0,growthrates[0]/w0,color='k')
        ax[i].annotate('{:.0f}%'.format(100*XI2[i]),xy=(0.05,0.95),xycoords='axes fraction',ha='left',va='top',**tnrfont)        
    ax[0].set_xlim(0,20)
    ax[0].set_ylim(0,0.15)
    fig.supxlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
    fig.supylabel('Growth Rate '+r'$[\Omega_i]$',**tnrfont)
    fig.savefig('../freq_growth_PIC-concentrations_{}.png'.format(XI2),bbox_inches='tight')
    plt.show()
    return None


if __name__=='__main__':
    homes = {
        'lowkperp_T':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Tritons/",
        'highkperp_T':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Tritons_high_kperp/",
        'lowkperp_noT':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons/",
        'highkperp_noT':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons_high_kperp/",        
    }

    # # (1)
    # homeloc=homes.get('highkperp_T')
    # XI2 = np.array([i/100 for i in range(45,95,5)])
    # sollocs = [homeloc+'run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(i) for i in XI2]
    # # # k2d singular (0%)
    # # k2d_zeropercent(sollocs[0],_xlim=(0,25))
    # # k2d multi
    # plot_k2d_growth_multipanel(sollocs=sollocs,loop=XI2,cmap='summer',clims=(0,0.15),collim=(0,25),\
    #                         _ylim=(-0.5,0.5),_xlim=(13,15))

    # # (2) and (4)
    # homeloc=homes.get('highkperp_T')
    # XI2 = [i/100 for i in np.arange(0,100,10)]
    # # XI2.append(0.95)
    # sollocs = [homeloc+'run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(i) for i in XI2]
    # growth_combined(sollocs,labels=XI2,_xlim=(5,12))

    # # (3)
    # homeloc=homes.get('highkperp_T')
    # XI2 = [0.0, 0.01, 0.11, 0.25, 0.38, 0.5]
    # sollocs = [homeloc+'run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(i) for i in XI2]
    # plot_growth_vs_JET_power(sollocs,labels=XI2,_xlim=(0,11),_ylim=(0,0.1))

    # (4)
    homeloc=homes.get('highkperp_T')
    XI2 = [i/200 for i in range(0,200,5)]
    sollocs = ['run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(i) for i in XI2]
    trends_get_peak_frqs(homeloc,sollocs=sollocs,XI2=XI2,maxnormf=18,_xlimA=(5,11),_xlimB=(9,11))
    
    # # (5)
    # homeloc=homes.get('highkperp_T')
    # XI2 = [0.0,0.01,0.11,0.25,0.38,0.5]
    # sollocs = [homeloc+'run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(i) for i in XI2]
    # get_growth_isoangle(sollocs,XI2)
    