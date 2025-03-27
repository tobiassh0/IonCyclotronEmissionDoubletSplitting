
import os,sys
from makeplots import *
import PRLplots as prl

def frequency_angle_growth(homes,rowlim=(-4,4),collim=(0,25),_xlim=(0,25),_ylim=(0,180)):
    rowlim = np.array(rowlim)
    collim = np.array(collim)

    # load sollocs
    xi2_noT, sollocs_noT, xi2_T, sollocs_T = prl.GETLOCSANDXI2()
    sollocs = [sollocs_T,sollocs_noT]
    xi2 = [xi2_T,xi2_noT]
    XI2 = [i/100 for i in np.arange(0,100,10)]
    # dont have to worry about sorting because in this case they already are in the right order (0.1, 0.2, 0.3 ...)
    
    sollocs_T = [homes.get('highkperp_T')+'/'+str(i) for i in sollocs_T if float(i.split('_')[2]) in XI2]
    sollocs_noT = [homes.get('highkperp_noT')+'/'+str(i) for i in sollocs_noT if float(i.split('_')[8]) in XI2]
    fignames=['withoutT','withT']

    angles = np.linspace(0,180,1000)
    lsize=1000

    i=0
    for sollocs in [sollocs_noT,sollocs_T]:

        fig,axs = plt.subplots(figsize=(8,10),nrows=5,ncols=2,sharey=True,sharex=True) #,layout='constrained')
        # fig.subplots_adjust(wspace=0.05,hspace=0.05)
        ax = axs.ravel()

        j=0
        for solloc in sollocs:
            data = read_all_data(solloc)
            w0,k0,w,dw,kpara,kperp = data

            fig_single, ax_single = plt.subplots(figsize=(8,6))

            Z = make2D(kpara,kperp,dw,rowlim=rowlim*k0,collim=collim*k0,dump=False,name=solloc+'k2d_growth')
            Zfreq = make2D(kpara,kperp,w,rowlim=rowlim*k0,collim=collim*k0,dump=False,name=solloc+'k2d_freq')
            freq, zi, zisum = get_frq_growth_angles(Z,Zfreq,rowlim=rowlim*k0,collim=collim*k0,norm=(w0,k0),angles=angles,lsize=lsize)
            FREQS = np.zeros((len(angles),lsize))
            GROWTHS = np.zeros((len(angles),lsize))
            ANGLES = np.zeros((len(angles),lsize))

            for k in range(len(angles)):
                angle = angles[k]
                FREQS[k,:] = freq[k]
                GROWTHS[k,:] = zi[k]
                ANGLES[k,:] = [angle]*lsize
            #     plt.scatter(freq[k][:]/w0,[angles[k]]*len(freq[k]),c=zi[k][:]/w0,edgecolor='none',cmap='summer')

            # plt.show()
            # sys.exit()
            ANGLES = np.array(ANGLES)
            Z=make2D(ANGLES,FREQS,GROWTHS,rowlim=(0,180),collim=collim*w0,bins=(1000,1000),dump=True,name=solloc+'freq_angle_growth')

            # imshow and 90deg line
            im=ax[j].imshow(Z/w0,**imkwargs,cmap='summer',extent=[collim[0],collim[-1],0,180],clim=(0,0.15))
            ax[j].axhline(90,color='k',linestyle='--')
            # single panel
            ax_single.imshow(Z/w0,**imkwargs,cmap='summer',extent=[collim[0],collim[-1],0,180],clim=(0,0.15))
            ax_single.axhline(90,color='k',linestyle='--')
            ax_single.set_ylabel("Angle"+ "  "+"[deg]",**tnrfont)
            ax_single.set_xlabel("Frequency"+ "  "+r"$[\Omega_i]$",**tnrfont)
            fig_single.savefig(solloc+'frequency_angle_growth.png',bbox_inches='tight')

            # annotate label
            xi2label = '{:.0f}%'.format(100*XI2[j])
            if j == 0:
                xi2label = r'$\xi_T=$'+xi2label
            ax[j].annotate(xi2label,xy=(0.025,0.975),xycoords='axes fraction',ha='left',va='top',**tnrfont)
            
            # limits and xybins
            ax[j].set_ylim(_ylim) ; ax[j].set_xlim(_xlim)
            ax[j].locator_params(nbins=4,axis='both')
        
            j+=1

        # colorbar
        p0 = ax[0].get_position().get_points().flatten() # [left bottom right top]
        p3 = ax[-1].get_position().get_points().flatten()
        cbar = fig.add_axes([p3[2]+0.02, p3[1], 0.01, p0[-1]-p3[1]]) # [left bottom width height]
        plt.colorbar(im, cax=cbar, orientation='vertical')
        cbar.set_ylabel(r'$\gamma/\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
        fig.supylabel("Angle"+ "  "+"[deg]",**tnrfont)
        fig.supxlabel("Frequency"+ "  "+r"$[\Omega_i]$",**tnrfont)
        # savefigs
        print(os.getcwd())
        fig.savefig('frequency_angle_growth_{}.png'.format(fignames[i]),bbox_inches='tight')
        i+=1
    return None

if __name__=='__main__':
    from makeplots import *
    import PRLplots as prl

    homes=getHomes()
    frequency_angle_growth(homes)