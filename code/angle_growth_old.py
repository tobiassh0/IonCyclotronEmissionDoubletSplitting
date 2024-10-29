
from makeplots import *

# plot freq vs. growth for a given (range of) angle(s)
def plot_frq_growth_angles_OLD(loc,kpara,kperp,w,dw,maxnormf=None,norm=[None,None],angles=[88.,88.5,89.,89.5],\
                            labels=['',''],clims=[0,0.5],percentage=0.0025):
    thresh = (w < maxnormf*norm[0]) & (dw > 0) # less than maxnormf & growth rates greater than 0
    kpara = kpara[thresh] ; w = w[thresh] ; dw = dw[thresh]
    for ang in angles: # TODO; dont have to loop over angles, could loop over once and assign growths based on array of angles given
        tkpara = np.zeros(len(kpara))
        # tkperp = np.zeros(len(kperp))
        tw = np.zeros(len(w))
        tdw = np.zeros(len(dw))
        fig,ax=plt.subplots(figsize=(8,6))
        ang *= np.pi/180 # radians
        for k in range(len(kpara)): # provide error bars between as grid is not 100% accurate to angles
            if np.arctan(kperp[k]/kpara[k]) < ang*(1+percentage) and np.arctan(kperp[k]/kpara[k]) > ang*(1-percentage):
                tkpara[k] = kpara[k]
                tw[k] = w[k]
                tdw[k] = dw[k]
        sc = ax.scatter(tw/norm[0],tdw/norm[0],c=tkpara/norm[1],edgecolor='none')#,vmin=clims[0],vmax=clims[1])
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(r'$k_\parallel v_A/\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
        ax.set_xlabel(labels[0],**tnrfont)
        ax.set_ylabel(labels[1],**tnrfont)
        ax.set_xlim(0,maxnormf)
        ax.set_ylim(0,0.15)
        sw, sdw = make1D(tw,tdw,norm=(norm[0],norm[0]),maxnormx=maxnormf,bins=200) # very small No. bins
        ax.plot(sw/norm[0],sdw/norm[0],color='k')
        ax.annotate(r"$\theta=($"+r"${:.2f}\pm{:.2f}$".format(ang*180/np.pi,ang*percentage*180/np.pi)+r"$)^\circ$", xy=(0.125,0.95),\
                    ha='left',va='top',xycoords='axes fraction',**tnrfont)
        fig.savefig(loc+'/freq_growth_{:.1f}.png'.format(ang*180/np.pi),bbox_inches='tight')
        # plt.show()
        plt.clf()
    return None


if __name__=='__main__':

    homeloc = '/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons/'
    XI2 = [0,0.95]
    sollocs = ['run_2.07_0.0_-0.646_0.01_0.01_15.0_3.5_0.0_1.0_4.0_1.7e19_0.00015_1024/','run_2.07_0.0_-0.646_0.01_0.01_15.0_3.5_0.95_1.0_4.0_2.5034019661858546e19_0.00015_1024/']
    print(sollocs)
    l3 = "Frequency"+ "  "+r"$[\Omega_i]$"      # r'$\omega/\Omega_i$'
    l4 = "Growth Rate"+ "  "+r"$[\Omega_i]$"    # r'$\gamma/\Omega_i$'
    for solloc in sollocs:
        loc=homeloc+solloc
        alldata=read_all_data(homeloc+solloc)
        w0,k0,w,dw,kpara,kperp = alldata
        plot_frq_growth_angles_OLD(loc,kpara,kperp,w,dw,maxnormf=np.max(kperp)/k0,norm=[w0,k0],angles=[89.,90.],\
                            labels=[l3,l4],clims=[0,0.5],percentage=0.0025)

