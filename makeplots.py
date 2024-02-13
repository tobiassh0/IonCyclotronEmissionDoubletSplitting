"""
    Written by Tobias Slade-Harajda for the purpose of analysing LMV 
    (https://github.com/jwscook/IonCyclotronEmissionDoubletSplitting)
    solutions2D.jld files. 
    Functions dumpfiles() and read_pkl() are taken from my Thesis 
    code to analyse EPOCH sims.
"""

tnrfont = {'fontsize':20,'fontname':'Times New Roman'}

## Dump pkl files
def dumpfiles(array, quant):
    print('Pickling '+quant+'...')
    with open(quant+'.pkl', 'wb') as f:
        pickle.dump(array,f)
    return None

## Read pkl files
def read_pkl(quant):
    with open(quant+'.pkl', 'rb') as f:
        print('Loading '+quant+'...')
        array = pickle.load(f)
    print('Done.')
    # automatically closed when loaded due to "with" statement
    return array

# extract peaks in a dataset (used for power spectra comparisons)	
def extractPeaks(data,Nperw=1,prominence=0.3,plateau_size=None):
	# tune till Nperw encapsulates all peaks (visually)
	return signal.find_peaks(data,distance=Nperw,prominence=prominence,plateau_size=plateau_size)[0]

def paraload(i,f,sols):
    print(i)
    wi,dwi=f[sols[i]][()][0]
    kparai,kperpi=f[sols[i]][()][1]
    return [wi,dwi,kparai,kperpi]

def paraload_int(I):
    loc,i=I
    f = h5py.File(loc+'solutions2D_.jld',"r")
    sols = f["plasmasols"][()]
    wi,dwi=f[sols[i]][()][0]
    kparai,kperpi=f[sols[i]][()][1]
    return [wi,dwi,kparai,kperpi]

def jld_to_pkl(loc='',frac=1,parallel=False):
    f = h5py.File(loc+'solutions2D_.jld',"r")
    keys=f.keys()
    # for k in keys: print(k)
    w0 = f["w0"][()]
    k0 = f["k0"][()]
    sols = f["plasmasols"][()]
    solshape = sols.shape[0] ; print(solshape)
    w = np.zeros(int(solshape/frac+1)) ; dw = np.zeros(int(solshape/frac+1))
    kpara = np.zeros(int(solshape/frac+1)) ; kperp = np.zeros(int(solshape/frac+1))
    # parallel loop
    if parallel:
        arr = [[loc, i] for i in range(len(sols))]
        pool = mp.Pool(mp.cpu_count())
        res = np.array(pool.map_async(paraload_int,arr).get(99999))
        pool.close()
        w,dw,kpara,kperp=np.split(res,4,axis=1)
        print(len(w),len(kpara))
    # linear loop
    if not parallel:
        for i in range(len(sols[::frac])):
            item = sols[i]
            w[i],dw[i] = f[item][()][0]
            kpara[i],kperp[i] = f[item][()][1]
    # dumpfiles
    dumpfiles(w,'frequency')
    dumpfiles(dw,'growthrates')
    dumpfiles(kpara,'parallelwavenumber')
    dumpfiles(kperp,'perpendicularwavenumber')
    dumpfiles([w0,k0],'w0k0')    
    return w0,k0,w,dw,kpara,kperp

def read_all_data(loc=''):
    try: # load all relevant data
        w0,k0=read_pkl('w0k0')
        w=read_pkl('frequency')
        dw=read_pkl('growthrates')
        kpara=read_pkl('parallelwavenumber')
        kperp=read_pkl('perpendicularwavenumber')
    except: # calculate and dump data
        w0,k0,w,dw,kpara,kperp = jld_to_pkl()
    data = w0,k0,w,dw,kpara,kperp
    return data

def make2D(rowval,colval,val,rowlim=(None,None),collim=(None,None),bins=(None,None),limits=False):
    if rowval.shape != colval.shape and rowval.shape != val.shape: # make sure same shape
        raise SystemError
    if rowlim[0] != None or collim[0] != None: # check if limits applied
        # thresh to limit size
        thresh = (rowlim[0]<rowval) & (rowval<rowlim[1]) & (collim[0]<colval) & (colval<collim[1]) 
        rowval = rowval[thresh] ; colval = colval[thresh] ; val = val[thresh]
    if bins[0] == None: # no bins
        # unique values
        urow,urowind = np.unique(rowval,return_index=True)
        ucol,ucolind = np.unique(colval,return_index=True)
        nx,ny=[len(urow),len(ucol)]
        # min, max values from data
        rowmin, rowmax = [np.min(rowval),np.max(rowval)]
        colmin, colmax = [np.min(colval),np.max(colval)]
    else: # set bin size
        nx,ny=bins
        # min max values from user input
        rowmin, rowmax = rowlim
        colmin, colmax = collim
    # arrays between max and min values
    rowarr = np.linspace(rowmin,rowmax,nx)
    colarr = np.linspace(colmin,colmax,ny)
    Z = np.zeros((len(rowarr),len(colarr)))
    for k in range(len(rowval)):
        i = np.where(rowval[k] >= rowarr)[0][-1] # last index corresponding to row
        j = np.where(colval[k] >= colarr)[0][-1] # last index corresponding to column
        if Z[i,j] < val[k]:
            Z[i,j]=val[k] # assign highest growth rate
    if limits:
        return Z, [colmin,colmax,rowmin,rowmax]
    else:
        return Z

def make1D(xdata,ydata,maxnormx=None,norm=(1,1),bins=1000):
    normx, normy = norm
    if not maxnormx:
        maxnormx = np.max(xdata)
    thresh = xdata < maxnormx*normx
    xdata = xdata[thresh] ; ydata = ydata[thresh]
    # bin freq (x) axes
    minx, maxx = [np.min(xdata),np.max(xdata)]
    miny, maxy = [np.min(ydata),np.max(ydata)]
    xarr = np.linspace(minx,maxx,bins)
    Z = np.zeros(bins)
    # find max value (y) in bin
    for k in range(len(xdata)):
        i = np.where(xdata[k] >= xarr)[0][-1] # last index corresponding to row
        if Z[i] < ydata[k]:
            Z[i]=ydata[k] # assign highest growth rate
    return xarr, Z

def plot_k2d_growth(kpara,kperp,dw,norm=[None,None],cmap='summer',clims=(None,None),labels=['','']):
    # make 2d matrix
    Z,extents=make2D(kpara,kperp,dw,rowlim=(-4*norm[1],4*norm[1]),collim=(0,15*norm[1]),\
                        bins=(1000,1000),limits=True) # y, x, val
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=np.array(extents)/norm[1],cmap=cmap,clim=clims)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    fig.savefig('kperp_kpara_growthrates.pdf',bbox_inches='tight')
    del Z
    print('plotted k2d')
    return None

def plot_frq_kpara(kpara,w,dw,maxw=None,norm=[None,None],cmap='summer',labels=['','']):
    # make 2d matrix
    thresh = w/norm[0] < maxw
    Z,extents=make2D(kpara[thresh],w[thresh],dw[thresh],rowlim=(-4*norm[1],4*norm[1]),\
                    collim=(0,15*norm[0]),bins=(1000,1000),limits=True)
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    wmin,wmax,kmin,kmax = np.array(extents)
    im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=[wmin/norm[0],wmax/norm[0],kmin/norm[1],kmax/norm[1]],cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    fig.savefig('freq_kpara_growthrates.pdf',bbox_inches='tight')
    del Z
    print('plotted freq kpara')
    return None

def plot_frq_kperp(kperp,w,dw,maxk=None,maxw=None,norm=[None,None],cmap='summer',labels=['',''],clims=(None,None)):
    # make 2d matrix
    thresh = w/norm[0] < maxw
    Z,extents=make2D(kperp,w,dw,rowlim=(0,maxk*norm[1]),collim=(0,maxw*norm[0]),\
                    bins=(1000,1000),limits=True)
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    wmin,wmax,kmin,kmax = np.array(extents)
    im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=[wmin/norm[0],wmax/norm[0],kmin/norm[1],kmax/norm[1]],\
                    clim=clims,cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    ax.plot([0,maxw],[0,maxw],linestyle='--',color='white')
    ax.set_ylim(0,maxk)
    ax.set_xlim(0,maxw)
    fig.savefig('freq_kperp_growthrates.pdf',bbox_inches='tight')
    del Z
    print('plotted freq kperp')
    return None

def plot_frq_growth(w,dw,kpara,maxw=None,norm=[None,None],clims=(-1.5,1.5),labels=['','']):
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    thresh = (w/norm[0] < maxw) & (dw/norm[0] > 0)
    sc = ax.scatter(w[thresh]/norm[0],dw[thresh]/norm[0],c=kpara[thresh]/norm[1],edgecolor='none',vmin=clims[0],vmax=clims[1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r'$k_\parallel v_A/\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    ax.set_xlim(0,maxw)
    ax.set_ylim(0,0.15)
    fig.savefig('freq_growth.png',bbox_inches='tight')
    print('plotted freq growth')
    return None

def make_all_plots(alldata=None,maxnormfreq=15,maxnormkperp=15,cmap='summer'):
    w0,k0,w,dw,kpara,kperp = alldata
    # labels
    l1 = r'$k_\perp v_A/\Omega_i$'
    l2 = r'$k_\parallel v_A/\Omega_i$'
    l3 = r'$\omega/\Omega_i$'
    l4 = r'$\gamma/\Omega_i$'
    plot_k2d_growth(kpara,kperp,dw,norm=[w0,k0],clims=(0,0.15),cmap=cmap,labels=[l1,l2])
    plot_frq_growth(w,dw,kpara,maxw=maxnormfreq,norm=[w0,k0],labels=[l3,l4])
    plot_frq_kpara(kpara,w,dw,maxw=maxnormfreq,norm=[w0,k0],cmap=cmap,labels=[l3,l2])
    plot_frq_kperp(kperp,w,dw,maxk=maxnormkperp,maxw=maxnormfreq,norm=[w0,k0],cmap=cmap,labels=[l3,l1])
    return None

def loop_over_xiT(homeloc=''):
    # loop over concentrations
    freqs = []
    gamma = []
    txi2 = []
    maxnormf = 18
    fbins = 800
    fig,ax=plt.subplots(figsize=(15,2))
    ax.set_facecolor('#008066') # first color in summer heatmap

    XI2 = np.array([i/200 for i in range(0,200,5) if (i/2)%5!=0])
    for xi2 in XI2:
        print(xi2)
        solloc = homeloc+"run_2.07_{}_-0.646_0.01_0.01_15.0_3.5__1.0_4.0_1.7e19_0.00015_1024".format(xi2)
        os.chdir(solloc)
        data=read_all_data(loc=solloc)
        # make_all_plots(alldata=data)

        w0,k0,w,dw,kpara,kperp = data
        farr, growth = make1D(w,dw,norm=(w0,w0),maxnormx=maxnormf,bins=fbins)
        peaks = extractPeaks(growth,Nperw=8,plateau_size=0.5)

        txi2.append([xi2]*len(peaks))
        gamma.append(growth[peaks]/w0)
        freqs.append(farr[peaks]/w0)
        
        ## 3d plot
        # ax.scatter(farr[peaks]/w0,xi2*np.ones(len(peaks)),growth[peaks]/w0,color='b')
        # ax.plot(farr/w0,xi2*np.ones(len(farr)),growth/w0,color='k')

    # plot integer deuteron harmonics
    for i in range(0,maxnormf+1,1):
        ax.axvline(i,color='darkgrey',linestyle='--')
    # plot freqs vs. xi2 and gamma as color
    for i in range(len(gamma)):
        im = ax.scatter(freqs[i],txi2[i],c=gamma[i],marker='s',s=25,vmin=0,vmax=0.15,cmap='summer',edgecolor='none')
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel('Frequency'+' '+r'$[\Omega_i]$',**tnrfont)
    ax.set_ylabel(r'$\xi_T$',**tnrfont)
    ax.set_xlim(0,18.1)
    ax.set_ylim(0,1)
    plt.show()
    fig.savefig('../freq_xiT_growth_summer.png',bbox_inches='tight')
    return None

#-#-#
if __name__ == '__main__':
    ## PACKAGES ## 
    # standard
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('classic')
    plt.tight_layout()
    plt.rcParams['axes.formatter.useoffset'] = False
    import pickle
    import h5py
    from scipy import stats, signal
    # parallelising
    import multiprocessing as mp
    from functools import partial
    # other
    import os,sys

    ## BODY ## 

    ## one run file
    # data=read_all_data(loc=LOC_FILE)
    # make_all_plots(alldata=data)

    # multiple run files
    loop_over_xiT(homeloc="/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/")


