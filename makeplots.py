"""
    Written by Tobias Slade-Harajda for the purpose of analysing LMV () solutions2D.jld files.
    Functions dumpfiles() and read_pkl() are taken from my Thesis code to analyse EPOCH sims.
"""
tnrfont = {'fontsize':20,'fontname':'Times New Roman'}

## Dump pkl files
def dumpfiles(array, quant):
	print('Pickling '+quant+'...')
	with open(quant+'.pkl', 'wb') as f:
		pickle.dump(array,f)

## Read pkl files
def read_pkl(quant):
	with open(quant+'.pkl', 'rb') as f:
		print('Loading '+quant+'...')
		array = pickle.load(f)
	print('Done.')
	# automatically closed when loaded due to "with" statement
	return array

def paraload(i,f,sols):
    print(i)
    wi,dwi=f[sols[i]][()][0]
    kparai,kperpi=f[sols[i]][()][1]
    return [wi,dwi,kparai,kperpi]

def jld_to_pkl(loc='',frac=1):
    f = h5py.File(loc+'solutions2D_.jld',"r")
    keys=f.keys()
    # for k in keys: print(k)
    w0 = f["w0"][()]
    k0 = f["k0"][()]
    sols = f["plasmasols"][()]
    solshape = sols.shape[0] ; print(solshape)
    w = np.zeros(int(solshape/frac+1)) ; dw = np.zeros(int(solshape/frac+1))
    kpara = np.zeros(int(solshape/frac+1)) ; kperp = np.zeros(int(solshape/frac+1))

    #######################
    # TODO; parallelise this
    arr = [i for i in range(len(sols))]
    pool = mp.Pool(mp.cpu_count())
    par = partial(paraload,f=f,sols=sols) # utilise functools partial package
    res = np.array(pool.map_async(par,arr).get(99999))
    pool.close()
    w,dw,kpara,kperp=np.split(res,4,axis=1)
    print(len(w),len(kpara))
    sys.exit()

    #######################
    for i in range(len(sols[::frac])):
        item = sols[i]
        print(item)
        w[i],dw[i] = f[item][()][0]
        kpara[i],kperp[i] = f[item][()][1]
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

# def make2D(data=None):
#     w0,k0,w,dw,kpara,kperp = data
#     if kpara.shape != kperp.shape and kpara.shape != dw.shape:
#         raise SystemError
#     # unique values
#     ukpara,kparaind = np.unique(kpara,return_index=True)
#     ukperp,kperpind = np.unique(kperp,return_index=True)
#     # kind = np.sort(np.concatenate((kparaind,kperpind)))
#     ny,nx=[len(ukpara),len(ukperp)]
#     # min, max values
#     kparamin, kparamax = [np.min(kpara),np.max(kpara)]
#     kperpmin, kperpmax = [np.min(kperp),np.max(kperp)]
#     print(kparamin/k0, kparamax/k0)
#     print(kperpmin/k0, kperpmax/k0)
#     # arrays between max and min values
#     kperparr = np.linspace(kperpmin,kperpmax,nx)
#     kparaarr = np.linspace(kparamin,kparamax,ny)
#     Z = np.zeros((len(kperparr)+1,len(kparaarr)+1))
#     for k in range(len(kpara)):
#         i = np.where(kpara[k] >= kparaarr)[0][-1] # last index corresponding to row
#         j = np.where(kperp[k] >= kperparr)[0][-1] # last index corresponding to column
#         if Z[i,j] < dw[k]: 
#             Z[i,j]=dw[k] # assign highest growth rate
#     plt.imshow(Z/w0,origin='lower',aspect='auto',cmap='jet')
#     plt.colorbar()
#     plt.show()
#     return None

def make2D(rowval,colval,val,limits=True):
    if rowval.shape != colval.shape and rowval.shape != val.shape:
        raise SystemError
    # unique values
    urow,urowind = np.unique(rowval,return_index=True)
    ucol,ucolind = np.unique(colval,return_index=True)
    nx,ny=[len(urow),len(ucol)]
    # min, max values
    rowmin, rowmax = [np.min(rowval),np.max(rowval)]
    colmin, colmax = [np.min(colval),np.max(colval)]
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

def plot_k2d_growth(kpara,kperp,dw,norm=[None,None],cmap='summer',labels=['','']):
    # make 2d matrix
    Z,extents=make2D(kpara,kperp,dw) # y, x, val
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=np.array(extents)/norm[1],cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    fig.savefig('kperp_kpara_growthrates.pdf',bbox_inches='tight')
    del Z
    print('plotted k2d')
    return None

def plot_frq_kpara(kpara,w,dw,norm=[None,None],cmap='summer',labels=['','']):
    # make 2d matrix
    Z,extents=make2D(kpara,w,dw)
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    wmin,wmax,kmin,kmax = np.array(extents)
    im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=[wmin/w0,wmax/w0,kmin/k0,kmax/k0],cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    fig.savefig('freq_kpara_growthrates.pdf',bbox_inches='tight')
    del Z
    print('plotted freq kpara')
    return None

def plot_frq_kperp(kperp,w,dw,norm=[None,None],cmap='summer',labels=['','']):
    # make 2d matrix
    Z,extents=make2D(kperp,w,dw)
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    wmin,wmax,kmin,kmax = np.array(extents)
    im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=[wmin/w0,wmax/w0,kmin/k0,kmax/k0],cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    fig.savefig('freq_kperp_growthrates.pdf',bbox_inches='tight')
    del Z
    print('plotted freq kperp')
    return None

def plot_frq_growth(w,dw,kpara,norm=[None,None],labels=['','']):
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    sc = ax.scatter(w/w0,dw/w0,c=kpara/norm[1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r'$k_\parallel v_A/\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    fig.savefig('freq_growth.pdf',bbox_inches='tight')
    print('plotted freq growth')
    return None

def make_all_plots(alldata=None,cmap='summer'):
    w0,k0,w,dw,kpara,kperp = alldata
    # labels
    l1 = r'$k_\perp v_A/\Omega_i$'
    l2 = r'$k_\parallel v_A/\Omega_i$'
    l3 = r'$\omega/\Omega_i$'
    l4 = r'$\gamma/\Omega_i$'
    plot_k2d_growth(kpara,kperp,dw,norm=[w0,k0],cmap=cmap,labels=[l1,l2])
    plot_frq_growth(w,dw,kpara,norm=[w0,k0],labels=[l3,l4])
    # TODO; memory intensive, utilise 2d histograms (fixed #bins)
    # plot_frq_kpara(kpara,w,dw,norm=[w0,k0],cmap=cmap,labels=[l3,l2])
    # plot_frq_kperp(w,dw,kperp,norm=[w0,k0],cmap=cmap,labels=[l3,l1])
    return None

#-#-#
if __name__ == '__main__':
    # standard
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('classic')
    plt.tight_layout()
    plt.rcParams['axes.formatter.useoffset'] = False
    import pickle
    import h5py
    # parallelising
    import multiprocessing as mp
    from functools import partial
    # other
    import os,sys
    home = "/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/"
    XI2 = np.arange(0,1,0.05)
    for xi2 in XI2:
        print(xi2)
        solloc = "/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/run_2.07_{}_-0.646_0.01_0.01_15.0_3.5__1.0_4.0_1.7e19_0.00015_1024/".format(xi2)
        os.chdir(solloc)
        data=read_all_data(loc=solloc)
        make_all_plots(alldata=data)
