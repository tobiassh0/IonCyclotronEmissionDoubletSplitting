"""
    Written by Tobias Slade-Harajda for the purpose of analysing LMV 
    (https://github.com/jwscook/IonCyclotronEmissionDoubletSplitting)
    solutions2D.jld files. 
    Functions dumpfiles() and read_pkl() are taken from my Thesis 
    code to analyse EPOCH sims.
"""

## PACKAGES ## 
# standard
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
from matplotlib import cm
from matplotlib.gridspec import GridSpec
plt.style.use('classic')
plt.tight_layout()
plt.rcParams['axes.formatter.useoffset'] = False
from scipy import stats, signal
# parallelising
import multiprocessing as mp
from functools import partial
# other
import os,sys

tnrfont = {'fontsize':20,'fontname':'Times New Roman'}
## 

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

# ignore y axes tick labels
def ignorex(lax):
    for ax in lax:
        ax.tick_params(labelbottom=False)

# ignore y axes tick labels
def ignorey(lax):
    for ax in lax:
        ax.tick_params(labelleft=False)

# paraload using functools partial load of f and sols, pkl doesn't work on HDF5 files
def paraload(i,f,sols):
    print(i)
    wi,dwi=f[sols[i]][()][0]
    kparai,kperpi=f[sols[i]][()][1]
    return [wi,dwi,kparai,kperpi]

# paraload integer function, reloads f and solutions each time (slower than linear loop)
def paraload_int(I):
    loc,i=I
    f = h5py.File(loc+'solutions2D_.jld',"r")
    sols = f["plasmasols"][()]
    wi,dwi=f[sols[i]][()][0]
    kparai,kperpi=f[sols[i]][()][1]
    return [wi,dwi,kparai,kperpi]

# get Alfven velocity based on w0 and k0 (assuming k0 is normalisation of Va/w0)
def getVa(w0,k0):
    return w0/k0

# convert .jld to .pkl file type
def jld_to_pkl(loc='',frac=1,parallel=False):
    f = h5py.File(loc+'solutions2D_.jld',"r")
    keys=f.keys()
    # for k in keys: print(k)
    w0 = f['w0'][()]
    k0 = f['k0'][()]
    sols = f['plasmasols'][()]
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
    dumpfiles(w,loc+'frequency')
    dumpfiles(dw,loc+'growthrates')
    dumpfiles(kpara,loc+'parallelwavenumber')
    dumpfiles(kperp,loc+'perpendicularwavenumber')
    dumpfiles([w0,k0],loc+'w0k0')
    return w0,k0,w,dw,kpara,kperp

# try reading data, otherwise convert to pkl
def read_all_data(loc=''):
    try: # load all relevant data
        w0,k0=read_pkl(loc+'w0k0')
        w=read_pkl(loc+'frequency')
        dw=read_pkl(loc+'growthrates')
        kpara=read_pkl(loc+'parallelwavenumber')
        kperp=read_pkl(loc+'perpendicularwavenumber')
    except: # calculate and dump data
        w0,k0,w,dw,kpara,kperp = jld_to_pkl(loc=loc)
    data = w0,k0,w,dw,kpara,kperp
    return data

# make 1 array, defined by shape nx,ny or binx,biny, into (smaller) 2d array
def make2D(rowval,colval,val,rowlim=(None,None),collim=(None,None),bins=(1000,1000),limits=False,dump=False,name=''):
    if rowval.shape != colval.shape and rowval.shape != val.shape: # make sure same shape
            raise SystemError
    if rowlim[0] != None or collim[0] != None: # check if limits applied
        # thresh to limit size
        thresh = (rowlim[0]<rowval) & (rowval<rowlim[1]) & (collim[0]<colval) & (colval<collim[1]) 
        rowval = rowval[thresh] ; colval = colval[thresh] ; val = val[thresh]
        # min, max values from data
        rowmin, rowmax = [np.min(rowval),np.max(rowval)]
        colmin, colmax = [np.min(colval),np.max(colval)]
    # if dumping, then expect to load a file, otherwise calculate
    try:
        if not dump:
            Z = read_pkl(name)
            [colmin,colmax,rowmin,rowmax] = read_pkl(name+'_ext')
        else:
            read=False
            raise SystemError
    except: # can't load data
        if bins[0] == None: # no bins
            # unique values
            urow,urowind = np.unique(rowval,return_index=True)
            ucol,ucolind = np.unique(colval,return_index=True)
            nx,ny=[len(urow),len(ucol)]
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
        
        # boolean if wanting to pkl Z and (ext)ents
        if dump: 
            dumpfiles(Z,name)
            dumpfiles([colmin,colmax,rowmin,rowmax],name+'_ext')

    if limits:
        return Z, [colmin,colmax,rowmin,rowmax]
    else:
        return Z

# make 2 arrays of nx,ny length into shorter binned arrays 
def make1D(xdata,ydata,maxnormx=15,norm=(1,1),bins=800):
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

# plot the contour lines of integer multiples of the cyclotron frequency (not smooth)
def plotCycContours(fig,ax,norm=[1,1],maxnormf=18,rowlim=(None,None),collim=(None,None),bins=(1000,1000),alpha=0.5,\
                    levels=None,color='white'):
    if not levels:
        levels = range(0,maxnormf+1,1)
    # get frequency meshgrid as per FAW dispersion (Eq. 9 in DOI: 10.1088/1361-6587/ac8ba4)
    VA = getVa(norm[0],norm[1])
    kpara = np.linspace(rowlim[0],rowlim[1],1000)
    kperp = np.linspace(collim[0],collim[1],1000)
    KPERP, KPARA = np.meshgrid(kperp,kpara)
    K2 = KPARA**2 + KPERP**2
    W2 = ((VA**2)/2)*(K2 + KPARA**2 + (K2*KPARA**2)*((VA**2)/(norm[0]**2)) + \
            ((K2 + KPARA**2 + (K2*KPARA**2)*(VA**2)/(norm[0]**2))**2 - 4*K2*KPARA**2)**0.5)
    extents = np.array([collim[0],collim[1],rowlim[0],rowlim[1]])/norm[1]

    ax.contour(np.sqrt(W2)/norm[0],levels=levels,origin='lower',colors=color,alpha=alpha,extent=extents)
    ax.plot([0,maxnormf],[0,0],color='darkgrey',linestyle='--',alpha=alpha)
    return fig, ax

# plot kperp vs. kpara with growth rate heatmap 
def plot_k2d_growth(kpara,kperp,dw,w,norm=[None,None],cmap='summer',clims=(None,None),labels=['',''],contours=False,dump=True,\
                    rowlim=(-4,4),collim=(0,15),maxnormf=15):
    # make 2d matrix
    Z,extents=make2D(kpara,kperp,dw,rowlim=np.array(rowlim)*norm[1],collim=np.array(collim)*norm[1],\
                     bins=(1000,1000),limits=True,dump=dump,name='k2d_growth') # y, x, val
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    im = ax.imshow(Z/norm[0],aspect='auto',origin='lower',extent=np.array(extents)/norm[1],cmap=cmap,clim=clims)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    if contours:
        fig,ax=plotCycContours(fig,ax,norm=norm,rowlim=np.array(rowlim)*norm[1],collim=np.array(collim)*norm[1],maxnormf=maxnormf)
    fig.savefig('kperp_kpara_growthrates.pdf',bbox_inches='tight')
    del Z
    print('plotted k2d')
    return None

# plot kperp vs. kpara with growth rate heatmap for a range (loop) of concentrations
def plot_k2d_growth_combined(sollocs=[''],loop=[],cmap='summer',clims=(0,0.15),rowlim=(-4,4),collim=(0,15)):
    rowlim = np.array(rowlim)
    collim = np.array(collim)

    fig = plt.figure(figsize=(10,10))
    
    # base (largest fig)
    gs1 = GridSpec(4, 10, bottom=0.53, top=0.98, left=0.1, right=0.95, wspace=0.0, hspace=0.05)# bottom spacing creates space for gs2 
    ax1 = fig.add_subplot(gs1[:, :])                                # 0
    # loop (zoomed figs)
    gs2 = GridSpec(8, 10, bottom=0.075, top=0.49, left=0.1, right=0.95, wspace=0.15, hspace=0.35) # nrows, ncols, l, r, wsp, hsp
    ax11 = fig.add_subplot(gs2[:4, :2])                             # 1
    ax12 = fig.add_subplot(gs2[:4, 2:4],sharey=ax11)                # 2
    ax13 = fig.add_subplot(gs2[:4, 4:6],sharey=ax11)                # 3
    ax14 = fig.add_subplot(gs2[:4, 6:8],sharey=ax11)                # 4
    ax15 = fig.add_subplot(gs2[:4, 8:10],sharey=ax11)               # 5
    ax21 = fig.add_subplot(gs2[4:, :2],sharey=ax11)                 # 6
    ax22 = fig.add_subplot(gs2[4:, 2:4],sharey=ax11)                # 7
    ax23 = fig.add_subplot(gs2[4:, 4:6],sharey=ax11)                # 8 
    ax24 = fig.add_subplot(gs2[4:, 6:8],sharey=ax11)                # 9 
    ax25 = fig.add_subplot(gs2[4:, 8:10],sharey=ax11)               # 10

    allax = fig.axes
    ignorex(allax)
    ignorey(allax)

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
        if i==0:
            fig,ax=plotCycContours(fig,ax,norm=norm,rowlim=rowlim*norm[1],collim=collim*norm[1],maxnormf=collim[-1])
        else:
            fig,ax=plotCycContours(fig,ax,norm=norm,rowlim=rowlim*norm[1],collim=collim*norm[1],levels=[12,13,14,15])        
        # ax.plot([0,15],[-1,1])
        ax.annotate("{:.2f}".format(loop[i]), xy=(0.0125,0.9), xycoords='axes fraction',**tnrfont)
        i+=1

    # formatting    
    for i in range(len(allax)):
        if i == 0:
            allax[i].tick_params(labelleft=True,labelbottom=True)
        if i == 1:
            allax[i].tick_params(labelleft=True)
            allax[i].set_yticks([-1,-0.5,0,0.5,1])
        if i == (len(loop)-1)//2+1:
            allax[i].tick_params(labelleft=True,labelbottom=True)
            allax[i].set_yticks([-1,-0.5,0,0.5,1])
        if i > (len(loop)-1)//2+1:
            allax[i].tick_params(labelbottom=True)
        if i > 0:
	         allax[i].set_xticks([11,12,13,14,15])

        if i == len(allax)-1:
            allax[i].set_xticks([11,12,13,14,15])
            allax[i].set_xticklabels(['11','12','13','14','15'])
        elif i >= (len(loop)-1)//2+1:
            allax[i].set_xticks([11,12,13,14])
            allax[i].set_xticklabels(['11','12','13','14'])
        else:
            None

        if i != 0:
            allax[i].set_xlim(11,15)
            allax[i].set_ylim(-1,1)
        else:
            allax[i].set_xlim(collim)
            allax[i].set_ylim(rowlim)

    # ax_group = fig.add_subplot(gs2[-1, 0:10])
    # ax_group.set_xticks([])
    # ax_group.set_yticks([])
    # ax_group.set_frame_on(False)
    # ax_group.set_xlabel("Group label!", labelpad=20)

    # colorbar
    p0 = ax1.get_position().get_points().flatten()
    p1 = allax[-1].get_position().get_points().flatten()
    # im = ax1.imshow([[0,1],[0,1]],aspect='auto',origin='lower')
    cbar = fig.add_axes([p0[2]+0.02, p1[1], 0.01, p0[3]-p1[1]]) # [left bottom width height]
                      # [0.97,0.05,0.01,0.93]
    plt.colorbar(im, cax=cbar, orientation='vertical')
    cbar.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    fig.supylabel('Parallel Wavenumber '+'  '+r'$[\Omega_i/V_A]$',**tnrfont)
    fig.supxlabel('Perpendicular Wavenumber '+'  '+r'$[\Omega_i/V_A]$',**tnrfont)
    fig.savefig('../kperp_kpara_growthrates_combined.pdf',bbox_inches='tight')
    # plt.show()

    return None

# plot frequency vs. kpara with growth rate heatmap
def plot_frq_kpara(kpara,w,dw,maxnormf=None,norm=[None,None],cmap='summer',labels=['','']):
    # make 2d matrix
    thresh = w/norm[0] < maxnormf
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

# plot frequency vs. kperp with growth rate heatmap
def plot_frq_kperp(kperp,w,dw,maxk=None,maxnormf=None,norm=[None,None],cmap='summer',labels=['',''],clims=(None,None)):
    # make 2d matrix
    thresh = w/norm[0] < maxnormf
    Z,extents=make2D(kperp,w,dw,rowlim=(0,maxk*norm[1]),collim=(0,maxnormf*norm[0]),\
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
    ax.plot([0,maxnormf],[0,maxnormf],linestyle='--',color='white')
    ax.set_ylim(0,maxk)
    ax.set_xlim(0,maxnormf)
    fig.savefig('freq_kperp_growthrates.pdf',bbox_inches='tight')
    del Z
    print('plotted freq kperp')
    return None

# plot frequency vs growth rate (2d spectra)
def plot_frq_growth(w,dw,kpara,maxnormf=None,norm=[None,None],clims=(-1.5,1.5),labels=['','']):
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    thresh = (w/norm[0] < maxnormf) & (dw/norm[0] > 0)
    sc = ax.scatter(w[thresh]/norm[0],dw[thresh]/norm[0],c=kpara[thresh]/norm[1],edgecolor='none',vmin=clims[0],vmax=clims[1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r'$k_\parallel v_A/\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    ax.set_xlim(0,maxnormf)
    ax.set_ylim(0,0.15)
    fig.savefig('freq_growth.png',bbox_inches='tight')
    print('plotted freq growth')
    return None

# plot freq vs. growth for a given (range of) angle(s)
def plot_frq_growth_angles(kpara,kperp,w,dw,maxnormf=None,norm=[None,None],angles=[88.,88.5,89.,89.5],labels=['',''],clims=[0,0.5],\
                            percentage=0.0025,smooth=True):
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
        if smooth:
            sw, sdw = make1D(tw,tdw,norm=(norm[0],norm[0]),maxnormx=maxnormf,bins=200) # very small No. bins
            ax.plot(sw/norm[0],sdw/norm[0],color='k')
        ax.annotate(r"${:.2f}\pm{:.2f}$".format(ang*180/np.pi,ang*percentage*180/np.pi), xy=(0.0125,0.9), xycoords='axes fraction',**tnrfont)
        fig.savefig('freq_growth_{:.1f}.png'.format(ang*180/np.pi),bbox_inches='tight')
        # plt.show()
        plt.clf()
        print('plotted freq growth ang {:.1f}'.format(ang*180/np.pi))
    return None

# plot 2d or 3d over loop over y (xi2) in x (freq) and z (growth rate) space
def get_peak_freqs(solloc=[''],loop=[],maxnormf=18,fbins=1000,**kwargs):
    plot_2D=kwargs.get('plot_2D')
    plot_3D=kwargs.get('plot_3D')
    plot_hm=kwargs.get('plot_hm')
    if sum(filter(None,[plot_2D,plot_3D,plot_hm])) > 1 or sum(filter(None,[plot_2D,plot_3D,plot_hm])) < 1:
        print('# ERROR # :: defaulting to heatmap')
        plot_hm=True ; plot_2D=False ; plot_3D=False
    
    if plot_2D:
        # 2d colormap
        fig2d,ax2d=plt.subplots(figsize=(15,2))
        ax2d.set_facecolor('#008066') # first color in summer heatmap
        x=[];y=[]
    if plot_3D:
        # 3d surface
        fig3d,ax3d=plt.subplots(figsize=(10,6),subplot_kw={'projection':'3d'})
        x = np.linspace(0,maxnormf,fbins)	 
        X,Y = np.meshgrid(x,loop)
    if plot_hm:
        # imshow array
        fighm,axhm=plt.subplots(figsize=(15,2))
        x=[];y=[]
    z=[]
    growth_hm=np.zeros((len(loop),fbins))
    # loop over concentrations
    for i in range(len(loop)):
        print(loop[i])
        os.chdir(solloc[i])
        data=read_all_data(loc=solloc[i])
        w0,k0,w,dw,kpara,kperp = data
        xarr, zarr = make1D(w,dw,norm=(w0,w0),maxnormx=maxnormf,bins=fbins)
        # 2D plot
        peaks = extractPeaks(zarr,Nperw=8,plateau_size=0.5)
        x.append(xarr[peaks]/w0)
        y.append([loop[i]]*len(peaks))# xi2 concentration constant
        z.append(zarr[peaks]/w0)
        growth_hm[i,:]=zarr/w0
        ## 3D plot
        # ax.scatter(farr[peaks]/w0,l*np.ones(len(peaks)),growth[peaks]/w0,color='b')
        # ax.plot(farr/w0,l*np.ones(len(farr)),growth/w0,color='k')
    
    if plot_3D:
        z = np.array(z)
        Z = z.reshape(X.shape)
        ax3d.plot_surface(X,Y,Z,cmap=cm.summer,vmin=0,vmax=0.15,antialiased=True)
        ax3d.set_xlabel('Frequency'+' '+r'$[\Omega_i]$',**tnrfont)
        ax3d.set_ylabel(r'$\xi_T$',**tnrfont)
        ax3d.set_zlabel('Growth Rate'+'  '+r'$[\Omega_i]$',**tnrfont)
        for az in range(0,359,1):
            print(az)
            ax3d.view_init(elev=30,azim=az)
            fig3d.savefig('../az_%d.png' % az)
    if plot_2D:
        # plot integer deuteron harmonics
        for i in range(0,maxnormf+1,1):
            ax2d.axvline(i,color='darkgrey',linestyle='--')
        # plot freqs vs. xi2 and gamma as color
        for i in range(len(z)):
            im = ax2d.scatter(x[i],y[i],c=z[i],marker='s',s=25,vmin=0,vmax=0.15,cmap='summer',edgecolor='none')
        # plot two trend lines
        ax2d.plot([9.5,10],[1,0],linestyle='--',color='k',alpha=0.75)
        ax2d.annotate('A',xy=(9.3,0.8),xycoords='data')
        ax2d.plot([4.5,10],[1.05,0],linestyle='--',color='k',alpha=0.75)
        ax2d.annotate('B',xy=(6,0.6),xycoords='data')
        # colorbar, labelling and other formatting
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
        ax2d.set_xlabel('Frequency'+' '+r'$[\Omega_i]$',**tnrfont)
        ax2d.set_ylabel(r'$\xi_T$',**tnrfont)
        ax2d.set_xlim(0,maxnormf+0.1)
        ax2d.set_ylim(0,1)
        fig2d.savefig('../freq_xiT_growth_peaks_labelled.png',bbox_inches='tight')
    if plot_hm:
        # plot integer deuteron harmonics
        for i in range(0,maxnormf+1,1):
            axhm.axvline(i,color='darkgrey',linestyle='--')
        im = axhm.imshow(growth_hm,origin='lower',aspect='auto',extent=[0,maxnormf,0,1],cmap='summer',interpolation='none',clim=(0,0.15))
        # cbar = fig.colorbar(im, orientation='vertical', pad=0.1)
        cbar=plt.colorbar(im)
        cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
        # for i in range(len(x)):
            # im = axhm.scatter(x[i],y[i],facecolor='none',marker='s',s=12,edgecolor='k',alpha=0.5)
        axhm.set_xlabel('Frequency'+' '+r'$[\Omega_i]$',**tnrfont)
        axhm.set_ylabel(r'$\xi_T$',**tnrfont)
        fighm.savefig('../freq_xiT_growth_all.png',bbox_inches='tight')
        # plt.show()
    plt.clf()
    return None

# reproduce main plots of the JWS.Cook LMV Julia code
def make_all_plots(alldata=None,cmap='summer'):
    w0,k0,w,dw,kpara,kperp = alldata
    maxnormfreq = np.max(w)/w0
    maxnormkperp = np.max(kperp)/k0
    maxnormkpara = np.max(kpara)/k0
    # labels
    l1 = "Perpendicular Wavenumber"+ "  "+r"$[\Omega_i/V_A]$"   # r'$k_\perp v_A/\Omega_i$'
    l2 = "Parallel Wavenumber"+ "  "+r"$[\Omega_i/V_A]$"        # r'$k_\parallel v_A/\Omega_i$'
    l3 = "Frequency"+ "  "+r"$[\Omega_i]$"                      # r'$\omega/\Omega_i$'
    l4 = "Growth Rate"+ "  "+r"$[\Omega_i]$"                    # r'$\gamma/\Omega_i$'
    # plotting scripts
    # plot_k2d_growth(kpara,kperp,dw,w,norm=[w0,k0],clims=(0,0.15),cmap=cmap,labels=[l1,l2],contours=False,\
    #                 rowlim=(-maxnormkpara,maxnormkpara),collim=(0,maxnormkperp),maxnormf=maxnormkperp,dump=False) # load
    # plot_frq_growth(w,dw,kpara,maxnormf=maxnormkperp,norm=[w0,k0],labels=[l3,l4])
    plot_frq_growth_angles(kpara,kperp,w,dw,maxnormf=maxnormkperp,norm=[w0,k0],angles=[88.,88.5,89.,89.5,90.],labels=[l3,l4],\
                    clims=[0,0.5],percentage=0.0005)
    # plot_frq_kpara(kpara,w,dw,maxnormf=maxnormfreq,norm=[w0,k0],cmap=cmap,labels=[l3,l2])
    # plot_frq_kperp(kperp,w,dw,maxk=maxnormkperp,maxnormf=maxnormfreq,norm=[w0,k0],cmap=cmap,labels=[l3,l1])
    return None

# parallel transferral of multiple runs data from HDF5 to pkl
def para_runs(loc):
    data=read_all_data(loc=loc)
    return None

# parallel calculate and make all of the pkl files, optional bool to make plots as well in linear loop
def para_calc(home,plot=True):
    sollocs = getsollocs(home)
    # parallel run to load/calculate data
    pool = mp.Pool(2**(round(np.log2(mp.cpu_count()))-1)) # find nearest multiple of 2, then decrease by factor 1
    pool.map(para_runs,sollocs).get(99999)
    pool.close()
    if plot:
        for i in range(len(sollocs)):
            print(sollocs[i])
            os.chdir(sollocs[i])
            data=read_all_data(loc=sollocs[i])
            w0,k0,w,dw,kpara,kperp = data
            make_all_plots(alldata=data)
    return None

# return list of sorted run solution files in home directory (defaults to cwd)
def getsollocs(home=''):
    if home == '': home = os.getcwd()
    sollocs = [home+i+'/' for i in os.listdir(home) if 'run' in i]
    return np.sort(sollocs)

#-#-#
if __name__ == '__main__':
    
    import kernel_doppler as kd
    import line_doppler as ld

    ## BODY ## 
    homes = {
        'lowkperp_T':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/",
        'highkperp_T':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration_high_kperp/",
        'lowkperp_noT':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons/",
        'highkperp_noT':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons_high_kperp/",        
    }
    """
    :: examples  :: 
        # one file
    LOC_FILE = homes.get(NAME) # if using dict
    data=read_all_data(loc=LOC_FILE)
    make_all_plots(alldata=data)

        # multiple files
    para_calc(LOC_FILE,plot=True)
    """

    homeloc = homes.get('lowkperp_T')

    # # DT runs
    # XI2 = [i/200 for i in range(0,200,5)]
    # print(XI2)
    # # XI2 = [i/100 for i in range(45,95,5)]
    # # XI2.append(0)
    # # XI2 = np.sort(XI2)
    # sollocs = getsollocs(homeloc)
    # XI2 = []
    # for sol in sollocs: # missing 11% in XI2 array
    #     XI2.append(float(sol.split('run')[1].split('_')[2]))
    # XI2 = np.array(XI2)
    # # kd.plot_doppler_kernel(sollocs=[sollocs[-1]],labels=[XI2[-1]])
    # ld.plot_doppler_line(sollocs=sollocs,labels=XI2) # [str(i*100)+'%' for i in XI2])
    # # plot_k2d_growth_combined(sollocs=sollocs,loop=XI2,cmap='summer',clims=(0,0.15),collim=(0,25))
    # sys.exit()

    # D runs
    # homeloc = homes.get('highkperp_noT')
    # runs = getsolloc(homeloc)
    # nearr=[]
    # names=[]
    # # separate to find parameters
    # for run in runs:
    #    params = run.split("_")
    #    # if name not given, this will return ValueError
    #    _, B0, xiT, pitch, vthperp, vthpara, kperpmax, EminMeV, name, tempkeV, kparamax, ne, ximin, ngridpoints = params
    #    nearr.append(ne)
    #    names.append(name)
    # # sort names & nearr (sort by increasing name (xi2))
    # nearr = [x for _,x in sorted(zip(names,nearr))]
    # names = np.sort(names)
    # sollocs = np.array([homeloc+'run_2.07_0.0_-0.646_0.01_0.01_25.0_3.5_{}_1.0_4.0_{}_0.00015_2048/'.format(names[i],nearr[i]) for i in range(len(names))])

    # energy runs
    homeloc = '/home/space/phrmsf/Documents/ICE_DS/JET26148/D_T_energy_scan/'
    # get sollocs
    sollocs = getsollocs(homeloc)
    pitches = []
    energies= []
    for sol in sollocs: # missing 11% in XI2 array
        pitches.append(float(sol.split('run')[1].split('_')[3]))
        energies.append(float(sol.split('run')[1].split('_')[7]))
    labels=[pitches,energies]
    names=['maxvdop_pitches','maxvdop_energy']
    for i in range(2):
        ld.plot_doppler_line(sollocs=sollocs,labels=labels[i],plot_grad=True,name=names[i])
    sys.exit()



"""
    TODO ; 
        - (efficiency) list of angles given to plot angle loop, dont loop over N len arrays each time
            - could make number of empty arrays based on len(angles) and loop over kperp/kpara once and see if
                angle in list of angles
"""
