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
import scipy
from scipy import stats, signal
# parallelising
import multiprocessing as mp
from functools import partial
# other
import os,sys

# separate python files
import kernel_doppler as kd
import line_doppler as ld

tnrfont = {'fontsize':20,'fontname':'Times New Roman'}
imkwargs = {'origin':'lower','interpolation':'none','aspect':'auto'}

## 
class constants():
    def __init__(self):
        self.c = 2.9979246E+08		# ms^[-1] #

const = constants()

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

# put ticks on the outside of the figure
def outside_ticks(fig,_axis='both'):
	for i, ax in enumerate(fig.axes):
		ax.tick_params(axis=_axis,direction='out',top=False,right=False,left=True,bottom=True)

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

# get frequency from kpara, kperp, Va and wci
def getFreq(tkpara,tkperp,w0,k0):
    Va = getVa(w0,k0)
    Va2 = Va**2
    kpara2 = tkpara**2
    ktot2 = kpara2 + tkperp**2
    omega2 = (Va2/2)*(ktot2 + kpara2 + (ktot2*kpara2*Va2/(w0**2)) + \
                ((ktot2 + kpara2 + (ktot2*kpara2*Va2/(w0**2)))**2 - 4*ktot2*kpara2)**0.5)
    return omega2**0.5

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

# make 1 array, defined by shape (nx,ny) or (binx,biny) into *smaller* 2d array
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
            if Z[i,j] < val[k]: # change to abs(val[k]) and will aid in extracting kpara in 2d map
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
                    levels=[None],color='white'):
    if levels[0] == None:
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
        ax.annotate(r"$\xi_T=$"+"{:.0f}%".format(100*loop[i]), xy=(0.0125,0.975), xycoords='axes fraction',**tnrfont,va='top',ha='left')
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
    fig.savefig('../kperp_kpara_growthrates_combined-1.pdf',bbox_inches='tight')
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
def plot_frq_growth(w,dw,kpara,maxnormf=None,maxnormg=0.15,norm=[None,None],clims=(-1.5,1.5),labels=['','']):
    # setup figure & plot
    fig,ax=plt.subplots(figsize=(8,5))
    thresh = (w/norm[0] < maxnormf) & (dw/norm[0] > 0)
    sc = ax.scatter(w[thresh]/norm[0],dw[thresh]/norm[0],c=kpara[thresh]/norm[1],edgecolor='none',vmin=clims[0],vmax=clims[1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r'$k_\parallel v_A/\Omega_i$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel(labels[0],**tnrfont)
    ax.set_ylabel(labels[1],**tnrfont)
    ax.set_xlim(0,maxnormf)
    ax.set_ylim(0,maxnormg)
    fig.savefig('freq_growth.png',bbox_inches='tight')
    print('plotted freq growth')
    return None

# extract frequency value on the grid with pixel (x,y) coordinates
def getZfreq(Zfreq,x,y):
    """
    IN : 
        Zfreq : make2D() array of kpara,kperp,w
        x, y  : arrays of pixel coordinates for a line drawn on the (kperp,kpara,w) grid
    OUT:
        freq  : array of frequencies of pixel coordinates corresponding to (x,y) coordinates
    """
    freq = scipy.ndimage.map_coordinates(Zfreq,np.vstack((y,x)))
    return freq

# get freq vs. growth for a given angle
def get_frq_growth_angles(Z,Zfreq,wmax=15,rowlim=(None,None),collim=(None,None),norm=[1,1],angles=[90.]):
    Ny, Nx = Z.shape
    # lsize = np.sqrt(Nx**2+Ny**2) # maximum length of array
    # limits of box, normalised units
    rowlim = np.array(rowlim) ; collim = np.array(collim)
    kperpmin, kperpmax = (collim/norm[1])
    kparamin, kparamax = (rowlim/norm[1])
    extents = [kperpmin,kperpmax,kparamin,kparamax]
    # centre of lines
    Xn = 0 # starts at 0
    Yn = Ny/2 # centred on kpara = 0, symmetric about axes
    # find intersection points between line and box (pixel coords)
    Ystart = Ny ; Yend = 0
    # empty arrays
    freqs=[] ; zi=[] ; zisum=[]
    # loop through angles
    for j in range(len(angles)):
        xlim, ylim = ld.LineBoxIntersection(Ystart,Yend,Xn,Yn,Nx,Ny,abs(angles[j])*np.pi/180)
        # find data points along line
        lsize = np.sqrt((xlim[-1]-xlim[0])**2 + (ylim[-1]-ylim[0])**2)
        x = np.linspace(xlim[0],xlim[1],int(lsize))
        y = np.linspace(ylim[0],ylim[1],int(lsize))
        zi.append(scipy.ndimage.map_coordinates(Z,np.vstack((y,x)))) # growth rates (rad/s)
        # convert all to real coordinates
        xlim = (collim[0]/norm[1])+xlim*((collim[1]-collim[0])/norm[1])/Nx
        ylim = (rowlim[0]/norm[1])+ylim*((rowlim[1]-rowlim[0])/norm[1])/Ny
        # flip line direction is pointed down (end point flips so need to revert)
        if xlim[-1]<0.00001 and ylim[-1]<0.00001: # should be equal to 0, this is valid for difference in angles greater than a degree 
            zi[-1] = np.flip(zi[-1]) # opposite direction of line
        # summate all points # "growth per (dkpara,dw) cell"
        zisum.append(np.sum(zi[-1])/len(zi[-1])) # normalise to number of cells along line
        # freq from kpara, kperp
        tkpara = np.linspace(0,np.max(np.abs(ylim))*norm[1],len(zi[-1]))
        tkperp = np.linspace(0,np.max(np.abs(xlim))*norm[1],len(zi[-1]))
        freqs.append(getFreq(tkpara,tkperp,norm[0],norm[1]))
        """ alternative method for extracting frequencies, using grid values rather than FAW dispersion relation
            >>> freqs.append(getZfreq(Zfreq,x,y))
        """
    return freqs, zi, zisum

# plot peak frequencies in 2d xi2 space for a given angle
def plot_peak_frq_angle(home,sollocs,XI2=[0.1,0.3,0.5,0.7],wmax=15,rowlim=(None,None),collim=(None,None),angle=89.0,\
                        plateau_size=0.5,Nperw=10,growth_thresh=0.0):
    fig,ax=plt.subplots(figsize=(15,2))
    ax.set_facecolor('#008066') # first color in summer heatmap
    # loop through each xi2
    for i in range(len(XI2)):
        print(XI2[i])
        os.chdir(home+sollocs[i])
        # load data
        data=read_all_data(loc=home+sollocs[i])
        w0,k0,w,dw,kpara,kperp = data
        if i == 0:
            rowlim = np.array(rowlim)*k0
            collim = np.array(collim)*k0
        # make 2d data (kperp,kpara)
        Z = make2D(kpara,kperp,dw,rowlim=rowlim,collim=collim,dump=False,name='k2d_growth')
        Zfreq = make2D(kpara,kperp,w,rowlim=rowlim,collim=collim,dump=False,name='k2d_freq')
        # get growth rate along angle
        freqs,growthrates,_=get_frq_growth_angles(Z=Z,Zfreq=Zfreq,wmax=wmax,rowlim=rowlim,collim=collim,norm=[w0,k0],angles=[angle])
        xarr = freqs[-1]
        zarr = growthrates[-1]
        # plt.plot(xarr,zarr) ; plt.show()
        # get peaks of growth rates
        peaks = extractPeaks(zarr)
        xarp = xarr[peaks]/w0
        zarp = zarr[peaks]/w0
        thresh = zarp > growth_thresh
        xarp = xarp[thresh] ; zarp = zarp[thresh]
        # plt.plot(xarr/w0,zarr/w0) ; plt.scatter(xarp,zarp,color='r') ; plt.show()
        im = ax.scatter(xarp,[XI2[i]]*len(xarp),c=zarp,marker='s',s=25,vmin=0,vmax=0.1,cmap='summer',edgecolor='none')
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
    ax.set_xlabel('Frequency'+' '+r'$[\Omega_i]$',**tnrfont)
    ax.set_ylabel(r'$\xi_T$',**tnrfont)
    ax.set_xlim(0,15)
    ax.set_ylim(0,np.max(XI2))
    ax.annotate(r'${:.1f}^\circ$'.format(angle),xy=(0.01,0.95),xycoords='axes fraction',va='top',ha='left',**tnrfont)
    # fig.savefig(home+'freq_xiT_growth_peaks_angle_{}.png'.format(angle),bbox_inches='tight')
    plt.show()
    return None

# plot freq vs. growth for a given (range of) angle(s)
def plot_frq_growth_angles(kpara,kperp,dw,w,wmax=15,rowlim=(None,None),collim=(None,None),norm=[1,1],angles=[None],\
                            anglabels=None,colorarr=None,clims=(0,0.10)):
    Z = make2D(kpara,kperp,dw,rowlim=np.array(rowlim),collim=np.array(collim),dump=True,name='k2d_growth')
    Zfreq = make2D(kpara,kperp,w,rowlim=np.array(rowlim),collim=np.array(collim),dump=True,name='k2d_freq')
    Ny, Nx = Z.shape
    # lsize = np.sqrt(Nx**2+Ny**2) # maximum length of array
    if angles[0] == None:
        angles = np.array([-80,-85,-90,-95,-100])
        anglabels = np.array([r'$80^\circ$',r'$85^\circ$',r'$90^\circ$',r'$95^\circ$',r'$100^\circ$'])
    # color array
    if colorarr != None:
        colors = plt.cm.rainbow(np.linspace(0,1,len(angles)))
    else:
        colors = ['k']*len(angles)

    # limits of box, normalised units
    rowlim = np.array(rowlim) ; collim = np.array(collim)
    kperpmin, kperpmax = (collim/norm[1])
    kparamin, kparamax = (rowlim/norm[1])
    extents = [kperpmin,kperpmax,kparamin,kparamax]
    # collection of growths for given range of angles
    fig_line,ax_line=plt.subplots(figsize=(4,int(2*len(angles))),nrows=len(angles),sharex=True)
    # heatmap of kpara vs freq
    fig_hm,ax_hm=plt.subplots(figsize=(10,5))
    ax_hm.imshow(Z/norm[0],**imkwargs,cmap='summer',clim=clims,\
                    extent=[collim[0]/norm[1],collim[1]/norm[1],rowlim[0]/norm[1],rowlim[1]/norm[1]])
    # fig_hm,ax_hm=plotCycContours(fig_hm,ax_hm,norm=norm,maxnormf=wmax,rowlim=rowlim,collim=collim,bins=(1000,1000))
    ax_line[0].set_xlim(0,wmax)
    # centre of lines
    Xn = 0 # starts at 0
    Yn = Ny/2 # centred on kpara = 0, symmetric about axes
    Ystart = Ny ; Yend = 0
    # get freqs, zi and zisum
    freqs, zi, zisum = get_frq_growth_angles(Z,Zfreq=Zfreq,wmax=15,rowlim=rowlim,collim=collim,norm=norm,angles=angles)
    # loop through angles
    for j in range(len(angles)):
        fig_single,ax_single = plt.subplots(figsize=(8,6))
        # find intersection points between line and box (pixel coords)
        xlim, ylim = ld.LineBoxIntersection(Ystart,Yend,Xn,Yn,Nx,Ny,abs(angles[j])*np.pi/180)
        # convert all to real coordinates
        xlim = (collim[0]/norm[1])+xlim*((collim[1]-collim[0])/norm[1])/Nx
        ylim = (rowlim[0]/norm[1])+ylim*((rowlim[1]-rowlim[0])/norm[1])/Ny
        # combined line plot
        ax_line[j].set_ylim(0,clims[-1]) # no negative growths
        ax_line[j].locator_params(axis='y',nbins=5)
        ax_line[j].annotate(anglabels[j],xy=(0.1,0.9),xycoords='axes fraction',va='top')
        ax_line[j].plot(freqs[j]/norm[0],zi[j]/norm[0],color=colors[j]) # np.linspace(xlim[0],xlim[1],len(zi))
        # plot lines on heatmap
        ax_hm.plot(xlim,ylim,'k--')
        # annotate angular lines
        print(angles[j],ylim)
        if ylim[-1] != 0:
            ax_hm.annotate(anglabels[j],xy=(15,ylim[-1]-0.01),xycoords='data',va='top',ha='right')
        else:
            ax_hm.annotate(anglabels[j],xy=(15,ylim[0]+0.01),xycoords='data',va='bottom',ha='right')
        # single plot
        ax_single.plot(freqs[j]/norm[0],zi[j]/norm[0],color='k') # np.linspace(xlim[0],xlim[1],len(zi))
        ax_single.set_xlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
        ax_single.set_ylabel('Growth rate '+r'$[\Omega_i]$',**tnrfont)
        ax_single.set_xlim(0,wmax)
        ax_single.set_ylim(0,clims[-1])
        # fig_single.savefig('freq_growth_{:.1f}.png'.format(angles[j]),bbox_inches='tight')

    ax_hm.set_xlabel('Perpendicular Wavenumber'+ '  '+r'$[\Omega_i/V_A]$',**tnrfont)
    ax_hm.set_ylabel('Parallel Wavenumber'+'  '+r'$[\Omega_i/V_A]$',**tnrfont)
    ax_hm.set_ylim(-2,2) ; ax_hm.set_xlim(0,15)
    fig_hm.savefig('kpara_kperp_angle_lines.png',bbox_inches='tight')
    ax_line[-1].set_xlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
    fig_line.supylabel('Growth rate '+r'$[\Omega_i]$',**tnrfont,x=-0.1)
    # fig_line.savefig('freq_growth_angles_combi.png',bbox_inches='tight')
    # plt.show()
    return None

# plot 2d or 3d over XI2 over y (xi2) in x (freq) and z (growth rate) space
def get_peak_frqs(home,sollocs=[''],XI2=[],maxnormf=18,fbins=800,plateau_size=0.5,Nperw=10,**kwargs):
    plot_2D=kwargs.get('plot_2D')
    plot_3D=kwargs.get('plot_3D')
    plot_hm=kwargs.get('plot_hm')
    # if sum(filter(None,[plot_2D,plot_3D,plot_hm])) > 1 or sum(filter(None,[plot_2D,plot_3D,plot_hm])) < 1:
    #     print('# ERROR # :: defaulting to heatmap')
    #     plot_hm=True ; plot_2D=False ; plot_3D=False
    if plot_2D:
        # 2d colormap
        fig2d,ax2d=plt.subplots(figsize=(15,2))
        ax2d.set_facecolor('#008066') # first color in summer heatmap
        # ax2d.set_facecolor('#FFFFFF') # white
        x=[];y=[]
    if plot_3D:
        # 3d surface
        fig3d,ax3d=plt.subplots(figsize=(10,6),subplot_kw={'projection':'3d'})
        x = np.linspace(0,maxnormf,fbins)	 
        X,Y = np.meshgrid(x,XI2)
    if plot_hm:
        # imshow array
        fighm,axhm=plt.subplots(figsize=(15,2))
        x=[];y=[]
    
    z=[]
    growth_hm=np.zeros((len(XI2),fbins))

    try:
        x=read_pkl(home+'freqpeaks_{}_{}'.format(Nperw,fbins))
        y=read_pkl(home+'xi2peaks_{}_{}'.format(Nperw,fbins))
        z=read_pkl(home+'growthpeaks_{}_{}'.format(Nperw,fbins))
    except:
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
            ## 3D plot
            # growth_hm[i,:]=zarr/w0
            # ax.scatter(farr[peaks]/w0,l*np.ones(len(peaks)),growth[peaks]/w0,color='b')
            # ax.plot(farr/w0,l*np.ones(len(farr)),growth/w0,color='k')
        dumpfiles(x,'freqpeaks_{}_{}'.format(Nperw,fbins))
        dumpfiles(y,'xi2peaks_{}_{}'.format(Nperw,fbins))
        dumpfiles(z,'growthpeaks_{}_{}'.format(Nperw,fbins))

    if plot_3D:
        # plot 3d map of peak growth rates (messy)
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
        # plt.show()
    if plot_2D:
        # plot integer deuteron harmonics
        for i in range(0,maxnormf+1,1):
            ax2d.axvline(i,color='darkgrey',linestyle='--')
        # plot freqs vs. xi2 and gamma as color
        for i in range(len(z)):
            im2d = ax2d.scatter(x[i],y[i],c=z[i],marker='s',s=25,vmin=0,vmax=0.15,cmap='summer',edgecolor='none') # 'summer'
        # plot two trend lines
        ax2d.plot([4.5,10],[1.05,0],linestyle='--',color='w')#,alpha=0.75)
        ax2d.annotate('A',xy=(6,0.6),xycoords='data',color='w')
        ax2d.plot([9.5,10],[1,0],linestyle='--',color='w')#,alpha=0.75)
        ax2d.annotate('B',xy=(9.7,0.8),xycoords='data',color='w')
        # colorbar, labelling and other formatting
        cbar2d = fig2d.colorbar(im2d)
        cbar2d.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
        ax2d.set_xlabel('Frequency'+' '+r'$[\Omega_i]$',**tnrfont)
        ax2d.set_ylabel(r'$\xi_T$',**tnrfont)
        ax2d.set_xlim(0,maxnormf)
        ax2d.set_ylim(0,np.max(XI2))
        fig2d.savefig(home+'freq_xiT_growth_peaks_Nperw_{}.png'.format(Nperw),bbox_inches='tight')
    if plot_hm:
        # plot integer deuteron harmonics
        for i in range(0,maxnormf+1,1):
            axhm.axvline(i,color='darkgrey',linestyle='--')
        imhm = axhm.imshow(growth_hm,origin='lower',aspect='auto',extent=[0,maxnormf,0,np.max(XI2)],cmap='summer',interpolation='gaussian',clim=(0,0.15))
        # cbar = fig.colorbar(im, orientation='vertical', pad=0.1)
        cbarhm = fighm.colorbar(imhm)
        cbarhm.ax.set_ylabel('Growth Rate'+' '+r'$[\Omega_i]$',**tnrfont,rotation=90.,labelpad=20)
        # for i in range(len(x)):
            # im = axhm.scatter(x[i],y[i],facecolor='none',marker='s',s=12,edgecolor='k',alpha=0.5)
        axhm.set_xlabel('Frequency'+' '+r'$[\Omega_i]$',**tnrfont)
        axhm.set_ylabel(r'$\xi_T$',**tnrfont)
        fighm.savefig(home+'freq_xiT_growth_feathered.png',bbox_inches='tight')
        # plt.show()
    plt.clf()
    return x, y, z, growth_hm

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
    plot_k2d_growth(kpara,kperp,dw,w,norm=[w0,k0],clims=(0,0.015),cmap=cmap,labels=[l1,l2],contours=False,\
                    rowlim=(-maxnormkpara,maxnormkpara),collim=(0,maxnormkperp),maxnormf=maxnormkperp,dump=True) # load
    plot_frq_growth(w,dw,kpara,maxnormf=maxnormkperp,maxnormg=0.015,norm=[w0,k0],labels=[l3,l4])
    plot_frq_growth_angles(kpara,kperp,dw,w,rowlim=(-4*k0,4*k0),collim=(0,15*k0),norm=[w0,k0],clims=(0,0.012))
    plot_frq_kpara(kpara,w,dw,maxnormf=maxnormfreq,norm=[w0,k0],cmap=cmap,labels=[l3,l2])
    plot_frq_kperp(kperp,w,dw,maxk=maxnormkperp,maxnormf=maxnormfreq,norm=[w0,k0],cmap=cmap,labels=[l3,l1])
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
    sollocs = [i+'/' for i in os.listdir(home) if 'run' in i]
    return np.sort(sollocs)

#-#-#
if __name__ == '__main__':
    ## BODY ## 
    
    homes = {
        'lowkperp_T':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Tritons/",
        'highkperp_T':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Tritons_high_kperp/",
        'lowkperp_noT':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons/",
        'highkperp_noT':"/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons_high_kperp/",        
    }
    
    """
    :: examples  :: 
        # one file
    LOC_FILE = homes.get(NAME) # if using dict
    LOC_FILE = '/name/of/dir/to/solutions/file/here'
    data=read_all_data(loc=LOC_FILE)
    make_all_plots(alldata=data)

        # multiple files
    para_calc(LOC_FILE,plot=True)
    """

    # # p-B11
    # homeloc = '/home/space/phrmsf/Documents/ICE_DS/p_B11/'
    # sollocs = [i+'/' for i in os.listdir(homeloc) if 'run' in i]
    # XI2 = [100*float(i.split('_')[2]) for i in sollocs]
    # # sort sollocs and XI2 based on XI2
    # XI2 = np.array([x/100 for x,_ in sorted(zip(XI2,sollocs))])
    # sollocs = np.array([x for _,x in sorted(zip(XI2,sollocs))])
    # for i in range(len(sollocs)):
    #     os.chdir(homeloc+sollocs[i])
    #     data=read_all_data(homeloc+sollocs[i])
    #     make_all_plots(data)
    # _=get_peak_frqs(home=homeloc,sollocs=sollocs,loop=XI2,maxnormf=15,plot_2D=True,plot_hm=True)
    # sys.exit()

    # # D-T
    # homeloc = homes.get('lowkperp_noT')
    # sollocs = getsollocs(homeloc)
    # XI2 = [float(i.split('_')[8]) for i in sollocs]
    # # sollocs = [getsollocs(homeloc)[3]]
    # angles = [85.0,88.0,89.0,90.0,91.0,92.0,95.0]
    # for angle in angles:
    #     plot_peak_frq_angle(homeloc,sollocs,XI2=XI2,wmax=10,rowlim=(-4,4),collim=(0,15),angle=angle,\
    #                         plateau_size=0.5,Nperw=10)

    # sollocs = ['run_2.07_0.0_-0.646_0.01_0.01_15.0_3.5_0.0_1.0_4.0_1.7e19_0.00015_1024/']
    # for i in range(len(sollocs)):
    #     os.chdir(homeloc+sollocs[i])
    #     w0,k0,w,dw,kpara,kperp = read_all_data(loc=homeloc+sollocs[i])
    #     k2d = np.sqrt(kpara**2 + kperp**2)
    #     Z = make2D(w,k2d,dw,rowlim=(0,15*w0),collim=(0,np.max(k2d)),bins=(512,512))
    #     plt.imshow(Z/w0,aspect='auto',origin='lower',extent=[0,np.max(k2d)/k0,0,15],cmap='summer')
    #     plt.show()
    #     sys.exit()
    #     # Z = make2D(kpara,kperp,dw,rowlim=(-4*k0,4*k0),collim=(0,15*k0))
    #     plot_frq_growth_angles(kpara,kperp,dw,w,wmax=11,rowlim=(-4*k0,4*k0),collim=(0,15*k0),norm=[w0,k0],clims=(0,0.045))
    # sys.exit()

    homeloc=homes.get('highkperp_T')
    sollocs = getsollocs(homeloc)
    XI2 = [i/200 for i in range(0,200,5)]
    print(XI2)
    # XI2 = [i/200 for i in range(0,200,5)]
    # XI2 = [i/100 for i in range(45,95,5)]
    # XI2.append(0)
    # XI2 = np.sort(XI2)
    
    # XI2 = [i/100 for i in range(45,95,5)]
    # XI2.append(0)
    # XI2 = np.sort(XI2)
    # plot_k2d_growth_combined(sollocs=sollocs,loop=XI2,cmap='summer',clims=(0,0.15),collim=(0,25))
    
    sollocs = ['run_2.07_{}_-0.646_0.01_0.01_25.0_3.5__1.0_4.0_1.7e19_0.00015_2048/'.format(i) for i in XI2]
    get_peak_frqs(homeloc,sollocs=sollocs,XI2=XI2,maxnormf=18,plot_2D=True)#,plot_hm=True 
    # for sol in sollocs: # all runs (not in order) use list comprehension for ordered and specific XI2  
    #     XI2.append(float(sol.split('run')[1].split('_')[2]))
    # XI2 = np.array(XI2)
    # kd.plot_doppler_kernel(sollocs=[sollocs[-1]],labels=[XI2[-1]])
    # ld.plot_all(sollocs=sollocs,labels=XI2,name='maxvdop_xiT',plot_angles=True,xlabel=r'$\xi_T$') # [str(i*100)+'%' for i in XI2])
    # # plot peak frequencies # if plot heatmap then only use xi2=5, 10, 15...95 

    sys.exit()

    # # D runs
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
        pitches.append(float(sol.split('_')[3]))
        energies.append(float(sol.split('_')[7]))
    labels=[pitches,energies]
    names=['maxvdop_pitches','maxvdop_energy']
    for i in range(2):
        ld.plot_all(homeloc=homeloc,sollocs=sollocs,labels=labels[i],plot_grad=True,name=names[i],xlabel=r'$E_\alpha$')
    sys.exit()


"""
    TODO ; 
"""
