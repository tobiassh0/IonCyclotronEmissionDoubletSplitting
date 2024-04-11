
from makeplots import *
import numpy as np
import matplotlib.pyplot as plt

def line_integrate(Z,xposini=[],angles=None):
    """
        In:
            Z : 2d matrix of data points to calculate integral over
            xposini : x positions, in units of x-axis, as to where take line integral
            angles : angles over which to loop, defaults to 0 -> 90deg
        Out:
            dopGang : array of angles per xposini of the total intensity along a line, 2d shape of 
            dopGangmax : array of angles per xposini that correspond to the strongest total intensity along the line 
    """
    # normalise Z so colour map between 0 and 255
    
    if not angles:
        angles = np.linspace(0,np.pi,100)
    dopGang=np.zeros((len(xposini),len(angles))) ; dopGangmax=np.zeros(len(xposini))
    # loop through angles
    for i in range(len(xposini)):
        intensity = np.zeros(len(angles))
        for j in range(len(angles)):
            # find min max y and x bounds of line

            # find data points along line

            # summate all points 
            intensity[j] = 
        lines_intensity[i,:] = intensity
        dopGangmax[i] = angles[np.argmax(intensity)]

    return dopGang, dopGangmax

def get_line_doppler(rowdata,coldata,data,datanorm=1,norm=[1,1],rowlim=(None,None),collim=(None,None),thresh_growth=0.001):
    # get 2d map of y vs x (row vs col)
    Z,extents=make2D(rowdata,coldata,data,rowlim=rowlim,collim=collim,limits=True,dump=False,name='',bins=(512,512))
    
    # normalise extents
    extents=[extents[0]/norm[0],extents[1]/norm[0],extents[2]/norm[1],extents[3]/norm[1]]

    # remove anomalous
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i,j]/datanorm < thresh_growth:
                Z[i,j] = 0
    plt.imshow(Z,origin='lower',interpolation='none',aspect='auto',cmap='summer',extent=extents)# ; plt.show() ; sys.exit()

    # find gradients using integral along line method
    levels = range(0,16)
    dopGangle = line_integrate(Z,xposini=levels)

    # tan kernel angles
    tandopGangle = np.tan(dopGangle)
    return None

def plot_doppler_line(sollocs=[''],labels=['']):
    for i in range(len(sollocs)):
        w0,k0,w,dw,kpara,kperp=read_all_data(loc=sollocs[i])
        dkparadw = get_line_doppler(kpara,w,dw,datanorm=w0,rowlim=(-4*k0,4*k0),collim=(0,15*w0),norm=[w0,k0])
        dkparadw = dkparadw.flatten()
        counts,bins,_=plt.hist(dkparadw,bins=1000,range=(-10,-0.1),density=True) # np.log10
        dsv = bins[np.argmax(counts)] # doppler shift velocity in units of vA
        warr = np.linspace(0,16,100)
        for i in range(0,15):
            plt.plot(warr,(warr-i)*dsv,color='k',linestyle='--',alpha=0.5)
            plt.text(i,0.1,"{:.2f}".format(i))

    line_integrate()
    return None
