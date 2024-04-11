
from makeplots import *
import numpy as np
import matplotlib.pyplot as plt

# 2d kernel convolution with either Sobel or Scharr kernel, returns gradient and magnitude array 
def Kernel(img,kernel='sobel',plot=True):
	# convert img to 0-255 color
	img = 255*(img/np.nanmax(img))
	if kernel == 'scharr':
		Gx=np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
		Gy=np.array([[3,10,3],[0,0,0],[-3,-10,-3]])
	if kernel == 'sobel':
		Gx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
		Gy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	if kernel == 'custom':
		Gx=np.array([[10,0,-10],[3,0,-3],[10,0,-10]])
		Gy=np.array([[10,3,10],[0,0,0],[-10,-3,-10]])
	# magnitude and angle arrays
	kGmag = np.zeros(img.shape)
	kGangle = np.zeros(img.shape)
	# position relative to centre cell
	for i in range(img.shape[0]-1):
		for j in range(img.shape[1]-1):
			NORTH=True; SOUTH=True; WEST=True; EAST=True
			# four conditions on square image
			if i+1 > img.shape[0]: # reached end of img EAST
				EAST = False
			if i-1 < 0: # reached end of img WEST
				WEST = False
			if j+1 > img.shape[1]: # reached end of img NORTH
				NORTH = False
			if j-1 < 0: # reached end of img SOUTH
				SOUTH = False
			##
			if NORTH: N = img[i,j+1]
			else: N = 0
			#
			if SOUTH: S = img[i,j-1]
			else: S = 0
			#
			if WEST: W = img[i-1,j]
			else: W = 0
			#
			if EAST: E = img[i+1,j]
			else: E = 0
			#
			if NORTH and WEST: NW = img[i-1,j+1]
			else: NW = 0
			#
			if NORTH and EAST: NE = img[i+1,j+1]
			else: NE = 0
			#
			if SOUTH and WEST: SW = img[i-1,j-1]
			else: SW = 0
			#
			if SOUTH and EAST: SE = img[i+1,j-1]
			else: SE = 0
			## 
			timg = np.array([[NW,N,NE],[W,img[i,j],E],[SW,S,SE]])
			kG = np.array([np.sum(Gx*timg),np.sum(Gy*timg)])
			kGmag[i,j] = np.sqrt(kG[0]**2 + kG[1]**2)
			kGangle[i,j]= np.arctan2(kG[1],kG[0]) # radians between +- pi
	kGangle = np.nan_to_num(kGangle,posinf=np.nan,neginf=np.nan) # change +-inf vals to nan
	kGmag = np.nan_to_num(kGmag,posinf=np.nan,neginf=np.nan) 	 # change +-inf vals to nan		
	return kGmag, kGangle

# get kernel extracted gradient of kpara vs freq plots 
def get_doppler_kernel(rowdata,coldata,data,datanorm=1,norm=[1,1],rowlim=(None,None),collim=(None,None),thresh_growth=0.001):
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

    # find gradients using kernel method
    dopGmag, dopGangle = Kernel(Z, plot=False)

    # # convert all angles to negative (easier to calculate gradient), pi out of phase
    # for i in range(dopGangle.shape[0]):
    #     for j in range(dopGangle.shape[1]):
    #         if dopGangle[i,j] > 0:
    #             dopGangle[i,j]-=np.pi

    # tan kernel angles
    tandopGangle = np.tan(dopGangle)

    # remove tan(angles) larger than some thresh value (20)
    for i in range(tandopGangle.shape[0]):
        for j in range(tandopGangle.shape[1]):
            if np.abs(tandopGangle[i,j]) > 20:
                tandopGangle[i,j] = np.nan
    dx = ((collim[1]-collim[0])/Z.shape[1])/norm[0]
    dy = ((rowlim[1]-rowlim[0])/Z.shape[0])/norm[1]
    print(dx,dy,dy/dx)

    fig,axs=plt.subplots(figsize=(8,10),nrows=3)
    axs[0].imshow(dopGmag,origin='lower',interpolation='none',aspect='auto',cmap='gray',extent=extents)
    axs[1].imshow(dopGangle,origin='lower',interpolation='none',aspect='auto',cmap='bwr',extent=extents)
    axs[2].imshow(tandopGangle,origin='lower',interpolation='none',aspect='auto',cmap='bwr',clim=(-3,3),extent=extents)
    for ax in axs: 
        ax.set_xlim(0,15)
        ax.set_ylim(-1.5,1.5)
    plt.show() ; sys.exit()
    return tandopGangle #*(dy/dx) # units of gradient on plane (1/VA)

# plot kernel extracted gradient of kpara vs freq plots 
def plot_doppler_kernel(sollocs=[''],labels=['']):
    # load kpara v freq 2d map
    # calculate doppler shift
    # plot as function of labels
    for i in range(len(sollocs)):
        w0,k0,w,dw,kpara,kperp=read_all_data(loc=sollocs[i])
        dkparadw = get_doppler_kernel(kpara,w,dw,datanorm=w0,rowlim=(-4*k0,4*k0),collim=(0,15*w0),norm=[w0,k0])
        dkparadw = dkparadw.flatten()
        counts,bins,_=plt.hist(dkparadw,bins=1000,range=(-10,-0.1),density=True) # np.log10
        dsv = bins[np.argmax(counts)] # doppler shift velocity in units of vA
        warr = np.linspace(0,16,100)
        for i in range(0,15):
            plt.plot(warr,(warr-i)*dsv,color='k',linestyle='--',alpha=0.5)
            plt.text(i,0.1,"{:.2f}".format(i))
        print(counts,bins,dsv)
        plt.xlim(0,15)
        plt.ylim(-4,4)
        plt.show()
    return None