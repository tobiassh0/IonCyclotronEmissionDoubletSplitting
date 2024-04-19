
from makeplots import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.odr as odr


def func_linear(p,x):
	m,c = p
	return m*x + c

def ODR_fit(x,y,sx=[],sy=[],beta0=[1,0],curve='linear'):
	"""
	fit a linear/quadratic function based off of the ODR approach in the scipy package
	(https://docs.scipy.org/doc/scipy/reference/odr.html)
		
		IN
			x : x array of points to fit
			y : y array "				"
			sx, sy : error arrays of the x and y values
			beta0 : an estimation on the parameters to fit the curve with (grad, intercept)
			curve : name of the curve to fit (linear,quad)
		OUT
			params : the parameters of the curve defined
			params_err : errors on the determined parameters
	"""
	if curve == 'linear':
		func = func_linear
	elif curve=='quad':
		func = func_quad
	else:
		print('## ERROR ## :: curve function has not been defined')

	# check if errors are present, use std otherwise
	if sx == []:
		sx = np.std(x)
	if sy == []:
		sy = np.std(y)

	# fit ODR line of best fit with errors
	linear_model = odr.Model(func)
	data = odr.RealData(x=x,y=y,sx=sx,sy=sy)
	myodr = odr.ODR(data, linear_model, beta0=[1,-0.1])
	myout = myodr.run()
	# myout.pprint()

	params = myout.beta
	params_err = myout.sd_beta
	return params, params_err

def plotSemiCirclePolarCoords(ax,angles,data):
    """
        In:
            angles : the angle corresponding to the data
            data : in effect the radius of the plot
    """
    if np.max(np.abs(angles)) != np.pi:
        print('# ERROR # :: Not a semi-circle') # TODO; make it so will plot over all angles provided
    data_norm = data/np.max(data)
    # plot as function of angles
    ax.plot(data_norm*np.sin(angles),data_norm*np.cos(angles))
    # plot integer radius lines (1/4, 1/2, 3/4 and 1 * maxdata)
    for i in [0.25,0.5,0.75,1]:
        radius = i
        rangles = np.linspace(-np.pi,0,100)
        if i == 1:
            alpha = 1
            linestyle='-'
        else:
            alpha = 0.5
            linestyle='--'
        ax.plot(radius*np.sin(rangles),radius*np.cos(rangles),color='darkgrey',linestyle=linestyle,alpha=alpha)
        # ax.annotate("{:.3f}".format(radius*np.max(data))+r'$\Omega_i$',xy=(-radius,0.1),xycoords='data')
    # plot constant angles
    for j in [0,1/8,0.25,1/3,0.5,2/3,0.75,7/8,1]:
        radius = np.array([0,1])
        ax.plot(radius*np.sin(j*(-np.pi)),radius*np.cos(j*(-np.pi)),linestyle='--',color='darkgrey',alpha=0.75)
        ax.annotate("{:.1f}".format(-j*np.pi*180/np.pi)+r"$^\circ$",xy=(np.sin(j*(-np.pi)),np.cos(j*(-np.pi))),xycoords='data')
    ax.set_xlim(-1.1,0)
    ax.set_ylim(-1.1,1.1)
    ax.tick_params(axis='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labeltop=False,labelleft=False,labelright=False)
    return ax

def LineBoxIntersection(Ys,Ye,Xn,Yn,XL,YL,theta):
    """
    Find intersection points between a line with eqn (Y-Yn)=(X-Xn)/tan(theta)
        In:
            Ys, Ye : start and end point of line on y-axis
            Yn, Xn : offset of line mid-point
            YL, XL : total length of box in Y and X (pixel coordinates)
            theta : positive angle clockwise from north, represents 1/gradient of line
    """
    # +ve angle clockwise from north
    Xs = (Ys-Yn)*np.tan(theta)+Xn ; Xe = (Ye-Yn)*np.tan(theta)+Xn
    # check initial co-ordinates, if wrong then use other bounds 
    if Xs < 0:
        Xs = 0
        Ys = Yn+((Xs-Xn)/np.tan(theta))
    if Xs > XL:
        Xs = XL
        Ys = Yn+((Xs-Xn)/np.tan(theta))
    if Xe > XL:
        Xe = XL
        Ye = Yn+((Xe-Xn)/np.tan(theta)) 
    if Xe < 0:
        Xe = 0
        Ye = Yn+((Xe-Xn)/np.tan(theta)) 
    xlim = [Xs,Xe] ; ylim = [Ys,Ye]
    return np.around(xlim), np.around(ylim)

def line_integrate(Z,xposini=[],angles=[],rowlim=(-4,4),collim=(0,15),norm=[1,1],lsize=None,label='',plot=False):
    """
        In:
            Z : 2d matrix of data points to calculate integral over
            xposini : x positions, in units of x-axis, as to where take line integral
            angles : angles over which to loop, defaults to 0 -> 90deg
        Out:
            dopangmax : array of angles per xposini that correspond to the strongest total intensity along the line 
    """
    Ny, Nx = Z.shape
    if not lsize:
        lsize = np.sqrt(Nx**2+Ny**2) # max length of line within imshow square (px coords)
    if angles==[]:
        angles = np.linspace(0,-2*np.pi,100)
    dopmaxang=np.zeros(len(xposini))
    zi = np.zeros((len(angles),int(lsize)))
    intensity = np.zeros((len(xposini),len(angles)))
    # limits of box, normalised units
    rowlim = np.array(rowlim) ; collim = np.array(collim)
    wmin, wmax = (collim/norm[0])
    kparamin, kparamax = (rowlim/norm[1])
    extents = [wmin,wmax,kparamin,kparamax]
    # transform to pixel coords
    px_xposini = (np.array(xposini)/(wmax-wmin))*Nx
    if len(xposini)//4 < 1:
        nrows = 1
    else:
        nrows = len(xposini)//4
    if plot:
        fig_semi,ax_semi=plt.subplots(figsize=(8,10),nrows=nrows,ncols=4)
        ax_semi=ax_semi.ravel()
        fig,ax=plt.subplots(nrows=2)
    # loop through angles
    for i in range(len(px_xposini)):
        Xn = px_xposini[i] # number of cells in x
        Yn = Ny/2 # centred on kpara = 0
        for j in range(len(angles)):
            # find intersection points between line and box (pixel coords)
            Ystart = Ny ; Yend = 0
            xlim, ylim = LineBoxIntersection(Ystart,Yend,Xn,Yn,Nx,Ny,angles[j])
            # find data points along line
            x = np.linspace(xlim[0],xlim[1],int(lsize))
            y = np.linspace(ylim[0],ylim[1],int(lsize))
            zi[j,:] = scipy.ndimage.map_coordinates(Z/norm[0],np.vstack((y,x))) # normalised growth rates
            # convert all to real coordinates
            xlim = (collim[0]/norm[0])+xlim*((collim[1]-collim[0])/norm[0])/Nx
            ylim = (rowlim[0]/norm[1])+ylim*((rowlim[1]-rowlim[0])/norm[1])/Ny
            # summate all points # "growth per (dkpara,dw) cell" 
            intensity[i,j] = np.sum(zi[j,:])/len(zi[j,:]) # normalise to number of cells along line
            # example plot
            if plot:
                ax[0].imshow(Z,origin='lower',interpolation='none',aspect='auto',cmap='summer',extent=extents)
                ax[0].plot(xlim,ylim,'ko--',alpha=j/len(angles)) # color=colors[j])
                ax[1].plot(np.linspace(xlim[0],xlim[1],len(zi[j,:])),zi[j,:],alpha=j/len(angles)) # ,color=colors[j])#

        # find maximum intensity as a function of angle per xposini
        maxintarg = np.argmax(intensity[i,:])
        maxang = angles[maxintarg]
        print('Max intensity angle [rad]: ',maxang)
        dopmaxang[i] = maxang
        if plot:
            ax[0].set_ylim(kparamin,kparamax) ; ax[0].set_xlim(wmin,wmax)
            ax[0].set_xlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
            ax[0].set_ylabel('Parallel Wavenumber '+r'$[\Omega_i/V_A]$',**tnrfont)
            ax[1].set_xlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
            ax[1].set_ylabel('Growth Rate '+r'$[\Omega_i]$',**tnrfont)
            fig.savefig('all_lines_XI2_{}.png'.format(label))
            ax_semi[i] = plotSemiCirclePolarCoords(ax_semi[i],angles,intensity[i,:])
            ax_semi[i].annotate("{:.0f}".format(xposini[i])+r"$\Omega_i$",xy=(0.05,0.05),xycoords='axes fraction',va='bottom',ha='left')
    # determine if want to plot
    if plot:
        fig_semi.savefig('semicircle_intensity_angle_XI2_{}.png'.format(label),bbox_inches='tight')
        plt.clf()
    return intensity, dopmaxang

def plotSumGrowthAngle(angles,intensity,levels,label):
    # plot mock "hist"
    fig,ax=plt.subplots(figsize=(8,6))
    for i in range(intensity.shape[0]):
        ax.fill_between(angles,intensity[i,:],color='r',alpha=1/len(levels))
    ax.set_xlim(np.min(angles),np.max(angles))
    ax.set_xlabel(r'$\theta$'+' '+'[rad]',**tnrfont)
    ax.set_ylabel(r'Growth Rate per cell '+r'$[\Omega_i]$',**tnrfont)
    ax.set_ylim(0,0.012) # don't plot negative growths # hardcoded
    fig.savefig('intensity_angle_XI2_{}.png'.format(label),bbox_inches='tight')
    plt.clf()
    return None

def get_line_doppler(rowdata,coldata,data,datanorm=1,norm=[1,1],rowlim=(None,None),collim=(None,None),thresh_growth=0.001,\
                    label='',levels=range(0,16),angles=[]):
    # get 2d map of y vs x (row vs col)
    Z,extents=make2D(rowdata,coldata,data,rowlim=rowlim,collim=collim,limits=True,dump=False,name='',bins=(700,512))
    # normalise extents
    extents=[extents[0]/norm[0],extents[1]/norm[0],extents[2]/norm[1],extents[3]/norm[1]]
    # remove anomalous
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i,j]/datanorm < thresh_growth:
                Z[i,j] = 0
    # plt.imshow(Z,origin='lower',interpolation='none',aspect='auto',cmap='summer',extent=extents) ; plt.show() ; sys.exit()
    # find gradients using integral along line method
    intensity, dopmaxang = line_integrate(Z,xposini=levels,angles=angles,rowlim=rowlim,collim=collim,norm=norm,label=label)
    # plotSumGrowthAngle(angles,intensity,levels,label)
    return intensity, dopmaxang

def plot_doppler_line(sollocs=[''],labels=[],levels=range(0,16),angles=[],plot_angles=False,plot_grad=True,name=''):
    if name == '':
        if plot_angles: name = 'maxangle_xiT'
        if plot_dop: name = 'maxvdop_xiT'
    if plot_grad == plot_angles:
        plot_angles = False
        plot_grad = True

    if angles == []:
        angles = np.linspace(-0.6,0,100) # limit to find dopplershift
    if len(sollocs) != len(labels):
        labels = np.arange(0,len(sollocs),1)
    dopmaxangles = np.zeros((len(sollocs),len(levels))) # max angle per label
    dopmeanangles= np.zeros(len(sollocs))
    dopstdangles = np.zeros(len(sollocs)) # std angle per label    
    allabels = np.zeros((len(sollocs),len(levels))) # 2d allabels
    fig,ax = plt.subplots(figsize=(8,6))
    for i in range(len(sollocs)):
        w0,k0,w,dw,kpara,kperp=read_all_data(loc=sollocs[i])
        intensity, dopmaxang=get_line_doppler(kpara,w,dw,datanorm=w0,rowlim=(-4*k0,4*k0),collim=(0,15*w0),\
                                                norm=[w0,k0],label=labels[i],levels=levels,angles=angles)
        # assign large arrays
        dopmaxangles[i,:] = dopmaxang
        allabels[i,:] = [labels[i] for r in range(len(dopmaxang))]
        dopmeanangles[i]= np.mean(dopmaxang)
        dopstdangles[i] = np.std(dopmaxang) # for all levels, given solloc

    if plot_angles:
        # fit straight line odr
        ax.scatter(allabels.flatten(),dopmaxangles.flatten(),color='k',alpha=0.25) #1/len(levels))
        params, params_err = ODR_fit(allabels.flatten(),dopmaxangles.flatten(),beta0=[-0.15,-0.4],curve='linear')
        # params, params_err = ODR_fit(labels,dopmeanangles,sy=dopstdangles,beta0=[-0.15,-0.4],curve='linear')
        print('grad and intercept :',params,'errors: ',params_err)
        ax.plot(labels,labels*params[0]+params[1],color='r',linestyle='--')
        ax.set_ylabel(r'$\theta_{max}$'+' '+'[rad]',**tnrfont)
    if plot_grad:
        ax.scatter(allabels.flatten(),np.tan(dopmaxangles.flatten()),color='k',alpha=0.25) #1/len(levels))
        ax.set_ylabel(r'$v_{dop}$'+' '+'['+r'$V_A$'+']',**tnrfont)
    ax.set_xlim(labels[0],labels[-1])
    ax.set_xlabel(r'$\xi_T$',**tnrfont)
    plt.show()
    fig.savefig(name+'.png',bbox_inches='tight')
    return None
