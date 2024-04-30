
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.odr as odr
from makeplots import *

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

def plotSemiCirclePolarCoords(angles,intensity,levels,label):
    """
        In:
            axes : ravelled ax
            angles : the angle corresponding to the data
            intensity : in effect the radius of the plot, shape [len(angles), len(extracted growth rates)]
            levels : number of plots, decided by # harmonics 
            label : label (or rather name) of the plot (typically xiT)
    """
    if len(levels)//4 < 1:
        nrows = 1
    else:
        nrows = len(levels)//4

    fig,axes=plt.subplots(figsize=(8,10),nrows=nrows,ncols=4)
    axes=axes.ravel()

    if np.max(np.abs(angles)) != 180.0:
        print('# ERROR # :: Not a semi-circle') # TODO; make it so will plot over all angles provided
    for i in range(intensity.shape[0]):
        data = intensity[i,:]
        ax = axes[i] # already axes.ravel()
        data_norm = data/np.max(data)
        # plot as function of angles
        ax.plot(data_norm*np.sin(angles*np.pi/180),data_norm*np.cos(angles*np.pi/180))
        # plot integer radius lines (1/4, 1/2, 3/4 and 1 * maxdata)
        for i in [0.25,0.5,0.75,1]:
            radius = i
            rangles = np.linspace(-180,0,100)
            if i == 1:
                alpha = 1
                linestyle='-'
            else:
                alpha = 0.5
                linestyle='--'
            ax.plot(radius*np.sin(rangles*np.pi/180),radius*np.cos(rangles*np.pi/180),color='darkgrey',linestyle=linestyle,alpha=alpha)
            # ax.annotate("{:.3f}".format(radius*np.max(data))+r'$\Omega_i$',xy=(-radius,0.1),xycoords='data')
        # plot constant angles
        for j in [0,1/8,0.25,1/3,0.5,2/3,0.75,7/8,1]:
            radius = np.array([0,1])
            ax.plot(radius*np.sin(j*(-np.pi)),radius*np.cos(j*(-np.pi)),linestyle='--',color='darkgrey',alpha=0.75)
            ax.annotate("{:.1f}".format((-j*np.pi)*180/np.pi)+r"$^\circ$",xy=(np.sin(j*(-np.pi)),np.cos(j*(-np.pi))),xycoords='data')
        ax.set_xlim(-1.1,0)
        ax.set_ylim(-1.1,1.1)
        ax.tick_params(axis='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labeltop=False,labelleft=False,labelright=False)
        ax.annotate("{:.0f}".format(levels[i])+r"$\Omega_i$",xy=(0.05,0.05),xycoords='axes fraction',va='bottom',ha='left')
    fig.savefig('semicircle_intensity_angle_XI2_{}.png'.format(label),bbox_inches='tight')
    plt.clf()
    return None

def plot_doppler_line(angles,Zij,levels,level=0):
    ith_level = int(len(levels)*level/levels[-1])
    fig_line,ax_line=plt.subplots(figsize=(4,int(1.5*len(angles))),nrows=len(angles),sharex=True)
    ax_line[0].set_xlim(wmin,wmax)
    # ax_line[-1].imshow(Z/norm[0],origin='lower',interpolation='none',aspect='auto',cmap='summer',extent=extents)
    # loop over angles
    for j in range(len(angles)):
        ax_line[j].set_ylim(0) # no negative growths
        ax_line[j].annotate(rangles[j],xy=(0.1,0.9),xycoords='axes fraction',va='top')
        ax_line[j].locator_params(axis='y',nbins=4)
        ax_line[j].plot(np.linspace(xlim[0],xlim[1],len(Zii[ith_level,j,:])),Zij[ith_level,j,:],color=colors[j],alpha=1/len(px_xposini))
    ax_line[-1].set_xlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
    fig_line.supylabel('Growth rate '+r'$[\Omega_i]$',**tnrfont,x=-0.1)
    fig_line.savefig('lines_XI2_{}_n_{}_.png'.format(label,xposini[-1]),bbox_inches='tight')
    pass

def LineBoxIntersection(Ys,Ye,Xn,Yn,XL,YL,theta):
    """
    Find intersection points between a line with eqn (Y-Yn)=(X-Xn)/tan(theta)
        In:
            Ys, Ye : start and end point of line on y-axis
            Yn, Xn : offset of line mid-point
            YL, XL : total length of box in Y and X (pixel coordinates)
            theta : positive angle clockwise from north [deg], represents 1/gradient of line
    """
    # +ve angle clockwise from north
    Xs = (Ys-Yn)*np.tan(theta*np.pi/180)+Xn ; Xe = (Ye-Yn)*np.tan(theta*np.pi/180)+Xn
    # check initial co-ordinates, if wrong then use other bounds 
    if Xs < 0:
        Xs = 0
        Ys = Yn+((Xs-Xn)/np.tan(theta*np.pi/180))
    if Xs > XL:
        Xs = XL
        Ys = Yn+((Xs-Xn)/np.tan(theta*np.pi/180))
    if Xe > XL:
        Xe = XL
        Ye = Yn+((Xe-Xn)/np.tan(theta*np.pi/180)) 
    if Xe < 0:
        Xe = 0
        Ye = Yn+((Xe-Xn)/np.tan(theta*np.pi/180)) 
    xlim = [Xs,Xe] ; ylim = [Ys,Ye]
    return np.around(xlim), np.around(ylim)

def line_integrate(Z,xposini=[],angles=[],rowlim=(-4,4),collim=(0,15),norm=[1,1],lsize=None,label='',colorarr=True):
    """
        In:
            Z : 2d matrix of data points to calculate integral over
            xposini : x positions, in units of x-axis, as to where take line integral
            angles : angles over which to loop, defaults to 0 -> 90deg
        Out:
            intensity : summated line intensity normalised to the number of cells within the line (sum(zi)/len(zi))
            dopangmax : array of angles per xposini that correspond to the strongest total intensity along the line
            Zij       : 3d array of the line intensity [shape of (len(xposini), len(angles), len(zi))]
    """
    Ny, Nx = Z.shape
    if not lsize:
        lsize = np.sqrt(Nx**2+Ny**2) # max length of line within imshow square (px coords)
    # set angle array if empty
    if angles==[]:
        angles = np.linspace(0,-180,360)
    # color array
    if colorarr:
        colors = plt.cm.rainbow(np.linspace(0,1,len(angles)))
    else:
        colors = ['k']*len(angles)

    # setup empty arrays
    dopmaxang=np.zeros(len(xposini))
    zi = np.zeros((len(angles),int(lsize)))
    intensity = np.zeros((len(xposini),len(angles)))
    rangles = np.zeros(len(angles)) # real angles
    Zij = np.zeros((len(xposini),len(angles),int(lsize)))
    
    # limits of box, normalised units
    rowlim = np.array(rowlim) ; collim = np.array(collim)
    wmin, wmax = (collim/norm[0])
    kparamin, kparamax = (rowlim/norm[1])
    extents = [wmin,wmax,kparamin,kparamax]
    
    # transform to pixel coords
    px_xposini = (np.array(xposini)/(wmax-wmin))*Nx

    # loop through angles
    for i in range(len(px_xposini)):
        Xn = px_xposini[i] # number of cells in x
        Yn = Ny/2 # centred on kpara = 0
        for j in range(len(angles)):
            # find intersection points between line and box (pixel coords)
            Ystart = Ny ; Yend = 0
            xlim, ylim = LineBoxIntersection(Ystart,Yend,Xn,Yn,Nx,Ny,angles[j])
            # lsize = np.sqrt((xlim[-1]-xlim[0])**2+(ylim[-1]-ylim[0])**2)
            # find data points along line
            x = np.linspace(xlim[0],xlim[1],int(lsize))
            y = np.linspace(ylim[0],ylim[1],int(lsize))
            zi[j,:] = scipy.ndimage.map_coordinates(Z/norm[0],np.vstack((y,x))) # normalised growth rates
            Zij[i,j,:]=zi[j,:]
            # convert all to real coordinates
            xlim = (collim[0]/norm[0])+xlim*((collim[1]-collim[0])/norm[0])/Nx
            ylim = (rowlim[0]/norm[1])+ylim*((rowlim[1]-rowlim[0])/norm[1])/Ny
            rangles[j] = np.arctan((xlim[-1]-xlim[0])/(ylim[-1]-ylim[0]))*180/np.pi # [deg]
            # summate all points # "growth per (dkpara,dw) cell" 
            tzi = zi[j,:]
            tzi[tzi < 0] = 0 # no negative growth rates affecting intensity extraction
            intensity[i,j] = np.sum(tzi)/len(tzi) # normalise to number of cells along line
            # # example plot
            # if plot:
            #     ax[0].imshow(Z,origin='lower',interpolation='none',aspect='auto',cmap='summer',extent=extents)
            #     ax[0].plot(xlim,ylim,color=colors[j],linestyle='--')#,alpha=1/len(angles))
            #     ax[1].plot(np.linspace(xlim[0],xlim[1],len(zi[j,:])),zi[j,:],color=colors[j],label=rangles[j])#,alpha=1/len(angles))

        # find maximum intensity as a function of angle per xposini
        maxintarg = np.argmax(intensity[i,:])
        maxang = rangles[maxintarg]
        print('Max intensity angle [deg]: ',maxang)
        dopmaxang[i] = maxang
        # # example plot
        # ax[0].set_ylim(kparamin,kparamax) ; ax[0].set_xlim(wmin,wmax)
        # # ax[0].set_xlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
        # ax[0].set_ylabel('Parallel Wavenumber '+r'$[\Omega_i/V_A]$',**tnrfont)
        # ax[1].set_xlabel('Frequency '+r'$[\Omega_i]$',**tnrfont)
        # ax[1].set_ylabel('Growth Rate '+r'$[\Omega_i]$',**tnrfont)
        # ax[1].legend(loc='best')
        # fig.savefig('all_lines_XI2_{}.png'.format(label))
    return intensity, dopmaxang, Zij

def plotSumGrowthAngle(angles,intensity,levels,label):
    # plot mock "hist"
    fig,ax=plt.subplots(figsize=(8,6))
    for i in range(intensity.shape[0]):
        ax.fill_between(angles,intensity[i,:],color='r',alpha=1/len(levels))
    ax.set_xlim(np.min(angles),np.max(angles))
    ax.set_xlabel(r'$\theta$'+' '+'[deg]',**tnrfont)
    ax.set_ylabel(r'Growth Rate per cell '+r'$[\Omega_i]$',**tnrfont)
    ax.set_ylim(0)#,0.012) # don't plot negative growths # hardcoded
    fig.savefig('intensity_angle_XI2_{}.png'.format(label),bbox_inches='tight')
    plt.clf()
    return None

def get_line_doppler(rowdata,coldata,data,datanorm=1,norm=[1,1],rowlim=(None,None),collim=(None,None),thresh_growth=0.001,\
                    label='',levels=range(0,16),angles=[],bins=(512,512)):
    # get 2d map of y vs x (row vs col)
    Z,extents=make2D(rowdata,coldata,data,rowlim=rowlim,collim=collim,limits=True,dump=False,name='',bins=bins)
    # normalise extents
    extents=[extents[0]/norm[0],extents[1]/norm[0],extents[2]/norm[1],extents[3]/norm[1]]
    # remove anomalous
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i,j]/datanorm < thresh_growth:
                Z[i,j] = 0
    # plt.imshow(Z,origin='lower',interpolation='none',aspect='auto',cmap='summer',extent=extents) ; plt.show() ; sys.exit()
    # find gradients using integral along line method
    intensity, dopmaxang, Zij = line_integrate(Z,xposini=levels,angles=angles,rowlim=rowlim,collim=collim,norm=norm,label=label)
    return intensity, dopmaxang, Zij

def plot_all(sollocs=[''],labels=[],levels=range(0,16),angles=[],plot_angles=False,plot_grad=False,plot=False,name='',\
            xlabel=r'$\xi_T$',bins=(512,512),fit_ODR=True):
    # check all params at start
    if name == '':
        if plot_angles: name = 'maxangle_xiT'
        if plot_grad: name = 'maxvdop_xiT'
    # plotting v_dop gradient or angles
    if plot_grad == plot_angles:
        plot_angles = False
        plot_grad = True
    
    if angles == []:
        angles = np.linspace(-180,0,360) # limit to find dopplershift
    if len(sollocs) != len(labels):
        labels = np.arange(0,len(sollocs),1)
    # maximum length of line (No. cells)
    lsize = int(np.sqrt(bins[0]**2 + bins[1]**2))
    dopmaxangles = np.zeros((len(sollocs),len(levels))) # max angle per label
    dopmeanangles= np.zeros(len(sollocs))
    dopstdangles = np.zeros(len(sollocs)) # std angle per label    
    allabels = np.zeros((len(sollocs),len(levels))) # 2d allabels
    for i in range(len(sollocs)):
        w0,k0,w,dw,kpara,kperp=read_all_data(loc=sollocs[i])
        Va_c = getVa(w0,k0)/const.c # unitless
        try:
            Zij       = read_pkl(name+'_{}_Zij_lvlf_{}_angf_{}_lsize_{}'.format(i,levels[-1],angles[-1],lsize))
            intensity = read_pkl(name+'_{}_intensity_{}_{}'.format(i,levels[-1],angles[-1]))
            dopmaxang = read_pkl(name+'_{}_dopmaxang_{}'.format(i,levels[-1]))
        except:
            print("{:.2f}%".format(100*i/len(sollocs)))
            intensity, dopmaxang, Zij = get_line_doppler(kpara,w,dw,datanorm=w0,rowlim=(-4*k0,4*k0),collim=(0,15*w0),\
                                                    norm=[w0,k0],label=labels[i],levels=levels,angles=angles)
            dumpfiles(Zij,name+'_{}_Zij_lvlf_{}_angf_{}_lsize_{}'.format(i,levels[-1],angles[-1],lsize))
            dumpfiles(intensity,name+'_{}_intensity_{}_{}'.format(i,levels[-1],angles[-1]))
            dumpfiles(dopmaxang,name+'_{}_dopmaxang_{}'.format(i,levels[-1]))
        # assign large arrays
        allabels[i,:] = [labels[i] for r in range(len(dopmaxang))]
        dopmaxangles[i,:] = dopmaxang
        dopmeanangles[i]= np.mean(dopmaxang)
        dopstdangles[i] = np.std(dopmaxang) # for all levels, given solloc
        # make plots
        if plot:
            # plot semi circles
            plotSemiCirclePolarCoords(angles,intensity,levels,labels[i])
            # plot line doppler
            if len(angles) <= 12:
                plot_doppler_line(angles,Zij,levels,level=0)
            # plot summated growth per harmonics through angles
            plotSumGrowthAngle(angles,intensity,levels,labels[i])
    # # #
    fig,ax = plt.subplots(figsize=(8,6))
    dopmaxangles = dopmaxangles.flatten()
    allabels = allabels.flatten()
    if plot_angles:
        # fit straight line odr
        ax.scatter(allabels,dopmaxangles,color='k',alpha=1/len(levels))
        if fit_ODR:
            lowerb, upperb = [-60,-40] # lower, upper bound angles
            thresh = (dopmaxangles>lowerb) & (dopmaxangles<upperb)
            params, params_err = ODR_fit(allabels[thresh],dopmaxangles[thresh],beta0=[-1.0,-45.0],curve='linear')
            # params, params_err = ODR_fit(labels,dopmeanangles,sy=dopstdangles,beta0=[-0.15,-0.4],curve='linear')
            print('grad and intercept :',params,'errors: ',params_err)
            ax.plot(labels,labels*params[0]+params[1],color='r',linestyle='--')
            ax.annotate(r'$d\theta/d\xi_T=$'+'{:.2f}'.format(params[0])+r'$\pm$'+'{:.2f}'.format(params_err[0]),\
                        xy=(labels[1],labels[1]*params[0]+params[1]+upperb),xycoords='data',ha='left',va='bottom',color='r',**tnrfont)
        ax.set_ylabel(r'$\theta_{max}$'+' '+'[deg]',**tnrfont)
        ax.set_ylim(np.min(angles),np.max(angles))
        name=name+'_angles'
    if plot_grad:
        ax.scatter(allabels,np.tan(dopmaxangles)*Va_c,color='k',alpha=1/len(levels)) #1/len(levels))
        ax.set_ylabel(r'$v_{dop}/c$',**tnrfont)
        ax.set_ylim(np.min(np.tan(dopmaxangles))*Va_c,np.max(np.tan(dopmaxangles))*Va_c)
        if fit_ODR:
            lowerb, upperb = [-0.125,0.125] # lower bound, upper
            thresh = (np.tan(dopmaxangles)*Va_c>lowerb) & (np.tan(dopmaxangles)*Va_c<upperb)
            params, params_err = ODR_fit(allabels[thresh],np.tan(dopmaxangles[thresh])*Va_c,beta0=[-1.0,0.0],curve='linear')
            print('grad and intercept :',params,'errors: ',params_err)
            ax.plot(labels,labels*params[0]+params[1],color='r',linestyle='--')
            ax.annotate(r'$dv_{dop}/d\xi_T=$'+'{:.2f}'.format(params[0])+r'$\pm$'+'{:.2f}'.format(params_err[0]),\
                        xy=(labels[1],labels[1]*params[0]+params[1]+upperb),xycoords='data',ha='left',va='bottom',color='r',**tnrfont)
        name=name+'_grad'
    if fit_ODR:
        ax.axhline(lowerb,color='r',linestyle='--')
        ax.axhline(upperb,color='r',linestyle='--')
    ax.set_xlim(np.min(labels),np.max(labels))
    ax.set_xlabel(xlabel,**tnrfont)
    plt.show()
    fig.savefig(name+'.png',bbox_inches='tight')
    return None

if __name__=='__main__':
    homeloc = '/home/space/phrmsf/Documents/ICE_DS/JET26148/D_T_energy_scan/' # /home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Tritons/
    sollocs = [getsollocs(homeloc)[0]]
    print(sollocs)
    labels = ['1MeV']
    angles = np.linspace(-180,0,360) # np.array([-80,-85,-90,-95,-100])
    levels = range(0,16)
    # for i in range(len(sollocs)):
    os.chdir(sollocs[0])
    w0,k0,w,dw,kpara,kperp = read_all_data(loc=sollocs[0])
    Z = make2D(kpara,w,dw,rowlim=(-4*k0,4*k0),collim=(0,15*w0))
    intensity, dopmaxang, Zij = line_integrate(Z,xposini=levels,angles=angles,rowlim=(-4*k0,4*k0),collim=(0,15*w0),norm=[w0,k0],label=labels[0])
    # plotSumGrowthAngle(angles,intensity,levels,labels[0])
    fig,ax=plt.subplots(figsize=(8,6))
    ax.imshow(Z,origin='lower',interpolation='none',aspect='auto',cmap='summer',extent=[0,15,-4,4])
    for i in range(len(levels)):
        # plotSemiCirclePolarCoords(ax,angles,intensity[i,:])
        print(dopmaxang[i])
        warr = np.linspace(0,15,100)
        karr = (warr-i)*(1/np.tan(dopmaxang[i]*np.pi/180))
        ax.plot(warr,karr,'k--',alpha=0.5)
    ax.set_xlim(0,15)
    ax.set_ylim(-4,4)
    plt.show()
    sys.exit()

