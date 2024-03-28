
def loadJETdata(loc=''):
	JETdata = np.loadtxt(loc+'JET26148_ICE_POWER.txt',delimiter=',')
	JETpower, JETfreqs = JETdata[:,1], JETdata[:,0] # 2 columns, N rows
	wnorm = 2*np.pi*17E6
	JETfreqs = 2*np.pi*JETfreqs*1e6/(wnorm) # convert MHz to wcD (or wcalpha)
	maxJETfreq = round(max(JETfreqs))
	return maxJETfreq, JETfreqs, JETpower

def peakpowerratio(xpeaks,ypeaks):
    yppr = np.roll(ypeaks,1)/ypeaks
    yppr = yppr[1:] # exclude 0th/-1th peak
    xppr = xpeaks[1:]
    return xppr, yppr

def makeKDE(xdata,ydata,bins=(None,100),weight=0.7,bw=None,plot=False):
    if not bw:
        bw = len(xdata)**(-1/6)
    data = np.vstack([xdata,ydata])
    xmin = xdata.min() ; xmax = xdata.max()
    ymin = ydata.min() ; ymax = ydata.max()
    xlen = complex(0,int(xmax-xmin)+1) #; print(xlen)
    X,Y = np.mgrid[xmin:xmax:10j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(),Y.ravel()])
    kde = st.gaussian_kde(data,bw) # if None bandwidth then scipy calculates using Scott method
    Z = np.reshape(kde(positions).T, X.shape)
    # normalise total grid to sum to 1
    floor = 1/((ymax-ymin)*(xmax-xmin))
    Z = (1-weight)*floor + Z*weight
    Z /= np.sum(Z)
    print(np.sum(Z))

    if plot:
        plt.plot(xdata,ydata,'w.')
        plt.imshow(Z.T,origin='lower',aspect='auto',extent=[xmin,xmax,ymin,ymax],interpolation='none')
        plt.show()
    # p,_,_= np.histogram2d(xdata,ydata,bins=bins,density=True,range=[[min(xdata),max(xdata)],[min(ydata),max(ydata)]])
    # floor = (max(ydata)-min(ydata))*(max(xdata)-min(xdata)) # area of x and y plane
    # Z = (1-weight)*floor + weight*p
    return Z, bw

def getTauSquared(KDE,xvals,yvals,KDEbounds=[None,None,None,None],floor=1e-8):
    # find position of xval & yval in KDE
    # extract probability density
    # summate for all tau^2
    # return array of tau^2 and final summated tau^2

    xmin = KDEbounds[0] ; xmax = KDEbounds[1]
    ymin = KDEbounds[2] ; ymax = KDEbounds[-1]

    tau2arr = np.zeros(len(xvals))
    tau2sum = 0
    N,M = KDE.shape
    for i in range(len(xvals)):
        xbin = (N*(xvals[i]-xmin)/(xmax-xmin))-1
        ybin = (M*(yvals[i]-ymin)/(ymax-ymin))-1
        if xbin > N or ybin > M: # check if bin in range of calculated KDE 
            p = floor 
        else:
            p = KDE[int(xbin),int(ybin)]
        tau2=-2*np.log(p)
        tau2arr[i]=tau2
        tau2sum+=tau2

    return tau2arr, tau2sum # not normalised to number of ppr

if __name__=='__main__':
    from makeplots import *
    import scipy.stats as st
    from sklearn.neighbors import KernelDensity

    home = '/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/'
    sollocs = getsollocs(home)
    # get labels of sims
    xi2 = np.array([sol.split('_')[2] for sol in np.sort(os.listdir(home)) if 'run' in sol],dtype=float)
    print(xi2)

    # load JET 26148 data and calculate PPR peaks
    maxJETfreq, JETfreqs, JETpower = loadJETdata(loc='/home/space/phrmsf/Documents/thesis_code/power/')
    # turn JET PPR peaks into KDE
    JETpeaks = extractPeaks(JETpower,Nperw=8,plateau_size=0.5)
    xJETppr, yJETppr = peakpowerratio(JETfreqs[JETpeaks],JETpower[JETpeaks])
    xmin = np.min(xJETppr) ; xmax = np.max(xJETppr)
    ymin = np.min(yJETppr) ; ymax = np.max(yJETppr)
    print(xmin,xmax,ymin,ymax)
    # setup kde
    KDE, bw = makeKDE(xJETppr,yJETppr,bins=(maxJETfreq-1,1000))
    print(KDE.shape)

    tau2sumsollocs = []
    # loop over sollocs
    for i in range(len(sollocs)):
        data=read_all_data(loc=sollocs[i])
        w0,k0,w,dw,kpara,kperp = data
        # make sim data smooth
        w, dw = make1D(w,dw,norm=(w0,w0),bins=800)
        # thresh to limit of JET data
        thresh = w/w0 < maxJETfreq
        w = w[thresh] ; dw = dw[thresh]
        # turn sim peaks into ppr
        peaks = extractPeaks(dw,Nperw=8,plateau_size=0.5)
            # plt.plot(w/w0,dw/w0,color='b')
            # plt.scatter(w[peaks]/w0,dw[peaks]/w0)
        xppr, yppr = peakpowerratio(w[peaks],dw[peaks])
            # plt.plot(xppr/w0,yppr,'k.')
        # calculate tau2 for this given run
        tau2arr, tau2sum = getTauSquared(KDE,xppr/w0,yppr,KDEbounds=[xmin,xmax,ymin,ymax],floor=np.min(KDE))
        # plt.scatter(xppr/w0,tau2arr)
        tau2sumsollocs.append(tau2sum/len(xppr))
    # --- # 
    print(xi2,tau2sumsollocs)
    plt.scatter(xi2,tau2sumsollocs,facecolor='k',edgecolor='none')
    plt.ylabel(r'$\tau^2/N_{peaks}$',**tnrfont)
    plt.xlabel(r'$\xi_T$',**tnrfont)
    plt.savefig(home+'tau2_xi2_bw_{}.png'.format(bw),bbox_inches='tight')
    sys.exit()


        # calc tau-squared between model (JET 26148) and data (LMV runs)
            # find which bin data falls in model
            # summate probability/tau-squared per bin
        # plot tau-squared as function of tritium-concentration


