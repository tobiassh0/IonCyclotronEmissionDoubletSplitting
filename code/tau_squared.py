
def loadJETdata(loc=''):
	JETdata = np.loadtxt(loc+'JET26148_ICE_POWER.txt',delimiter=',')
	JETpower, JETfreqs = JETdata[:,1], JETdata[:,0] # 2 columns, N rows
	wnorm = 2*np.pi*17E6
	JETfreqs = 2*np.pi*JETfreqs*1e6/(wnorm) # convert MHz to wcD (or wcalpha)
	maxJETfreq = round(max(JETfreqs))
	return maxJETfreq, JETfreqs, JETpower

def makeKDE(xdata,ydata,bins=(None,100),weight=0.7):
    data = np.vstack((xdata,ydata)).T
    
    kde = KernelDensity(bandwidth=0.3, kernel='gaussian').fit(data)
    logprob = kde.score_samples(prob[:])
    print(logprob)

    p,_,_= np.histogram2d(xdata,ydata,bins=bins,density=True,range=[[min(xdata),max(xdata)],[min(ydata),max(ydata)]])
    floor = (max(ydata)-min(ydata))*(max(xdata)-min(xdata)) # area of x and y plane
    p = (1-weight)*floor + weight*p
    
    return p

def peakpowerratio(xpeaks,ypeaks):
    yppr = np.roll(ypeaks,1)/ypeaks
    yppr = yppr[1:] # exclude 0th/-1th peak
    xppr = xpeaks[1:]
    return xppr, yppr

if __name__=='__main__':
    from makeplots import *
    from sklearn.neighbors import KernelDensity
    home = '/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/'
    sollocs = getsollocs(home)
    # load JET 26148 data and calculate PPR peaks
    maxJETfreq, JETfreqs, JETpower = loadJETdata(loc='/home/space/phrmsf/Documents/thesis_code/power/')
    JETpeaks = extractPeaks(JETpower,Nperw=8,plateau_size=0.5)
    xJETppr, yJETppr = peakpowerratio(JETfreqs[JETpeaks],JETpower[JETpeaks])
    prob = makeKDE(xJETppr,yJETppr,bins=(maxJETfreq-1,1000))
    # setup kde
    # turn JET PPR peaks into KDE
    sys.exit()

    # for i in range(len(sollocs)):
    #     print(sollocs[i])
    #     data=read_all_data(loc=sollocs[i])
    #     w0,k0,w,dw,kpara,kperp = data
    #     xarr, zarr = make1D(w,dw,norm=(w0,w0)) # marnormf = 18
    #     # 2D plot
    #     peaks = extractPeaks(zarr,Nperw=8,plateau_size=0.5)
    #     plt.plot(xarr,zarr,color='b')
    #     plt.scatter(xarr[peaks],zarr[peaks],color='r')
    #     xpeak  = xarr[peaks] ; zpeak = zarr[peaks]

        # calc tau-squared between model (JET 26148) and data (LMV runs)
            # find which bin data falls in model
            # summate probability/tau-squared per bin
        # plot tau-squared as function of tritium-concentration


