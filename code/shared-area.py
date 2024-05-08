
def SharedArea2d(d1,d2,dx=1,dy=1,normsa=True):
    # 2D shared area method between two datasets (matrices)
    if d1.shape != d2.shape:
        # TODO; reshape
        raise SystemExit
    sa = np.zeros(d1.shape)
    for rx in range(d1.shape[0]):
        rxd2 = np.roll(d2,rx,axis=0)
        for ry in range(d1.shape[1]):
            rd2 = np.roll(rxd2,ry,axis=1)
            argd1_d2 = d1 < rd2
            argd2_d1 = argd1_d2 == False
            sa[rx,ry] = np.sum((d1*argd1_d2 + rd2*argd2_d1)*dx*dy)
    # sort and arrange
    nrows,ncols=sa.shape
    tsa = np.zeros(sa.shape)
    tsa[:nrows//2,:ncols//2] = sa[nrows//2:,ncols//2:] # A
    tsa[:nrows//2,ncols//2:] = sa[nrows//2:,:ncols//2] # B
    tsa[nrows//2:,:ncols//2] = sa[:nrows//2,ncols//2:] # C
    tsa[nrows//2:,ncols//2:] = sa[:nrows//2,:ncols//2] # D
    if normsa:
        tsa = (tsa-np.min(tsa))/(np.max(tsa)-np.min(tsa))

    peakind = None # np.unravel_index(sa.argmax(),sa.shape)
    # x = np.linspace(0,dx*sa.shape[0],sa.shape[0])
    # y = np.linspace(0,dy*sa.shape[1],sa.shape[1])
    # peakval = [x[peakind[0]],y[peakind]]
    return tsa, peakind

if __name__=='__main__':
    from makeplots import *

    # # 2d shared area example
    # d1 = np.zeros((100,100))
    # d1[10:20,10:20] = 1
    # d2 = np.zeros((100,100))
    # d2[50:60,50:60] = 1
    # sa, peak = SharedArea2d(d2,d1)
    # plt.imshow(sa,**imkwargs,cmap='binary',extent=(-50,50,-50,50)) ; plt.show()
    # sys.exit()
    
    homeloc = '/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration/'
    sollocs = getsollocs(homeloc)
    XI2 = []
    for sol in sollocs: # missing 11% in XI2 array
        XI2.append(float(sol.split('run')[1].split('_')[2]))
    XI2 = np.array(XI2)

    # load 0th solloc
    w0,k0,w,dw,kpara,kperp = read_all_data(loc=sollocs[0])
    Zsol0 = make2D(kpara,w,dw,rowlim=(-4*k0,4*k0),collim=(0,15*w0),bins=(512,512))
    # plt.imshow(Zsol0,**imkwargs,extent=[0,15,-4,4])

    i=0
    # loop through sollocs
    for sol in sollocs:
        w0,k0,w,dw,kpara,kperp = read_all_data(loc=sol)
        Zsoli = make2D(kpara,w,dw,rowlim=(-4*k0,4*k0),collim=(0,15*w0),bins=(512,512))
        sa,peak=SharedArea2d(Zsol0,Zsoli)
        plt.imshow(sa,**imkwargs,extent=[-7.5,7.5,-4,4],clim=(0,1))
        plt.colorbar()
        plt.savefig('2dsa_{:.3f}.png'.format(XI2[i]))
        plt.clf()
        i+=1
