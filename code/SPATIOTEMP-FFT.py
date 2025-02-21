
from makeplots import *

if __name__=='__main__':
    homeloc = '/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Tritons/'
    solloc = 'run_2.07_0.0_-0.646_0.01_0.01_15.0_3.5__1.0_4.0_1.7e19_0.00015_1024/'
    data = read_all_data(homeloc+solloc)
    w0,k0,w,dw,kpara,kperp = data

    ktot = np.sqrt(kpara**2 + kperp**2)
    plt.plot([0,np.max(ktot/k0)],[0,np.max(ktot/k0)],color='k',linestyle='--')

    frac = 1000
    # thresh = dw/w0 > 1e-2
    thresh=np.argsort(dw[::frac]/w0)
    plt.scatter(ktot[thresh]/k0, w[thresh]/w0, c=dw[thresh]/w0, edgecolor='none', cmap='summer')

    # plt.scatter(ktot[::frac]/k0, w[::frac]/w0, c=dw[::frac]/w0, edgecolor='none', cmap='summer')
    plt.show()