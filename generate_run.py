

# constants
global me, mu0, qe
me = 9.1093829E-31 # kg #
mu0 = 1.2566371E-06 # Hm^[-1] #
qe = 1.6021766E-19 # C #

# names : mass, charge
mc = {
	'Deuterons':[3671.5*me, 1],
	'Tritons':[5497.93*me, 1],
	'Helium3':[5497.885*me, 2],
	'He3':[5497.885*me, 2],
	'Protons':[1836.2*me, 1],
	'Alphas':[7294.3*me, 2],
}


def loop_generate(**kwargs): # filename, btext, narr, Barr, Eminarr):
	fname = kwargs.get('filename')
	btext = kwargs.get('_btext')
	Barr = kwargs.get('_Barr')
	narr = kwargs.get('_narr')
	Eminarr = kwargs.get('_Eminarr')
	Tarr = kwargs.get('_Tarr')
	xiarr = kwargs.get('_xiarr')
	B0 = kwargs.get('_B0')
	n0 = kwargs.get('_n0')
	xi2arr = kwargs.get('_xi2arr')
	xi3 = kwargs.get('_xi3')

	# singular loop over concentrations
	with open(fname+'.txt','a') as file:
		for xi2 in xi2arr:
			print(xi2)
			# rho0 = xi1*m1 + xi2*m2 + xi3*m3
			# vA = B0/np.sqrt(mu0*n0*rho0)
			# vperp = vA*vperp_vA
			# vpara = np.sqrt(v0**2 - vperp**2)
			# vpara_v0 = vpara/v0
			# print(vpara_v0)
			# # quasi-neutrality
			# n2 = xi2*n0
			# n1 = (n0/Z1)*(1-xi2*Z2-xi3*Z3)
			# xi1 = n1/n0
			# if not (n0 == (n1*Z1+n2*Z2+n3*Z3)): #  @assert quasi-neutrality
			# raise SystemExit
			file.write(btext+'--secondfuelionconcentrationratio'+' '+str(xi2)+'\n') # +' --nameextension '+str(np.around(Emin,2))+'MeV'+'\n'

	# # multiple loops over parameters
	# with open(fname+'.txt', 'a') as file:
	# 	for n0 in narr:
	# 		for B in Barr:
	# 			for Emin in Eminarr:
	# 				file.write(btext+str(Emin)+' --magneticField '+str(B)+' --electronDensity '+str(n0)+'\n') # +' --nameextension '+str(np.around(Emin,2))+'MeV'+'\n'

	return None

if __name__=='__main__':
	import numpy as np
	import os,sys
	# #D-He3
	# name = 'D_He3_p_0'
	# btext = '../julia-1.9.3/bin/julia --proj LMV.jl --temperaturekeV 10.0 --secondfuelionconcentrationratio 0.25 --minorityenergyMeV '
	# majspec = 'Deuterons'
	# maj2spec = 'Helium3'
	# minspec = 'Protons'
	# Emin = 14.68 # MeV

	# D-T (JET like)
	name  = 'D_T_JET'
	btext = '../julia-1.9.3/bin/julia --proj LMV.jl '
	majspec = 'Deuterons'
	maj2spec = 'Tritons'
	minspec = 'Alphas'
	Emin = 3.5 # MeV

	# --- #

	# get mass and charge
	m1, Z1 = mc.get(majspec)
	m2, Z2 = mc.get(maj2spec)
	m3, Z3 = mc.get(minspec)

	# plasma parameters
	B0 = 3.7 # T
	n0 = 5e19 # 1/m^3
	xi3 = 1e-3 #
	n3 = xi3*n0 # 1/m^3
	# v0 = np.sqrt(2*Emin*qe/m3) # m/s
	# vperp_vA = 0.9 #

	xi2arr = np.arange(0,1.1,0.05) # concentrations
	print(xi2arr)
	loop_generate(filename=name,_btext=btext,_n0=n0,_B0=B0,_xi3=xi3,_xi2arr=xi2arr)

	# # thermal spread and percentage
	# th_spread_beam = ' 0.01'   # with spacing before
	# th_spread_ring = ' 0.001 ' # "			" and after
	# th_spreads = th_spread_beam+th_spread_ring
	# upperlog = 3 ; lowerlog = -6
	# upperlin = 15.0; lowerlin = 1.0
	# llog = upperlog-lowerlog # 10^l MeV == 10^(l+6) eV
	# llin = upperlin-lowerlin # l MeV == l * 10^6 eV
	# #Eminarr = np.logspace(lowerlog,upperlog,int(llog+1))
	# #Eminarr = np.linspace(lowerlin,upperlin,int(2*llin+1))
	# Ep = 14.68
	# varr = np.array([v0/4,v0/2,v0])
	# Eminarr = np.around((0.5*m3*varr**2)/qe/1e6,4) # MeV energies for proton speeds 1/4, 1/2 and 1 of v0
	# Barr = [2.5,5]
	# narr = [1e19,4e19]
	# Temp = 10.0

