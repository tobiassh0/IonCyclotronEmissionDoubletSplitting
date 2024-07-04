

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
	'B11':[19707.25*me, 5],
	'Alphas':[7294.3*me, 2],
}


def loop_generate(species,**kwargs): # filename, btext, narr, Barr, Eminarr):
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
	pitcharr = kwargs.get('_pitcharr')
	print(Eminarr,pitcharr)

	# singular loops
	m1, Z1 = mc.get(species[0])
	m2, Z2 = mc.get(species[1])
	m3, Z3 = mc.get(species[2])

	# over concentration and pitch
	with open(fname+'.txt','a') as file:
		for i in range(len(xi2arr)):
			print(xi2arr[i],pitcharr[i])
			# xi1 = (1/Z1)*(1-Z2*xi2-Z3*xi3) # varies
			# xi1prime = (1/Z1)*(1-xi3*Z3) # const
			# upper = m2-(Z2/Z1)*m1
			# lower = (m1/Z1)+xi3*(m3-(Z3/Z1)*m1)
			# neprime = n0*(1+xi2*(upper/lower))
			# # VA_T = B0/np.sqrt(mu0*n0*(xi1*m1+xi2*m2+xi3*m3))
			# # VA_noT = B0/np.sqrt(mu0*neprime*(xi1prime*m1+xi3*m3))
			# file.write(btext+'--electronDensity '+str(neprime)+' --nameextension '+str(np.around(xi2,3))+'\n')
			file.write(btext+'--pitch '+str(pitcharr[i])+' --secondfuelionconcentrationratio '+str(xi2arr[i])+'\n') # +' --nameextension '+str(np.around(Emin,2))+'MeV'+'\n'

	# over energy and pitch
	with open(fname+'.txt','a') as file:
		for i in range(len(Eminarr)):
			file.write(btext+'--minorityenergyMeV '+str(Eminarr[i])+' --pitch '+str(pitcharr[i])+'\n')

	# # over electron number density
	# with open(fname+'.txt','a') as file:
	# 	for ne in narr:			
	# 		file.write(btext+'--electronDensity'+' '+str(ne)+'\n')

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

	# # D-T (JET like)
	# name  = 'DT_JET_Energy'
	# btext = '../julia-1.9.3/bin/julia --proj LMV.jl --secondfuelionconcentrationratio 0.25 --minorityenergyMeV 14.68 --pitch 0.9893994377513348 '
	# majspec = 'Deuterons'
	# maj2spec = 'Tritons'
	# minspec = 'Alphas'
	# # Emin = 3.5 # MeV

	# # D-He3
	# m1, Z1 = mc.get('Deuterons')
	# m2, Z2 = mc.get('He3')
	# m3, Z3 = mc.get('Protons')
	# with open('D_He3_test.txt','a') as file:
	# 	file.write(btext+'\n')
	# sys.exit()
	
	# p-B11
	name = 'p_B11'
	btext = '../julia-1.9.3/bin/julia --proj LMV.jl --minorityenergyMeV 5.5 --magneticField 5.18 --temperaturekeV 25 --electronDensity 1e20 '
	majspec = 'Protons'
	maj2spec = 'B11'
	minspec = 'Alphas'
	# --- #


	# get mass and charge
	m1, Z1 = mc.get(majspec)
	m2, Z2 = mc.get(maj2spec)
	m3, Z3 = mc.get(minspec)

	# loop over tritium concentration for fixed electron number density
	umin = np.sqrt(2*5.5e6*qe/m3)
	xi2arr = np.array([i for i in np.arange(0,0.2*(1-2*1.5e-4),0.01)])# if (i/2)%5!=0])
	xi1arr = np.array([1-Z2*xi2arr[i]-Z3*1.5e-4 for i in range(len(xi2arr))])
	vA = np.array([5.18/np.sqrt(mu0*1e20*(m1*xi1arr[i]+m2*xi2arr[i]+m3*1.5e-4)) for i in range(len(xi2arr))])
	uperp_vA = 0.98
	uperp = np.array([uperp_vA*vA[i] for i in range(len(vA))])
	upara = np.array([np.sqrt(umin**2 - uperp[i]**2) for i in range(len(uperp))])
	pitcharr = np.array([upara[i]/umin for i in range(len(upara))])
	print(xi2arr)
	loop_generate([majspec,maj2spec,minspec],filename=name,_btext=btext,_xi2arr=xi2arr,_n0=1e20,_B0=5.18,_xi3=1.5e-4,_pitcharr=pitcharr)
	sys.exit()

	# loop over minority energetic particle energy
	energies_MeV = np.array([i/100 for i in range(100,2025,25)])
	# plasma parameters
	B0 = 2.07 # T
	n0 = 1.7e19 # 1/m^3
	xi3 = 1.5e-4 #
	n3 = xi3*n0 # 1/m^3
	xi2 = 0.11
	xi1 = (1/Z1)*(1-Z2*xi2-Z3*xi3)
	v0 = np.sqrt(2*energies_MeV*1e6*qe/m3)# m/s
	vperp_vA = 0.9
	vA = B0/np.sqrt(mu0*n0*(m1*xi1+m2*xi2+m3*xi3))
	vpara = (v0**2 - (vperp_vA*vA)**2)**0.5
	pitchcosine = vpara/v0
	print(energies_MeV,pitchcosine)
	loop_generate(species=[majspec,maj2spec,minspec],filename=name,_btext=btext,_Eminarr=energies_MeV,_pitcharr=pitchcosine)

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

