import os
import numpy as np
import pandas as pd

from src.covarience import cov_mtx_general, roberts_scaled_cov, diag_cov_from_steps
from src.priors import E1_prior, M1_prior, nld_prior
from src.models import E1_model, M1_model, nld_model
from src.likelihood import loglikelihood_general
from src.converters import model_to_internal
from src.sampler import metropolis_unified
from src.diagnostics import mcmc_diagnostics_array, convert_model_to_internal


def energy_axis (m,b,data_size):
    energy_data = []
    for i in range(data_size):
        erg = m*i+b
        energy_data.append(erg)
    energy_data = np.array(energy_data)
    return energy_data


if __name__ == "__main__":
	#######################
	# Read in base models #
    #######################

	#NLD
	df_nld_talys=pd.read_csv("Talys_Models/NLD_models/ld5_59Cu.csv")
	#df_nld_talys= df_nld_talys.drop(df_nld_talys.index[:2]) #drop weird dip at beginning
	energy_nld_talys=df_nld_talys["Ex"].values
	nld_talys=df_nld_talys["NLD"].values 
	#ySF
	df_ysf_talys=pd.read_csv("Talys_Models/ySF_models/s10m3un.csv")
	energy_ysf_talys=df_ysf_talys["E"].values 
	ysf_E1_talys=df_ysf_talys['f(E1)'].values 
	ysf_M1_talys=df_ysf_talys["f(M1)"].values
	# Find GDR center (E_E1)
	ysf_total = ysf_E1_talys + ysf_M1_talys
	center_E = energy_ysf_talys[np.argmax(ysf_total)]
	################
	# Load in Data #
    ################	
	# Set up energy axis
    	# m & b from stregnth.cpp
	m = 0.4200 # 420 keV/bin
	b = -0.8390
	#NLD
	df_nld_data=pd.read_csv("/mnt/projects/SuNgroup/jordan/BOslo_spectra/Ga60_data/Data/30keV/420keV/low_Ey_841/rhopaw.cnt",header=None)
	nld_size = len(df_nld_data)//2
	nld_data=df_nld_data[:nld_size].values
	err_nld=df_nld_data[nld_size:].values

	energy_nld = energy_axis(m,b,nld_size)

	energy_data = np.array(energy_nld).flatten()
	nld_data = nld_data.flatten()
	err_nld = err_nld.flatten()

	#ySF
	df_ysf_data=pd.read_csv("/mnt/projects/SuNgroup/jordan/BOslo_spectra/Ga60_data/Data/30keV/420keV/low_Ey_841/strength.nrm",header=None,index_col=False) #420 keV/bin data

	ysf_size = len(df_ysf_data)//2
	energy_ysf = energy_axis(m,b,ysf_size)

	ysf_scale = 0.00018262 #Normalizing factor used when comparing data to nearby nuclei'
	ysf_data=df_ysf_data[:ysf_size]
	ysf_data_scaled=np.array(ysf_data*ysf_scale).flatten()
	error_data=df_ysf_data[ysf_size:]
	error_data_scaled= np.array(error_data*ysf_scale).flatten()


	########################
	# Sampler Requirements #
	########################

	# NLD params
	nld_params = [2.3, 0.0] # p_table, c_table
	nld_steps = [0.1, 0.1]
	#ySF params
	E1_params = np.array([0.8,0.,0.45]) # w_table, E_table, f_table
	M1_params = np.array([3e-8, 1.0, 0.75, 0.0]) #upbendc, upbende, upbendf, beta2
	E1_steps = [0.05, 0.01, 0.01]
	M1_steps = [ 7e-9, 0.03, 0.05, 0.01]
	# All params
	all_params = np.concatenate((E1_params,M1_params),axis=0)
	all_steps = np.concatenate((E1_steps,M1_steps),axis=0)

	# To pass tables into sampler. Used in models.py
	nld_args = {"energy_talys": energy_nld_talys,"nld_talys": nld_talys}
	ysf_args = {"energy_ysf": energy_ysf_talys, "ysf_E1": ysf_E1_talys,"ysf_M1": ysf_M1_talys,"center_E": center_E}

	# Parameter ranges described as sigma
	#Sigmas calculated as (upper_bound - lower_bound)/2 for ~6 sigma of param range. Bounds from Talys 2.0
	nld_sigmas = np.array([20.,20.])/2. # c_table & p_table ranges
	E1_sigmas = np.array([10.,20.,9.9])/2.0 # w_table, E_table, & f_table parameter ranges
	scale_sigma = np.log(1e-5)-np.log(1e-12) # better to treat M1 scale in log space due to size
	M1_sigmas = np.array([10.,20.,1.5,scale_sigma])/2.

	# Make cov matricies
	# If acceptance is near 0% for a block → proposed steps are too large → shrink covariance or ScalarTuner alpha.
	# If acceptance is very high (≫50%) but ESS still low → steps are too small → increase covariance.  
	# ESS is printed at the end in diagnostics
	cov_nld = cov_mtx_general(nld_sigmas)  # uses 2-d defaults
	cov_E1 = cov_mtx_general(E1_sigmas)
	cov_M1 = cov_mtx_general(M1_sigmas)

	#Make priors
	prior_arguments_nld = [nld_params, cov_nld]
	prior_arguments_E1 = [E1_params, cov_E1]
	prior_arguments_M1 = [M1_params, cov_M1]
	prior_arguments_all = [all_params, [cov_E1, cov_M1]]

	# Make Blocks 
	'''
		Blocks group parameters for step logic. If diagnostics show large t_int or small ESS (< ~50),
    try regrouping them. Some parameters may perform better when grouped differently or alone. 
	If one parameter has τ_int much larger than others → isolate it into its own block and give it a tailored proposal.
	t_int is printed at the end in Diagnostics.
	'''
	nld_blocks = [[0,1]] 
	ysf_blocks = [[0,1,2],[3],[4],[5,6]]  
	target_accept_nld = [0.234] #acceptance rate to aim for in each block
	target_accept_ysf = [0.234, 0.44, 0.44, 0.234] 
	ysf_mix = { # for sticky parameters (high t_int/low ESS) use this to push them to explore better
    			'index': [3,4],   # which parameter(s) to perturb (upbendc [3] and upbendf [4] hate to move)
    			'prob' : 0.3, # probability of using the local proposal. Can boost index param ESS at the cost of other param ESS
    			'scale': [1.,1.]  # scale (standard deviation) of the perturbation
				}
	# nld_mix = {'index':0,'prob':0.0,'scale':0}
	
	# Special when running E1 & M1
	all_model = [E1_model, M1_model]
	all_prior = [E1_prior, M1_prior]

	#NLD fit settings
	energy_nld_high = 10.
	energy_nld_low = 0.

	nld_mask = (energy_nld>energy_nld_low) & (energy_nld<energy_nld_high) & (nld_data > 0)
	erg_nld_dat = energy_nld[nld_mask]
	nld_dat = nld_data[nld_mask]
	err_nld_dat = err_nld[nld_mask]

	#ySF fit settings
	energy_ysf_high = 10.
	energy_ysf_low = 0.
	data_mask = ysf_data_scaled > 1e-10
	energy_mask = (energy_ysf>energy_ysf_low) & (energy_ysf<energy_ysf_high)
	ysf_mask = data_mask & energy_mask
	ysf_mask = np.ravel(ysf_mask)
	energy_ysf_data = energy_ysf[ysf_mask]
	ysf_data = ysf_data_scaled[ysf_mask]
	error_ysf_data = error_data_scaled[ysf_mask]


	#################
	# MCMC Sampler  #
	#################

	"""
	Pilot runs
	This is a short run that gives us an idea of the space we are exploring.
	We will use the posterior of this to create 'evidence' based posterior covariance matrices.
	These will be used in a longer run later.
	"""
	print("\n=== NLD pilot run ===")
	print(f'Initial NLD params: p_table = {nld_params[0]}; c_table {nld_params[1]}')

	chains_nld_pilot, post_nld_pilot, chi_nld_pilot, acc_nld_pilot = metropolis_unified(
		burn=10000,
		data=[erg_nld_dat,nld_dat],
		sigma=err_nld_dat,
		prior=nld_prior,
		prior_arguments=prior_arguments_nld,
		likelihood=loglikelihood_general,
		model=nld_model,
		model_args = nld_args,
		num_iterations=50000,
		step_size= nld_steps,
		block_idxs= nld_blocks,
		cov_bases=None,
		adapt_interval=5000,
		adapt_window=5000,
		target_accept=target_accept_nld,# Good choice for 2 param setup
		log_params=None,
		mixture_local=None,
		random_seed=42,
	    )
	print(f"NLD Pilot acceptance: {acc_nld_pilot:.3f}%")

	print("\n=== ySF pilot (E1+M1) run  ===")
	print(f'Initial ySF params \nE1 Parameters: w_table = {E1_params[0]}; E_table {E1_params[1]}; f_table {E1_params[2]} \nUpbend Parameters: upbendc = {M1_params[0]}; upbende = {M1_params[1]}; upbendf = {M1_params[2]}; beta2 = {M1_params[3]}')
	
	chains_ysf_pilot, post_ysf_pilot, chi_ysf_pilot, acc_ysf_pilot = metropolis_unified(
	    burn=10000,
	    data=[energy_ysf_data,ysf_data],
	    sigma=error_ysf_data,
	    prior=all_prior,
	    prior_arguments=prior_arguments_all,
	    likelihood=loglikelihood_general,
	    model=[E1_model, M1_model],
		model_args = ysf_args,
	    num_iterations=50000,
	    step_size=all_steps,
	    block_idxs=ysf_blocks,
	    cov_bases=None,
	    adapt_interval=5000,
		adapt_window=5000,
	    target_accept=target_accept_ysf,
	    log_params=[3],
	    mixture_local = ysf_mix,
	    random_seed=123,
	)
	print(f"ySF pilot acceptance: {acc_ysf_pilot:.3f}%")

	#Build empirical covariance matrices from Pilot runs

	cov_pilot_nld = np.cov(chains_nld_pilot.T)
	cov_emp_nld = []
	for idxs in nld_blocks:
		sub = cov_pilot_nld[np.ix_(idxs, idxs)].copy()
		# regularize to ensure Postive Definite (PD)
		jitter = 1e-8 * np.eye(sub.shape[0])
		sub += jitter
		cov_emp_nld.append(sub)
		
	cov_pilot_ysf = np.cov(chains_ysf_pilot.T)
	cov_emp_ysf = [] #Final cov matrix is made differently since parameters are split into blocks
	for idxs in ysf_blocks:
		sub = cov_pilot_ysf[np.ix_(idxs, idxs)].copy()
	    # regularize to ensure Postive Definite (PD)
		jitter = 1e-8 * np.eye(sub.shape[0])
		sub += jitter
		cov_emp_ysf.append(sub)


	""" 
	Production runs
	Longer runs that should explore large range of parameter space. 
	Increased burn and interations. 
	This provides final parameter posterior that is written to file
	"""

	print("\n=== NLD production run ===")
	chains_nld_final, post_nld_final, chi_nld_final, acc_nld_final = metropolis_unified(
		burn=20000,
		data=[erg_nld_dat,nld_dat],
		sigma=err_nld_dat,
		prior=nld_prior,
		prior_arguments=prior_arguments_nld,
		likelihood=loglikelihood_general,
		model=nld_model,
		model_args = nld_args,
		num_iterations=50000,
		step_size=nld_steps,
		block_idxs=nld_blocks,
		cov_bases=[cov_emp_nld],
		adapt_interval=500,
		adapt_window=2000,
		target_accept=target_accept_nld,
		log_params=None,
		mixture_local=None,
		random_seed=42,
	    )
	print(f"NLD Production acceptance: {acc_nld_final:.3f}%")


	print("\n=== ySF Production (E1+M1) run  ===")
	chains_ysf_final, post_ysf_final, chi_ysf_final, acc_ysf_final = metropolis_unified(
	    burn=20000,
	    data=[energy_ysf_data,ysf_data],
	    sigma=error_ysf_data,
	    prior=[E1_prior, M1_prior],
	    prior_arguments=prior_arguments_all,
	    likelihood=loglikelihood_general,
	    model=all_model,
		model_args = ysf_args,
	    num_iterations=50000,
	    step_size=all_steps,
	    block_idxs=ysf_blocks,
	    cov_bases=cov_emp_ysf,
	    adapt_interval=500,
		adapt_window=2000,
	    target_accept=target_accept_ysf,
	    log_params=[3],
	    mixture_local=ysf_mix,
	    random_seed=123,
	)
	print(f"ySF Production acceptance : {acc_ysf_final:.3f}%")

	###############
	# Diagnostics #
	###############

	print('\nPosterior Diagnostics \n')
	print('Check that t_int << ESS. ESS > ~1k good > ~2k great \n')
	print('NLD')
	chains_internal_nld = convert_model_to_internal(chains_nld_final, log_params=None)

	diag_nld = mcmc_diagnostics_array(chains_internal_nld, burn=0, chi2_col=None, max_lag=500)
	print("param indices:", diag_nld['param_idxs'])
	print("tau_int:", np.round(diag_nld['tau_int'], 2))
	print("ESS:", np.round(diag_nld['ess'], 1))
	print("N used:", diag_nld['N'])

	print('\nySF')
	chains_internal_ysf = convert_model_to_internal(chains_ysf_final, log_params=[3])

	diag_ysf = mcmc_diagnostics_array(chains_internal_ysf, burn=0, chi2_col=None, max_lag=500)
	print("param indices:", diag_ysf['param_idxs'])
	print("tau_int:", np.round(diag_ysf['tau_int'], 2))
	print("ESS:", np.round(diag_ysf['ess'], 1))
	print("N used:", diag_ysf['N'])


	#################
	# Write to File #
	#################

	# Write posteriors to file for visualization or further analysis
	df_nld_output = pd.DataFrame({'ptable':chains_nld_final[:,0],'ctable':chains_nld_final[:,1], 'LogLikelihood':chi_nld_final})
	df_nld_output = df_nld_output.drop_duplicates()
	df_nld_output.to_csv('posterior_files/Bayes_nld_post_params.csv', index=False, )
	

	df_ysf_output = pd.DataFrame({' wtable':chains_ysf_final[:,0], 'etable':chains_ysf_final[:,1], 'ftable':chains_ysf_final[:,2], 'upbendc':chains_ysf_final[:,3], 'upbende':chains_ysf_final[:,4], 'upbendf':chains_ysf_final[:,5], 'beta2': chains_ysf_final[:,6], 'LogLikelihood':chi_ysf_final})
	df_ysf_output = df_ysf_output.drop_duplicates()
	df_ysf_output.to_csv('posterior_files/Bayes_ysf_post_params.csv', index=False)
	print('\nPosterior parameter distribution files written to posterior_files directory.')
	print('IMPORTANT: Make sure to use the visualization tools in create_figures.py to check fit quality!\n')
### END ###
