###### This file contains the functions for adjusting E1, M1 and NLD for tabulated TALYS models ######################
'''
If you only want to fit  E1 or M1 you will need to adjust the function to add in the other component to compare to ysf data
If you want to fit both ensure that M1 does not add in E1 and vice versa
'''
import numpy as np

def nld_model(params,energy,model_args,prior_len): #prior len doesn't matter here but I need to keep everthing clean

	if prior_len != 1:
		return ValueError("More than one prior has passed into function. Did you mean to run ySF instead? NLD fit only takes 1 prior: nld_prior")

	param0,param1 = params
	talys_energy = model_args["energy_talys"]
	talys_nld = model_args["nld_talys"]
    #Calculate new Talys with c p suggestion
    #ptable shift
	shifted_energy = energy-param0 
	shifted_nld = np.interp(shifted_energy,talys_energy, talys_nld,left=np.nan,right=np.nan)
	#ctable shift
	exp = np.exp(param1*np.sqrt(np.clip(shifted_energy,0,None)))

	new_nld = exp*shifted_nld

	return new_nld


def E1_model(params,energy,model_args,prior_length):
	w, E_shift, scale = params
	talys_energy = model_args["energy_ysf"]
	talys_E1 = model_args["ysf_E1"]
	talys_M1 = model_args["ysf_M1"]
	center_E = model_args["center_E"]
	# Compute mapped energies for data grid
	mapped_energy = center_E + w * (energy - center_E) + E_shift

	# Interpolate from TALYS onto mapped energy positions
	new_E1 = np.interp(mapped_energy, talys_energy, talys_E1,
		               left=np.nan, right=np.nan)
	if prior_length == 1:
		new_M1 = np.interp(mapped_energy, talys_energy, talys_M1, # use when only fitting E1 model
		               left=np.nan, right=np.nan)
		scaled = scale * new_E1 + new_M1 
	else:
		scaled = scale * new_E1
	return scaled

def M1_model(params,energy,model_args,prior_length): # upbend manipulation
	param0,param1,param2,param3 = params
	talys_energy = model_args["energy_ysf"]
	talys_E1 = model_args["ysf_E1"]
	talys_M1 = model_args["ysf_M1"]
	#param0 - upbendc; param1 - upbende; param2-upbendf. I may need to add a param3 - beta2 for deformation
	# new_f(M1) = upbendf * exp(-upbendc*abs(deformation) ) * exp(-upbende*Energy ) maybe???? TALYS 2.0 manual pg.56
	new_m1= param0*np.exp(-param1*energy)*np.exp(-param2*np.abs(param3))

	int_m1 = np.interp(energy, talys_energy, talys_M1, left=np.nan, right=np.nan)
	if prior_length == 1:	
		int_E1 = np.interp(energy, talys_energy, talys_E1,
		               left=np.nan, right=np.nan)
		return new_m1+int_m1+ int_E1
	else:
		return new_m1+int_m1 
