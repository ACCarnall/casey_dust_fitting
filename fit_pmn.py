import numpy as np 
import pymultinest as pmn
import corner
import matplotlib.pyplot as plt

from glob import glob
from subprocess import call

"""

Model taken from Casey (2012)

Parameters:

Nbb - black body normalisation
Npl - power law normalisation
beta - spectral emissivity index (generally assumed 1.5, but can range from 1 to 2.5)
alpha - power law slope
lambda_0 - wavelength at which optical depth is 1. (can be fixed at 200 um.)
lambda_c - wavelength at which MIR power law turns over 
T - temperature

"""

# Input data
wavs   = np.array([8., 24., 100., 160., 250., 450., 850.])
fluxes = np.array([3.94000000e-05, 1.82737185e-04, 1.40000000e-01, 1.40000000e-01, 6.79557736e-01, 1.88463960e-02, 4.85758671e-03]) # Janskys
errors = np.array([2.21000000e-06, 7.39953076e-06, 7.60000000e-02, 9.70000000e-02, 5.22661289e-01, 6.47668353e-03, 4.98186312e-03])


# Implement error floor at 5 sigma
for i in range(fluxes.shape[0]):
	if fluxes[i]/errors[i] > 5.:
		errors[i] = fluxes[i]/5.

fit_params = ["Nbb", "Npl", "T", "alpha", "beta"]

# Prior limits on parameter values
fit_limits = []
fit_limits.append((-40., -30.)) # Prior limits on log_10(Nbb)
fit_limits.append((-10., 20.)) # Prior limits on log_10(Npl)
fit_limits.append((0., 273.)) # Prior limits on T
fit_limits.append((0.5, 5.5)) # Prior limits on alpha
fit_limits.append((0.5, 2.5)) # Prior limits on beta



def model(param):
	""" Function to evaluate the model with parameters: param, and at wavelengths: wavs. """

	wavs_m = wavs*10**-6 # Convert wavelength values from microns to metres

	Nbb = 10**param[0] # Convert normalisations from log space to linear space
	Npl = 10**param[1]
	T = param[2]
	alpha = param[3]
	beta = param[4]

	lambda_0 = 200.
	lambda_c = 0.75*((26.68 + 6.246*alpha)**-2. + ((1.905*10**-4) + ((7.243*10**-5)*alpha))*T)**-1.

	hc_k = (6.626*10**-34)*(3*10**8)/(1.38*10**-23)

	# Ended up fitting Npl separately, but this is the correct way of calculating the value they fix it to in the paper
	# Npl = Nbb*(1-np.exp(-(lambda_0/lambda_c)**beta))*((3*10**8)**3/((lambda_c*10**-6)**(3+alpha)))/(np.exp(hc_k/T/(lambda_c*10**-6)) - 1.)

	# At certain points in parameter space, this factor gets too big to take the exponential of, resulting in math errors
	greybody_factor = hc_k/T/wavs_m

	S_jy_greybody = np.zeros(wavs_m.shape[0])

	# Therefore only actually evaluate the exponential if greybody_factor is greater than 300, otherwise greybody flux is zero
	S_jy_greybody[greybody_factor < 300.] = Nbb*(1-np.exp(-(lambda_0*10**-6/wavs_m[greybody_factor < 300.])**beta))*(((3*10**8)/wavs_m[greybody_factor < 300.])**3)/(np.exp(greybody_factor[greybody_factor < 300.]) - 1.)

	S_jy_powerlaw = Npl*(wavs_m**alpha)*np.exp(-(wavs_m/(lambda_c*10**-6))**2)

	return S_jy_greybody + S_jy_powerlaw



def model_highsampling(param):
	""" Function to evaluate the model with parameters: param, at high wavelength sampling. """

	wavs_m = np.arange(1., 1000.)*10**-6 # Convert wavelength values from microns to metres

	Nbb = 10**param[0] # Convert normalisations from log space to linear space
	Npl = 10**param[1]
	T = param[2]
	alpha = param[3]
	beta = param[4]

	lambda_0 = 200.
	lambda_c = 0.75*((26.68 + 6.246*alpha)**-2. + ((1.905*10**-4) + ((7.243*10**-5)*alpha))*T)**-1.

	hc_k = (6.626*10**-34)*(3*10**8)/(1.38*10**-23)

	# Ended up fitting Npl separately, but this is the correct way of calculating the value they fix it to in the paper
	# Npl = Nbb*(1-np.exp(-(lambda_0/lambda_c)**beta))*((3*10**8)**3/((lambda_c*10**-6)**(3+alpha)))/(np.exp(hc_k/T/(lambda_c*10**-6)) - 1.)

	# At certain points in parameter space, this factor gets too big to take the exponential of, resulting in math errors
	greybody_factor = hc_k/T/wavs_m

	S_jy_greybody = np.zeros(wavs_m.shape[0])

	# Therefore only actually evaluate the exponential if greybody_factor is greater than 300, otherwise greybody flux is zero
	S_jy_greybody[greybody_factor < 300.] = Nbb*(1-np.exp(-(lambda_0*10**-6/wavs_m[greybody_factor < 300.])**beta))*(((3*10**8)/wavs_m[greybody_factor < 300.])**3)/(np.exp(greybody_factor[greybody_factor < 300.]) - 1.)

	S_jy_powerlaw = Npl*(wavs_m**alpha)*np.exp(-(wavs_m/(lambda_c*10**-6))**2)

	return S_jy_greybody + S_jy_powerlaw



def prior_transform(cube, ndim, nparams):
	""" Prior function for MultiNest algorithm, converts unit cube to uniform prior between set limits. """
	
	for i in range(len(fit_limits)):
		cube[i] = fit_limits[i][0] + (fit_limits[i][1] - fit_limits[i][0])*cube[i]

	return cube



def log_like(param, ndim, nparams):
	""" Function to return the log_likelihood values at given parameter values: param. """

	log_like = -0.5*np.sum(np.log(2.*np.pi*errors**2)) - 0.5*np.sum(((fluxes - model(param))**2)/((errors)**2))

	return log_like



ndim = len(fit_limits)


pmn.run(log_like, prior_transform, ndim, importance_nested_sampling = False, verbose = True, sampling_efficiency = "parameter", n_live_points = 400, outputfiles_basename= "dust_fit-")


a = pmn.Analyzer(n_params = ndim, outputfiles_basename="dust_fit-")
s = a.get_stats()


mode_evidences = []
for i in range(len(s["modes"])):
    mode_evidences.append(s["modes"][i]["local log-evidence"])


# Best fit values taken to be the highest point in probability surface
best_fitvals = s["modes"][np.argmax(mode_evidences)]["maximum"]


# Median values of the marginalised posterior distributions
posterior_median = np.zeros(ndim)
for j in range(ndim):
    posterior_median[j] = s["marginals"][j]["median"]


# Confidence interval - tuple with 16th and 84th percentile values for each parameter
conf_int = []
for j in range(ndim):
    conf_int.append((s["marginals"][j]["1sigma"][0], s["marginals"][j]["1sigma"][1]))


print " "
print "Min Chisq Reduced: ", np.sum(((fluxes - model(best_fitvals))**2)/(errors**2))
print " "
print "Confidence interval:"
for x in range(ndim):
    print str(np.round(conf_int[x], 4)), np.round(best_fitvals[x], 4), fit_params[x]
print " "


posterior = np.loadtxt("dust_fit-post_equal_weights.dat")[:,:-1]

fig = corner.corner(posterior, labels=["$N_{bb}$", "$N_{pl}$", "$T$", "$\\alpha$", "$\\beta$"], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}, smooth=1)

fig.savefig("corner.pdf")

#plt.show()


posterior_spectra = np.zeros((999, posterior.shape[0]))

for i in range(posterior.shape[0]):
	posterior_spectra[:,i] = model_highsampling(posterior[i,:])

# Plot the model posterior
plt.figure()
plt.errorbar(wavs, fluxes, yerr=errors, lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=9, color="black")
plt.scatter(wavs, fluxes, color="blue", s=75, zorder=10, linewidth=1, facecolor="blue", edgecolor="black")
plt.fill_between(np.arange(1., 1000.), np.percentile(posterior_spectra, 16, axis=1), np.percentile(posterior_spectra, 84, axis=1), color="navajowhite")
plt.plot(np.arange(1., 1000.), np.percentile(posterior_spectra, 50, axis=1), color="darkorange")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Wavelength ($\mathrm{\mu}$m)")
plt.ylabel("Flux (Jy)")
plt.ylim(10**-5, 2.)
plt.xlim(2., 1000.)

plt.savefig("posterior.pdf")

#plt.show()



fnames = glob("dust_fit-*")

for fname in fnames:
	call(["rm", fname])

