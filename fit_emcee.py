import numpy as np 
import emcee
import corner
import matplotlib.pyplot as plt

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
wavs   = np.array([8., 24., 100., 160., 250., 450., 850.]) # Microns
fluxes = np.array([3.94000000e-05, 1.82737185e-04, 1.40000000e-01, 1.40000000e-01, 6.79557736e-01, 1.88463960e-02, 4.85758671e-03]) # Janskys
errors = np.array([2.21000000e-06, 7.39953076e-06, 7.60000000e-02, 9.70000000e-02, 5.22661289e-01, 6.47668353e-03, 4.98186312e-03]) # Janskys


# Implement error floor at 10 sigma
for i in range(fluxes.shape[0]):
	if fluxes[i]/errors[i] > 10.:
		errors[i] = fluxes[i]/10.


# Prior limits on parameter values
param_limits = []
param_limits.append((-45., -30.)) # Prior limits on log_10(Nbb)
param_limits.append((-10., 20.)) # Prior limits on log_10(Npl)
param_limits.append((0., 273.)) # Prior limits on T
param_limits.append((0.5, 5.5)) # Prior limits on alpha
param_limits.append((0.5, 2.5)) # Prior limits on beta


def model(param, wavs_micron):
	"""Function to evaluate the Casey (2012) model with parameters: param, and at wavelengths: wavs:"""

	wavs = wavs_micron*10**-6 # Convert wavelength values from microns to metres

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
	greybody_factor = hc_k/T/wavs

	S_jy_greybody = np.zeros(wavs.shape[0])

	# Therefore only actually evaluate the exponential if greybody_factor is greater than 300, otherwise greybody flux is zero
	S_jy_greybody[greybody_factor < 300.] = Nbb*(1-np.exp(-(lambda_0*10**-6/wavs[greybody_factor < 300.])**beta))*(((3*10**8)/wavs[greybody_factor < 300.])**3)/(np.exp(greybody_factor[greybody_factor < 300.]) - 1.)

	S_jy_powerlaw = Npl*(wavs**alpha)*np.exp(-(wavs/(lambda_c*10**-6))**2)

	return S_jy_greybody + S_jy_powerlaw




def log_prior(param):
	""" Function to return the log_prior value at a given set of parameter values: param, assumed flat prior across given intervals. """

	# If the parameter values stray outside the ranges given, return a huge negative value for the prior probability
	for i in range(param.shape[0]):
		if not param_limits[i][0] < param[i] < param_limits[i][1]:
			return -9.9*10**99

	else:
		return 0.


# Function to return the log_likelihood values at given parameter values: param, given x_values, y_values and y_sigmas.
def log_like(param, x_values, y_values, y_sigmas):

	log_like = -0.5*np.sum(np.log(2.*np.pi*y_sigmas**2)) - 0.5*np.sum(((y_values - model(param, x_values))**2)/(y_sigmas**2))

	return log_like


# The un-normalised posterior probability density at parameter values: param, given x_values, y_values and y_sigmas.
def log_prob(param, x_values, y_values, y_sigmas):
	
	log_pri = log_prior(param)

	# Stops the log_like function being evaluated for parameter values outside the allowed range, as if it returns nan emcee crashes
	if log_pri == -9.9*10**99:
		return log_pri
	
	return log_like(param, x_values, y_values, y_sigmas) + log_pri



# Plot an example dust model over the data - basically I fiddled around with the parameters until I got something that looked reasonable
plt.figure()
plt.errorbar(wavs, fluxes, yerr=errors, lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=9, color="black")
plt.scatter(wavs, fluxes, color="blue", s=75, zorder=10, linewidth=1, facecolor="blue", edgecolor="black")
plt.plot(np.arange(1., 1000.), model([-36.58843886, 1.88639532, 36.8513886, 1.23338528, 0.99589455], np.arange(1., 1000.)))
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Wavelength ($\mathrm{\mu}$m)")
plt.ylabel("Flux (Jy)")
plt.ylim(10**-5, 1.)
plt.show()

raw_input("Press enter to continue.")

# Set the number of dimensions, walkers, steps for emcee
ndim, nwalkers, nsteps = 5, 200, 1000

# Set up the array of starting points for the walkers
p0 = np.zeros((nwalkers, ndim))

# Manually enter a "good fit" starting point - this was the best fit from a previous run
start_point = [-36.58843886, 1.88639532, 36.8513886, 1.23338528, 0.99589455]

# Randomly distribute your walkers around that starting point
for i in range(ndim):
	p0[:,i] = 0.001*np.random.randn(nwalkers) + start_point[i]

# Generate and run the sampler object
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(wavs, fluxes, errors))

sampler.run_mcmc(p0, nsteps)

# Extract the second half of the chain, allowing the first half as a burn-in period
samples = sampler.chain[:, (nsteps/2):, :].reshape((-1, ndim))

# Generate an array to hold minimum reduced chi sqaured values
chisq_reduced = np.zeros(nwalkers)

# Calculate the minimum reduced chi squared value for the final position of each walker
for i in range(nwalkers):
	chisq_reduced[i] = np.sum(((fluxes - model(sampler.chain[i, -1 ,:], wavs))**2)/(errors**2))/(wavs.shape[0] - ndim)

print "Minimum reduced chi-squared value:", np.min(chisq_reduced)

print "Best fit parameters:", sampler.chain[np.argmin(chisq_reduced), -1 ,:]

# Generate a corner plot of the posterior distribution
fig = corner.corner(samples, labels=["$N_{bb}$", "$N_{pl}$", "$T$", "$\\alpha$", "$\\beta$"], smooth=1)
plt.show()

# Plot the model posterior
plt.figure()
plt.errorbar(wavs, fluxes, yerr=errors, lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=9, color="black")
plt.scatter(wavs, fluxes, color="blue", s=75, zorder=10, linewidth=1, facecolor="blue", edgecolor="black")
for i in range(nwalkers):
	plt.plot(np.arange(1., 1000.), model(sampler.chain[i, -1 ,:], np.arange(1., 1000.)), alpha=0.05, color="gray")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Wavelength ($\mathrm{\mu}$m)")
plt.ylabel("Flux (Jy)")
plt.ylim(10**-5, 1.)
plt.show()

