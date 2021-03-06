#Kendall Sullivan

#EMCEE VERSION OF MCMC CODE

#TO DO: Write add_disk function, disk/dust to possible fit params

#20190522: Added extinction with a fixed value to model fitting (prior to fit), updated models to theoretical PHOENIX BT-SETTL models with the 
#CFIST line list downloaded from the Spanish Virtual Observatory "Theoretical Tools" resource. 
#20190901 (ish) Updated model spectra to be the PHOENIX BT-SETTL models with the CFIST 2011 line list downloaded from France Allard's website
#and modified from the FORTRAN format into more modern standard text file format before using
#20200514 Commented everything that's out of beta/active development (or at the very least mostly done) - that goes through the end of get_spec
#the remaining content probably needs to be pared down and likely should be troubleshot fairly carefully

"""
.. module:: model_fit_tools_v2
   :platform: Unix, Mac
   :synopsis: Large package with various spectral synthesis and utility tools.

.. moduleauthor:: Kendall Sullivan <kendallsullivan@utexas.edu>

Dependencies: numpy, synphot, matplotlib, astropy, scipy, PyAstronomy, emcee, corner, extinction.
"""

import numpy as np
# import pysynphot as ps
import synphot;from specutils import Spectrum1D
import matplotlib; matplotlib.use('Agg'); from matplotlib import pyplot as plt
from astropy.io import fits
import os 
from glob import glob
from astropy import units as u
#from matplotlib import rc
from itertools import permutations 
import time, sys, getopt
import scipy.stats
# from mpi4py import MPI
import timeit
from PyAstronomy import pyasl
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy import ndimage
import emcee
import corner
import extinction
import time
import multiprocessing as mp
import matplotlib.ticker
# from schwimmbad import MPIPool
# from scipy.optimize import differential_evolution
# from labellines import labelLine, labelLines
import matplotlib.cm as cm
import pyphot; lib = pyphot.get_library()

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def bccorr(wl, bcvel, radvel):
	"""Calculates a barycentric velocity correction given a barycentric and/or a radial velocity (set any unused value to zero)

	Args: 
		wl (array): wavelength vector.
		bcvel (float): a barycentric or heliocentric velocity in km/s
		radvel (float): a systemic radial velocity in km/s

	Returns: 
		a wavelength vector corrected for barycentric and radial velocities.

	"""
	return np.array(wl) * (1. + (bcvel - radvel)/3e5)

def extinct(wl, spec, av, rv = 3.1, unit = 'aa'):
	"""Uses the package "extinction" to calculate an extinction curve for the given A_v and R_v, 
	then converts the extinction curve to a transmission curve
	and uses that to correct the spectrum appropriately.
	Accepted units are angstroms ('aa', default) or microns^-1 ('invum').

	Args:
		wl (array): wavelength array
		spec (array): flux array
		av (float): extinction in magnitudes
		rv (float): Preferred R_V, defaults to 3.1
		unit (string): Unit to use. Accepts angstroms "aa" or inverse microns "invum". Defaults to angstroms.

	Returns:
		spec (list): a corrected spectrum vwith no wavelength vector. 

	"""
	ext_mag = extinction.ccm89(wl, av, rv, unit)
	spec = extinction.apply(ext_mag, spec)
	return np.array(spec)

def get_radius(teff, matrix):
	#assume I'm using MIST 5 gyr model

	aage = matrix[:, 1]

	teff5, lum5 = matrix[:,4][np.where(np.array(aage) == 9.3000000000000007)], matrix[:,6][np.where(np.array(aage) == 9.3000000000000007)]

	lum5 = [10**l for l in lum5]; teff5 = [10**t for t in teff5]

	intep = interp1d(teff5[:220], lum5[:220]); lum = intep(teff)

	sigma_sb = 5.670374e-5 #erg/s/cm^2/K^4
	lsun = 3.839e33 #erg/s 
	rsun = 6.955e10
	rad = np.sqrt(lum*lsun/(4 * np.pi * sigma_sb * teff**4))/rsun #solar radii

	return rad

def find_nearest(array, value):
	"""finds index in array such that the array component at the returned index is closest to the desired value.
	
	Args: 
		array (list): Array to search.
		value (float or int): Value to find closest value to.

	Returns: 
		idx (int): index at which array is closest to value

	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def chisq(model, data, var):
	"""Calculates chi square values of a model and data with a given variance.

	Args:
		model (list): model array.
		data (list): data array. Must have same len() as model array.
		variance (float or list): Data variance. Defaults to 10.

	Returns: 
		cs (float): Reduced chi square value.

	"""
	#make sure that the two arrays are comparable
	# print(np.where(var == 0), np.where(var < 0))

	if len(data) == len(model):
		#if there's a variance array, iterate through it as i iterate through the model and data
		if np.size(var) > 1 or type(var) == list:
			#calculate the chi square vector using xs = (model - data)^2/variance^2 per pixel
			xs = [((model[n] - data[n])**2)/var[n]**2 for n in range(len(model))]
		#otherwise do the same thing but using the same variance value everywhere
		else:
			xs = [((model - data)**2)/var**2 for n in range(len(model))]
		#return the chi square vector
		return np.asarray(xs)#np.sum(xs)/len(xs)
	#if the two vectors aren't the same length, yell at me
	else:
		return('data must be equal in length to model')

def shift(wl, spec, rv, bcarr, **kwargs):
	"""for bccorr, use bcarr as well, which should be EITHER:
	1) the pure barycentric velocity calculated elsewhere OR
	2) a dictionary with the following entries (all as floats, except the observatory name code, if using): 
	{'ra': RA (deg), 'dec': dec (deg), 'obs': observatory name or location of observatory, 'date': JD of midpoint of observation}
	The observatory can either be an observatory code as recognized in the PyAstronomy.pyasl.Observatory list of observatories,
	or an array containing longitude, latitude (both in deg) and altitude (in meters), in that order.

	To see a list of observatory codes use "PyAstronomy.pyasl.listobservatories()".
	
	Args:
		wl (list): wavelength array
		spec (list): flux array
		rv (float): Rotational velocity value
		bcarr (list): if len = 1, contains a precomputed barycentric velocity. Otherwise, should 
			be a dictionary with the following properties: either an "obs" keyword and code from pyasl
			or a long, lat, alt set of floats identifying the observatory coordinates.  

	Returns:
		barycentric velocity corrected wavelength vector using bccorr().

	"""
	if len(bcarr) == 1:
		bcvel = bcarr[0]
	elif len(bcarr) > 1:
		if isinstance(bcarr['obs'], str):
			try:
				ob = pyasl.observatory(bcarr['obs'])
			except:
				print('This observatory code didn\'t work. Try help(shift) for more information')
			lon, lat, alt = ob['longitude'], ob['latitude'], ob['altitude']
		if np.isarray(bcarr['obs']):
			lon, lat, alt = bcarr['obs'][0], bcarr['obs'][1], bcarr['obs'][2]
		bcvel = pyasl.helcorr(lon, lat, alt, bcarr['ra'], bcarr['dec'], bcarr['date'])[0]

	bc_wl = bccorr(wl, bcvel)

	return bc_wl

def broaden(even_wl, modelspec_interp, res, vsini = 0, limb = 0, plot = False):
	"""Adds resolution, vsin(i) broadening, taking into account limb darkening.

	Args: 
		even_wl (list): evenly spaced model wavelength vector
		modelspec_interp (list): model spectrum vector
		res (float): desired spectral resolution
		vsini (float): star vsin(i)
		limb (float): the limb darkening coeffecient
		plot (boolean): if True, plots the full input spectrum and the broadened output. Defaults to False.

	Returns:
		a tuple containing an evenly spaced wavelength vector spanning the width of the original wavelength range, and a corresponding flux vector

	"""

	#regrid by finding the smallest wavelength step 
	# mindiff = np.inf

	# for n in range(1, len(even_wl)):
	# 	if even_wl[n] - even_wl[n-1] < mindiff:
	# 		mindiff = even_wl[n] - even_wl[n-1]

	# #interpolate the input values
	# it = interp1d(even_wl, modelspec_interp)

	# #make a new wavelength array that's evenly spaced with the smallest wavelength spacing in the input wl array
	# w = np.arange(min(even_wl), max(even_wl), mindiff)

	# sp = it(w)
	#do the instrumental broadening and truncate the ends because they get messy
	broad = pyasl.instrBroadGaussFast(even_wl, modelspec_interp, res, maxsig=5)
	broad[0:5] = broad[5] 
	broad[len(broad)-10:len(broad)] = broad[len(broad) - 11]

	#if I want to impose stellar parameters of v sin(i) and limb darkening, do that here
	if vsini != 0 and limb != 0:
		rot = pyasl.rotBroad(even_wl, broad, limb, vsini)#, edgeHandling='firstlast')
	#otherwise just move on
	else:
		rot = broad

	#Make a plotting option just in case I want to double check that this is doing what it's supposed to
	if plot == True:

		plt.figure()
		plt.plot(even_wl, sp, label = 'model')
		plt.plot(even_wl, broad, label = 'broadened')
		plt.plot(even_wl, rot, label = 'rotation')
		plt.legend(loc = 'best')
		plt.xlabel('wavelength (angstroms)')
		plt.ylabel('normalized flux')
		plt.savefig('rotation.pdf')

	#return the wavelength array and the broadened flux array
	return np.array(even_wl), np.array(rot)

def redres(wl, spec, factor):
	"""Imposes instrumental resolution limits on a spectrum and wavelength array
	Assumes evenly spaced wl array

	"""
	#decide the step size by using the median original wl spacing and then increase it by the appropriate factor
	new_stepsize = np.median([wl[n] - wl[n-1] for n in range(1, len(wl))]) * factor

	#make a new wl array using the new step size
	wlnew = np.arange(min(wl), max(wl), new_stepsize)

	#interpolate the spectrum so it's on the new wavelength scale 
	i = interp1d(wl, spec)
	specnew = i(wlnew)

	#return the reduced spectrum and wl array
	return np.array(wlnew), np.array(specnew)

def rmlines(wl, spec, **kwargs):
	"""Edits an input spectrum to remove emission lines

	Args: 
		wl (list): wavelength
		spec (list): spectrum.
		add_line (boolean): to add more lines to the linelist (interactive)
		buff (float): to change the buffer size, input a float here. otherwise the buffer size defaults to 15 angstroms
		uni (boolean): specifies unit for input spectrum wavelengths (default is microns) [T/F]
		conv (boolean): if unit is true, also specify conversion factor (wl = wl * conv) to microns

	Returns: 
		spectrum with the lines in the linelist file removed if they are in emission.

	"""
	#reads in a linelist, which contains the names, transition, and wavelength of each emission line
	names, transition, wav = np.genfromtxt('linelist.txt', unpack = True, autostrip = True)
	#define the gap to mark out
	space = 1.5e-3 #15 angstroms -> microns

	#check the kwargs
	for key, value in kwargs.items():
		#if i want to add a line, use add_lines, which then lets me append a new line
		if key == add_line:
			wl.append(input('What wavelengths (in microns) do you want to add? '))
		#if i want to change the size of region that's removed, use a new buffer
		if key == buff:
			space = value
		#if i want to change the unit, use uni to do so
		if key == uni:
			wl = wl * value

	diff = wl[10] - wl[9]

	#for each line, walk trhough and remove the line, replacing it with the mean value of the end points of the removed region
	for line in wav:
		end1 = find_nearest(wl, line-space)
		end2 = find_nearest(wl, line+space)
		if wl[end1] > min(wl) and wl[end2] < max(wl) and (end1, end2)> (0, 0) and (end1, end2) < (len(wl), len(wl)):
			for n in range(len(wl)):
				if wl[n] > wl[end1] and wl[n] < wl[end2] and spec[n] > (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2:
					spec[n] = (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2
	#return the spectrum
	return spec

def make_reg(wl, flux, waverange):
	"""given some wavelength range as an array, output flux and wavelength vectors within that range.

	Args:
		wl (list): wavelength array
		flux (list): flux array
		waverange (list): wavelength range array

	Returns: 
		wavelength and flux vectors within the given range

	"""
	#find the smallest separation in the wavelength array

	#interpolate the input spectrum
	wl_interp = interp1d(wl, flux)
	#make a new wavelength array that's evenly spaced with the minimum spacing
	wlslice = np.arange(min(waverange), max(waverange), wl[1]-wl[0])
	#use the interpolation to get the evenly spaced flux
	fluxslice = wl_interp(wlslice)
	#return the new wavelength and flux
	return np.array(wlslice), np.array(fluxslice)

def interp_2_spec(spec1, spec2, ep1, ep2, val):
	"""Args: 
		spec1 (list): first spectrum array (fluxes only)
		spec2 (list): second spectrum array (fluxes only)
		ep1 (float): First gridpoint of the value we want to interpolate to.
		ep2 (float): Second gridpoint of the value we want to interpolate to.
		val (float): a value between ep1 and ep2 that we wish to interpolate to.

	Returns: 
		a spectrum without a wavelength parameter

	"""	
	ret_arr = []
	#make sure the two spectra are the same length
	if len(spec1) == len(spec2):
		#go through the spectra
		# for n in range(len(spec1)):
			#the new value is the first gridpoint plus the difference between them weighted by the spacing between the two gridpoints and the desired value.
			#this is a simple linear interpolation at each wavelength point
		ret_arr = ((np.array(spec2) - np.array(spec1))/(ep2 - ep1)) * (val - ep1) + np.array(spec1)
			# ret_arr.append(v)
		#return the new interpolated flux array
		return ret_arr

	#otherwise yell at me because i'm trying to interpolate things that don't have the same length
	else:
		return('the spectra must have the same length')

def make_varied_param(init, sig):
	"""randomly varies a parameter within a gaussian range based on given std deviation

	Args:
		init (float): initial value
		sig (float): std deviation of gaussian to draw from

	Returns: 
		the varied parameter.

	"""
	#initialize the variance list to return
	var = []
	#loop through the std devs 
	for s in sig:
		#check to make sure that the std devs are physical, and if they aren't print them out
		if any(a < 0 for a in s):
			print(s, np.where(sig == s), init[np.where(sig == s)])
	#then, loop through each of the input parameters
	for n in range(len(init)):
		#and if it's a single value just perturb it and put it into the array to return
		try:
			var.append(np.random.normal(init[n], sig[n]))
		#if the input parameter is itself an array, perturb each value in the array with the appropriate std dev and then save it
		except:
			var.append(list(np.random.normal(init[n], sig[n])))
	#return the perturbed values
	return var

def find_model(temp, logg, metal, models = 'btsettl'):
	"""Finds a filename for a phoenix model with values that fall on a grid point.
	Assumes that model files are in a subdirectory of the working directory, with that subdirectory called "SPECTRA"
	and that the file names take the form "lte{temp}-{log g}-{metallicity}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
	The file should contain a flux column, where the flux is in units of log(erg/s/cm^2/cm/surface area). There should also be a
	wavelength file in the spectra directory called WAVE_PHOENIX-ACES-AGSS-COND-2011.fits, with wavelength in Angstroms.
	THE NEW SPECTRA are from Husser et al 2013.

	Args: 
		temperature (float): temperature value
		log(g) (float): log(g) value
		metallicity (float): Metallicity value

	Note:
		Values must fall on the grid points of the model grid. Only supports log(g) = 4 with current spectra directory.

	Returns: 
		file name of the phoenix model with the specified parameters.

	"""
	#if using the hires phoenix models call using the correct formatting
	if models == 'hires':
		temp = str(int(temp)).zfill(5)
		metal = str(float(metal)).zfill(3)
		logg = str(float(logg)).zfill(3)
		file = glob('SPECTRA/lte{}-{}0-{}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits.txt'.format(temp, logg, metal))[0]
		return file
	#or if using BT-Settl, the other supported model, do the same
	#but here assume that we want the metallicity = 0 scenario (since those are the only ones I have downloaded)
	elif models == 'btsettl':
		# temp = str(temp/1e2).zfill(5)
		temp = str(int(temp/1e2)).zfill(3)
		metal = 0.0
		logg = str(logg)
		file = glob('../../../../Research/BT-Settl_M-0.0a+0.0/lte{}-{}-0.0a+0.0.BT-Settl.spec.7.txt'.format(temp, logg))[0]
		# file = glob('BT-Settl_M-0.5_a+0.2/lte{}-{}-0.5a+0.2.BT-Settl.spec.7.txt'.format(temp, logg))[0]
		return file

def spec_interpolator(w, trange, lgrange, specrange, npix = 3, resolution = 10000, metal = 0, write_file = True, models = 'btsettl'):
	'''Runs before emcee, to read in files to memory

	Args:
		trange (array): minimum and maximum temperature limits to be read in
		lgrange (array): minimum and maximum log(g) limits to be read in
		specrange (array): minimum and maximum wavelengths to use (in Angstroms)
		npix (int): factor by which to reduce the resolution. Default is 3
		resolution (int): resolution at which to store the spectra - should be larger than the final desired resolution
		metal (float): metallicity to use. Defaults to 0, which is also the only currently supported value.

	'''
	#first, read in the wavelength vector
	if models == 'hires':
		with open('SPECTRA/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits.txt', 'r') as wave:
			spwave = []
			for line in wave:
				spwave.append(float(line))
			wave.close()
			spwave = np.array(spwave)
		#and find the indices where it's within the desired wavelength range - because this will be SET
		idx = np.where((spwave >= min(specrange)) & (spwave <= max(specrange)))
		spwave = spwave[idx]

		#now we need to collect all the files within the correct temperature and log(g) range
		#get the files
		files = glob('SPECTRA/lte*txt')
		#initialize temperature and log(g) arrays
		t = []
		l = []
		#sort through and pick out the temperature and log(g) value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[0].split('e')[1]
			mu = float(files[n].split('-')[1])

			if len(nu) < 4:
				nu = int(nu) * 1e2
			else:
				nu = int(nu)

			#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
			if nu not in t and nu >= min(trange) and nu <= max(trange):
				t.append(nu)
			if mu not in l and mu >= min(lgrange) and mu <= max(lgrange):
				l.append(mu)
		#now, make a dictionary where each name is formatted as 'teff, log(g)' and the entry is the spectrum
		specs = {}

		#for each (temperature, log(g)) combination, we need to read in the spectrum
		#select out the correct wavelength region
		#and downsample
		for n in range(len(t)):
			for k in range(len(l)):
				print(n, k)
				#find the correct file
				file = find_model(t[n], l[k], metal, models = models)
				#read it in
				with open(file, 'r') as f1:
					spec1 = []
					for line in f1:
						spec1.append(float(line))
				#select the right wavelength region
				spec1 = np.array(spec1)[idx]

				#downsample - default is 3 pixels per resolution element
				res_element = np.mean(spwave)/resolution
				spec_spacing = spwave[1] - spwave[0]
				if npix * spec_spacing < res_element:
					factor = (res_element/spec_spacing)/npix
					wl, spec1 = redres(spwave, spec1, factor)

				#next, we just add it to the dictionary with the correct (temp, log(g)) tuple identifying it

				specs['{}, {}'.format(t[n], l[k])] = np.array(spec1)

		specs['wl'] = np.array(wl)

	if models == 'btsettl':
		files = glob('../../../../Research/BT-Settl_M-0.0a+0.0/lte*')

		#initialize temperature and log(g) arrays
		t = []
		l = []
		#sort through and pick out the temperature and log(g) value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[2].split('e')[1]
			mu = float(files[n].split('-')[3])

			nu = int(float(nu) * 1e2)

			#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
			if nu not in t and nu >= min(trange) and nu <= max(trange):
				t.append(nu)
			if mu not in l and mu >= min(lgrange) and mu <= max(lgrange):
				l.append(mu)
		#now, make a dictionary where each name is formatted as 'teff, log(g)' and the entry is the spectrum
		specs = {}; wls = {}
		wl = np.arange(min(specrange), max(specrange), 0.2)
		#for each (temperature, log(g)) combination, we need to read in the spectrum
		#select out the correct wavelength region
		#and downsample
		for n in range(len(t)):
			for k in range(len(l)):
				print(n, k)
				#find the correct file and read it in
				spold, spec1 = [], []
				#then find the correct file and save all the values that fall within the requested spectral range
				with open(find_model(t[n], l[k], metal, models = 'btsettl')) as file:
					for line in file:
						li = line.split(' ')
						if float(li[0]) >= min(specrange) - 100 and float(li[0]) <= max(specrange) + 100:
							spold.append(float(li[0])); spec1.append(float(li[1]))

				# spold, spec1 = np.genfromtxt(find_model(t[n], l[k], metal, models = 'btsettl')).T
				# spold, spec1 = spold[np.where((spold >= min(specrange) - 100) & (spold <= max(specrange) + 100))], spec1[np.where((spold >= min(specrange) - 100) & (spold <= max(specrange) + 100))]
				
				#next, we just add it to the dictionary with the correct (temp, log(g)) tuple identifying it
				specs['{}, {}'.format(t[n], l[k])] = np.array(spec1)
				wls['{}, {}'.format(t[n], l[k])] = np.array(spold)

		#now, for each spectrum we need to impose instrumental broadening
		for k in specs.keys():
			#select a spectrum and create an interpolator for it
			itep = interp1d(wls[k], specs[k])
			#then interpolate it onto the correct wavelength vcector
			spflux = itep(wl)
			#then instrumentally broaden the full spectrum
			ww, brd = broaden(wl[np.where((wl >= min(w))&(wl <= max(w)))], spflux[np.where((wl >= min(w))&(wl <= max(w)))], resolution)
			#we want to save the spectrum at original resolution where we can for better photometry
			#so create a new spectrum at original resolution outside the data spectrum range and at the data resolution inside it 
			newsp = np.concatenate((spflux[np.where(wl < min(w))], brd, spflux[np.where(wl > max(w))]))
			#and save to the dictionary
			specs[k] = newsp

		#fix the wavelength vector so it matches the spectrum wavelength scale
		wlnew = np.concatenate((wl[np.where(wl<min(w))], ww, wl[np.where(wl>max(w))]))
		#save the wavelength vector since it's now a common vector amongst all the spectra
		specs['wl'] = np.array(wlnew)
	#return the spectrum dictionary
	return specs

def get_spec(temp, log_g, reg, specdict, metallicity = 0, normalize = False, wlunit = 'aa', pys = False, plot = False, models = 'btsettl', resolution = 1000, reduce_res = False, npix = 3):
	"""Creates a spectrum from given parameters, either using the pysynphot utility from STScI or using a homemade interpolation scheme.
	Pysynphot may be slightly more reliable, but the homemade interpolation is more efficient (by a factor of ~2).
	
	TO DO: add a path variable so that this is more flexible, add contingency in the homemade interpolation for if metallicity is not zero

	Args: 
		temp (float): temperature value
		log_g (float): log(g) value
		reg (list): region array ([start, end])
		metallicity (float): Optional, defaults to 0
		normalize (boolean): Optional, defaults to True
		wlunit: Optional, wavelength unit. Defaults to angstroms ('aa'), also supports microns ('um').
		pys (boolean): Optional, set to True use pysynphot. Defaults to False.
		plot (boolean): Produces a plot of the output spectrum when it is a value in between the grid points and pys = False (defaults to False).
		resolution (int): Spectral resolution to broaden the spectrum to. Default is 3000.
		reduce_res (boolean): Whether to impose "pixellation" onto the spectrum using a designated number of pixels per resolution element. Default is True.
		npix (int): Number of pixels per resolution element if pixellating the spectrum.

	Returns: 
		a wavelength array and a flux array, in the specified units, as a tuple. Flux is in units of F_lambda (I think)

	Note:
		Uses the Phoenix models as the base for calculations. 

	"""
	if pys == True:
	#grabs a phoenix spectrum using Icat calls via pysynphot (from STScI) defaults to microns
	#get the spectrum
		sp = ps.Icat('phoenix', temp, metallicity, log_g)
		#put it in flambda units
		sp.convert('flam')
		#make arrays to eventually return so we don't have to deal with subroutines or other types of arrays
		spflux = np.array(sp.flux, dtype='float')
		spwave = np.array(sp.wave, dtype='float')

	if pys == False:
		#we have to:
		#read in the synthetic spectra
		#pick our temperature and log g values (assume metallicity is constant for now)
		#pull a spectrum 
		#initialize a time variable if i want to check how long this takes to run
		time1 = time.time()
		#list all the spectrum files
		if models == 'hires':
			files = glob('SPECTRA/lte*txt')

			#initialize a tempeature array
			t = []
			#sort through and pick out the temperature value from each file name
			for n in range(len(files)):
				nu = files[n].split('-')[0].split('e')[1]
				if len(nu) < 4:
					nu = int(nu) * 1e2
					t.append(nu)
				else:
					t.append(int(nu))
			#sort the temperature array so it's in order
			t = sorted(t)
			#initialize a non-redundant array
			temps = [min(t)]

			#go through the sorted array and if the temperature isn't already in the non-redundant array, put it in
			for n, tt in enumerate(t):
				if tt > temps[-1]:
					temps.append(tt)

		if models == 'btsettl':
			files = glob('../../../../Research/BT-Settl_M-0.0a+0.0/lte*')
			#initialize temperature and log(g) arrays
			t = []
			l = []
			#sort through and pick out the temperature and log(g) value from each file name
			for n in range(len(files)):
				nu = files[n].split('-')[2].split('e')[1]
				nu = int(float(nu) * 1e2)

				#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
				if nu not in t:
					t.append(nu)

				temps = sorted(t)

		#find the closest temperature to the input value
		t1_idx = find_nearest(temps, temp)

		#if that input value is on a grid point, make the second spectrum the same temperature
		if temps[t1_idx] == temp:
			t2_idx = t1_idx
		#if the nearest temp value is above the input value, the other temperature should fall below
		elif temps[t1_idx] > temp:
			t2_idx = t1_idx - 1
		#otherwise the second temperature should fall above
		else:
			t2_idx = t1_idx + 1

		#temp1 and temp2 have been selected to enclose the temperature, or to be the temperature exactly if the temp requested is on a grid point
		temp1 = temps[t1_idx]
		temp2 = temps[t2_idx]

		#now do the same thing for log(g)
		if models == 'btsettl':
			l = sorted([float(files[n].split('-')[3]) for n in range(len(files))])
		if models == 'hires':
			l = sorted([float(files[n].split('-')[1]) for n in range(len(files))])

		lgs = [min(l)]

		for n, tt in enumerate(l):
			if tt > lgs[-1]:
				lgs.append(tt)

		lg1_idx = find_nearest(lgs, log_g)
		 
		if lgs[lg1_idx] == log_g:
			lg2_idx = lg1_idx
		elif lgs[lg1_idx] > log_g:
			lg2_idx = lg1_idx - 1
		else:
			lg2_idx = lg1_idx + 1

		lg1 = lgs[lg1_idx]
		lg2 = lgs[lg2_idx]

		#so now I have four grid points: t1 lg1, t1 lg2, t2 lg1, t2 lg2. now i have to sort out whether some of those grid points are the same value
		#first, get the wavelength vector for everything 
		spwave = specdict['wl']
		#define a first spectrum using t1 lg1 and unpack the spectrum 

		#if I'm looking for a log(g) and a temperature that fall on a grid point, things are easy
		#just open the file 

		#the hires spectra are in units of erg/s/cm^2/cm, so divide by 1e8 to get erg/s/cm^2/A
		if lg1 == lg2 and temp1 == temp2:
			spflux = specdict['{}, {}'.format(temp1, lg1)]
			if models == 'hires':
				spflux /= 1e8

		#If the points don't all fall on the grid points, we need to get the second spectrum at point t2 lg2, as well as the cross products
		#(t1 lg2, t2 lg1)
		else:
			#find all the spectra 
			spec1 = specdict['{}, {}'.format(temp1, lg1)]
			spec2 = specdict['{}, {}'.format(temp2, lg2)]
			t1_inter = specdict['{}, {}'.format(temp1, lg2)]
			t2_inter = specdict['{}, {}'.format(temp2, lg1)]

			#if using hires correct the models to get the correct spectrum values
			if models == 'hires':
				spec1, spec2, t1_inter, t2_inter = spec1/1e8, spec2/1e8, t1_inter/1e8, t2_inter/1e8

			#if t1 and t2 AND lg1 and lg2 are different, we need to interpolate first between the two log(g) points, then the two teff points
			if lg1 != lg2 and temp1 != temp2:
				t1_lg = interp_2_spec(spec1, t1_inter, lg1, lg2, log_g)
				t2_lg = interp_2_spec(t2_inter, spec2, lg1, lg2, log_g)

				tlg = interp_2_spec(t1_lg, t2_lg, temp1, temp2, temp)

			#or if we're looking at the same log(g), we only need to interpolate in temperature
			elif lg1 == lg2 and temp1 != temp2:
				tlg = interp_2_spec(spec1, spec2, temp1, temp2, temp)

			#similarly, if we're using the same temperature but different log(g), we only interpolate in log(g)
			elif temp1 == temp2 and lg1 != lg2:
				tlg = interp_2_spec(spec1, spec2, lg1, lg2, log_g)
			#if you want, make a plot of all the different spectra to compare them
			#this only plots the final interpolated spectrum and the two teff points that are interpolated, after the log(g) interpolation has occurred
			if plot == True:
				wl1a, tla = make_reg(spwave, tlg, [1e4, 1e5])
				wl1a, t1l1a = make_reg(spwave, t1_lg, [1e4, 1e5])
				wl1a, t1l2a = make_reg(spwave, t2_lg, [1e4, 1e5])
				plt.loglog(wl1a, tla, label = 'tl')
				plt.loglog(wl1a, t1l1a, label = 't1l1')
				plt.loglog(wl1a, t1l2a, label = 't1l2')
				plt.legend()
				plt.show()
			
			#reassign some variables used above to match with the environment outside the if/else statement
			# spwave = spwave
			spflux = tlg

	#convert the requested region into angstroms to match the wavelength vector
	reg = np.array(reg)*1e4

	# #make sure the flux array is a float not a string
	# spflux = [float(s) for s in spflux]
	#and truncate the wavelength and flux vectors to contain only the requested region
	spwave, spflux = spwave[np.where((spwave >= min(reg)) & (spwave <= max(reg)))], spflux[np.where((spwave >= min(reg)) & (spwave <= max(reg)))] #make_reg(spwave, spflux, reg)
	#you can choose to normalize to a maximum of one
	if normalize == True:
		spflux /= max(spflux)

	#this is the second time object in case you want to check runtime
	# print('runtime for spectral retrieval (s): ', time.time() - time1)
	#broaden the spectrum to mimic the dispersion of a spectrograph using the input resolution
	# spwave, spflux = broaden(spwave, spflux, resolution)

	#depending on the requested return wavelength unit, do that, then return wavelength and flux as a tuple
	if wlunit == 'aa': #return in angstroms
		return np.array(spwave), np.array(spflux)
	elif wlunit == 'um':
		spwave = spwave * 1e-4
		return np.array(spwave), np.array(spflux)
	else:
		factor = float(input('That unit is not recognized for the return unit. \
			Please enter a multiplicative conversion factor to angstroms from your unit. For example, to convert to microns you would enter 1e-4.'))
		spwave = [s * factor for s in spwave]

		return np.array(spwave), np.array(spflux)

def get_transmission(f, res):
	'''f = filter name, with system if necessary
	'''
	#first, we have to figure out what filter this is
	#make sure it's lowercase
	f = f.lower().strip(',')
	#get the system and filter from the input string
	try:
		if ',' in f:
			syst, fil = f.split(','); syst = syst.strip(); fil = fil.strip()
		#be lazy and hard-code some of these because it's easier and there's a limited number of options in the Furlan paper
		#and everything has a slightly different file format because of course this should be as hard as possible
		else:
			fil = f
			if fil in 'i':
				syst = 'cousins'
			elif fil in 'ubvr':
				syst = 'johnson'
			elif fil == 'kp':
				syst = 'keck'
			elif fil in 'jhks':
				syst = '2mass'
			elif fil in '562 692 880':
				syst = 'dssi'
			elif fil in 'kepler':
				syst = 'kep'
	except:
		print('Please format your filter as, e.g., "Johnson, V". The input is case insensitive.')
	# print(syst, fil)
	#now get the fits file version of the transmission curve from the "bps" directory
	#which should be in the same directory as the code
	#many of these have their own weird format so hard code some of them 
	if fil == 'lp600' or fil == 'LP600': #got the transmission curve from Baranec et al 2014 (ApJL: High-efficiency Autonomous Laser Adaptive Optics)
		filtfile = np.genfromtxt('bps/lp600.csv', delimiter = ',')
		t_wl, t_cv = filtfile[:,0]* 10, filtfile[:,1]
	elif syst == 'kep':
		t_wl, t_cv = np.genfromtxt('bps/Kepler_Kepler.K.dat', unpack = True)
	elif syst == '2mass' or syst == '2MASS':  
		if fil.strip() == 'j' or fil.strip() == 'h':
			filtfile = fits.open('bps/2mass_{}_001_syn.fits'.format(fil.strip()))[1].data
			t_wl, t_cv = filtfile['WAVELENGTH'], filtfile['THROUGHPUT']
		if fil.strip() == 'k' or fil.strip() == 'ks':
			filtfile = np.genfromtxt('bps/2MASS_2MASS.Ks.dat')
			t_wl, t_cv = filtfile[:,0], filtfile[:,1]/max(filtfile[:,1])
	elif syst == 'dssi': #from the SVO transmission curve database for DSSI at Gemini North
		filtfile = np.genfromtxt('bps/DSSI_{}nm.dat'.format(fil))
		t_wl = filtfile[:,0]; t_cv = filtfile[:,1]
	elif syst == 'sdss':
		t_wl, t_cv = np.genfromtxt('bps/SLOAN_SDSS.{}prime_filter.dat'.format(fil)).T
		# filtfile = Table(fits.getdata('bps/sdss_{}_005_syn.fits'.format(fil)))
		# t_wl = np.array(filtfile['WAVELENGTH']); t_cv = np.array(filtfile['THROUGHPUT'])
	elif syst == 'sloan':
		fname = np.array(['u\'', 'g\'', 'r\'', 'i\'', 'z\''])
		n = np.where(fil + '\'' == fname)[0][0]
		filtfile = Table(fits.open('bps/sdss.fits')[n+1].data)
		t_wl = np.array(filtfile['wavelength']); t_cv = np.array(filtfile['respt'])	
	elif syst == 'keck': #taken from the keck website I think? or from SVO
		filtfile = np.genfromtxt('bps/keck_kp.txt')
		t_wl, t_cv = filtfile[:,0], filtfile[:,1]	
		t_wl *= 1e4
	else:
		filtfile = fits.open('bps/{}_{}_002.fits'.format(syst, fil))[1].data

		t_wl, t_cv = filtfile['WAVELENGTH'], filtfile['THROUGHPUT']

	#calculate the size of the mean resolution element and then the number of total resolution elements 
	#this was for way back when I was down-weighting the spectrum by this value, but I don't do that anymore
	#so I suppose this is deprecated now, but it's still built in as a nuisance parameter for all the calls to this function so might as well leave it in for now
	res_element = np.mean(t_wl)/res
	n_resel = (max(t_wl) - min(t_wl))/res_element

	#return the wavelength array, the transmission curve, the number of resolution elements in the bandpass, and the central wavelength
	return t_wl, t_cv, n_resel, np.mean(t_wl)

def make_composite(teff, logg, rad, distance, contrast_filt, phot_filt, r, specs, ctm, ptm, tmi, tma, vs, normalize = False, res = 1000, npix = 3, models = 'btsettl', plot = False, iso = False):
	"""add spectra together given an array of spectra and flux ratios

	Args: 
		teff (array): array of temperature values (floats)
		logg (array): array of log(g) values (floats) 
		rad (array): set of primary radius guess and all radius ratios (if distance is known) or just the set of radius ratios (if distance is not known)
		distance (float): distance to system, in pc
		contrast_filt (array): array of strings of different filters to calculate contrasts for
		phot_filt (array): array of strings of different filters in which to calculate unresolved photometry.\
		Supported systems are 2MASS, Bessell, Cousins, Johnson, Landolt, SDSS, and Stromgren.\
		For standard UBVRIJHK defaults to Johnson (UBVRI) and 2MASS (JHKs). Should be entered as, e.g., 'Johnson, U'. Not case sensitive.
		range (array): minimum and maximum limits for the spectrum, in units of microns.
		specs (dict): dictionary of reference spectra to feed to the spectrum generator
		normalize (boolean): Normalize the spectra before adding them (default is True)
		res (int): spectra resolution to use

	Returns: 
		wl, spec (tuple): wavelength and spectrum (or central wavelength and synthetic photometry) for a spectrum (if in "spec" mode)
		set of synthetic colors (if in 'sed' mode)

	"""

	#initialize variables for keeping track of the total minimum and maximum wavelengths requested for all filters


	#what i need to do: get the primary and secondary spectra, use a given radius to make a normalization constant for the secondary
	#take the individual spectra and calculate a contrast in the relevant filters
	#add the spectra for a composite spectrum
	#do synthetic photometry in the relevant filters
	#return the wl, spec, contrasts, and photometry 

	#unpack the contrast and photometry lists
	#wls and tras are the wavelength and transmission arrays for the contrast list; n_res_el is the number of resolution elements in each filter, and 
	#cwl is the central wavelength for each contrast filter
	wls, tras, n_res_el, cwl = ctm
	#same order for these four variables but for the photometry, not the contrasts
	phot_wls, phot_tras, phot_resel, phot_cwl = ptm

	#now find the wavelength global extrema by checking the min and max of each transmission curve wavelength vector
	wlmin, wlmax = np.inf, 0
	for w in wls:
		if min(w) < wlmin:
			wlmin = min(w)
		if max(w) > wlmax:
			wlmax = max(w)
	for p in phot_wls:
		if min(p) < wlmin:
			wlmin = min(p)
		if max(p) > wlmax:
			wlmax = max(p)

	#the "plot" keyword doesn't plot anything, it's here as a flag for when the function call is for plotting
	#as things are set up I have it to also calculate the kepler magnitude when called with "plot = True" because when creating diagnostic plots I use it for other analysis
	if plot == True:
		ran, tm, a, b = get_transmission('kepler', res)
		if min(ran) < wlmin:
			wlmin = min(ran)
		if max(ran) > wlmax:
			wlmax = max(ran)

	#get the primary star wavelength array and spectrum 
	#the min and max wavelength points will be in Angstroms so we need to make them microns for the function call
	#returns in erg/s/cm^2/A/surface area
	pri_wl, pri_spec = get_spec(teff[0], logg[0], [min(min(r), tmi/1e4, wlmin/1e4) - 1e-4, max(max(r), tma/1e4, wlmax/1e4) + 1e-4], specs, normalize = False, resolution = res, npix = npix, models = models)
	#convert spectrum to a recieved flux at earth surface: multiply by surface area (4pi r^2) to get the luminosity, then divide by distance (4pi d^2) to get flux
	if not type(distance) == bool: #if we're fitting for distance convert to a flux
		di = 1/distance
		pri_spec *= (rad[0]*6.957e+10/(di * 3.086e18))**2

	#now we need to get the secondary (and possibly higher-order multiple) spectra
	#given the way the spectral retrieval code works, as long as the wavelength range is the same the spectra will be on the same grid
	#so I'm just going to stack them with the primary wavelength and spectrum - going to save the wavelength just in case I need to troubleshoot later
	for n in range(1, len(teff)):
		sec_wl, sec_spec = get_spec(teff[n], logg[n], [min(min(r), tmi/1e4, wlmin/1e4) - 1e-4, max(max(r), tma/1e4, wlmax/1e4) + 1e-4], specs, normalize = False, resolution = res, npix = npix, models = models)
		if not type(distance) == bool: #if fitting for distance convert to flux
			di = 1/distance
			sec_spec = sec_spec * (rad[0]*rad[n]*6.957e+10/(di * 3.086e18))**2 
		else: #otherwise need to alter the flux based on the radius ratio
			#we've set the primary radius to 1, so this radius is just the square of the radius ratio in cm
			sec_spec = sec_spec * rad[0]**2

		#save the secondary wavelength and spectrum in the total system arrays 
		#this is to provide flexibility if I want to fit for a triple intead of a binary
		pri_wl = np.row_stack((pri_wl, sec_wl)); pri_spec = np.row_stack((pri_spec, sec_spec))

	"""
	calculate contrast magnitudes
	"""
	#define an array to hold all my "instrumental" fluxes
	mag = np.zeros((len(contrast_filt), len(teff)))

	#loop through each filter 
	for n, f in enumerate(contrast_filt):
		#get the wavelength range and transmission curve
		ran, tm = wls[n], tras[n]
		#pick out the region of stellar spectrum that matches the curve
		w = pri_wl[0, :][np.where((pri_wl[0,:] <= max(ran)) & (pri_wl[0,:] >= min(ran)))]
		#and interpolate the transmission curve so that it matches the stellar spectral resolution
		intep = interp1d(ran, tm)
		tran = intep(w)
		#now for each star
		for k in range(len(teff)):
			#pick out the spectrum over the appropriate region
			s = pri_spec[k][np.where((pri_wl[0,:] <= max(ran)) & (pri_wl[0,:] >= min(ran)))]
			t_spec = [s[p] * tran[p] for p in range(len(s))]
			#put it through the filter by multiplying by the transmission, then integrate to finally get the instrumental flux
			m = np.trapz(t_spec, w)
			#and add it to the array in the appropriate place for this star and filter
			mag[n][k] = -2.5 * np.log10(m)
	#now we have a set of synthetic magnitudes for each filter with a known flux ratio 
	#don't need to apply filter zp etc because all that matters is the differential measurement
	#now we need to calculate the contrast
	#i'm only going to set up the contrast calculation for a binary right now
	#need to just take the flux ratio from flux (secondary/primary)
	contrast = [mag[n][1] - mag[n][0] for n in range(len(contrast_filt))]

	#make the composite spectrum
	spec1 = pri_spec[0,:] + pri_spec[1,:]

	#pri_wl[0] and spec1 are the composite wavelength and spectrum vectors
	#now we have to go through each filter in phot_filt and do the photometry using phot_wls and phot_tras
	phot_phot = []
	#2mass zero points (values from Cohen+ 2003); sdss filter values from SVO filter profile service
	#[r, i, z, J, H, Ks]
	zp_jy = [3112.91, 2502.62, 1820.98, 1594, 1024, 666.7] #zero point in Jy (10^-23 erg/s/cm^2/Hz)
	cw = [6246.98, 7718.28, 10829.83, 1.235e4, 1.662e4, 2.159e4] #A
	bp_width = [1253.71, 1478.93, 4306.72, 1620, 2509, 2618]
	zp = [zp_jy[n]*bp_width[n]/(3.336e4 * cw[n]**2) for n in range(len(zp_jy))] #convert to a zero point in flux
	#conversion from ab mag to vega mag for the SDSS filters 
	mab_minus_mvega = [0.16,0.37,0.54] #to get m_vega, mab - mvega = 0.08 -> mvega = mab - 0.08
	# t1 = time.time()
	# spec = Spectrum1D(spectral_axis=pri_wl[0,:]*u.Unit('AA'), flux=spec1*u.Unit('erg s-1 cm-2 AA-1'))
	#if only using 2MASS mags truncate the zero point array
	if len(phot_filt) == 3:
		zp = zp[-3:]
		fs = ['2MASS_J', '2MASS_H', '2MASS_Ks']
	else:
		fs = ['SDSS_r', 'SDSS_i', 'SDSS_z', '2MASS_J', '2MASS_H', '2MASS_Ks']
	# t1 = time.time()

	# fluxes = [lib[fs[n]].get_flux(pri_wl[0,:] * pyphot.unit('AA'), spec1.pyphot.unit('erg/s/cm**2/AA')) for n in range(len(fs))]
	# phot_phot = np.array([-2.5*np.log10((fluxes[n]/lib[fs[n]].Vega_zero_flux).value) for n in range(len(fs))])
	# phot_phot[:4] -= np.array(mab_minus_mvega)

	for n in range(len(phot_filt)):
		# t1 = time.time()
		#get the wavelength range and transmission curve for each filter
		# ran, tm = np.array(phot_wls[n]), np.array(phot_tras[n])
		# # interpolate the transmission curve
		# intep = interp1d(ran, tm)
		# data_tm = intep(pri_wl[0,:][np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))])
		'''
		the few commented out lines below are for if I want to use synphot instead of calculating myself - I found this to be too slow to be useful in testing
		'''
		# vega_tm = intep(vs['WAVELENGTH'][np.where((vs['WAVELENGTH'] >= min(ran)) & (vs['WAVELENGTH'] <= max(ran)))])
		# bp = phot_tras[n] #synphot.SpectralElement(synphot.models.Empirical1D, points = ran*u.Unit('AA'), lookup_table = tm * u.dimensionless_unscaled, keep_neg = True)
		# obs = synphot.Observation(spec, bp)
		
		# if len(phot_filt) - n < 4:
		# 	mag = obs.effstim(flux_unit = 'vegamag', vegaspec = vs)
		# else:
		# 	mag = obs.effstim(flux_unit = 'abmag')
		# phot_phot.append(float(mag.value))

		# get the internal default library of passbands filters
		f = lib[fs[n]] #['SDSS_g']#pyphot.Filter(phot_wls[n], bp, dtype = 'photon', unit = 'Angstrom')
		# compute the integrated flux through the filter f
		# note that it work on many spectra at once
		# t1 = time.time()
		fluxes = f.get_flux(pri_wl[0,:]*pyphot.unit('AA'), spec1*pyphot.unit('erg/s/cm**2/AA'))

		if '2MASS' in fs[n]:
			# convert to vega magnitudes
			mags = -2.5 * np.log10((fluxes/f.Vega_zero_flux).value)
		else:
			mags = -2.5 * np.log10((fluxes/f.AB_zero_flux).value)	
		phot_phot.append(mags)
		# #if it's a 2MASS mag (one of the last three filters), just calculate the vega mag 
		# if len(phot_filt) - n < 4:
		# 	source_phot = np.sum(np.array(spec1)[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * data_tm)/zp[n] 
		# 	# vega_phot = np.sum(vs['FLUX'][np.where((vs['WAVELENGTH'] >= min(ran)) & (vs['WAVELENGTH'] <= max(ran)))] * vega_tm)/zp[n]
		# 	source_vegamag = -2.5*np.log10(source_phot)
		# 	phot_phot.append(source_vegamag)
		# #otherwise calculate the abmag and then convert to vegamag
		# else:
		# 	#convert to janskys by first converting to erg/s/cm^2/Hz then multiplying by 1e23 to get Jy
		# 	spec_jy = np.array(spec1)[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * ((pri_wl[0,:][np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))])**2) #* 3.336e4

		# 	# spec_jy = spec_hz * 1e23
		# 	#convolve with transmission curve to retrieve the flux in Jy, then divide by the zero point convolved with the transmission curve
		# 	spec_jy_filt = np.sum(spec_jy * data_tm)#/(zp_jy[n]*len(data_tm))
		# 	#calculate the ab mag and convert to a vegamag
		# 	mab = -2.5 * np.log10(spec_jy_filt/phot_cwl[n]**2/2.99792458e+18) -48.46#- mab_minus_mvega[n]
		# 	phot_phot.append(mab)
		# print(phot_phot)
	# print('time for photometry:', time.time()-t1)

	#if I'm calling this during the plot function, I also want to get the kepler photometry for each component
	if plot == True:
		#so get the kep tm curve
		ran, tm, a, b = get_transmission('kepler', res)

		#interpolate the transmission curve to the data wavelength scale
		intep = interp1d(ran, tm); data_tm = intep(pri_wl[0,:][np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))])

		#calculate the photometry for each component by convolving with the tm curve, integrating, and dividing by the zero point
		pri_phot = np.trapz(np.array(pri_spec[0,:])[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * data_tm, pri_wl[0,:][np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))])
		sec_phot = np.trapz(np.array(sec_spec)[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * data_tm, pri_wl[0,:][np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))])

		#convert to vega mag from flux values
		pri_vegamag = -2.5*np.log10(pri_phot); sec_vegamag = -2.5*np.log10(sec_phot)

		#return the wavelength array, the composite spectrum, the component spectra, and the component kepler magnitudes
		return np.array(pri_wl[0,:]), np.array(spec1), np.array(pri_spec[0,:]), np.array(sec_spec), pri_vegamag, sec_vegamag

	#If I'm calling the function to make an CMD, I've decided I want to do it in H-K vs K space
	elif iso == True:
		#so read in the H and Ks transmission curves
		hran, htm, ha, hb = get_transmission('2mass,h', res)
		kran, ktm, ka, kb = get_transmission('2mass,ks', res)

		#and calculate the zero point in erg/s/cm^2/A
		zp_jy = [1024, 666.7] #zero point in Jy (10^-23 erg/s/cm^2/Hz)
		cw = [1.662e4, 2.159e4] #A
		bp_width = [2509, 2618]
		zp = [zp_jy[n]*bp_width[n]/(3.336e4 * cw[n]**2) for n in range(len(zp_jy))] #convert to a zero point in flux

		#create the transmission curves for H and K
		intep_h = interp1d(hran, htm); 
		data_tm_h = intep_h(pri_wl[0,:][np.where((pri_wl[0,:] >= min(hran)) & (pri_wl[0,:] <= max(hran)))])
		intep_k = interp1d(kran, ktm); 
		data_tm_k = intep_k(pri_wl[0,:][np.where((pri_wl[0,:] >= min(kran)) & (pri_wl[0,:] <= max(kran)))])

		#calculate the H band photometry
		pri_phot_h = np.sum(np.array(pri_spec[0,:])[np.where((pri_wl[0,:] >= min(hran)) & (pri_wl[0,:] <= max(hran)))] * data_tm_h)/zp[0]
		sec_phot_h = np.sum(np.array(sec_spec)[np.where((pri_wl[0,:] >= min(hran)) & (pri_wl[0,:] <= max(hran)))] * data_tm_h)/zp[0]

		#calculate the K band photometry
		pri_phot_k = np.sum(np.array(pri_spec[0,:])[np.where((pri_wl[0,:] >= min(kran)) & (pri_wl[0,:] <= max(kran)))] * data_tm_k)/zp[1]
		sec_phot_k = np.sum(np.array(sec_spec)[np.where((pri_wl[0,:] >= min(kran)) & (pri_wl[0,:] <= max(kran)))] * data_tm_k)/zp[1]

		#calculate the H and K magnitudes for each component
		pri_hmag = -2.5*np.log10(pri_phot_h); sec_hmag = -2.5*np.log10(sec_phot_h)
		pri_kmag = -2.5*np.log10(pri_phot_k); sec_kmag = -2.5*np.log10(sec_phot_k)

		#return the H and K component magnitudes
		return pri_hmag, sec_hmag, pri_kmag, sec_hmag
	
	else:
		return np.array(pri_wl[0,:]), np.array(spec1), [c for c in contrast], np.array([float(p) for p in phot_cwl]), np.array([p for p in phot_phot])

def opt_prior(vals, pval, psig):
	"""Imposes a gaussian prior using a given prior value and standard deviation
	"""
	#initialize a list for likelihood values
	pp = []
	#for each value we're calculating the prior for
	if len(pval) == 1 or type(pval) == float:
		try:
			pp.append((float(vals)-float(pval))/float(psig)**2)
		except:
			pp.append((vals[0]-pval[0])/psig[0]**2)
	else:
		for k, p in enumerate(pval):
			#as long as the prior is defined
			if p != 0:
				#calculate the likelihood
				like = (vals[k] - pval[k])/(psig[k])**2
				#and save it
				pp.append(like)

	#return the sum of the likelihoods
	return np.sum(pp)

def fit_spec(n_walkers, dirname, wl, flux, err, reg, t_guess, lg_guess, av, rad_guess, fr_guess, specs, tlim, llim, distance, ctm, ptm, tmi, tma, vs, matrix, cs = 2, steps = 200, burn = 1, conv = True, models = 'btsettl', dist_fit = True, rad_prior = False):
	"""Performs a modified gibbs sampler MCMC using a reduced chi-square statistic.

	Args:
		n_walkers (int): number of walkers.
		wl (list): wavelength array.
		flux (list): spectrum array.
		reg (list): Two value array with start and end points for fitting.
		t_guess (array): Initial teff guesses.
		lg_guess (array): initial log(g) guesses.
		extinct (float): [currently] known extinction value - [future] initial ext guess.
		fr (list): flux ratio array. has structure like [[0.2], ['johnson, r']].
		sp (dict): dictionary including wavelength vector ('wl') and teff/log(g) pairs ('3000, 4.5').
		tlim (tuple): min and max allowed temperature values.
		llim (tuple): min and max allowed log(g) values.
		cs (int): cutoff chi square to decide convergence. Default: 2.
		steps (int): maximum steps to take after the burn-in steps. Default: 200.
		burn (int): number of burn-in steps. Default: 20.
		conv (Bool): Use chi-square for convergence (True) or the number of steps (False). Default: True.
	"""
	#make sure wl is in Angstroms 
	wl *= 1e4

	#unpack the distance as long as I'm using it 
	if not type(distance) == bool:
		dist, pprior, psig = distance

	#unpack the initial extinction guess (although I currently don't fit for Av)
	extinct_guess, eprior, esig = av

	#note that fr_guess[0] = contrasts, fr_guess[1] = contrast errors, [2] = filters, [3] = unres phot values, [4] = errors, [5] = filters
	#phot is in flux, contrast is in mags

	#make an initial guess spectrum, contrast, and photometry using absolute radii if using a distance and using a radius ratio otherwis
	if not type(distance) == bool:
		wave1, init_cspec, contrast, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, dist, fr_guess[2], fr_guess[5], reg, specs, ctm, ptm, tmi, tma, vs, models = models)
	else:
		wave1, init_cspec, contrast, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, distance, fr_guess[2], fr_guess[5], reg, specs, ctm, ptm, tmi, tma, vs, models = models)
	
	#extinct the initial spectrum and photometry guesses
	# init_cspec = extinct(wave1, init_cspec, extinct_guess)
	# init_phot = extinct(phot_cwl, phot, extinct_guess)

	#interpolate the model onto the data wavelength vector
	intep = interp1d(wave1, init_cspec)
	init_cspec = intep(wl)

	#normalize the model to match the median of the data
	init_cspec*=np.median(flux)/np.median(init_cspec)

	#calculate the chi square value of the spectrum fit
	ic = chisq(init_cspec, flux, err)
	iic = np.sum(ic)/len(ic)

	#calculate the chi square for the contrast fit
	chi_contrast = chisq(contrast, fr_guess[0], fr_guess[1])
	icontrast = np.sum(chi_contrast)

	#if using any distance calculate the photometry chi square and then the total chi square after weighting the spectrum chi square appropriately
	if not type(distance) == bool:
		ip = chisq(phot, fr_guess[3], fr_guess[4])
		iphot = np.sum(ip)
		init_cs = np.sum((iic*(len(chi_contrast) + len(ip)), icontrast, iphot))
	#otherwise calculate the total chi square as the sum of the contrast and the spectrum chi squares after weighting the spectrum chi^2 appropriately
	else:
		init_cs = np.sum((iic*len(chi_contrast), icontrast))
	#if the distance is from Gaia, impose a distance prior
	if not type(distance) == bool and dist_fit == True:
		init_cs += opt_prior([dist], [pprior], [psig])

	if rad_prior == True:
		model_radius1 = get_radius(t_guess[0], matrix)
		model_radius2 = get_radius(t_guess[1], matrix)
		#assume sigma is 5%, which is a pretty typical radius measurement uncertainty
		# print(rad_guess)

		init_cs += opt_prior(rad_guess, [model_radius1, model_radius2/model_radius1], [0.05*r for r in rad_guess])



	#that becomes your comparison chi square
	chi = init_cs
	#make a random seed
	#this is necessary because we initialized everything from the same node, so each core needs to generate its own seed to make sure that the random calls are independent
	r = np.random.RandomState()

	#savechi will hang on to the chi square value of each fit
	savechi = [init_cs]
	#savetest will hang on to the test values for each guess
	savetest = [t_guess, lg_guess, extinct_guess, rad_guess]

	"""
	This section is for if the fit will use any sort of distance measurement 
	"""
	if not type(distance) == bool:

		#sp will hang on to the tested set of parameters at the end of each iteration
		#so will eventually become 2D but for now it just holds the initial guess
		sp = [t_guess, lg_guess, extinct_guess, rad_guess, dist]

		#si is the std dev for the gaussian calls the function makes to vary the test parameters
		#initially this is very coarse so we explore the parameter space rapidly
		si = [[250, 250], [0.1, 0.1], [0.05], [0.1 * r for r in rad_guess], [0.02 * dist]]
		#gi is the guess for an individual function call so right now it's just the initial guess
		#this is what will vary at each function call and be saved in sp, which was initialized above
		gi = [t_guess, lg_guess, extinct_guess, rad_guess, dist]

		#initialize a step counter and a cutoff counter
		#the cutoff it to make sure that if the function goes bonkers it will still eventually end 
		n = 0
		total_n = 0
		#as long as both counters are below the set limits 
		while n < steps + burn and total_n < (10 * steps) + burn:

			#if we're halfway through reduce the step size significantly to refine the fit in chi^2 surface assuming the coarse step got us to the approximate correct minimum
			if n > (burn + steps/2):
				si = [[20, 20], [0.05, 0.05], [0.01], [0.05 * r for r in rad_guess], [0.005*dist]]

			#and then vary all the parameters simultaneously using the correct std dev
			var_par = make_varied_param(gi, si)

			# print(var_par, tlim, llim, total_n)

			# print(var_par, tlim, llim)
			#make sure that everything that got varied was inside the parameter limits
			if all(min(tlim) < v < max(tlim) for v in var_par[0]) and all(min(llim) < v < max(llim) for v in var_par[1]) \
				and 0 < var_par[2] < 0.5 and all(0.05 < r for r in var_par[3]) and 1/100 > var_par[4] > 1/2000:
				#we made it through, so increment the counters by 1 to count the function call
				total_n += 1
				n += 1

				#create a test data set using the guess parameters
				test_wave1, test_cspec, test_contrast, test_phot_cwl, test_phot = make_composite(var_par[0], var_par[1], var_par[3], float(var_par[4]), fr_guess[2], fr_guess[5], reg, specs, ctm, ptm, tmi, tma, vs, models = models)

				#extinct the spectrum and photometry
				# test_cspec = extinct(test_wave1, test_cspec, var_par[2])
				# test_phot = extinct(test_phot_cwl, test_phot, var_par[2])

				#interpolate the test spectrum onto the data wavelength vector
				intep = interp1d(test_wave1, test_cspec)
				test_cspec = intep(wl)

				#normalize the test spectrum to match the data normalization
				test_cspec*=np.median(flux)/np.median(test_cspec)

				#calculate the reduced chi square between data spectrum and guess spectrum 
				tc = chisq(test_cspec, flux, err)
				ttc = np.sum(tc)/len(tc)

				#calculate the contrast chi square
				chi_contrast = chisq(test_contrast, fr_guess[0], fr_guess[1])
				tcontrast = np.sum(chi_contrast)

				#calculate the photometry chi square - this is a loop that is only for distance-provided inputs, so photometry will always be relevant
				chi_phot = chisq(test_phot, fr_guess[3], fr_guess[4])
				tphot = np.sum(chi_phot)

				#create the total chi square by summing the chi^2 after weighting the spectrum chi^2 appropriately
				test_cs = np.sum((ttc*(len(chi_contrast) + len(chi_phot)), tcontrast, tphot))

				#if we're using a real distance instead of a fake distance, add in a distance prior to the guess 
				if dist_fit == True:
					test_cs += opt_prior([var_par[4]], [pprior], [psig])

				if rad_prior == True:
					model_radius1 = get_radius(var_par[0][0], matrix)
					model_radius2 = get_radius(var_par[0][1], matrix)
					#assume sigma is 10% or 5%, which is a pretty typical radius measurement uncertainty
					# print(init_cs, opt_prior(var_par[3], [model_radius1, model_radius2/model_radius1], np.array(si[3])/2), [model_radius1, model_radius2/model_radius1], var_par[3], si[3])
					test_cs += opt_prior(var_par[3], [model_radius1, model_radius2/model_radius1], np.array(si[3])/2)

				#now, if the test chi^2 is better than the prevous chi^2
				if test_cs < chi:
					#replace the old best-fit parameters with the new ones
					gi = var_par
					#save the new chi^2
					chi = test_cs 
					#if we're more than halfway through, just go back to the small variations, don't start all over again
					if n > (steps/2 + burn):
						n = steps/2 + burn + 1
					#but if we're less than halfway through, start the fit over because we want to go until we've tried n guesses without a better guess
					else:
						n = 0

				#save everything to the appropriate variables
				sp = np.vstack((sp, gi))
				savechi.append(chi)
				savetest.append(test_cs)

			#if any guess is outside of the limits, vary the offending parameter until it isn't anymore 
			else:
				#but still increment the total count by one, because these calls count too
				total_n += 1

				#temperatures
				while any(v < min(tlim) for v in var_par[0]):
					total_n += 1
					var_par[0][np.where(var_par[0]<min(tlim))] += 100
				while any(v > max(tlim) for v in var_par[0]):
					total_n += 1
					var_par[0][np.where(var_par[0] > max(tlim))] -= 100

				while var_par[0][0] < var_par[0][1]:
					total_n += 1
					var_par[0][1] -= 100

				#surface gravities
				while any(v < min(llim) for v in var_par[1]):
					total_n += 1
					var_par[1][np.where(var_par[1]<min(llim))] += 0.1
				while any(v > max(llim) for v in var_par[1]):
					total_n += 1
					var_par[1][np.where(var_par[1] > max(llim))] -= 0.1

				#extinction
				while var_par[2] > 0.5:
					total_n += 1
					var_par[2] -= 0.1
				while var_par[2] < 0:
					total_n += 1
					var_par[2] += 0.1

				#radius
				while any(v < 0.05 for v in var_par[3]):
					total_n += 1
					var_par[3][np.where(var_par[3]<0.05)] += 0.01

				#distance (as parallax)
				while var_par[4] > 1/100:
					total_n += 1
					var_par[4] -= 0.05 * np.abs(var_par[4])
				while var_par[4] < 1/2000:
					total_n += 1
					var_par[4] += 0.05*np.abs(var_par[4])

		#save all the guessed best-fit parameters to a file 
		f = open(dirname + '/params{}.txt'.format(n_walkers), 'a')
		for n in range(1, len(savechi)):
			f.write('{} {} {} {} {} {} {} {}\n'.format(sp[:][n][0][0], sp[:][n][0][1], sp[:][n][1][0], sp[:][n][1][1], float(sp[:][n][2]), sp[:][n][3][0], sp[:][n][3][1], float(sp[:][n][4])))
		f.close()
		#save all the best-fit chi^2 values to a file
		f = open(dirname + '/chisq{}.txt'.format(n_walkers), 'a')
		for n in range(1, len(savechi)):
			f.write('{} {}\n'.format(savechi[n], savetest[n]))
		f.close()

		#and then return the final best-fit value and chi^2 
		return '{} {} {} {} {} {} {} {}\n'.format(gi[0][0], gi[0][1], gi[1][0], gi[1][1], float(gi[2]), gi[3][0], gi[3][1], float(gi[4])), savechi[-1]

	#If there is no distance being used

	else:
		#sp will hang on to the tested set of parameters at the end of each iteration
		#it will eventually become multidimensional
		sp = [t_guess, lg_guess, extinct_guess, rad_guess]

		#si holds the standard deviation for each parameter during the variance process
		si = [[250, 250], [0.1, 0.1], [0.05], [0.1 * r for r in rad_guess]]
		#gi is the current best-fit guess
		gi = [t_guess, lg_guess, extinct_guess, rad_guess]

		#initialize a counter for how many guesses have gone by without an improvement
		n = 0
		#and initialize another counter for the total number of steps taken
		total_n = 0

		#make sure that both counters fall within limits: the number of guesses that have gone by without improvement must be less than the cutoff number of test steps
		#and the total number of iterations must be below the cutoff
		while n < steps + burn and total_n < (20 * steps) + burn:

			#if the algorithm is halfway through, switch to using much smaller variations for each guess after assuming we've approximately found the chi^2 minimum
			if n > (burn + steps/2):
				si = [[20, 20], [0.05, 0.05], [0.01], [0.05 * r for r in rad_guess]]

			#vary the old best-fit parameters to make a new guess
			var_par = make_varied_param(gi, si)

			#as long as everything falls in the appropriate limits
			if all(min(tlim) < v < max(tlim) for v in var_par[0]) and all(min(llim) < v < max(llim) for v in var_par[1]) \
				and 0 < var_par[2] < 0.5 and all(0.05 < r for r in var_par[3]):
				#bump each counter up by one
				total_n += 1
				n += 1

				#create a test data set using the test parameters
				test_wave1, test_cspec, test_contrast, test_phot_cwl, test_phot = make_composite(var_par[0], var_par[1], var_par[3], False, fr_guess[2], fr_guess[5], reg, specs, ctm, ptm, tmi, tma, vs, models = models)

				#extinct the test spectrum - no photometry here because we're not using a distance
				# test_cspec = extinct(test_wave1, test_cspec, var_par[2])

				#interpolate the test spectrum onto the data wavelength vector
				intep = interp1d(test_wave1, test_cspec)
				test_cspec = intep(wl)

				#normalize the test spectrum to match the data spectrum
				test_cspec *= np.median(flux)/np.median(test_cspec)

				#calculate the reduced chi square between data spectrum and the guess spectrum
				tc = chisq(test_cspec, flux, err)
				ttc = np.sum(tc)/len(tc)

				#calculate the chi^2 for the contrast data set
				chi_contrast = chisq(test_contrast, fr_guess[0], fr_guess[1])
				tcontrast = np.sum(chi_contrast)

				#calculate the total chi^2 after weighting the spectrum chi^2 appropriately
				test_cs = np.sum((ttc*len(chi_contrast), tcontrast))

				if rad_prior == True:
					model_radius1 = get_radius(var_par[0][0], matrix)

					#assume sigma is 10% or 5%, which is a pretty typical radius measurement uncertainty
					test_cs += opt_prior(var_par[3], [model_radius1], si[3])

				#if the guess is better than the previous best guess
				if test_cs < chi:
					#save the new best-fit values
					gi = var_par
					chi = test_cs 
					#if we're halfway through, just go back to halfway through again (since we want to keep exploring in a fine-grained way)
					if n > (steps/2 + burn):
						n = steps/2 + burn + 1
					#otherwise fully start over again at the new best-fit position
					else:
						n = 0

				#save all the best-fit parameters 
				sp = np.vstack((sp, gi))
				savechi.append(chi)
				savetest.append(test_cs)

			#if any of the guess values fall outside the defined limits
			else:
				#still increment the counter for the total number of guesses
				total_n += 1

				#and then vary the offending parameter until it reaches the appropriate range
				#temperatures
				while any(v < min(tlim) for v in var_par[0]):
					total_n += 1
					var_par[0][np.where(var_par[0]<min(tlim))] += 100
				while any(v > max(tlim) for v in var_par[0]):
					total_n += 1
					var_par[0][np.where(var_par[0] > max(tlim))] -= 100

				#surface gravities
				while any(v < min(llim) for v in var_par[1]):
					total_n += 1
					var_par[1][np.where(var_par[1]<min(llim))] += 0.1
				while any(v > max(llim) for v in var_par[1]):
					total_n += 1
					var_par[1][np.where(var_par[1] > max(llim))] -= 0.1

				#extinction
				while var_par[2] > 0.5:
					total_n += 1
					var_par[2] -= 0.1
				while var_par[2] < 0:
					total_n += 1
					var_par[2] += 0.1

				#radius ratio
				while any(v < 0.05 for v in var_par[3]):
					total_n += 1
					var_par[3][np.where(var_par[3]<0.05)] += 0.01
				while any(v > 1 for v in var_par[3]):
					total_n += 1
					var_par[3][np.where(var_par[3] > 1)] -= 0.01

		#at the end, save the best-fit parameters during the full fitting process to a file
		f = open(dirname + '/params{}.txt'.format(n_walkers), 'a')
		for n in range(len(savechi)):
			f.write('{} {} {} {} {} {}\n'.format(sp[:][n][0][0], sp[:][n][0][1], sp[:][n][1][0], sp[:][n][1][1], float(sp[:][n][2]), float(sp[:][n][3][0])))
		f.close()
		#and save all the best-fit chi^2 values to another file
		f = open(dirname + '/chisq{}.txt'.format(n_walkers), 'a')
		for n in range(len(savechi)):
			f.write('{} {}\n'.format(savechi[n], savetest[n]))
		f.close()

		#return the final best-fit parameters and chi^2 
		return '{} {} {} {} {} {}\n'.format(gi[0][0], gi[0][1], gi[1][0], gi[1][1], float(gi[2]), float(gi[3][0])), savechi[-1]

def loglikelihood(p0, fr, nspec, ndust, data, err, broadening, r, specs, ctm, ptm, tmi, tma, vs, w = 'aa', pysyn = False, dust = False, norm = True, mode = 'spec', av = True, optimize = False, models = 'btsettl'):
	#unpack data tuple into wavelength and data arrays
	wl, spec = np.array(data)

	#unpack the guess array using the number of spectra to fit to (hardcoded as two in the len call right now)
	#if distance is included in the guess unpack it
	if len(p0) == 8:
		t_guess, lg_guess, extinct_guess, rad_guess, dist_guess = p0[:nspec], p0[nspec:2*nspec], p0[2*nspec], p0[2*nspec+1:3*nspec + 1], p0[3*nspec+1]
		#create the composite spectrum that corresponds to the guess values
		# t1 = time.time()
		wave1, init_cspec, contrast, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, dist_guess, fr[2], fr[5], r, specs, ctm, ptm, tmi, tma, vs, models = models)
		# print('time for composite: ', time.time() - t1)
	#otherwise just get the other terms 
	else:
		t_guess, lg_guess, extinct_guess, rad_guess = p0[:nspec], p0[nspec:2*nspec], p0[2*nspec], p0[2*nspec+1:3*nspec + 1]
		#and again create the appropriate composite and other data
		wave1, init_cspec, contrast, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, False, fr[2], fr[5], r, specs, ctm, ptm, tmi, tma, vs, models = models)

	#if the extinction value is True extinct the spectrum and photometry using the guess value
	if av == True:
		init_cspec = extinct(wave1, init_cspec, extinct_guess)
		init_phot = extinct(phot_cwl, phot, extinct_guess)
	#or if I set the extinction to a specific value, extinct the spectrum and photometry using that value
	elif av != 0 and type(av) == float:
		init_cspec = extinct(wave1, init_cspec, av)
		init_phot = extinct(phot_cwl, phot, av)
	#otherwise just transfer the photometry to another variable for consistency and proceed without extincting anything
	else:
		init_phot = phot

	#interpolate the model onto the date wavelength scale
	intep = interp1d(wave1, init_cspec)
	init_cspec = intep(wl * 1e4)

	#normalize the model
	init_cspec *= np.median(spec)/np.median(init_cspec)

	#calculate the chi square value of that fit
	ic = chisq(init_cspec, spec, err)
	iic = np.sum(ic)/len(ic)

	#calculate the chi square for the contrast fit
	chi_contrast = chisq(contrast, fr[0], fr[1])
	icontrast = np.sum(chi_contrast)

	#if there is photometry involved (i.e., a distance)
	if len(p0) == 8:
		#calculate the chi square for the photometry
		chi_phot = chisq(init_phot, fr[3], fr[4])
		iphot = np.sum(chi_phot)
		#and the total chi square is the sum of the chi squares, where the spectrum is weighted to be as important as the combined photometry and contrasts
		init_cs = np.sum((iic*(len(chi_contrast) + len(chi_phot)), icontrast, iphot))
	#if we're not using a distance at all, just weight the spectrum by the contrast chi square and then calculate the total chi square
	else:
		# iic *= (len(chi_contrast))
		init_cs = np.sum((iic*len(chi_contrast), icontrast))

	#if i'm running a simple optimization I just need to return a chi square
	if optimize == True:
		return init_cs
	#but usually I want to return the log likelihood for emcee 
	else:
		if np.isnan(init_cs):
			return -np.inf
		else:
			return -0.5 * init_cs

def logprior(p0, nspec, ndust, tmin, tmax, lgmin, lgmax, matrix, prior = 0, ext = True, dist_fit = True, rad_prior = False):
	#get the guess temperatures and surface gravities
	temps = p0[0:nspec]
	lgs = p0[nspec:2 * nspec]
	#now there are a bunch of different situations and this is definitely not as pretty as it could be but I suppose it works fine
	#first, calculate if I'm using a distance 
	if len(p0) == 8 and dist_fit == True:
		#if i'm fitting for extinction, get Av, radii, and distance from the guess array
		if ext == True:
			extinct = p0[2*nspec]
			rad = p0[2*nspec + 1:3*nspec + 1]
			dist = p0[3*nspec + 1]
		#otherwise just get the radii and distance
		else:
			rad = p0[2*nspec:3*nspec]
			dist = p0[3*nspec + 1]

		#this is a remnant of old stuff that I never used
		#supposed to include blackbody radiation to correspond with veiling or disk emission
		if ndust > 0:
			dust = p0[2 * nspec + 1:]

		pp = [0]

		#now check to make sure that everything falls within appropriate boundaries
		#and if it doesn't, return -inf which will force a new draw in the MCMC chain (corresponds to a zero probability for the guess)
		if any(t > tmax for t in temps) or any(t < tmin for t in temps) or any(l < lgmin for l in lgs) or any(l > lgmax for l in lgs)\
		or any(r < 0.05 for r in rad)  or rad[0] > 2 or dist < 1/2000 or dist > 1/100:
			return  -np.inf
		if ext == True and (extinct < 0 or extinct > 0.5):
			return  -np.inf 
		#and now if we're applying a nonzero non-uniform (gaussian) prior with values we entered 

		if prior != 0:
			#unpack the various priors and standard deviations from the input prior variable
			tprior = prior[:nspec]
			tpsig = prior[nspec:2*nspec]
			lprior = prior[2*nspec:3*nspec]
			lpsig = prior[3*nspec:4*nspec]
			distprior = [prior[-2]]
			distsig = [prior[-1]]
			eprior = prior[4*nspec]
			epsig = prior[4*nspec+1]
			rprior = prior[4*nspec+2:5*nspec+2]
			rsig = prior[5*nspec+2:6*nspec+2]

			#concatenate the various priors and standard deviations in a different way than they were input (this is so clunky!)
			ps = tprior + lprior 
			ss = tpsig + lpsig

			ps += [eprior] + rprior + distprior
			ss += [epsig] + rsig + distsig

			#now for every prior where the stddev is not zrro calculate the likelihood 
			for k, p in enumerate(ps):
				if p != 0:
					like = -0.5 * ((p0[k] - p)/ss[k])**2
					pp.append(like)

		if rad_prior == True:
			model_radius1 = get_radius(temps[0], matrix)
			model_radius2 = get_radius(temps[1], matrix)

			for k, p in enumerate([model_radius1, model_radius2/model_radius1]):
				like = -0.5 * ((rad[k] - p)/(0.02*p))**2
				# print(model_radius1, model_radius2/model_radius1, rad, like)
				pp.append(like)

		#if all values in the prior array are zero, just return zero (equivalent to just adding one uniformly to the likelihood function)
		return np.sum(pp)

	#now if we're calculating the prior for a system where there is a distance entered but we're not using it (e.g., the current fitting system for stars without a Gaia parallax)
	elif len(p0) == 8 and dist_fit == False:
		#first calculate the radius ratio
		#this is what we actually want to assess, since we technically don't have any constraints on the distance so any absolute radii are essentially nonphysical
		rad = p0[-2]
		rad1 = p0[-3]
		#get the extinction guessif I need it
		if ext == True:
			extinct = p0[-4]

		#again get the dust temperature if I'm using it (which currently nothing does)
		if ndust > 0:
			dust = p0[2 * nspec + 1:]
		
		#now assess that all the guess parameters fall within the preset limits and return -inf if they don't
		if any(t > tmax for t in temps) or any(t < tmin for t in temps) or any(l < lgmin for l in lgs) or any(l > lgmax for l in lgs)\
		or rad < 0.05 or rad1 < 0.05:
			return -np.inf
		if ext == True and (extinct < 0 or extinct > 0.5):
			return-np.inf

		pp = [0]
		#if there are any non-zero non-uniform priors, we need to assess them 
		#this assumes that all non-uniform priors are gaussian
		if prior != 0:
			#unpack the different values
			tprior = prior[:nspec]
			tpsig = prior[nspec:2*nspec]
			lprior = prior[2*nspec:3*nspec]
			lpsig = prior[3*nspec:4*nspec]
			eprior = prior[4*nspec]
			epsig = prior[4*nspec+1]
			rprior = prior[4*nspec+2:5*nspec+1]
			rsig = prior[5*nspec+2:6*nspec+1]

			#re-organize the different values into a values list and a stddev list
			ps = tprior + lprior 
			ss = tpsig + lpsig
			ps += [eprior] + rprior
			ss += [epsig] + rsig

			#evalute the log(prior) value for each defined prior entry
			for k, p in enumerate(ps):
				if p != 0:
					like = -0.5 * ((p0[k] - p)/ss[k])**2
					pp.append(like)

		if rad_prior == True:
			model_radius1 = get_radius(temps[0], matrix)
			model_radius2 = get_radius(temps[1], matrix)
			r = [rad1, rad]
			for k, p in enumerate([model_radius1, model_radius2/model_radius1]):
				like = -0.5 * ((r[k] - p)/(0.02*p))**2
				pp.append(like)
		return np.sum(pp)

	#finally, if I'm not even entering a distance (e.g., there's only a radius ratio in p0, with not even a "fake" distance)		
	else:
		#then we just get the radius ratio tacked on to the end of the guess
		rad = p0[-1]
		
		#grab the extinction guess if I'm dealing with extinction
		if ext == True:	
			extinct = p0[-2]

		#get the dust temperature guess if I'm dealing with dust/veiling/another random blackbody
		if ndust > 0:
			dust = p0[2 * nspec + 1:]
		
		#check that everything falls within the limits of the uniform prior and return -inf if anything doesn't
		if any(t > tmax for t in temps) or any(t < tmin for t in temps) or any(l < lgmin for l in lgs) or any(l > lgmax for l in lgs)\
		or rad < 0.05:
			return -np.inf
		if ext == True and (extinct < 0 or extinct > 0.5):
			return -np.inf

		pp = [0]
		#if I want a gaussian prior on anything
		if prior != 0:
			#unpack the prior array into invidiual values' mean and sigmas 
			tprior = prior[:nspec]
			tpsig = prior[nspec:2*nspec]
			lprior = prior[2*nspec:3*nspec]
			lpsig = prior[3*nspec:4*nspec]
			eprior = prior[4*nspec]
			epsig = prior[4*nspec+1]
			rprior = prior[4*nspec+2:5*nspec+1]
			rsig = prior[5*nspec+2:6*nspec+1]

			#rearrange the prior into a values array and a stddev array
			pp = []
			ps = tprior + lprior 
			ss = tpsig + lpsig
			ps += [eprior] + rprior
			ss += [epsig] + rsig

			#calculate the gaussian prior for any parameter that has a nonzero stddev
			for k, p in enumerate(ps):
				if p != 0:
					like = -0.5 * ((p0[k] - p)/ss[k])**2
					pp.append(like)

		if rad_prior == True:
			model_radius1 = get_radius(temps[0], matrix)
			model_radius2 = get_radius(temps[1], matrix)

			for k, p in enumerate([model_radius2/model_radius1]):
				like = -0.5 * ((rad - p)/(0.02*p))**2
				pp.append(like)
		return np.sum(pp)

def logposterior(p0, fr, nspec, ndust, data, err, broadening, r, specs, ctm, ptm, tmi, tma, vs, tmin, tmax, lgmin, lgmax, matrix, wu = 'aa', pysyn = False, dust = False, norm = True, prior = 0, a = True, models = 'btsettl', dist_fit = True, rad_prior = False):
	"""The natural logarithm of the joint posterior.

	Args:
		p0 (list): a sample containing individual parameter values. Then p0[0: n] = temp, p0[n : 2n] = lg, p0[2n+1] = flux ratio,\
			 p0[2n + 2] = extinction, p0[2n+3:-1] = dust temps
		nspec (int): number of spectra/stars
		ndust (int): number of dust continuum components
		data (list): the set of data/observations
		flux_ratio (array): set of flux ratios with corresponding wavelength value for location of ratio
		broadening (int): The instrumental resolution of the spectra
		r (list): region to use when calculating liklihood
		w (string): Wavelength unit, options are "aa" and "um". Default is "aa".
		pysyn (bool): Use pysynphot to calculate spectra. Default is False.
		dust (bool): Add dust continuum? Default is False.
		norm (bool): Normalize spectra when fitting. Default is True.

	Returns: 
		lh (float): The log of the liklihood of the fit being pulled from the model distribution.
	"""
	lp = logprior(p0, nspec, ndust, tmin, tmax, lgmin, lgmax, matrix, prior = prior, ext = a, dist_fit = dist_fit, rad_prior = rad_prior)
	# if the prior is not finite return a probability of zero (log probability of -inf)
	#otherwise call the likelihood function 
	if not np.isfinite(lp):
		return -np.inf
	else:
		# t1 = time.time()
		lh = loglikelihood(p0, fr, nspec, ndust, data, err, broadening, r, specs, ctm, ptm, tmi, tma, vs, w = wu, pysyn = False, dust = False, norm = True, av = a, optimize = False, models = models)
		# print('time for likelihood call: ', time.time() - t1)
		# return the likeihood times the prior (log likelihood plus the log prior)
		return lp + lh

def run_emcee(dirname, fname, nwalkers, nsteps, ndim, nburn, pos, fr, nspec, ndust, data, err, broadening, r, specs, value1, ctm, ptm, tmi, tma, vs, title_format, matrix, nthin=10, w = 'aa', pys = False, du = False, no = True, prior = 0, av = True, models = 'btsettl', dist_fit = True, rad_prior = False):
	"""Run the emcee code to fit a spectrum 

	Args:
		fname (string): input file name to use
		nwalkers (int): number of walkers to use
		nsteps (int): number of steps for each walker to take
		ndim (int): number of dimensions to fit to. For a single spectrum to fit temperature and log(g) for, ndim would be 2, for example. 
		nburn (int): number of steps to discard before starting the sampling. Should be large enough that the walkers are well distributed before sampling starts.
		pos (list): array containing the initial guesses for temperature, log g, flux ratio, and extinction
		nspec (int): number of spectra to fit to. For a single spectrum fit this would be 1, for a two component fit this should be 2.
		ndust (int): number of dust continuum components to fit to. (untested)
		data (list): the spectrum to fit to
		flux_ratio (list): an array with a subarray of flux ratios, followed by a subarray with the strings of the filter in which they were measured.
		broadening (float): the instrumental resolution of the input data, or the desired resolution to use to fit.
		r (list): a two valued array containing the region to fit within, in microns.
		nthin (int): the sampling rate of walker steps to save. Default is 10.
		w (string): the wavelength unit to use. Accepts 'um' and 'aa'. Default is 'aa'.
		pys (boolean): Whether to use pysynphot for spectral synthesis (if true). Default is False.
		du (boolean): Whether to fit to dust components. Default is False.
		no (boolean): Whether to normalize the spectra while fitting. Default is True.
	
	Note:
		This is still in active development and doesn't always work.

	"""

	# if which == 'pt':
	# 	ntemps = int(input('How many temperatures would you like to try? '))
	# 	sampler = emcee.PTSampler(ntemps, nwalkers, ndim, loglikelihood, logprior, threads=nwalkers, loglargs=[\
	# 	nspec, ndust, data, flux_ratio, broadening, r], logpargs=[nspec, ndust], loglkwargs={'w':w, 'pysyn': pys, 'dust': du, 'norm':no})

	# 	for p, lnprob, lnlike in sampler.sample(pos, iterations=nburn):
	# 		pass
	# 	sampler.reset()

	# 	#for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,lnlike0=lnlike, iterations=nsteps, thin=nthin):
	# 	#	pass

	# 	assert sampler.chain.shape == (ntemps, nwalkers, nsteps/nthin, ndim)

	# 	# Chain has shape (ntemps, nwalkers, nsteps, ndim)
	# 	# Zero temperature mean:
	# 	mu0 = np.mean(np.mean(sampler.chain[0,...], axis=0), axis=0)

	# 	try:
	# 		# Longest autocorrelation length (over any temperature)
	# 		max_acl = np.max(sampler.acor)
	# 		print('max acl: ', max_acl)
	# 		np.savetxt('results/acor.txt', sampler.acor)
	# 	except:
	# 		pass

	#first, define some limits for the prior based on the spectra we've read in
	#initialize a temperature and log(g) array
	t, l = [], []
	#go through the input model spectrum dictionary
	for s in specs.keys():
		#for all real spectra (not the wavelength vector entry)
		if not s == 'wl':
			#just take the key and split it to get a teff and a surface gravity for eahc entry
			p = s.split(', ')
			t.append(float(p[0]))
			l.append(float(p[1]))

	#just take the extrema of the range of entry values to be the upper and lower limits for the values allowed by the prior
	tmin, tmax, lgmin, lgmax = min(t), max(t), min(l), max(l)

	# count = mp.cpu_count()
	# with mp.Pool(processes = 75) as pool:
	# # with MPIPool() as pool:
	# # 	if not pool.is_master():
	# # 		pool.wait()
	# # 		sys.exit(0)
	# 	sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, threads=nwalkers, args=[fr, nspec, ndust, data, err, broadening, r, specs, ctm, ptm, tmi, tma, vs, tmin, tmax, lgmin, lgmax, matrix], \
	# 	kwargs={'pysyn': pys, 'dust': du, 'norm':no, 'prior':prior, 'a':av, 'models':models, 'dist_fit':dist_fit, 'rad_prior':rad_prior})
		
	# 	try:
	# 		sample = np.genfromtxt(os.getcwd()+'/{}/samples.txt'.format(dirname))
	# 		old_n = max(np.shape(sample))/nwalkers
	# 		state = np.median(sample, axis = 0)

	# 		state_init = emcee.utils.sample_ball(state, np.std(sample, axis = 0), size = nwalkers)

	# 		old_acl = np.inf
	# 		for n, s in enumerate(sampler.sample(state_init, iterations = nsteps)):
	# 			# print(n)
	# 			if n % nthin == 0:
	# 				with open('{}/{}_{}_results.txt'.format(dirname,fname, int(n+old_n)), 'ab') as f:
	# 					f.write(b'\n')
	# 					np.savetxt(f, s.coords)
	# 					f.close()
						
	# 				acl = sampler.get_autocorr_time(quiet = True)
	# 				macl = np.mean(acl)

	# 				with open('{}/{}_autocorr.txt'.format(dirname,fname), 'a') as f:
	# 					f.write(str(macl) + '\n')
				
	# 				if not np.isnan(macl):
	# 					# print('hi')
	# 					converged = np.all(acl * 50 < n)
	# 					# print(converged)
	# 					converged &= np.all((np.abs(old_acl - acl) / acl) < 0.1)
	# 					# print(converged)
	# 					if converged == True:
	# 						break

	# 				old_acl = acl

	# 		print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

	# 		samples = np.concatenate((sample, sampler.chain[:, :, :].reshape((-1, ndim))))

	# 		np.savetxt(os.getcwd() + '/{}/samples.txt'.format(dirname), samples)


	# 	except:
	# 		# t1 = time.time()
	# 		for n, s in enumerate(sampler.sample(pos, iterations = nburn)):
	# 			if n % nthin == 0:
	# 				with open('{}/{}_{}_burnin.txt'.format(dirname,fname, n), 'ab') as f:
	# 					f.write(b"\n")
	# 					np.savetxt(f, s.coords)
	# 					f.close() 
	# 			#f = open('results/{}_burnin.txt'.format(fname), "a")
	# 			#f.write(s.coords)
	# 			#f.close()

	# 		# t1 = time.time()
	# 		# sampler.run_mcmc(pos, nburn, progress=True)
	# 		# print('total burnin time: ', time.time() - t1, ' seconds')

	# 		state = sampler.get_last_sample()
	# 		sampler.reset()

	# 		old_acl = np.inf
	# 		for n, s in enumerate(sampler.sample(state, iterations = nsteps)):
	# 			# print(n)
	# 			if n % nthin == 0:
	# 				with open('{}/{}_{}_results.txt'.format(dirname,fname, n), 'ab') as f:
	# 					f.write(b'\n')
	# 					np.savetxt(f, s.coords)
	# 					f.close()
						
	# 				acl = sampler.get_autocorr_time(quiet = True)
	# 				macl = np.mean(acl)

	# 				with open('{}/{}_autocorr.txt'.format(dirname,fname), 'a') as f:
	# 					f.write(str(macl) + '\n')
				
	# 				if not np.isnan(macl):
	# 					# print('hi')
	# 					converged = np.all(acl * 50 < n)
	# 					# print(converged)
	# 					converged &= np.all((np.abs(old_acl - acl) / acl) < 0.1)
	# 					# print(converged)
	# 					if converged == True:
	# 						break

	# 				old_acl = acl
	# 		print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

	# 		samples = sampler.chain[:, :, :].reshape((-1, ndim))

	# 		np.savetxt(os.getcwd() + '/{}/samples.txt'.format(dirname), samples)
	# 		#f = open('results/{}_results.txt'.format(fname), 'a')
	# 		#f.write(s.coords)
	# 		#f.close()
	# 	#np.savetxt('results/{}_results.txt'.format(fname), sampler.flatchain)



	samples = np.genfromtxt(os.getcwd()+'/{}/samples.txt'.format(dirname)) #np.concatenate(([np.genfromtxt('results/run2_{}_results.txt'.format(r)) for r in np.arange(1000, 1300, 50)]))

	samples = np.hstack((samples[:,:4], samples[:,5:]))

	# for i in range(ndim):
	# 	plt.figure(i)
	# 	plt.hist(sampler.flatchain[:,i], histtype="step")
	# 	plt.title("Dimension {0:d}".format(i))
	# 	plt.savefig(os.getcwd() + '/{}/plots/{}_{}.pdf'.format(dirname,fname, i))
	# 	plt.close()

	# 	plt.figure(i)

	# 	# try:
	# 	for n in range(nwalkers):
	# 		plt.plot(np.arange(len(sampler.chain[n, :, i])),sampler.chain[n, :, i], color = 'k', alpha = 0.5)
	# 	plt.savefig(os.getcwd() + '/{}/plots/{}_chain_{}.pdf'.format(dirname,fname, i))
	# 	plt.close()
	# 	# except:
	# 	# 	pass

	if ndim == 8:
		samples[:,-1] *= 1e3

		if dist_fit == True:			

			## make the plots not suck

			plt.rcParams['lines.linewidth']   =2
			plt.rcParams['axes.linewidth']    = 1.5
			plt.rcParams['xtick.major.width'] =2
			plt.rcParams['ytick.major.width'] =2
			plt.rcParams['ytick.labelsize'] = 13
			plt.rcParams['xtick.labelsize'] = 13
			plt.rcParams['axes.labelsize'] = 18
			plt.rcParams['legend.numpoints'] = 1
			plt.rcParams['axes.labelweight']='semibold'
			plt.rcParams['mathtext.fontset']='stix'
			plt.rcParams['font.weight'] = 'semibold'
			plt.rcParams['axes.titleweight']='semibold'
			plt.rcParams['axes.titlesize']=9

 

			# corner.corner(CORNERDATA, labels=planetlabels,bins=50,quantiles=(0.16, 0.84),# levels=(1-np.exp(-0.5),), ## optional

			#         fill_contours=True, plot_datapoints=False,title_kwargs={"fontsize": 12},#title_fmt='.3f',## change the format based on the situation

			#         hist_kwargs={"linewidth": 2.5},levels=levels,smooth=0.75

			#     );
			figure = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], \
				labels = [r'T$_{eff,1}$', r'T$_{eff,2}$', r'Log(g)$_{1}$', r'Log(g)$_{2}$', r'R$_{1}$', r'R$_{2}$/R$_{1}$', r'$\pi$ (mas)'], show_titles = True,\
				 bins = 50, fill_contours = True, plot_datapoints = False, title_kwargs = {'fontsize':15}, hist_kwargs={"linewidth":2}, smooth = 0.75, title_fmt = title_format)
			
			if not all(v for v in value1) == 0:
				value1[-1] *= 1e3
				# Extract the axes
				axes = np.array(figure.axes).reshape((ndim-1, ndim-1))

				# Loop over the diagonal
				for i in range(ndim-1):
					ax = axes[i, i]
					ax.axvline(value1[i], color="g")

				# Loop over the histograms
				for yi in range(ndim-1):
					for xi in range(yi):
						ax = axes[yi, xi]
						ax.axvline(value1[xi], color="g")
						ax.axhline(value1[yi], color="g")
						ax.plot(value1[xi], value1[yi], "sg")

			figure.savefig(os.getcwd() + "/{}/plots/{}_corner.pdf".format(dirname, fname))
			plt.close()

		elif dist_fit == False:

			# samples[:,-3] = samples[:,-2]#/samples[:,-3]


			plt.rcParams['lines.linewidth']   =2
			plt.rcParams['axes.linewidth']    = 1.5
			plt.rcParams['xtick.major.width'] =2
			plt.rcParams['ytick.major.width'] =2
			plt.rcParams['ytick.labelsize'] = 13
			plt.rcParams['xtick.labelsize'] = 13
			plt.rcParams['axes.labelsize'] = 18
			plt.rcParams['legend.numpoints'] = 1
			plt.rcParams['axes.labelweight']='semibold'
			plt.rcParams['mathtext.fontset']='stix'
			plt.rcParams['font.weight'] = 'semibold'
			plt.rcParams['axes.titleweight']='semibold'
			plt.rcParams['axes.titlesize']=9

			# corner.corner(CORNERDATA, labels=planetlabels,bins=50,quantiles=(0.16, 0.84),# levels=(1-np.exp(-0.5),), ## optional

			#         fill_contours=True, plot_datapoints=False,title_kwargs={"fontsize": 12},#title_fmt='.3f',## change the format based on the situation

			#         hist_kwargs={"linewidth": 2.5},levels=levels,smooth=0.75

			#     );

			figure = corner.corner(samples[:,:-1], quantiles=[0.16, 0.5, 0.84], \
				labels = [r'T$_{eff,1}$', r'T$_{eff,2}$', r'Log(g)$_{1}$', r'Log(g)$_{2}$', r'R$_{1}$', r'R$_{2}$/R$_{1}$'], show_titles = True,\
				 bins = 50, fill_contours = True, plot_datapoints = False, title_kwargs = {'fontsize':15}, hist_kwargs={"linewidth":2}, smooth = 0.75, title_fmt = title_format)
			
			if not all(v for v in value1) == 0:
				# value1[-2] = value1[-2]/value1[-3]
				# Extract the axes
				axes = np.array(figure.axes).reshape((ndim-2, ndim-2))

				# Loop over the diagonal
				for i in range(ndim-2):
					ax = axes[i, i]

					# if i == max(range(ndim - 3)):
					# 	print(value1[-2], value1[-3])
					# 	ax.axvline(value1[-2]/value1[-3], color = 'g')
					# else:
					ax.axvline(value1[i], color="g")

				# Loop over the histograms
				for yi in range(ndim-2):
					for xi in range(yi):
						ax = axes[yi, xi]
						ax.axvline(value1[xi], color="g")
						ax.axhline(value1[yi], color="g")
						ax.plot(value1[xi], value1[yi], "sg")

			figure.savefig(os.getcwd() + "/{}/plots/{}_corner.pdf".format(dirname, fname))
			plt.close()
	else:
		figure = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],labels = [r'T$_{eff,1}$', r'T$_{eff,2}$', r'Log(g)$_{1}$', r'Log(g)$_{2}$', r'$\frac{Rad_{2}}{Rad_{1}}$'], show_titles = True, title_fmt = title_format)

		if not all(v for v in value1) == 0:
			# Extract the axes
			axes = np.array(figure.axes).reshape((ndim-1, ndim-1))

			# Loop over the diagonal
			for i in range(ndim -1 ):
				ax = axes[i, i]
				ax.axvline(value1[i], color="g")

			# Loop over the histograms
			for yi in range(ndim -1):
				for xi in range(yi):
					ax = axes[yi, xi]
					ax.axvline(value1[xi], color="g")
					ax.axhline(value1[yi], color="g")
					ax.plot(value1[xi], value1[yi], "sg")

		figure.savefig(os.getcwd() + "/{}/plots/{}_corner.pdf".format(dirname, fname))
		plt.close()
	return samples

def optimize_fit(dirname, data, err, specs, nwalk, fr, dist_arr, av, res, ctm, ptm, tmi, tma, vs, matrix, cutoff = 2, nstep = 200, nburn = 20, con = True, models = 'btsettl', err2 = 0, dist_fit = True, rad_prior = False):
	#we're going to do rejection sampling:
	#initialize a set number of random walkers randomly in parameter space (teff 1 + 2, log(g) 1 + 2)
	#for now, assume we've already fit for extinction - should be easy enough to add in an additional parameter eventually
	#do a reduced chi square to inform steps
	#take some assigned number of steps/until we hit convergence (chisq < 2 or something)
	#take the best 50% (or whatever) and use those as the MCMC walkers


	#first, get the temperature and log(g) range from sp by reading in the keys
	#we need this so we can initialize the random walkers in the correct parameter space
	t, l = [], []
	for s in specs.keys():
		if not s == 'wl':
			a = s.split(', ')
			t.append(float(a[0]))
			l.append(float(a[1]))

	tmin, tmax, lmin, lmax = min(t), max(t), min(l), max(l)

	rmin = 0.05; rmax = 1 #min and max radii in solar radius

	# #now we have the limits, we need to initialize the random walkers over the parameter space
	# #we need to assign four numbers: the two temps and the two log(g)s
	# #so randomly distribute the assigned number of walkers over the parameter space
	# #making sure that the secondary temperature is always less than the primary

	t1, l1, l2 = np.random.uniform(tmin, tmax, nwalk), np.random.uniform(lmin, lmax, nwalk), np.random.uniform(lmin, lmax, nwalk)
	t2 = []
	for tt in t1:
		tt2 = np.random.uniform(tmin, tt)
		t2.append(tt2)

	e1 = np.random.uniform(0, 0.5, nwalk)

	if not type(dist_arr) == bool:
		rg1 = np.random.uniform(rmin, rmax, nwalk)
		rg2 = []
		for r in rg1:
			r2 = np.random.uniform(rmin, r)
			rg2.append(r2/r)
	else:
		rad = np.random.uniform(rmin, rmax)

	dist = np.random.uniform(1/2000, 1/100, nwalk)

	# dist = np.ones(nwalk) * dist_arr[0]
	# t1, l1, l2, t2, e1, rg1, rg2 = np.ones(nwalk) * 3850, np.ones(nwalk) * 4.7, np.ones(nwalk) * 4.9, np.ones(nwalk) * 3325, np.ones(nwalk) * 0.2, np.ones(nwalk) * 0.5, np.ones(nwalk) * 0.261

	# dist = [1/d for d in dist]
	# dist = np.random.uniform(100, 1000, nwalk)
	# fit_spec(1, data[0], data[1], err, [min(data[0]), max(data[0])], [float(t1[0]), float(t2[0])], [float(l1[0]), float(l2[0])], e1[0], [float(rg1[0]), float(rg2[0])],\
	# fr, specs, [tmin, tmax], [lmin, lmax], dist[0], cs = cutoff, steps = nstep, burn = nburn, conv = con, models = models)

	#now we need to evaluate the chi square of each position until it's either hit a maximum number of steps or has converged 
	#use fit_test, which uses a metropolis-hastings algorithm to walk around the parameter space

	with mp.Pool(processes = 75) as pool:

		# if not type(dist_arr) == bool:
		out = [pool.apply_async(fit_spec, \
				args = (n, dirname, data[0], data[1], err, [min(data[0]), max(data[0])], [t1[n], t2[n]], [l1[n], l2[n]], [e1[n], av[0], av[1]], [rg1[n], rg2[n]], fr, specs, [tmin, tmax], [lmin, lmax], [dist[n], dist_arr[0], dist_arr[1]], ctm, ptm, tmi, tma, vs, matrix), \
				kwds = dict(cs = cutoff, steps = nstep, burn = nburn, conv = con, models = models, dist_fit = dist_fit, rad_prior = rad_prior)) for n in range(nwalk)]
		# else:
		# 	out = [pool.apply_async(fit_spec, \
		# 		args = (n, dirname, data[0], data[1], err, [min(data[0]), max(data[0])], [t1[n], t2[n]], [l1[n], l2[n]], [e1[n], av[0], av[1]], [rad], fr, specs, [tmin, tmax], [lmin, lmax], dist_arr, ctm, ptm, tmi, tma, vs), \
		# 		kwds = dict(cs = cutoff, steps = nstep, burn = nburn, conv = con, models = models)) for n in range(nwalk)]			

		a = [o.get() for o in out]

		for line in a:
			gi = line[0]; cs = line[1]

			with open(dirname + '/optimize_res.txt', 'a') as f:
				f.write(gi)
			with open(dirname + '/optimize_cs.txt', 'a') as f:
				f.write(str(cs) + '\n')

	return 

def plot_fit(run, data, sp, fr, ctm, ptm, tmi, tma, vs, models = 'btsettl', dist_fit = True):
	'''
	Plots for after the optimization step
	'''
	#find all the chisq and parameter files 
	cs_files = glob(run + '/chisq*txt')
	walk_files = glob(run + '/params*txt')

	#initialize a bunch of figures 
	#there was deifnitely an easier way to do this, but whatever
	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()
	fig3, ax3 = plt.subplots()
	fig4, ax4 = plt.subplots()
	fig5, ax5 = plt.subplots()
	fig6, ax6 = plt.subplots()
	fig7, ax7 = plt.subplots()
	fig8, ax8 = plt.subplots()
	fig9, ax9 = plt.subplots()
	fig10, ax10 = plt.subplots()

	#for each set of parameter files
	for f in walk_files:
		#open the file
		results = np.genfromtxt(f, dtype = 'S')

		#initialize a bunch of arrays to hold the parameters
		tem1, tem2, log1, log2, ext, rad1, rad2, dist = [], [], [], [], [], [], [], []

		#read in the file after saving each line
		for n, line in enumerate(results):
			try:
				tem1.append(float(line[0])); tem2.append(float(line[1]))
				log1.append(float(line[2])); log2.append(float(line[3]))
				ext.append(float(line[4])); rad1.append(float(line[5]))
				rad2.append(float(line[6])), dist.append(float(line[7]))

				#if we're on the last line, read in the final values to separate variables as well, since these are the best-fit values
				if n == len(results)-1:
					tt1, tt2, tl1, tl2, te, tr1, tr2, d = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7])
			#if there's only a radius ratio and no distance, the try clause above won't work so read in the shorter line correctly here
			except:
				tem1.append(float(line[0])); tem2.append(float(line[1]))
				log1.append(float(line[2])); log2.append(float(line[3]))
				ext.append(float(line[4]))
				rad1.append(float(line[5]))

		#plot the steps as a function of length for each variable
		ax1.plot(range(len(tem1)), tem1, color = 'k', alpha = 0.5)
		ax2.plot(range(len(tem2)), tem2, color = 'k', alpha = 0.5)
		ax3.plot(range(len(log1)), log1, color = 'k', alpha = 0.5)
		ax4.plot(range(len(log2)), log2, color = 'k', alpha = 0.5)
		ax5.plot(range(len(ext)), ext, color = 'k', alpha = 0.5)
		ax6.plot(range(len(rad1)), rad1, color = 'k', alpha = 0.5)
		try:
			ax7.plot(range(len(rad2)), rad2, color = 'k', alpha = 0.5)
			ax8.plot(range(len(dist)), dist, color = 'k', alpha = 0.5)
			ax9.plot(rad1, dist, color = 'k', alpha = 0.5)
			ax10.plot(len(rad1), rad2/rad1, color = 'k', alpha = 0.5)
		except:
			pass

	#once all the chains have been plotted, turn on minorticks first 
	plt.minorticks_on()
	#now try to make all the figures
	try:
		figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]

		#format everything nicely for each figure and save it
		for n, a in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]):
			labels = ['teff1', 'teff2', 'logg1', 'logg2', 'Av', 'rad1', 'rad2', 'dist', 'rad1vsdist', 'rad_ratio']
			a.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
			a.tick_params(bottom=True, top =True, left=True, right=True)
			a.tick_params(which='both', labelsize = "large", direction='in')
			a.tick_params('both', length=8, width=1.5, which='major')
			a.tick_params('both', length=4, width=1, which='minor')
			a.set_xlabel('Step number', fontsize = 13)
			a.set_ylabel('{}'.format(labels[n]), fontsize = 13)
			plt.tight_layout()
			figs[n].savefig(run + '/plots/fit_res_{}.png'.format(labels[n]))
			plt.close()
	#if there's no distance some of those arrays won't be defined, so come down here and make fewer figures		
	except:
		figs = [fig1, fig2, fig3, fig4, fig5, fig6]

		#format everything nicely for each figure and save it
		for n, a in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
			labels = ['teff1', 'teff2', 'logg1', 'logg2', 'Av', 'rad1', 'rad2', 'dist', 'rad1vsdist']
			a.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
			a.tick_params(bottom=True, top =True, left=True, right=True)
			a.tick_params(which='both', labelsize = "large", direction='in')
			a.tick_params('both', length=8, width=1.5, which='major')
			a.tick_params('both', length=4, width=1, which='minor')
			a.set_xlabel('Step number', fontsize = 13)
			a.set_ylabel('{}'.format(labels[n]), fontsize = 13)
			plt.tight_layout()
			figs[n].savefig(run + '/plots/fit_res_{}.png'.format(labels[n]))
			plt.close()


	#read in the best-fit parameters from each run 
	chisqs, pars = np.genfromtxt(run + '/optimize_cs.txt'), np.genfromtxt(run + '/optimize_res.txt')

	#make a best-fit data set using the parameters from the chi^2 minimum 
	if dist_fit == True and len(pars[0,:]) == 8:
		tt1, tt2, tl1, tl2, te, rad1, rad2, dist = pars[np.where(chisqs == min(chisqs))][0]
		w, spe, p1, p2, p3 = make_composite([tt1, tt2], [tl1, tl2], [rad1, rad2], dist, fr[2], fr[5], [min(data[0]), max(data[0])], sp, ctm, ptm, tmi, tma, vs, models = models)
	elif len(pars[0,:]) == 8 and dist_fit == False:
		tt1, tt2, tl1, tl2, te, rad1, rad2, dist = pars[np.where(chisqs == min(chisqs))][0]
		ratio = rad2/rad1
		w, spe, p1, p2, p3 = make_composite([tt1, tt2], [tl1, tl2], [ratio], False, fr[2], fr[5], [min(data[0]), max(data[0])], sp, ctm, ptm, tmi, tma, vs, models = models)
	else:
		tt1, tt2, tl1, tl2, te, ratio1 = pars[np.where(chisqs == min(chisqs))][0]
		w, spe, p1, p2, p3 = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(data[0]), max(data[0])], sp, ctm, ptm, tmi, tma, vs, models = models)

	#extinct the best-fit spectrum
	# spe = extinct(w, spe, te)

	#interpoalte the best-fit spectrum onto the data wavelength
	itep = interp1d(w, spe)
	spe = itep(data[0]*1e4)

	#normalize the best-fit spectrum to the data
	spe *= np.median(data[1])/np.median(spe)

	#plot the data and the best-fit spectrum, and save the figure
	plt.figure()
	plt.minorticks_on()
	plt.plot(data[0]*1e4, data[1], color = 'navy', linewidth = 1)#, label = 'data: 4250 + 3825; 4.2 + 4.3; 2')
	plt.plot(data[0]*1e4, spe, color = 'xkcd:sky blue', label = 'model: {:.0f} + {:.0f}; {:.1f} + {:.1f}; {:.2f}'.format(tt1, tt2, tl1, tl2, te), linewidth = 1)
	plt.xlim(max(min(w), min(data[0]*1e4)), min(max(w), max(data[0])*1e4))
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.xlabel('Wavelength (A)', fontsize = 13)
	plt.ylabel('Normalized flux', fontsize = 13)
	plt.legend(loc = 'best', fontsize = 13)
	plt.tight_layout()
	plt.savefig(run + '/plots/bestfit_spec.pdf')
	plt.close()

	return

def plot_results(fname, sample, run, data, sp, fr, ctm, ptm, tmi, tma, vs, real_val, distance, models = 'btsettl', res = 1000, dist_fit = True, tell = False):
	#unpack the sample array
	t1, t2, l1, l2, av, r1, r2, pl = sample.T
	#calculate the radius ratio
	ratio = r2/r1

	#calculate the median value for all the emcee chains
	a = np.median(sample, axis = 0)
	#draw a random sample of parameters from the emcee draws 
	random_sample = sample[np.random.choice(len(sample), size = 100, replace = False), :]

	############
	#FIND AND PLOT THE BEST-FIT PARAMETERS IN THE BIMODAL DISTRIBUTIONS
	###########
	#define the number of bins to use to create the PDFs
	nbins = 75

	#initialize some arrays 
	#*bins defines the histogram bins; *count is the PDF for the bins (non-normalized)
	t1_bins, t2_bins = np.linspace(min(t1), max(t1), nbins), np.linspace(min(t2), max(t2), nbins)
	r1_bins, r2_bins = np.linspace(min(r1), max(r1), nbins), np.linspace(min(r2), max(r2), nbins)
	ratio_bins = np.linspace(min(ratio), max(ratio), nbins); ratio_count = np.zeros(len(ratio_bins))

	t1_count, t2_count = np.zeros(len(t1_bins)), np.zeros(len(t2_bins))
	r1_count, r2_count = np.zeros(len(r1_bins)), np.zeros(len(r2_bins))

	#fill up the count bins 
	#temperature 1
	for t in t1:
		for b in range(len(t1_bins) - 1):
			if t1_bins[b] <= t < t1_bins[b + 1]:
				t1_count[b] += 1

	#Teff 2
	for t in t2:
		for b in range(len(t2_bins) - 1):
			if t2_bins[b] <= t < t2_bins[b + 1]:
				t2_count[b] += 1

	#radius 1
	for r in r1:
		for b in range(len(r1_bins) - 1):
			if r1_bins[b] <= r < r1_bins[b + 1]:
				r1_count[b] += 1

	#Radius 2
	for r in r2:
		for b in range(len(r2_bins) - 1):
			if r2_bins[b] <= r < r2_bins[b + 1]:
				r2_count[b] += 1

	#radius ratio
	for r in ratio:
		for b in range(len(ratio_bins) - 1):
			if ratio_bins[b] <= r < ratio_bins[b + 1]:
				ratio_count[b] += 1

	#####
	#Temperature 1
	#####
	try:
		#find the local min between the two gaussian temperature measurements 
		t1_localmin = int(np.mean([np.where(t1_count[np.where((t1_bins > min(t1)) & (t1_bins < max(t1)))] < 0.5*max(t1_count))]))
		#fit a bimodal gaussian to the distribution
		fit_t1, cov = curve_fit(bimodal, t1_bins, t1_count, [np.mean(t1_bins[t1_localmin:]), np.std(t1_bins[t1_localmin:]), max(t1_count[t1_localmin:]),\
			 np.mean(t1_bins[:t1_localmin]), np.std(t1_bins[:t1_localmin]), max(t1_count[:t1_localmin])])

		#plot the bimodal distribution, the best fit, and the corresponding component Gaussians
		plt.figure()
		plt.hist(t1, bins = t1_bins)
		# plt.hist(t1[np.where(t1 >= t1_bins[t1_localmin])], bins = t1_bins[t1_localmin:], color = 'g')
		# plt.hist(t1[np.where(t1 <= t1_bins[t1_localmin])], bins = t1_bins[:t1_localmin], color = '0.5')
		plt.axvline(t1_bins[t1_localmin], color = 'k', linewidth = 2)
		plt.plot(t1_bins, t1_count)
		plt.plot(t1_bins, bimodal(t1_bins, *fit_t1), color = 'b')
		plt.plot(t1_bins, gauss(t1_bins, *fit_t1[:3]))
		plt.plot(t1_bins, gauss(t1_bins, *fit_t1[3:]))
		plt.savefig(run + '/plots/bimodal_test_T1.pdf')

		#calculate the area under each curve as a percentage
		t1_v1 = np.trapz(gauss(t1_bins, *fit_t1[:3]))/np.trapz(bimodal(t1_bins, *fit_t1))
		t1_v2 = np.trapz(gauss(t1_bins, *fit_t1[3:]))/np.trapz(bimodal(t1_bins, *fit_t1))

		#pick the gaussian with the largest area and set the appropriate parameters to the corresponding values as established from the fit
		if t1_v1 > t1_v2:
			a[0] = fit_t1[0]; sigma_t1 = fit_t1[1]; 
		else:
			a[0] = fit_t1[3]; sigma_t1 = fit_t1[4]
		p_t1 = max(t1_v1, t1_v2)
	#if none of that works because it's not bimodal, just set the stddev to 0 and the best-fit value will stay at the median value from emcee	
	except:
		sigma_t1 = 0
		pass;

	#####
	#Temperature 2
	#####
	try:
		t2_localmin = int(np.mean([np.where(t2_count[np.where((t2_bins > min(t2)) & (t2_bins < max(t2)))] < 0.5*max(t2_count))]))

		fit_t2, cov = curve_fit(bimodal,t2_bins, t2_count, [np.mean(t2_bins[t2_localmin:]), np.std(t2_bins[t2_localmin:]), max(t2_count[t2_localmin:]), \
				np.mean(t2_bins[:t2_localmin]), np.std(t2_bins[:t2_localmin]), max(t2_count[:t2_localmin])])

		plt.figure()
		plt.hist(t2, bins = t2_bins)
		# plt.hist(t2[np.where(t2 >= t2_bins[t2_localmin])], bins = t2_bins[t2_localmin:], color = 'g')
		# plt.hist(t2[np.where(t2 <= t2_bins[t2_localmin])], bins = t2_bins[:t2_localmin], color = '0.5')
		plt.axvline(t2_bins[t2_localmin], color = 'k', linewidth = 2)
		plt.plot(t2_bins, t2_count)
		plt.plot(t2_bins, bimodal(t2_bins, *fit_t2), color = 'b')
		plt.plot(t2_bins, gauss(t2_bins, *fit_t2[:3]))
		plt.plot(t2_bins, gauss(t2_bins, *fit_t2[3:]))
		plt.savefig(run + '/plots/bimodal_test_T2.pdf')


		t2_v1 = np.trapz(gauss(t2_bins, *fit_t2[:3]))/np.trapz(bimodal(t2_bins, *fit_t2))
		t2_v2 = np.trapz(gauss(t2_bins, *fit_t2[3:]))/np.trapz(bimodal(t2_bins, *fit_t2))
		if t2_v1 > t2_v2:
			a[1] = fit_t2[0]; sigma_t2 = fit_t2[1]; 
		else:
			a[1] = fit_t2[3]; sigma_t2 = fit_t2[4]
		p_t2 = max(t2_v1, t2_v2)
	except:
		sigma_t2 = 0
		pass;

	#####
	#Radius 1
	#####
	try:
		r1_localmin = int(np.mean([np.where(r1_count[np.where((r1_bins > min(r1)) & (r1_bins < max(r1)))] < 0.5*max(r1_count))]))
		fit_r1, cov = curve_fit(bimodal, r1_bins, r1_count, [np.mean(r1_bins[r1_localmin:]), np.std(r1_bins[r1_localmin:]), max(r1_count[r1_localmin:]),\
			 np.mean(r1_bins[:r1_localmin]), np.std(r1_bins[:r1_localmin]), max(r1_count[:r1_localmin])])

		plt.figure()
		plt.hist(r1, bins = r1_bins)
		# plt.hist(t1[np.where(t1 >= t1_bins[t1_localmin])], bins = t1_bins[t1_localmin:], color = 'g')
		# plt.hist(t1[np.where(t1 <= t1_bins[t1_localmin])], bins = t1_bins[:t1_localmin], color = '0.5')
		plt.axvline(r1_bins[r1_localmin], color = 'k', linewidth = 2)
		plt.plot(r1_bins, r1_count)
		plt.plot(r1_bins, bimodal(r1_bins, *fit_r1), color = 'b')
		plt.plot(r1_bins, gauss(r1_bins, *fit_r1[:3]))
		plt.plot(r1_bins, gauss(r1_bins, *fit_r1[3:]))
		plt.savefig(run + '/plots/bimodal_test_R1.pdf')

		r1_v1 = np.trapz(gauss(r1_bins, *fit_r1[:3]))/np.trapz(bimodal(r1_bins, *fit_r1))
		r1_v2 = np.trapz(gauss(r1_bins, *fit_r1[3:]))/np.trapz(bimodal(r1_bins, *fit_r1))
		if r1_v1 > r1_v2:
			a[5] = fit_r1[0]; sigma_r1 = fit_r1[1]; 
		else:
			a[5] = fit_r1[3]; sigma_r1 = fit_r1[4]
		p_r1 = max(r1_v1, r1_v2)
	except:
		sigma_r1 = 0
		pass;

	#####
	#Radius ratio
	#####
	try:
		r2_localmin = int(np.mean([np.where(r2_count[np.where((r2_bins > min(r2)) & (r2_bins < max(r2)))] < 0.5*max(r2_count))]))
		fit_r2, cov = curve_fit(bimodal,r2_bins, r2_count, [np.mean(r2_bins[r2_localmin:]), np.std(r2_bins[r2_localmin:]), max(r2_count[r2_localmin:]), \
				np.mean(r2_bins[:r2_localmin]), np.std(r2_bins[:r2_localmin]), max(r2_count[:r2_localmin])])

		plt.figure()
		plt.hist(r2, bins = r2_bins)
		# plt.hist(t2[np.where(t2 >= t2_bins[t2_localmin])], bins = t2_bins[t2_localmin:], color = 'g')
		# plt.hist(t2[np.where(t2 <= t2_bins[t2_localmin])], bins = t2_bins[:t2_localmin], color = '0.5')
		plt.axvline(r2_bins[r2_localmin], color = 'k', linewidth = 2)
		plt.plot(r2_bins, r2_count)
		plt.plot(r2_bins, bimodal(r2_bins, *fit_r2), color = 'b')
		plt.plot(r2_bins, gauss(r2_bins, *fit_r2[:3]))
		plt.plot(r2_bins, gauss(r2_bins, *fit_r2[3:]))
		plt.savefig(run + '/plots/bimodal_test_R2R1.pdf')

		r2_v1 = np.trapz(gauss(r2_bins, *fit_r2[:3]))/np.trapz(bimodal(r2_bins, *fit_r2))
		r2_v2 = np.trapz(gauss(r2_bins, *fit_r2[3:]))/np.trapz(bimodal(r2_bins, *fit_r2))
		if r2_v1 > r2_v2:
			a[6] = fit_r2[0]; sigma_r2 = fit_r2[1]; 
		else:
			a[6] = fit_r2[3]; sigma_r2 = fit_r2[4]
		p_r2 = max(r2_v1, r2_v2)
	except:
		sigma_r2 = 0
		pass;


	############
	#CREATE THE PHOTOMETRY AND CONTRAST PLOTS
	###########

	# a = np.median(sample, axis = 0)


	plt.minorticks_on()
	wl, spec, err = data
	err /= np.median(spec); spec /= np.median(spec)

	#unpack the best-fit stellar parameters 
	if len(a) == 8:
		tt1, tt2, tl1, tl2, te, rad1, rad2, plx = np.median(sample, axis = 0)
		ratio1 = rad2
	else:
		tt1, tt2, tl1, tl2, te, ratio1 = a

	if len(a) == 8:
		# if dist_fit == True:
		# b = np.median(sample, axis = 0)	# print('composite')
		ww, ss, c, pcwl, phot = make_composite([tt1, tt2], [tl1, tl2], [rad1, rad2], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False)
		
		w, spe, pri_spec, sec_spec, pri_mag, sec_mag = make_composite([tt1, tt2], [tl1, tl2], [rad1, rad2], plx, fr[2], fr[5], [5000, 24000], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		new_wl, new_prispec = redres(w, pri_spec, 250)
		new_wl, new_spe = redres(w, spe, 250)
		new_wl, new_secspec = redres(w, sec_spec, 250)

		new_wl, new_spe, new_prispec, new_secspec = new_wl[np.where((new_wl >= 5315) & (new_wl <= 23652))], new_spe[np.where((new_wl >= 5315) & (new_wl <= 23652))], new_prispec[np.where((new_wl >= 5315) & (new_wl <= 23652))], new_secspec[np.where((new_wl >= 5315) & (new_wl <= 23652))]
		# else:
			# b = np.median(sample, axis = 0)
		# 	# print('composite')
			# ww, ss, c, pcwl, phot = make_composite([tt1, tt2], [tl1, tl2], [b[5], ratio1], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False)
			# print(tt1, tt2, tl1, tl2, b[5], b[6], phot)

		zp = [2.854074834606756e-09,1.940259205607388e-09,1.359859453789013e-09,3.1121838042516567e-10,1.1353317746392182e-10, 4.279017715611946e-11]
		phot_cwl = [6175, 7489, 8946, 12350, 16620, 21590]
		phot_width = np.array([[6175-5415, 6989-6175], [7489-6689, 8389-7489], [8946-7960, 10833-8946], [12350-10806, 14067-12350], [16620-14787, 18231-16620], [21590-19543, 23552-21590]]).T

		fig,ax = plt.subplots(nrows = 3, gridspec_kw = dict(hspace = 0, height_ratios = [3, 1.75, 1]), sharex = True, figsize = (7 , 6))
		e = ax[0].scatter(phot_cwl[:len(phot)], 10**(-0.4*phot)*zp[:len(phot)], color = 'seagreen', s = 100, marker = '.', label = 'Composite phot.')
		ax[0].errorbar(phot_cwl[:len(phot)], 10**(-0.4*phot)*zp[:len(phot)], xerr = phot_width[:len(phot)], color= 'seagreen', zorder = 0, linestyle = 'None')
		b = ax[0].scatter(phot_cwl[:len(phot)], 10**(-0.4*fr[3])*zp[:len(phot)], linestyle = 'None', color = 'k', marker = '.', s = 100, label = 'Data phot.')
		m = ax[0].plot(new_wl, new_spe, color = 'seagreen', linewidth = 1, zorder = 0, alpha = 0.5)
		# ax[0].set_ylim(max(phot) + 0.5, min(phot) - 0.5)	
		plt.minorticks_on()
		ax[0].set_xscale('log')
		ax[0].set_yscale('log')
		ax[0].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(which='both', labelsize = 12, direction='in')
		ax[0].tick_params('both', length=8, width=1.5, which='major')
		ax[0].tick_params('both', length=4, width=1, which='minor')
		# ax[0].set_xlabel(r'Wavelength (\AA)', fontsize = 14)
		ax[0].set_ylabel('{}'.format(r'Flux (erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)'), fontsize = 12)

		# y_minor = matplotlib.ticker.LinearLocator(numticks = int(len(np.arange(min(phot) -0.5, max(phot) + 0.75, 0.2))))
		# ax[0].yaxis.set_minor_locator(y_minor)
		# ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

		# ax2 = ax[0].twinx()
		f = ax[1].scatter(ctm[3], c, color = 'blue', marker = 'v', label = 'Model contrast', zorder = 2)
		ax[1].errorbar(ctm[3], fr[0], yerr = fr[1], linestyle = 'None', capsize = 4, capthick = 2, color = 'k', marker = 'v', zorder = 1)
		g = ax[1].scatter(ctm[3], fr[0], color = 'k', marker = 'v', label = 'Data contrast', zorder = 1)
		ax[1].plot(new_wl, 2.5*np.log10(new_prispec) - 2.5*np.log10(new_secspec), color = 'blue', linewidth = 1, zorder = 0, alpha = 0.5)
		ax[1].set_ylabel(r'$\Delta$ mag', fontsize = 12)

		# y_minor = matplotlib.ticker.LinearLocator(numticks = 15)
		# ax[1].yaxis.set_minor_locator(y_minor)
		# ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
		plt.minorticks_on()
		ax[1].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(which='both', labelsize = 12, direction='in')
		ax[1].tick_params('both', length=8, width=1.5, which='major')
		ax[1].tick_params('both', length=4, width=1, which='minor')

		ax[2].scatter(phot_cwl[:len(phot)], phot[:len(phot)]-fr[3][:len(phot)], color = 'seagreen', marker = 'x', s = 50, label = 'Phot. resid.')
		ax[2].axhline(0, color = '0.3', linestyle = '--', linewidth = 2, label = 'No resid.')
		ax[2].scatter(ctm[3], np.array(fr[0])-np.array(c), color = 'blue', marker = 'x', label = 'Cont. resid.',s = 50)
		plt.minorticks_on()
		ax[2].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[2].tick_params(bottom=True, top =True, left=True, right=True)
		ax[2].tick_params(which='both', labelsize = 12, direction='in')
		ax[2].tick_params('both', length=8, width=1.5, which='major')
		ax[2].tick_params('both', length=4, width=1, which='minor')
		ax[2].set_xlabel(r'Wavelength (\AA)', fontsize = 12)
		ax[2].set_ylabel('{}'.format(r'Resid. (mag)'), fontsize = 12)

		fig.align_ylabels(ax)

		handles, labels = plt.gca().get_legend_handles_labels()
		handles.extend([e,b,f,g])

		ax[0].legend(handles = handles, loc = 'best', fontsize = 10, ncol = 2)

		plt.tight_layout()
		plt.savefig(run + '/plots/{}_phot_scatter.pdf'.format(fname))
		plt.close()

	else:
		ww, ss, c, pcwl, phot = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False)
		fig,ax = plt.subplots(nrows = 2, gridspec_kw = dict(hspace = 0, height_ratios = [3, 1]), sharex = True, figsize = (7,6))
		f = ax[0].scatter(ctm[3], c, color = 'blue', marker = 'v', label = 'Model contrast', zorder = 2)
		g = ax[0].errorbar(ctm[3], fr[0], yerr = fr[1], linestyle = 'None', capsize = 4, capthick = 2, color = 'k', marker = 'v', label = 'Data contrast', zorder = 1)
		ax[0].set_ylabel('Contrast (mag)', fontsize = 12)
		# y_minor = matplotlib.ticker.LinearLocator(numticks = 15)
		# ax[0].yaxis.set_minor_locator(y_minor)
		# ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
		plt.minorticks_on()
		ax[0].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(which='both', labelsize = 14, direction='in')
		ax[0].tick_params('both', length=8, width=1.5, which='major')
		ax[0].tick_params('both', length=4, width=1, which='minor')

		d = ax[1].axhline(0, color = '0.3', linestyle = '--', linewidth = 2, label = 'No resid.')
		h = ax[1].scatter(ctm[3], np.array(fr[0])-np.array(c), color = 'blue', marker = 'x', label = 'Cont. resid.',s = 50)
		plt.minorticks_on()
		ax[1].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(which='both', labelsize = 14, direction='in')
		ax[1].tick_params('both', length=8, width=1.5, which='major')
		ax[1].tick_params('both', length=4, width=1, which='minor')
		ax[1].set_xlabel(r'Wavelength ($\AA$)', fontsize = 12)
		ax[1].set_ylabel('{}'.format(r'Residual (mag)'), fontsize = 12)
		ax[1].set_xscale('log')
		fig.align_ylabels(ax)

		handles, labels = plt.gca().get_legend_handles_labels()
		handles.extend([f,g])

		ax[0].legend(handles = handles, loc = 'best', fontsize = 10, ncol = 2)

		plt.tight_layout()
		plt.savefig(run + '/plots/{}_phot_scatter.pdf'.format(fname))
		plt.close()

	############
	#CREATE THE COMPOSITE + COMPONENET + DATA PLOT (all_spec)
	###########

	#contrast and phot (unresolved photometry) in flux
	#pwl is the central wavelength for the photometric filters
	if len(a) == 8 and dist_fit == True:
		w, spe, pri_spec, sec_spec, pri_mag, sec_mag = make_composite([tt1, tt2], [tl1, tl2], [rad1, rad2], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
	else:	
		w, spe, pri_spec, sec_spec, pri_mag, sec_mag = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)

	itep = interp1d(w, spe)
	spe = itep(wl*1e4)

	if len(a) == 8:
		ratio1 = rad2

	pri_ratio =  (np.median(spec)/np.median(spe))#*(1/(1+ratio1**2))
	sec_ratio = (np.median(spec)/np.median(spe)) #* (ratio1)

	# plt.plot(wl*1e4, spe, linewidth = 1)
	# plt.plot(w, pri_spec, linewidth = 1)
	# plt.plot(w, sec_spec, linewidth = 1)
	# plt.xlim(5500, 9000)
	# plt.savefig('tefF_test.pdf')

	i1 = interp1d(w, pri_spec)
	pri_spec = i1(wl*1e4)
	pri_spec *= pri_ratio
	i2 = interp1d(w, sec_spec)
	sec_spec = i2(wl*1e4)
	sec_spec *= sec_ratio

	spe *= np.median(spec)/np.median(spe)

	if tell == True:
		regions = [[6860, 6880], [7600, 7660], [8210, 8240]]

	with open(run + '/params.txt', 'w') as f:
		if len(a) == 8 and dist_fit == True:
			f.write('teff: {} +/- {} + {} +/- {}\nlog(g): {} + {}\nradius: {} +/- {} + {} +/- {}\nextinction: {}\nparallax: {}\nprimary Kep mag:{}\nsecondary Kep mag:{}'.format(tt1, sigma_t1, tt2, sigma_t2, tl1, tl2, rad1, sigma_r1, rad2, sigma_r2, te, plx, pri_mag, sec_mag))
		else:
			f.write('teff: {} +/- {} + {} +/- {}\nlog(g): {} + {}\nradius: {} +/- {}\nextinction: {}\nprimary Kep mag:{}\nsecondary Kep mag:{}'.format(tt1, sigma_t1, tt2, sigma_t2, tl1, tl2, rad2, sigma_r2, te, pri_mag, sec_mag))
	#make a nice figure with the different spectra
	fig, [ax, ax1] = plt.subplots(nrows = 2, gridspec_kw = dict(hspace = 0, height_ratios = [3, 1]), sharex = True, figsize = (7 , 6))

	if any(v != 0 for v in real_val):
		ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data: {:.0f}+{:.0f}K; {:.1f}+{:.1f}dex'.format(real_val[0], real_val[1], real_val[2], real_val[3]), color = 'k', zorder = 4)
	else:
		ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data', color = 'k', zorder = 4)
	ax.plot(wl*1e4, spe, linewidth = 1, label = 'Composite spectrum', color = 'seagreen', zorder=3.5)
	ax.plot(wl*1e4, pri_spec, linewidth = 1, label = 'Primary: {:.0f}K + {:.1f}dex'.format(tt1, tl1), color = 'darkblue', zorder = 3)
	ax.plot(wl*1e4, sec_spec, linewidth = 1, label = 'Secondary: {:.0f}K + {:.1f}dex'.format(tt2, tl2), color = 'darkorange', zorder = 3)
	if tell == True:
		[ax.axvspan(*r, alpha=0.3, color='0.4', zorder = 5) for r in regions]

	for n in range(len(random_sample)):
		# if n == 0:
		if len(a) == 8 and dist_fit == True:
			t1, t2, l1, l2, e, r1, r2, pl = random_sample[n]
			dist = 1/pl

			ww, sspe, ppri_spec, ssec_spec, ppri_mag, ssec_mag = make_composite([t1, t2], [l1, l2], [r1, r2], pl, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		elif len(a) == 8 and dist_fit == False:
			t1, t2, l1, l2, e, r1, r2, pl = random_sample[n]
			ratio1 = r2
			ww, sspe, ppri_spec, ssec_spec, ppri_mag, ssec_mag = make_composite([t1, t2], [l1, l2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)

		else:
			t1, t2, l1, l2, e, ratio1 = random_sample[n]
			ww, sspe, ppri_spec, ssec_spec, ppri_mag, ssec_mag = make_composite([t1, t2], [l1, l2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)

		# sspe = extinct(ww, sspe, e)
		ite = interp1d(ww, sspe)
		sspe = ite(wl*1e4)

		it1, it2 = interp1d(ww, ppri_spec), interp1d(ww, ssec_spec)
		ppri_spec, ssec_spec = it1(wl*1e4), it2(wl*1e4)

		if len(a) == 8:
			ratio = rad2

		ppri_spec *= pri_ratio; ssec_spec *= sec_ratio

		sspe *= np.median(spec)/np.median(sspe)

		ax.plot(wl*1e4, sspe, linewidth = 0.75, color = 'limegreen', alpha = 0.5, zorder = 2.5, rasterized = True)
		ax.plot(wl*1e4, ppri_spec, linewidth = 0.75, color = 'skyblue', alpha = 0.5, zorder = 2, rasterized = True)
		ax.plot(wl*1e4, ssec_spec, linewidth = 0.75, color = 'gold', alpha = 0.5, zorder = 2, rasterized = True)

		ax1.plot(wl*1e4, spec - sspe, linewidth = 0.5, color = '0.7', alpha = 0.5, zorder = 0, rasterized = True)

	plt.minorticks_on()
	ax.set_rasterization_zorder(0)

	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = 14, direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	# ax.set_xlim(6900, 7500)

	ax1.plot(wl*1e4, spec - spe, linewidth = 1, color = 'k', label = 'Data - composite', zorder = 2)
	if tell == True:
		[ax1.axvspan(*r, alpha=0.3, color='0.4', zorder = 5) for r in regions]
	ax1.axhline(0, label = 'No resid.', linestyle = '--', color ='k', linewidth = 1, zorder = 1)
	ax1.legend(loc = 'best', fontsize = 10, ncol = 2)
	ax1.tick_params(which='both', labelsize = 14, direction='in')

	ax1.set_xlabel(r'Wavelength ($\AA$)', fontsize = 14)
	ax.set_ylabel('{}'.format(r'Normalized Flux'), fontsize = 14)
	ax1.set_ylabel('Resid.', fontsize = 14)
	ax.legend(loc = 'best', fontsize = 12)
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_all_spec.pdf'.format(fname))

	############
	#CREATE THE COMPOSITE + DATA PLOT 
	###########

	if not real_val[0] == 0:
		if len(real_val) == 8 and dist_fit == True:
			rt1, rt2, rl1, rl2, rr1, rr2, rpl = real_val
			real_wl, rspec, rpspec, rsspec, pri_mag, sec_mag = make_composite([rt1, rt2], [rl1, rl2], [rr1, rr2], rpl, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		elif len(real_val) == 8 and dist_fit == False:
			rt1, rt2, rl1, rl2, rr1, rr2, rpl = real_val
			ratio = rr2
			real_wl, rspec, rpspec, rsspec, pri_mag, sec_mag = make_composite([rt1, rt2], [rl1, rl2], [ratio], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		else:
			rt1, rt2, rl1, rl2, ratio = real_val
			real_wl, rspec, rpspec, rsspec, pri_mag, sec_mag = make_composite([rt1, rt2], [rl1, rl2], [ratio], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		# rspec = extinct(real_wl, rspec, re)
		rspec *= np.median(spec)/np.median(rspec[np.where((real_wl < max(wl * 1e4)) & (real_wl > min(wl * 1e4)))])

	fig, ax = plt.subplots()
	ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data spectrum', color = 'navy', zorder = 0)
	ax.plot(wl*1e4, spe, linewidth = 1, label = 'Model: {:.0f}K + {:.0f}K; {:.1f}dex + {:.1f}dex'.format(tt1, tt2, tl1, tl2), color = 'xkcd:sky blue', zorder=1)
	if not real_val[0] == 0:
		ax.plot(real_wl, rspec, linewidth = 1, color = 'xkcd:grass green', label = 'B15 values: {:.0f}K + {:.0f}K; {:.1f}dex + {:.1f}dex'.format(rt1, rt2, rl1, rl2))
	# ax.set_xlim(max(min(w), min(wl*1e4)), min(max(w), max(wl)*1e4))
	ax.set_xlim(8500, 8700)
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	plt.xlabel(r'Wavelength (\AA)', fontsize = 13)
	plt.ylabel('Normalized flux', fontsize = 13)
	plt.legend(loc = 'best', fontsize = 13)
	plt.tight_layout()
	plt.savefig(run + '/plots/bestfit_spec_post_mcmc.pdf')
	plt.close()


	############
	#COMPUTE AND CREATE THE KEPLER CONTRAST AND CORRECTION FACTOR PLOTS
	###########

	kep_sample = sample[np.random.choice(len(sample), size = int(1200), replace = False), :]
	kep_contrast = []; kep_rad = []
	for n in range(len(kep_sample)):
		if len(a) == 8 and dist_fit == True:
			tt1, tt2, tl1, tl2, ex, rad1, rad2, plx = kep_sample[n,:]
			ratio1 = rad2
			w, spe, pri_spec, sec_spec, pri_mag, sec_mag = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		elif len(a) == 8 and dist_fit == False:
			tt1, tt2, tl1, tl2, ex, rad1, rad2, plx = kep_sample[n,:]
			ratio1 = rad2
			w, spe, pri_spec, sec_spec, pri_mag, sec_mag = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		else:	
			tt1, tt2, tl1, tl2, te, ratio1 = kep_sample[n,:]
			w, spe, pri_spec, sec_spec, pri_mag, sec_mag = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		
		kep_rad.append(ratio1)
		kep_contrast.append(sec_mag-pri_mag)

	kep_contrast = np.array(kep_contrast); kep_rad = np.array(kep_rad)

	nbins = 110
	contrast_bins = np.linspace(min(kep_contrast), max(kep_contrast), nbins)
	contrast_count = np.zeros(len(contrast_bins))


	for t in kep_contrast:
		for b in range(len(contrast_bins) - 1):
			if contrast_bins[b] <= t < contrast_bins[b + 1]:
				contrast_count[b] += 1

	kep_mean = float(np.array(contrast_bins)[np.where(np.array(contrast_count) == max(np.array(contrast_count)))[0]])
	kep_84 = np.abs(np.percentile(np.array(kep_contrast), 84))
	kep_16 = np.abs(np.percentile(np.array(kep_contrast),16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(kep_contrast, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(kep_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(kep_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(kep_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$\Delta$Kep = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(kep_mean, kep_84-kep_mean, kep_mean - kep_16))
	ax.set_xlabel(r'$\Delta$Kep (mag)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_delta_kep.pdf'.format(fname))
	plt.close()

	pri_corr = np.sqrt(1 + 10**(-0.4 * kep_contrast)) #from ciardi+2015; Furlan+2017
	sec_corr = kep_rad * np.sqrt(1 + 10**(0.4 * kep_contrast))

	pc_bins = np.linspace(min(pri_corr), max(pri_corr), nbins)
	sc_bins = np.linspace(min(sec_corr), max(sec_corr), nbins)
	pc_count = np.zeros(len(pc_bins))
	sc_count = np.zeros(len(sc_bins))
	for t in pri_corr:
		for b in range(len(pc_bins) - 1):
			if pc_bins[b] <= t < pc_bins[b + 1]:
				pc_count[b] += 1

	for t in sec_corr:
		for b in range(len(sc_bins) - 1):
			if sc_bins[b] <= t < sc_bins[b + 1]:
				sc_count[b] += 1

	pri_mean = float(np.array(pc_bins)[np.where(np.array(pc_count) == max(np.array(pc_count)))])
	pri_84 = np.abs(np.percentile(np.array(pri_corr), 84))
	pri_16 = np.abs(np.percentile(np.array(pri_corr),16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(pri_corr, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(pri_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(pri_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(pri_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$f_{{p, corr}}$ = {:.4f}$^{{+ {:.4f}}}_{{- {:.4f}}}$'.format(pri_mean, pri_84-pri_mean, pri_mean - pri_16))
	ax.set_xlabel(r'Corr. factor (primary)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_pri_corr.pdf'.format(fname))
	plt.close()

	sec_mean = float(np.array(sc_bins)[np.where(np.array(sc_count) == max(np.array(sc_count)))])
	sec_84 = np.abs(np.percentile(np.array(sec_corr), 84))
	sec_16 = np.abs(np.percentile(np.array(sec_corr),16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(sec_corr, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(sec_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(sec_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(sec_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$f_{{s, corr}}$ = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(sec_mean, sec_84-sec_mean, sec_mean - sec_16))
	ax.set_xlabel(r'Corr. factor (secondary)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_sec_corr.pdf'.format(fname))
	plt.close()

	############
	#CREATE ISOCHRONE + COMPONENT PLOT
	###########

	#make this in H-K vs. K space so I can use contrasts to get a mag for each component
	matrix = np.genfromtxt('mist_2mass_old.cmd', autostrip = True)
	#get the age and convert it into Gyr from log(years)
	aage = matrix[:, 1]

	teff5, lum5 = matrix[:,4][np.where(np.array(aage) == 9.7000000000000011)], matrix[:,6][np.where(np.array(aage) == 9.7000000000000011)]

	aage = np.array([(10**a)/1e9 for a in aage])

	#get the mass, log(luminosity), effective temp, and log(g)
	ma = matrix[:,3][np.where((aage > 0.1) & (aage < 8))]
	teff = matrix[:, 4][np.where((aage > 0.1) & (aage < 8))]
	lum = matrix[:, 6][np.where((aage > 0.1) & (aage < 8))]
	hmag = matrix[:, 15][np.where((aage > 0.1) & (aage < 8))]
	kmag = matrix[:, 16][np.where((aage > 0.1) & (aage < 8))]
	hk_color = np.array(hmag) - np.array(kmag)
	aage = aage[np.where((aage > 0.1) & (aage < 8))]

	lum, lum5 = [10**l for l in lum], [10**l for l in lum5]; teff, teff5 = [10**t for t in teff], [10**t for t in teff5]

	#remove redundant ages from the age vector
	a1 = [aage[0]]
	for n in range(len(aage)):
		if aage[n] != a1[-1]:
			a1.append(aage[n])

	#plot the age = 0 temperature vs. luminosity 
	fig, ax = plt.subplots()

	#now for all the other ages fill an array with the single valued age, get the temperature and convert it from log
	#then plot it versus the correct luminosity
	#tagging each one with the age and color coding it 
	for n in np.arange(0, len(a1), 4):
		a2 = np.full(len(np.where(aage == a1[n])[0]), a1[n])

		if n == 0:
			ax.plot(np.array(teff)[np.where(np.array(aage) == a1[n])], np.log10(np.array(lum)[np.where(np.array(aage) == a1[n])]), color = cm.plasma(a1[n]/10), zorder = 0, label = 'MS')#, label = '{}'.format(int(np.around(a1[n]))))
		else:
			ax.plot(np.array(teff)[np.where(np.array(aage) == a1[n])], np.log10(np.array(lum)[np.where(np.array(aage) == a1[n])]), color = cm.plasma(a1[n]/10), zorder = 0)#, label = '{}'.format(int(np.around(a1[n]))))

	#calculate the intrinsic H and K mags from the SED to get the color
	if len(a) == 8 and dist_fit == True:
		tt1, tt2, tl1, tl2, ex, rad1, rad2, plx = a
		ratio1 = rad2
		# pri_h, sec_h, pri_k, sec_k = make_composite([tt1, tt2], [tl1, tl2], [rad1, rad2], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False, iso = True)
	elif len(a) == 8 and dist_fit == False:
		tt1, tt2, tl1, tl2, ex, rad1, rad2, plx = a
		ratio1 = rad2
		# pri_h, sec_h, pri_k, sec_k = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False, iso = True)
	else:	
		tt1, tt2, tl1, tl2, te, ratio1 = a
		# pri_h, sec_h, pri_k, sec_k = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False, iso = True)
	# pri_hk = pri_h - pri_k; sec_hk = sec_h - sec_k

	# kcontrast = fr[0][-1]; composite_kmag = fr[3][-1]

	# pri_kmag = -2.5*np.log10((10**(-0.4*composite_kmag))/(1 + 10**(-0.4*kcontrast))) - 5*np.log10((1/plx)/10)
	# sec_kmag = -2.5*np.log10(10**(-0.4*composite_kmag) - (10**(-0.4*composite_kmag))/(1 + 10**(-0.4*kcontrast))) - 5*np.log10((1/plx)/10)

	# ax.scatter(pri_hk, pri_kmag, marker= 'x', color = 'darkblue', s = 50, label = 'primary')
	# ax.scatter(sec_hk, sec_kmag, marker ='x', color = 'darkorange', s = 50, label = 'secondary')
	# ax.set_xlim(-0.5, 1)
	# ax.set_ylim(9, 4)

	l_intep = interp1d(teff5[:200], lum5[:200]); pri_lum = l_intep(tt1)

	# print(pri_lum, tt1, teff5[find_nearest(teff5, tt1)], lum5[find_nearest(teff5[:100], tt1)],find_nearest(teff5, tt1), lum5[70:80], teff5[70:80], teff5[300:400])

	sigma_sb = 5.670374e-5 #erg/s/cm^2/K^4
	lsun = 3.839e33 #erg/s 
	rsun = 6.955e10
	pri_rad = np.sqrt(pri_lum*lsun/(4 * np.pi * sigma_sb * tt1**4)) #cm

	sec_rad = ratio1 * pri_rad
	sec_lum = (4 * np.pi * sec_rad**2 * sigma_sb * tt2**4)/lsun #solar luminosity

	ax.scatter(tt1, np.log10(pri_lum), marker = 'x', color = 'darkgray', s = 60, label = 'Primary')
	ax.scatter(tt2, np.log10(sec_lum), marker = 'x', color = 'darkorange', s = 50, label = 'Secondary')
	ax.set_xlabel(r'T$_{eff}$ (K)', fontsize = 16)
	ax.set_ylabel(r'$\log_{10}$(L (L$_{\odot}$))', fontsize = 16)
	# ax.set_yscale('log')
	ax.set_xlim(5000, 3000)
	ax.set_ylim(np.log10(1e-3), np.log10(1))
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = 16, direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.legend(loc = 'best', fontsize = 13)
	# plt.gca().invert_yaxis()
	fig.tight_layout()
	#save the figure in the run directory
	plt.savefig(run + '/plots/{}_isochrone.pdf'.format(fname))
	plt.close()

	return

def main(argv):

	argument_list = argv[1:]
	short_options = 'f:o:e:' #filename, optimize y/n, emcee y/n
	long_options = 'file =, optimize =, emcee ='
	arguments, values = getopt.getopt(argument_list, short_options, long_options)

	parkey, parfile = arguments[0]

	pardict = {}
	with open(parfile) as fi:
		for line in fi:
			if not line.startswith('#') and not line.strip() == '':
				# print(line.split(' ')[0:2])
				(key, val) = line.split(' ')[0:2]
				val = val.split('\t')[0]
				# print(key, val)
				pardict[str(key)] = val

	# print(pardict)
	models = pardict['models']; res = int(pardict['res'])
	try:
		mask = pardict['mask']
	except:
		mask = 'f'

	try:
		rp = pardict['rad_prior']
	except:
		rp = 'f'

	if 't' in rp.lower():
		rp = True
	else:
		rp = False

	# vs = Table(fits.getdata('Data/vegaspec.fits'))
	vs2 = synphot.spectrum.SourceSpectrum.from_file('Data/vegaspec.fits')
	matrix = np.genfromtxt('mist_2mass_old.cmd', autostrip = True)


	data_wl, dsp, de = np.genfromtxt(pardict['filename'], unpack = True)

	# data_wl /= 1e4

	if 't' in mask.lower():
		dsp = np.concatenate((dsp[np.where(data_wl <= 0.6860)], dsp[np.where((data_wl >= 0.6880) & (data_wl <= 0.7600))], dsp[np.where((data_wl >= 0.7660) & (data_wl <= 0.8210))], dsp[np.where(data_wl > 0.8240)]))
		de = np.concatenate((de[np.where(data_wl <= 0.6860)], de[np.where((data_wl >= 0.6880) & (data_wl <= 0.7600))], de[np.where((data_wl >= 0.7660) & (data_wl <= 0.8210))], de[np.where(data_wl > 0.8240)]))
		data_wl = np.concatenate((data_wl[np.where(data_wl <= 0.6860)], data_wl[np.where((data_wl >= 0.6880) & (data_wl <= 0.7600))], data_wl[np.where((data_wl >= 0.7660) & (data_wl <= 0.8210))], data_wl[np.where(data_wl > 0.8240)]))


	data_wl, dsp, de = data_wl[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))], \
		dsp[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))], \
			de[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))]

	de /= np.median(dsp)
	dsp /= np.median(dsp)

	data = [data_wl, dsp]

	# dwl = data_wl#np.arange(min(data_wl)-2e-4, max(data_wl)+2e-4, 5e-5)

	t1 = time.time()
	specs = spec_interpolator([float(pardict['spmin'])*1e4, float(pardict['spmax'])*1e4], [int(pardict['tmin']), int(pardict['tmax'])], [float(pardict['lgmin']),float(pardict['lgmax'])], \
		[int(pardict['specmin']), int(pardict['specmax'])], resolution = res, models = models)
	print('time to read in specs:', time.time() - t1)

	plx, plx_err, dist_true = float(pardict['plx']), float(pardict['plx_err']), pardict['dist_fit']
	if 't' in dist_true.lower():
		dist_true = True
	else:
		dist_true = False


	mags = list([float(p) for p in pardict['cmag'].strip('[]').split(',')])
	me = list([float(p) for p in pardict['cerr'].strip('[]').split(',')])
	filts = np.array([p.strip('\\') for p in pardict['cfilt'].strip('[] ').split('\'')])
	filts = np.array([p for p in filts if len(p) >= 1 and not p == ','])

	oldphot = list([float(p) for p in pardict['pmag'].strip('[]').split(',')])
	phot_err = list([float(p) for p in pardict['perr'].strip('[]').split(',')])
	phot_filt = np.array([p.strip('\\') for p in pardict['pfilt'].strip('[] ').split('\'')])
	phot_filt = np.array([p for p in phot_filt if len(p) >= 1 and not p == ','])
	phot = np.zeros(len(oldphot))


	#Now that I'm calculating photometry manually I don't think I need to do this, since the ZPs should be in ab mag
	ab_to_vega = {'u':0.91,'g':-0.08,'r':0.16,'i':0.37,'z':0.54} #m_AB-m_vega from Blanton et al 2007
	kic_to_sdss_slope = {'g':0.0921, 'r':0.0548, 'i':0.0696, 'z':0.1587}
	kic_to_sdss_int = {'g':-0.0985,'r':-0.0383,'i':-0.0583,'z':-0.0597}
	kic_to_sdss_color = {'g':'g-r', 'r':'r-i','i':'r-i','z':'i-z'}

	# #if we have mAB, m_vega = m_ab - ab_to_vega
	#now everything will be in vegamag
	if not 'synth' in parfile:
		for n, p in enumerate(phot_filt):
			if 'sdss' in p.lower():
				color = oldphot[np.where('sdss,' + kic_to_sdss_color[p.split(',')[1]].split('-')[0] == phot_filt)[0][0]] - oldphot[np.where('sdss,' + kic_to_sdss_color[p.split(',')[1]].split('-')[1] == phot_filt)[0][0]]
				phot[n] = kic_to_sdss_int[p.split(',')[1]] + kic_to_sdss_slope[p.split(',')[1]]*color + oldphot[n]
				phot[n] = phot[n] #- ab_to_vega[p.split(',')[1]]
			else:
				phot[n] = oldphot[n]
	else:
		phot = np.array(oldphot) #- np.array([-0.08,0.16,0.37,0.54,0,0,0]) #-(5 * np.log10((1/plx)/10) - 5)

	#1" parallax = 1/D (in pc)
	av = float(pardict['av']); av_err = float(pardict['av_err'])

	nwalk1, cutoff, nburn, nstep = int(pardict['nwalk']), 1, 1, int(pardict['nstep'])

	nspec, ndust = int(pardict['nspec']), int(pardict['ndust'])
	
	# phot_flux = [10**(-0.4 * c) for c in phot]
	# phot_err_flux = [np.median((10**(-0.4 *(phot[n] + phot_err[n])) - (10**(-0.4 * phot[n])), 10 ** (-0.4 * (phot[n] - phot_err[n])) - (10 ** (-0.4 * phot[n])))) for n in range(len(phot_err))] 
	# print(phot_err_flux)
	#flux ratio and photometry error
	# err2 = [np.median((10**(-0.4 *(mags[n] + me[n])) - (10**(-0.4 * mags[n])), 10 ** (-0.4 * (mags[n] - me[n])) - (10 ** (-0.4 * mags[n])))) for n in range(len(me))]
	#give everything in fluxes
	fr = [mags, me, filts, phot, phot_err, phot_filt]
	# fr = [[10**(-0.4 * m) for m in mags], filts, err2]

	tmi, tma = np.inf, 0

	wls, tras, n_res_el, cwl = [], [], [], []
	for f in fr[2]:
		w, t, re, c = get_transmission(f, res)
		wls.append(list(w)); tras.append(list(t)); n_res_el.append(re); cwl.append(c)
		if min(w) < tmi:
			tmi = min(w)
		if max(w) > tma:
			tma = max(w)

	phot_wls, phot_tras, phot_resel, phot_cwl = [], [], [], []
	for p in fr[5]:
		w, t, re, c = get_transmission(p, res)
		phot_wls.append(list(w)); phot_tras.append(list(t)); phot_resel.append(re); phot_cwl.append(c)
		if min(w) < tmi:
			tmi = min(w)
		if max(w) > tma:
			tma = max(w)

	# for n in range(len(phot_tras)):
	# 	ran, tm = np.array(phot_wls[n]), np.array(phot_tras[n])
	# 	phot_tras[n] = synphot.SpectralElement(synphot.models.Empirical1D, points = ran*u.Unit('AA'), lookup_table = tm * u.dimensionless_unscaled, keep_neg = True)

	# print(wls)
	ctm, ptm = [wls, tras, n_res_el, cwl], [phot_wls, phot_tras, phot_resel, phot_cwl]

	# for n, t2 in enumerate([3025,3225,3425,3625,3800]):#[3225, 3425, 3625, 3825, 4025, 4175]):#
	# 	t1 =  3850  # #4500  . 4200
	# 	r1 =  0.4994  # 0.6162 
	# 	r2 = [0.1546, 0.2149, 0.3048, 0.3910, 0.4745][n] #[0.2149, 0.3048, 0.3910, 0.4870, 0.5791, 0.6039][n] #
	# 	lg1 = 4.76 # 4.67 
	# 	lg2 = [5.16, 5.06, 4.96, 4.87, 4.79][n] #[5.06, 4.96, 4.87, 4.77, 4.70, 4.68][n] #

	# 	comp_wl, comp_spec, contrast, phot_cwl, photometry = make_composite([t1, t2], [lg1, lg2], [r1,r2], plx, ['880','sdss,r','J', 'K'], ['sdss,g','sdsss,r','sdss,i','sdss,z','J', 'H', 'K'], [min(data_wl),max(data_wl)], specs, ctm, ptm, tmi, tma, vs2)  
	# 	print(contrast, photometry, t2)
	# 	# print(min(comp_wl), min(data_wl), max(comp_wl), max(data_wl))
	# 	# # plt.plot(comp_wl, comp_spec/np.median(comp_spec[np.where((comp_wl < max(data_wl*1e4)) & (comp_wl > min(data_wl*1e4)))]), linewidth = 1)
	# 	# # plt.plot(data_wl*1e4, dsp, linewidth = 1)
	# 	# # plt.xlim(5500,9000)
	# 	# # plt.savefig('test_data.pdf')
	# 	itep = interp1d(comp_wl, comp_spec); comp_spec = itep(data_wl*1e4)
	# 	err = np.random.normal(0, 0.01*comp_spec)

	# 	np.savetxt('Data/synth_spec_{}_{}.txt'.format(t1, t2), np.column_stack((data_wl, comp_spec + err, err)))

	dirname, fname = pardict['dirname'], pardict['fname']

	try:
		os.mkdir(dirname)
	except:
		pass;
	try:
		os.mkdir(dirname + '/plots')
	except:
		pass;

	optkey, optval = arguments[1]
	if optval == 'True':
		# if dist_true == True:
		optimize_fit(dirname, data, de, specs, nwalk1, fr, [plx, plx_err], [av, av_err], res, ctm, ptm, tmi, tma, vs2, matrix, cutoff = cutoff, nstep = nstep, nburn = nburn, con = False, models = models, dist_fit = dist_true, rad_prior = rp)
		# else:
		# 	optimize_fit(dirname, data, de, specs, nwalk1, fr, False, [av, av_err], res, ctm, ptm, tmi, tma, vs, cutoff = cutoff, nstep = nstep, nburn = nburn, con = False, models = models)

		plot_fit(dirname, data, specs, fr, ctm, ptm, tmi, tma, vs2, models = models, dist_fit = dist_true)

		print('optimization complete')

	emceekey, emceeval = arguments[2]
	if emceeval == 'True':
		chisqs, pars = np.genfromtxt(dirname + '/optimize_cs.txt'), np.genfromtxt(dirname + '/optimize_res.txt')

		cs_idx = sorted(chisqs)[:int(len(chisqs)*1/3)]

		idx = [int(np.where(chisqs == c)[0]) for c in cs_idx]

		p0 = pars[idx]

		# p0 = np.hstack((p0[:,0:4], p0[:,5:]))
		#dimensions: t1, t2, lg1, lg2, av, normalization (of composite spectrum w data spectrum)

		# if dist_true == True:
		nwalkers, nsteps, ndim, nburn = len(p0), int(pardict['nsteps']), 8, int(pardict['nburn'])

		real_val = list([float(p) for p in pardict['real_values'].strip('[]\n').split(',')])
		title_format = '.0f .0f .2f .2f .2f .2f .2f'

		# if dist_true == True:
		# a = run_emcee(dirname, fname, nwalkers, nsteps, ndim, nburn, p0, fr, nspec, ndust, data, de, res, [min(data_wl), max(data_wl)], specs, real_val, ctm, ptm, tmi, tma, vs2, title_format, matrix,\
				# nthin=100, w = 'aa', pys = False, du = False, prior = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,plx,plx_err], models = models, av = 0, dist_fit = dist_true, rad_prior = rp)
		
		# else:
		# 	a = run_emcee(dirname, fname, nwalkers, nsteps, ndim, nburn, p0, fr, nspec, ndust, data, de, res, [min(data_wl), max(data_wl)], specs, real_val, ctm, ptm, tmi, tma, vs,\
		# 		nthin=50, w = 'aa', pys = False, du = False, prior = 0, models = models, av = 0, dist_fit = dist_true)
		
		# else:
		# 	nwalkers, nsteps, ndim, nburn = len(p0), int(pardict['nsteps']), 6, int(pardict['nburn'])

		# 	real_val = list([float(p) for p in pardict['real_values'].strip('[]\n').split(',')])

		# 	a = run_emcee(dirname, fname, nwalkers, nsteps, ndim, nburn, p0, fr, nspec, ndust, data, de, res, [min(data_wl), max(data_wl)], specs, real_val, ctm, ptm, tmi, tma, vs,\
		# 		nthin=50, w = 'aa', pys = False, du = False, prior = 0, models = models, av = 0, dist_fit = dist_true)

		a = np.genfromtxt(dirname + '/samples.txt')
		dw, ds, de = np.genfromtxt(pardict['filename']).T
		dw, ds, de = dw[np.where((dw > float(pardict['spmin'])) & (dw < float(pardict['spmax'])))], \
		ds[np.where((dw > float(pardict['spmin'])) & (dw < float(pardict['spmax'])))], \
		de[np.where((dw > float(pardict['spmin'])) & (dw < float(pardict['spmax'])))]
		data = [dw, ds, de]

		if 't' in mask.lower():
			tell_kw = True
		else:
			tell_kw = False

		plot_results(fname, a, dirname, data, specs, fr, ctm, ptm, tmi, tma, vs2, real_val, plx, models = models, dist_fit = dist_true, res = 1200, tell = tell_kw)
	return 

if __name__ == "__main__":
	# try:
	main(sys.argv)
	# except:
	# 	print('Exception! Must include a parameter file as a command line argument')
