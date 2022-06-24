# binary_fitting_2022b
Repo for the code and associated required data to reproduce the analysis for Sullivan et al. 2022

Requires a modified version of corner.py to produce corner plots that look correct. 

Example usage (after installing dependencies) is "python mft5.py -f paramfile -o T/F -e T/F". The -o keyword determines whether the optimization stage will run, while the -e keyword determines whether emcee will run. The keywords are position-dependent. The enclosed parameter file is an example of the needed inputs. Please contact me at kendallsullivan@utexas.edu if you have any questions or issues with the code! 

The most recent version of the code, which may not be compatible with the output/assumptions made in the paper, is available at github.com/kendallsullivan/mcmc_spec.