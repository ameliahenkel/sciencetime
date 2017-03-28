"""
@title: ifu_emission_line_measurment

@description: Best Emission Line Measurement Tool Ever

@author: Amelia Hinkel
         Greg Zeimann
"""

# Load modules necessary for code
# For Example -
from astropy.io import fits # To load fits files
import numpy as np # for math calculation
import sys 
import matplotlib.pyplot as plt # for plotting
import argparse as ap # for parsing terminal command line arguments
from astropy.modeling.models import Gaussian1D, Polynomial1D # for modelling
from astropy.modeling import fitting # for fitting
fitter = fitting.SLSQPLSQFitter() # load the fitter globally since we use it often

# As an example, use this file ...
cube = fits.open('/Users/Poop/Downloads/CuFe20161214T091745.1_056_sci_R.fits')

# This code is going to be run from the command line in a bash terminal,
# so we will need an argument parser to pass arguments from the command line
# to our code.

# Below is an example function to parse args from the command line
def parse_args(argv=None):
    """Parse the command line arguments

    Parameters
    ----------
    argv : list of string
        arguments to parse; if ``None``, ``sys.argv`` is used

    Returns
    -------
    Namespace
        parsed arguments


    Examples
    --------
    For help -
    
    bash$ python ifu_emission_line_measurement.py -h 
    
    An example use -
    
    bash$ python ifu_emission_line_measurement.py --filename CuFe_FILLIN.fits
    """
    
    description = " Emission Line Measurement Tool "
                     
    parser = ap.ArgumentParser(description=description,
                            formatter_class=ap.RawTextHelpFormatter)
                            
    parser.add_argument("--filename", nargs='?', type=str,
                        help='''File name of the CuFe* file to analyze''',
                        default=None)                        

    parser.add_argument("--redshift", nargs='?', type=float,
                        help='''Redshift of the galaxy''',
                        default=0.0) 
    
    args = parser.parse_args(args=argv)

    if args.filename is None:
        msg = 'Missing the filename as an input argmument'
        parser.error(msg)              

    return args

parse_args()
# We will need to load the data and prep any information needed for measuring
# our emission lines.  For example, calculating the wavelength
def prep_data(args):
    '''
    Parameters
    ----------
    arg : Namespace Object
        acquired through parse_args()
    
    Returns
    -------
    wave : numpy array
        calculated from the header of the fits file passed by args.filename
    shape : tuple
        3 value tuple with the shape of the 3d fits file 
    data : 3-d numpy array
        (naxis1, naxis2, naxis3) -> x, y, wave for the three dimensions
        
    '''
    fits_file = fits.open(args.filename)
    data = fits_file[0].data
    shape = data.shape
    header = fits_file[0].header
    print(header)
    print(data)
    # Now calculate the wavelength from the header info and the length of the 
    # data array.
    
    #wave = BLAH BLAH BLAH
    # # wave = CDELT3 * (np.arange(NAXIS3)+1-CRPIX3) + CRVAL3
    # Now return wave and shape for other use
    wave = fits_file[0].header['NAXIS3']
    
    
    return wave, shape, data
    
# We will want a code that measures the emission line flux (area under the 
# curve) for H-beta, [OIII], H-alpha, and [NII]    


def measure_emission_lines(args, wave, spectrum):
    '''
    Parameters
    ----------
    arg : Namespace Object
        acquired through parse_args()
    wave : numpy array
        wavelength for the spectrum
    spectrum : numpy array
        spectrum for the given pixel y,x when called from main()
    
    Returns
    -------
    flux : list
        the list includes Hb flux, OIII, flux, Ha flux, and NII flux
    '''
    # This is where we will measure the lines and this is to be filled in
    # with time, but an example start would be 
    #                   HB   OIII  OIII   NII   HA    NII
    rest_frame_wave = [4861, 4959, 5007, 6548, 6563, 6584]
    obs_wave = np.array(rest_frame_wave) * (1. + args.z)
    
    # Fit region 1 with HB and OIII
    buff = 20 # fit +/20 A above and below the lowest emission wavelengths
    wplow = np.searchsorted(wave, obs_wave[0]-20, side='left')
    wphigh = np.searchsorted(wave, obs_wave[2]+20, side='right')
    
    # Think of a way to guess at the amplitude of each gaussian and the sigma
    # the wavelength guess comes from the redshift input on the command line
    # and the rest-frame wavelength
    
    
    init_model = (Gaussian1D(amp_guess, obs_wave[0], sig_guess) + 
                  Gaussian1D(amp_guess, obs_wave[1], sig_guess) + 
                  Gaussian1D(amp_guess, obs_wave[2], sig_guess) + 
                  Polynomial1D(degree=1))
    init_model_2 = 
    #Do this for last 3 lines in rest_wave_frame
    
    # We may potentially want to fix some parameters or tie some together
    # We would do that here, before we fit (no examples given yet)
    # # What does this mean?
    
    model_fit = fitter(init_model, spectrum)
    HB_flux = model_fit.amplitude_0 / np.sqrt(2.*np.pi*model_fit.stddev_0**2)
    O3_flux = (model_fit.amplitude_1 / np.sqrt(2.*np.pi*model_fit.stddev_1**2) 
                 + model_fit.amplitude_2 / np.sqrt(2.*np.pi*model_fit.stddev_2**2))

    # Now write the same for HA_flux and N2_flux
    # # What is the significance of the amplitude_0-2 ?
    HA_flux = model_fit    

    # Return our solved fluxes

    return [HB_flux, O3_flux, HA_flux, N2_flux]    
    

def main():
    
    
    #Main Loop
    
    
    args = parse_args()
    
    # Run prep_data with the only input being args (an object that collected
    # information from the bash command line)
    wave, shape, data = prep_data(args)
    
    # We want to measure the emission line flux in each pixel so we want
    # to loop through shape[1] and shape[2] (y and x) and call a function
    # that measures emission lines

    HB = np.zeros((shape[1],shape[2]))
    O3 = np.zeros((shape[1],shape[2]))
    HA = np.zeros((shape[1],shape[2]))
    N2 = np.zeros((shape[1],shape[2]))
    for y in np.arange(shape[1]):
        for x in np.arange(shape[2]):
            HB[y,x], O3[y,x], HA[y,x], N2[y,x] = measure_emission_lines(args,
                                                             wave, data[:,y,x])


# This runs main() which runs everything else .... it is very c++ like
if __name__ == '__main__':
    main()  
    
