import copy
import datetime as dt
import emcee
import ephem
import numpy as np
import math
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt


from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Longitude

import matplotlib.cm as cm
import matplotlib.mlab as mlab
from mpl_toolkits.basemap import Basemap

# Define a Sight class. This is used to represent an actual sighting taken
# from a marine sextant, along with the date, time, index error, temperature,
# etc., which are used to adjust the sighting to get the true position in the
# celestial sphere.
class Sight:
    def __init__(self, body, date, time, Hs, WE, IE, height, temperature, pressure, DRv, DRbearing, latA, lonA):
        self.body = body

        # process date and time into a single datetime object
        year, month, day = date.split('/')
        hour, minute, second = time.split(':')
        self.datetime = dt.datetime(int(year), int(month), int(day), int(hour), int(minute), int(round(float(second))))
        self.Hs = Angle(Hs)

        self.WE = float(WE)
        self.IE = Angle(IE)
        self.height = float(height) # height in meters
        self.temperature = float(temperature)  # temperature in degrees C
        self.pressure = float(pressure) # pressure in mbar
        self.DRv = float(DRv) # velocity in knots
        self.DRbearing = Angle(DRbearing, unit=u.deg) # bearing in degrees
        self.latA = Angle(latA, unit=u.deg) # estimated lattitude in degrees
        self.lonA = Angle(lonA, unit=u.deg) # estimates longitude in degrees

        # compute GHA and DEC
        self.GHA = None
        self.DEC = None
        self.compute_GHA_DEC()
        
        # this method will update the corrections to the sextant sighting.
        # it is useful to segment the function this way so that when generating theoretical sights and
        # changing the lat/lon to be artificial values, a single method will update the rest of the Sight
        # object's attributes accordingly
        self.update_sight_corrections()

        return

    def update_sight_corrections(self):
        # compute corrections for the 'apparent altitude'
        self.dip_correction()
        self.index_correction()
        self.Ha = self.Hs + self.dip_corr + self.index_corr
        # compute the 'main correction'
        self.semidiameter_correction()
        self.get_horizontal_parallax()
        self.parallax_in_altitude_correction()
        self.atmo_correction()
        self.Ho = self.Ha + self.SD_corr + self.parallax_corr + self.atmo_corr
        return

    def get_horizontal_parallax(self):
        if self.body == 'SunLL' or self.body == 'SunUL':
            self.HP = Angle('0.0024d')
            return
        if self.body == 'MoonLL' or self.body == 'MoonUL':
            # calculate the moons horizontal paralax
            s = ephem.Moon()
            date = self.datetime.strftime('%Y/%m/%d')
            time = self.datetime.strftime('%H:%M:%S')
            date_string = date + ' ' + time
            s.compute(date_string, epoch=date_string)
            deg = 180.0/np.pi*(s.radius/0.272805950305)
            self.HP = Angle(deg, unit=u.deg)
            print('Moon HP is {}'.format(self.HP))
            return

        self.HP = Angle('0.0d')
        return

    def parallax_in_altitude_correction(self):
        self.parallax_corr = self.HP*np.cos(self.Ha.rad)*(1.-(np.sin(self.latA.rad)**2.0)/297.0)
        return

    def semidiameter_correction(self):
        # implement for lower and upper limb Sun sightings
        if self.body == 'SunLL' or self.body == 'SunUL':
            s = ephem.Sun()
            obs = ephem.Observer()
            # format strings from datetime object
            date = self.datetime.strftime('%Y/%m/%d')
            time = self.datetime.strftime('%H:%M:%S')
            date_string = date + ' ' + time
            obs.date = date_string

            s.compute(date_string, epoch=date_string)
            # compute SD of sun
            sds = s.radius / ephem.pi * 180.0  # degrees of arc
            if self.body == 'SunLL':
                self.SD_corr = Angle('{:.3f}d'.format(sds))
                return
            else:
                self.SD_corr = Angle('-{:.3f}d'.format(sds))
                return
        if self.body == 'MoonLL' or self.body == 'MoonUL':
            s = ephem.Moon()
            obs = ephem.Observer()
            # format strings from datetime object
            date = self.datetime.strftime('%Y/%m/%d')
            time = self.datetime.strftime('%H:%M:%S')
            date_string = date + ' ' + time
            obs.date = date_string

            s.compute(date_string, epoch=date_string)
            # compute SD of moon
            sds = s.radius / ephem.pi * 180.0  # degrees of arc
            if self.body == 'MoonLL':
                self.SD_corr = Angle('{:.3f}d'.format(sds))
                print('Moon SD correction is {}'.format(sds))
                return
            else:
                self.SD_corr = Angle('-{:.3f}d'.format(sds))
                print('Moon SD correction is -{}'.format(sds))
                return

        self.SD_corr = Angle('0d')
        return

    def dip_correction(self):
        dip_corr = Angle(-0.0293 * np.sqrt(self.height), unit=u.deg)
        self.dip_corr = dip_corr
        return

    def index_correction(self):
        self.index_corr = -1.0 * self.IE
        return

    def atmo_correction(self):
        Pmb = self.pressure
        TdegC = self.temperature
        Ha = self.Ha.deg
        self.atmo_corr = Angle(atmo_formula(Ha, TdegC, Pmb), unit=u.deg)
        return

    def compute_GHA_DEC(self):
        # sun
        if self.body == 'SunLL' or self.body == 'SunUL':
            celestial_body = ephem.Sun()
        # insert moon here
        elif self.body == 'MoonLL' or self.body == 'MoonUL':
            celestial_body = ephem.Moon()
        # planets
        elif self.body == 'Mars':
            celestial_body = ephem.Mars()
        elif self.body == 'Venus':
            celestial_body = ephem.Venus()
        elif self.body == 'Jupiter':
            celestial_body = ephem.Jupiter()
        elif self.body == 'Saturn':
            celestial_body = ephem.Saturn()
        elif self.body == 'Uranus':
            celestial_body = ephem.Uranus()
        elif self.body == 'Mercury':
            celestial_body = ephem.Mercury()
        # anything else must be a star, and its body can be interpreted via the string argument in ephem.star('')
        else:
            celestial_body = ephem.star(self.body)

        obs = ephem.Observer()
        # disable a correction for pressure, since we do this later
        obs.pressure = 0

        # format strings from datetime object
        date = self.datetime.strftime('%Y/%m/%d')
        time = self.datetime.strftime('%H:%M:%S')
        date_string = date + ' ' + time
        obs.date = date_string

        celestial_body.compute(date_string, epoch=date_string)

        deg = ephem.degrees(obs.sidereal_time() - celestial_body.g_ra).norm
        ghas = nadeg(deg)
        deg = celestial_body.g_dec
        decs = nadeg(deg)

        # convert GHA and DEC to Angle objects
        self.GHA = Angle(ghas, unit=u.deg)
        self.DEC = Angle(decs, unit=u.deg)
        print('GHA and DEC for body {} are {:.3f} / {:.3f}'.format(self.body, self.GHA.deg, self.DEC.deg))
        return

    def est_longitude(self):
        # use a root-finding algorithm to compute the longitude based on an
        # assumed latitude. here, the estimated longitude is to start the
        # root-finding algorithm

        print('Finding Zo for true sextant angle of ' + '{:.3f}'.format(self.Ho) + ' degrees...')
        print('GHA and DEC are {} {}'.format(self.GHA.deg, self.DEC.deg))
        start_H = compute_Hc(self.latA, self.lonA, self.GHA, self.DEC).deg
        lower_H = compute_Hc(self.latA, Angle(self.lonA.deg - 2, unit=u.deg), self.GHA, self.DEC).deg

        upper_H = compute_Hc(self.latA, Angle(self.lonA.deg + 2, unit=u.deg), self.GHA, self.DEC).deg

        print(
            'Starting from value ' + '{:.3f}'.format(start_H) + ' degrees, with lower {:.3f} and upper {:.3f}.'.format(
                lower_H, upper_H))

        fz = lambda x: (compute_Hc(self.latA, Angle(x, unit=u.deg), self.GHA, self.DEC).deg - self.Ho.deg)

        # Diagnostic-- Check that the function fz contains a root between
        # points a and b.
        print('fz(a) is {:.3f}, fz(b) is {:.3f}'.format(fz(self.lonA.deg-2), fz(self.lonA.deg+2)))

        sp_out = scipy.optimize.brentq(fz, self.lonA.deg-2.0, self.lonA.deg+2.0, maxiter=100)

        self.est_lon = Angle(sp_out, unit=u.deg)
        print('est lon is now {}'.format(self.est_lon))
        return

def atmo_formula(Ha, TdegC, Pmb):
        f = 0.28 * Pmb / (TdegC + 273.0)
        Ro = -0.0167 / np.tan(np.pi / 180.0 * (Ha + 7.31 / (Ha + 4.4)))
        return Ro*f

def compute_Hc(latA, lonA, GHA, DEC):
    # compute LHA from GHA and lonA
    LHA = GHA + lonA
    Hc_rad = np.arcsin(np.sin(DEC.radian)*np.sin(latA.radian) + np.cos(latA.radian)*np.cos(DEC.radian)*np.cos(LHA.radian))
    return Angle(Hc_rad, unit=u.radian)

def compute_Hc_fast(latA, lonA, GHA, DEC):
    # compute LHA from GHA and lonA
    LHA = GHA + lonA
    Hc_rad = np.arcsin(np.sin(DEC)*np.sin(latA) + np.cos(latA)*np.cos(DEC)*np.cos(LHA))
    return Hc_rad


def db_sights_preprocess(db_sights):
    # split db_sights input
    db_sights_split = db_sights.split('\n')
    # remove last element
    del db_sights_split[-1]

    db_list = []

    for i in range(len(db_sights_split)):
        # split the current line into raw string objects
        body, date, time, Hs, WE, IE, height, temp, pressure, DRv, DRbearing, latA, lonA = db_sights_split[i].split(
            ',')  # split into line elements
        # coerce values from strings to objects
        sight_obj = Sight(body.strip(), date, time, Hs, WE, IE, height, temp, pressure, DRv, DRbearing, latA, lonA)
        db_list.append(sight_obj)
    return db_list

def get_GHADEC_arrays(sights):
    ghadec_vals = np.zeros((len(sights),2))
    for i in range(len(sights)):
        ghadec_vals[i,0] = sights[i].GHA.deg
        ghadec_vals[i,1] = sights[i].DEC.deg

    return ghadec_vals

def get_latlon_DR(sights):
    ghadec_vals = np.zeros(len(sights),2)
    for i in range(len(sights)):
        ghadec_vals[i,0] = sights[i].GHA.deg
        ghadec_vals[i,1] = sights[i].DEC.deg

    return ghadec_vals

def get_timedeltas(sights):
    # compute an array of timedeltas resolved into seconds, multiplied by a velocity
    time_array = []
    for i in range(len(sights)):
        td = sights[i].datetime - sights[0].datetime
        time_array.append(td.total_seconds())

    return np.array(time_array)

def compute_distances(td, DRv):
    darray = td * DRv * 1852.0 / 3600.0
    return darray


def compute_displacement(lat, lon, bearing, distance):
    # computes an array of distination lat/longs from an initial lat/long pair,
    # a bearing, and an array of distances travelled.

    # define EARTH_RADIUS in meters for computing angular displacement
    EARTH_RADIUS = 6371000.00  # meters

    # rename arrays to be consistent with moveable-type.co.uk convention
    # should be arrays: phi1, theta, and delta
    phi1 = lat
    theta = bearing
    lambda1 = lon
    delta = distance / EARTH_RADIUS  # should both be in meters

    # phi2 -- new lattitudes
    # lambda2 -- new longitudes
    # EXCEL equivalent:
    # lat2: =ASIN(SIN(lat1)*COS(d/R) + COS(lat1)*SIN(d/R)*COS(brng))
    # lon2: =lon1 + ATAN2(COS(d/R)-SIN(lat1)*SIN(lat2), SIN(brng)*SIN(d/R)*COS(lat1))
    phi2 = np.arcsin(np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta))
    lambda2 = lambda1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(phi1),
                                   np.cos(delta) - np.sin(phi1) * np.sin(phi2))
    # outputs are both in radians
    return phi2, lambda2

def get_totalsightcorrection(sights):
    # form a numpy array of total corrections that convert Ho back to Hs
    corr_out = np.ones(len(sights))
    for i in range(len(sights)):
        corr = sights[i].SD_corr + sights[i].parallax_corr + sights[i].atmo_corr + sights[i].dip_corr + sights[i].index_corr
        corr_out[i] = corr.rad
    return corr_out

def Hs_predict(sights, DRv, DRbearing, latA, lonA):
    # predict an array of Hs values

    # get an array of GHA and DEC vals for each sighting
    ghadec_vals = get_GHADEC_arrays(sights)

    # compute the distance array, based on time and DR velocity
    td = get_timedeltas(sights)
    darray = compute_distances(td, DRv)

    # compute the lat/lon pairs for each time
    lat_in = np.ones(len(sights))*latA
    lon_in = np.ones(len(sights))*lonA

    # compute the displacement of lat/lon along a great circle, given the initial point, bearing, and distance travelled
    phi2, lambda2 = compute_displacement(lat_in, lon_in, DRbearing.rad, darray) # args should be in radians

    # compute the Hs values for each lat/lon pair and sighting
    Hc = compute_Hc_fast(phi2, lambda2, ghadec_vals[:,0]/180.0*np.pi, ghadec_vals[:,1]/180.0*np.pi) # everything must be in radians

    # get the array of corrections, so we can convert to predicted sextant readings -- these are based on the approximate
    # lat/lon values, but as long as the guess is remotely correct, should be more than accurate.
    corr_vals = get_totalsightcorrection(sights) # in radians

    # take the true altitudes and subtracted the correction estimates to get what should be observed on a sextant reading
    est_Hs = (Hc - corr_vals)*180.0/np.pi # convert to degrees

    return est_Hs

def Ho_predict(ghadec_vals, td, latA, lonA, DRv, DRbearing):
    # predict an array of Ho values

    # compute the distance array, based on time and DR velocity

    darray = compute_distances(td, DRv)

    # compute the lat/lon pairs for each time
    lat_in = np.ones(len(sights))*latA
    lon_in = np.ones(len(sights))*lonA

    # compute the displacement of lat/lon along a great circle, given the initial point, bearing, and distance travelled
    phi2, lambda2 = compute_displacement(lat_in, lon_in, DRbearing, darray) # args should be in radians

    # compute the Hs values for each lat/lon pair and sighting
    Hc = compute_Hc_fast(phi2, lambda2, ghadec_vals[:,0]/180.0*np.pi, ghadec_vals[:,1]/180.0*np.pi) # everything must be in radians

    # take the true altitudes and subtracted the correction estimates to get what should be observed on a sextant reading
    Ho = Hc*180.0/np.pi # convert to degrees

    return Ho


def nadeg(deg):
    # changes ephem.angle (RADIANS) to the format for the Angle class.
    theminus = ""
    if deg < 0:
        theminus = "-"
    g = int(math.degrees(deg))
    m = (math.degrees(deg) - g) * 60
    gm = "%s%sd%04.1fm" % (theminus, abs(g), abs(m))
    return gm

def get_Ho_altitudes(sights):
    ho = np.zeros(len(sights))
    for i in range(len(sights)):
        ho[i] = sights[i].Ho.deg
    return ho

def fsight(x, ghadec_vals, td, Ho_obs):
    sigma_s = 1.0/60.0/180.0*np.pi
    y_theory = Ho_predict(ghadec_vals, td, x[0], x[1], x[2], x[3])
    y_data = Ho_obs

    chi = np.power(y_theory - y_data, 2.0)/(2.0*sigma_s**2.0)

    return np.sum(chi)

def prior_logP(params, parrange):
    if np.any(params < parrange[0, :]) or np.any(params > parrange[1, :]):
        return -np.inf
    else:
        bearing_prior = -(params[3]-270.0/180.0*np.pi)**2.0/(2.0*((20.0/180.0*np.pi)**2.0))
        velocity_prior = -(params[2]-5.5)**2.0/(2.0*(3.0**2.0))
        return bearing_prior + velocity_prior

def fsight_logp(x, parrange, ghadec_vals, td, Ho_obs, sigma_s):
    # get the prior probability
    logprior = prior_logP(x, parrange)
    if logprior == -np.inf:
        # if out of bounds, just return -inf now
        return -np.inf
    #sigma_s = 0.5/60.0
    y_theory = Ho_predict(ghadec_vals, td, x[0], x[1], x[2], x[3])
    y_data = Ho_obs

    ## this form of log likelihood is standard and assumes a known error
    #chi = -np.power(y_theory - y_data,2.0)/(2.0*sigma_s*sigma_s)
    #logp = np.sum(chi) + logprior

    ## this form of log likelihood is a lot more conservative w/ respect to errors
    #Rk = (y_theory - y_data)/sigma_s
    #logp = np.sum(np.log( (1.0 - np.exp(-Rk**2.0 / 2.0))/ (Rk**2.0))) + logprior

    # this is the good data/bad data model of Box and Tiao (1978), from Sivia's 'Data Analysis'
    beta = 0.1 # calibrated from initial data
    gamma = 4.8 # calibrated from initial data
    Rk = (y_theory - y_data) / sigma_s
    logp = np.sum(np.log(beta/gamma*np.exp(-np.power(Rk, 2.0)/(2.0*np.power(gamma, 2.0))) +
                         (1.0-beta)*np.exp(-np.power(Rk, 2.0)/2))) + logprior
    return logp

def HPD_SIM(data, alpha):
    # adapted from MATLAB code originally created by Hang Qian, Iowa State University
    nobs = np.size(data)
    cut = int(np.round(nobs * alpha))
    span = int(nobs - cut)

    data_sorted = np.sort(data)
    # data_sorted[:] = data_sorted[::-1]
    idx = np.argmin(data_sorted[span:nobs] - data_sorted[0:cut])
    LB = data_sorted[idx]
    UB = data_sorted[idx + span]
    return LB, UB

def generate_sights_DR(sights):
    # this function will generate a set of sights from positional data PLUS dead reckoning.
    # its purpose is to provide perfect sightings for a given initial position and heading/velocity at sea

    # compute the timedelta array & displacements in lat/lon based on the nominal bearing and velocity
    td = get_timedeltas(sights)
    # compute the nominal distances, based on the first entry's velocity/bearing
    DRv = sights[0].DRv
    DRbearing = sights[0].DRbearing.rad
    latA = sights[0].latA.radian
    lonA = sights[0].lonA.radian
    darray = compute_distances(td, DRv)
    # now compute displacements

    # compute the lat/lon pairs for each time
    lat_in = np.ones(len(sights)) * latA
    lon_in = np.ones(len(sights)) * lonA

    # compute the displacement of lat/lon along a great circle, given the initial point, bearing, and distance travelled
    phi2, lambda2 = compute_displacement(lat_in, lon_in, DRbearing, darray)  # args should be in radians

    # write the new lat/lon values into the respective sights
    for i in range(len(sights)):
        sights[i].latA = Angle(phi2[i], unit=u.radian)
        sights[i].lonA = Angle(lambda2[i], unit=u.radian)

    # get an array of GHA and DEC vals for each sighting
    ghadec_vals = get_GHADEC_arrays(sights)

    # compute the Hs values for each lat/lon pair and sighting
    Hc = compute_Hc_fast(phi2, lambda2, ghadec_vals[:, 0] / 180.0 * np.pi,
                         ghadec_vals[:, 1] / 180.0 * np.pi)  # everything must be in radians
    # get the array of corrections, so we can convert to predicted sextant readings -- these are based on the approximate
    # lat/lon values, but as long as the guess is remotely correct, should be more than accurate.
    corr_vals = get_totalsightcorrection(sights)  # in radians

    # take the true altitudes and subtracted the correction estimates to get what should be observed on a sextant reading
    est_Hs = (Hc - corr_vals) * 180.0 / np.pi  # convert to degrees

    # now update the sights to have sightings at the values we calculated
    for i in range(len(sights)):
        sights[i].Hs = Angle(est_Hs[i], unit=u.deg)
        sights[i].update_sight_corrections()

    return sights

def generate_sights_positional(sights):
    # this function will generate a set of sights from positional data ONLY. this means that for each entry in the
    # sight database, the lattitude and longitude indicated are assumed correct, and the sight that should be recorded
    # for that exact position is computed and set as the observed sight in the database.
    #
    # its purpose is to provide perfect sightings for the assumed lattitude/longitude positions, which can help with
    # code testing and also determining whether actual observed sights are close to what we'd expect.

    # compute the timedelta array & displacements in lat/lon based on the nominal bearing and velocity
    for i in range(len(sights)):
        latA = sights[i].latA.radian
        lonA = sights[i].lonA.radian
        # get an array of GHA and DEC vals for each sighting
        ghadec_vals = get_GHADEC_arrays([sights[i]])

        # compute the Hs values for each lat/lon pair and sighting
        Hc = compute_Hc_fast(latA, lonA, ghadec_vals[:, 0] / 180.0 * np.pi,
                             ghadec_vals[:, 1] / 180.0 * np.pi)  # everything must be in radians
        # take the true altitudes and subtracted the correction estimates to get what should be observed on a sextant reading
        corr_vals = get_totalsightcorrection([sights[i]])  # in radians
        est_Hs = (Hc - corr_vals) * 180.0 / np.pi  # convert to degrees
        print('Est_Hs is {}'.format(est_Hs))
        # get the array of corrections, so we can convert to predicted sextant readings -- these are based on the approximate
        # lat/lon values, but as long as the guess is remotely correct, should be more than accurate.

        # now update the sights to have sightings at the values we calculated
        sights[i].Hs = Angle(est_Hs, unit=u.deg)
        sights[i].update_sight_corrections()

    return sights

def print_sights(sights):
    print('----------Sight Database----------')
    for i in range(len(sights)):
        print('SIGHT {:d} --- Body: {} --- Hs: {} Ha: {} Ho: {}'.format(i, sights[i].body, nadeg(sights[i].Hs.rad), nadeg(sights[i].Ha.rad), nadeg(sights[i].Ho.rad)))
        print('')

    return

def add_sighting_noise(sights, noise):
    # sights -- list of Sight objects
    # noise -- noise magnitude as a scalar, in units of arcseconds
    for j in range(len(sights)):
        sights[j].Hs = Angle(sights[j].Hs.deg + np.random.randn() * noise / 3600.0, unit=u.deg)
        sights[j].update_sight_corrections()
    return

def analyze_sights(sights, parrange):
    # prepare the parameter boundaries
    # start with MCMC simulation


    # prepare MCMC walkers using emcee

    ndim, nwalkers = 4, 64

    # create walkers in random positions within the hypercube of feasible points
    p0 = [np.random.rand(ndim) * (parrange[1, :] - parrange[0, :]) + parrange[0, :] for i in range(nwalkers)]

    # coerce any values outside of the bounds to the boundary
    for k, p in enumerate(p0):
        lidx = p0[k] < parrange[0, :]
        p0[k][lidx] = parrange[0, lidx]
        uidx = p0[k] > parrange[1, :]
        p0[k][uidx] = parrange[1, uidx]

    # now get the GHA/DEC values and store them, so we don't have to keep looking them up
    ghadec_vals = get_GHADEC_arrays(sights)

    # get the timedeltas and store them, so we don't have to keep looking them up
    td = get_timedeltas(sights)

    # get the observed altitudes
    Ho_obs = get_Ho_altitudes(sights)

    # weight each point by the known noise uncertainty
    #sigma_s = noise / 60.0 / 60.0
    sigma_s = 2.3 / 60.0
    fslambda = lambda x: fsight_logp(x, parrange, ghadec_vals, td, Ho_obs, sigma_s)

    # set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fslambda)

    # burnin/sample using emcee
    N_burnin = 4000
    N_final = 2000
    print('Starting burnin of {:d} samples...'.format(N_burnin))
    [pos, lnprobs, rstate] = sampler.run_mcmc(p0, N_burnin)
    print('Burn-in complete. Starting final MCMC run of {:d} samples...'.format(N_final))
    # sample from the posterior using emcee
    sampler.reset()
    [pos, lnprobs, rstate] = sampler.run_mcmc(pos, N_final, thin=10)

    return pos, lnprobs, rstate, sampler

def plot_position_estimate_and_contours(sights, sampler):
    # begin plotting
    mean_lat = np.mean(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi)
    mean_lon = np.mean(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi)
    std_lat = np.std(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi)
    std_lon = np.std(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi)
    # compute a lat/lon range for the map plot, with a minimum span of +- 45 arcminutes
    lat_span = np.max(np.array((std_lat * 7, 45 / 60.0)))
    lon_span = np.max(np.array((std_lon * 7, 45 / 60.0)))

    # adjust the aspect ratio for a better plot
    if lat_span/lon_span > 2.0:
        # lattitude is much greater than longitude -- adjust longitude to be at least half the lattitude
        lon_span = lat_span/2.0
    if lon_span/lat_span > 2.0:
        # longitude is much greater than lattitude-- adjust lattitude to be at least half the longitude
        lat_span = lon_span/2.0

    # miller projection
    map = Basemap(projection='mill',
                  llcrnrlat=mean_lat - lat_span,
                  urcrnrlat=mean_lat + lat_span,
                  llcrnrlon=mean_lon - lon_span,
                  urcrnrlon=mean_lon + lon_span
                  )
    # plot coastlines, draw label meridians and parallels.
    map.drawcoastlines()

    map.drawparallels(np.arange(np.floor(mean_lat)-5, np.ceil(mean_lat) + 5, 1), labels=[1, 0, 0, 0])
    map.drawmeridians(np.arange(np.floor(mean_lon)-5, np.ceil(mean_lon) + 5, 1), labels=[0, 0, 0, 1])
    # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    # use a Gaussian KDE
    gkde = scipy.stats.gaussian_kde(
        [sampler.chain[:, :, 1].flatten() * 180.0 / np.pi, sampler.chain[:, :, 0].flatten() * 180.0 / np.pi],
        bw_method=0.4)
    # create a grid
    lonarray = np.linspace(mean_lon - 8 * std_lon, mean_lon + 8 * std_lon, 300)
    latarray = np.linspace(mean_lat - 8 * std_lat, mean_lat + 8 * std_lat, 300)
    lons, lats = np.meshgrid(lonarray, latarray)

    # find the 68%, 95%, and 99% scores at percentiles
    sarray = scipy.stats.scoreatpercentile(gkde(gkde.resample(1000)), [1, 5, 32])

    z = np.array(gkde.evaluate([lons.flatten(), lats.flatten()])).reshape(lons.shape)
    x, y = map(lons, lats)

    map.contour(x, y, z, sarray, linewidths=2, alpha=1.0, colors='k')
    # convert lat/lons to x,y points
    # x, y = map(sampler.chain[0:100, 0:20, 1].flatten() * 180.0 / np.pi,
    # sampler.chain[0:100, 0:20, 0].flatten() * 180.0 / np.pi)
    # map.scatter(x, y, 3, marker='o', color='k')
    # plot the true value, which relies in the first Sight's nominal lon/lat
    x, y = map(sights[0].lonA.deg, sights[0].latA.deg)
    map.scatter(x, y, s=50, marker='o', color='r')
    plt.title('Position at {} is {} ± {}  /  {} ± {} '.format(
        sights[0].datetime.strftime("%Y/%m/%d -- %H:%M:%S"),
        nadeg(mean_lat / 180 * np.pi),
        nadeg(2.0 * std_lat / 180 * np.pi),
        nadeg(mean_lon / 180 * np.pi),
        nadeg(2.0 * std_lon / 180 * np.pi)),
        fontsize=12)
    plt.show()
    return

def positional_fix_vs_sextant_error(input_sights):
    # This function will execute a loop of Bayesian inference simulations on different sets of Sights lists that have
    # added noise
    #
    # sights - should be an input list of Sight objects
    #

    # array of noise magnitudes, in arcseconds -- adjust this for adding different amounts of noise
    #noise_magnitudes = np.array((1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0))
    noise_magnitudes = np.array((0.01,))
    # prepare an array to store the results
    results = np.zeros((np.size(noise_magnitudes), 5))

    for i in range(np.size(noise_magnitudes)):
        noise = noise_magnitudes[i]

        # deep copy the original list of Sights
        sights = copy.deepcopy(input_sights)
        # now iterate through each sight, and add the prescribed amount of noise to the sextant reading
        print('Printing sights before adding noise of {}...'.format(nadeg(noise/3600.0/180.0*np.pi)))
        print_sights(sights)
        add_sighting_noise(sights, noise)


        parrange = np.zeros((2, 4))
        parrange[0, :] = np.array(
            [sights[0].latA.rad - 10 / 180 * np.pi, sights[0].lonA.rad - 10 / 180 * np.pi, 0.0, 200 / 180.0 * np.pi])
        parrange[1, :] = np.array(
            [sights[0].latA.rad + 10 / 180 * np.pi, sights[0].lonA.rad + 10 / 180 * np.pi, 14, 360 / 180.0 * np.pi])

        pos, lnprobs, rstate, sampler = analyze_sights(sights, parrange)


        results[i, 0] = noise
        # compute the uncertainties in lat/lon
        #LB, UB = HPD_SIM(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi, 0.05)
        est_mean = np.mean(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi)
        est_std = 2.0*np.std(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi)
        results[i, 1] = est_mean
        results[i, 2] = est_std
        #LB, UB = HPD_SIM(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi, 0.05)
        est_mean = np.mean(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi)
        est_std = 2.0 * np.std(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi)
        results[i, 3] = est_mean
        results[i, 4] = est_std
        print('Analysis of noise magnitude {} complete.'.format(nadeg(noise/60/60/180.0*np.pi)))
        print('Lat: {}  +- {}   Lon: {} +- {}'.format(nadeg(results[i, 1]/180.0*np.pi), nadeg(results[i, 2]/180.0*np.pi), nadeg(results[i, 3]/180.0*np.pi), nadeg(results[i, 4]/180.0*np.pi) ))

    plot_position_estimate_and_contours(sights, sampler)

    return results


if __name__ == "__main__":
    # 'database' of sightings for reduction into a single lat/long pair + course speed/heading
    # 21-Feb-2015
    db_sights = """\
        Venus,2015/02/21,20:15:13,21d9.5m,0.0,2.3m,3.05,25,1010,5.5,270d,17d0m,-25d50m
        Sirius,2015/02/21,20:25:27,45d32.5m,0.0,2.3m,3.05,25,1010,5.5,270d,16d59m,-25d51m
        """
    # 22-Feb-2015
    # db_sights = """\
    # SunLL,2015/02/22,12:11:17,51d07.3m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d28m
    # SunLL,2015/02/22,12:12:13,51d22.1m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d28m
    # SunLL,2015/02/22,12:13:08,51d29.8m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d28m
    # SunLL,2015/02/22,12:13:55,51d39.5m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d28m
    # SunLL,2015/02/22,12:15:05,51d48.1m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d29m
    # SunLL,2015/02/22,14:21:17,62d42.3m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    # SunLL,2015/02/22,14:22:23,62d34.8m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    # SunLL,2015/02/22,14:23:11,62d36.4m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    # """

    # 24-Feb-2015
    #db_sights = """\
    #     SunLL,2015/02/24,12:11:13,48d7.5m,0.0,1.9m,2.44,25,1010,5.5,270d,16d13m,-33d5m
    #     SunLL,2015/02/24,12:12:15,48d22.3m,0.0,1.9m,2.44,25,1010,5.5,270d,16d13m,-33d5m
    #     SunLL,2015/02/24,12:13:08,48d29.9m,0.0,1.9m,2.44,25,1010,5.5,270d,16d13m,-33d5m
    #     SunUL,2015/02/24,16:08:15,54d16.8m,0.0,2.1m,3.05,25,1010,5.5,270d,16d13m,-33d5m
    #     SunUL,2015/02/24,16:09:27,54d4.2m,0.0,2.1m,3.05,25,1010,5.5,270d,16d13m,-33d5m
    #     SunUL,2015/02/24,16:11:05,53d46.0m,0.0,2.1m,3.05,25,1010,5.5,270d,16d13m,-33d5m
    #     """

    # 25-Feb-2015
    #db_sights = """\
    #        SunLL,2015/02/25,14:20:47,63d53.4m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:26:40,64d1.3m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:29:08,64d4.9m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:30:45,64d7.1m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:32:16,64d7.2m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:33:27,64d7.0m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:34:39,64d7.5m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:37:46,64d7.1m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:40:13,64d4.9m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:42:53,64d1.7m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:44:16,64d1.9m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:48:26,63d54m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        SunLL,2015/02/25,14:55:31,63d35.4m,0.0,1.8m,3.05,25,1010,5.5,270d,16d35m,-35d24m
    #        """

    # 26-Feb-2015
    # db_sights = """\
    #         Altair,2015/02/26,08:25:16,38d15.7m,0.0,2.0m,3.05,25,1010,5.5,270d,16d21m,-37d0m
    #         """
    # 26-Feb-2015 -- using dead reckoning with a subsequent sun sight to determine this position
    # db_sights = """\
    #         Altair,2015/02/26,08:25:16,38d15.7m,0.0,2.0m,3.05,25,1010,5.5,270d,16d21m,-37d0m
    #         SunUL,2015/02/27,14:48:17,65d58.2m,0.0,1.8m,3.05,25,1010,5.5,270d,16d21m,-37d0m
    #         """
    # 27-Feb-2015 -- using only the sight from 27 Feb (i.e. no dead reckoning)
    # db_sights = """\
    #         SunUL,2015/02/27,14:48:17,65d58.2m,0.0,1.8m,3.05,25,1010,5.5,270d,16d05m,-40d8m
    #         """
    # 28-Feb-2015
    # db_sights = """\
    #         SunLL,2015/02/28,12:57:13,50d4.7m,0.0,1.9m,3.05,25,1010,6,270d,16d48m,-42d36m
    #         SunLL,2015/02/28,16:27:08,57d56.4m,0.0,1.8m,3.05,25,1010,6,270d,16d48m,-42d36m
    #         """
    # 01-Mar-2015
    # db_sights = """\
    #         SunUL,2015/03/01,14:00:05,60d1.7m,0.0,2.0m,3.66,25,1010,6,270d,16d48m,-45d8m
    #         SunLL,2015/03/01,17:47:59,44d28.5m,0.0,2.2m,3.66,25,1010,6,270d,16d48m,-45d8m
    #         """
    # 02-Mar-2015 -- including previous day's sun sights -- seems to be a significant error in these sights
    # The second sight: SunLL,2015/03/01,17:47:59,44d28.5m,0.0,2.2m,3.66,25,1010,6,270d,16d48m,-45d08m
    # is bad -- the location data is wrong.
    # db_sights = """\
    #             SunUL,2015/03/01,14:00:05,60d1.7m,0.0,2.0m,3.66,25,1010,6,270d,16d48m,-45d8m
    #             SunLL,2015/03/01,17:47:59,44d28.5m,0.0,2.2m,3.66,25,1010,6,270d,16d48m,-45d08m
    #             SunUL,2015/03/02,11:10:54,24d26.1m,0.0,2.0m,3.66,25,1010,6,270d,15d26m,-47d16m
    #             SunLL,2015/03/02,16:38:10,60d41.7m,0.0,1.9m,3.66,25,1010,6,270d,15d23m,-47d56m
    #             """
    # 02-Mar-2015 -- Including ONLY today's sights.
    # db_sights = """\
    #             SunUL,2015/03/02,11:10:54,24d26.1m,0.0,2.0m,3.66,25,1010,6,270d,15d26m,-47d16m
    #             SunLL,2015/03/02,16:38:10,60d41.7m,0.0,1.9m,3.66,25,1010,6,270d,15d23m,-47d16m
    #             """
    # 03-Mar-2015
    # db_sights = """\
    #             SunLL,2015/03/03,13:59:01,57d50.8m,0.0,2.2m,3.05,25,1010,6,270d,15d10m,-50d10m
    #             SunUL,2015/03/03,18:03:12,47d34.2m,0.0,2.0m,3.05,25,1010,6,270d,15d07m,-51d18m
    #            """
    # 04-Mar-2015 -- watch error is 3 seconds fast, not 4 seconds.
    # db_sights = """\
    #             SunLL,2015/03/04,14:52:14,65d10.9m,0.0,2.1m,3.05,25,1010,6,270d,14d58m,-52d22m
    #             SunUL,2015/03/04,17:15:17,59d07.1m,0.0,2.2m,3.05,25,1010,6,270d,14d56m,-52d46m
    #             """
    # 05-Mar-2015
    # db_sights = """\
    #             SunLL,2015/03/05,14:32:49,61d25.6m,0.0,1.8m,3.05,25,1010,6,270d,14d46m,-54d42m
    #             SunLL,2015/03/05,18:40:21,43d18.7m,0.0,2.2m,2.44,25,1010,6,270d,14d45m,-55d14m
    #             """
    # 06-Mar-2015 -- morning sights only
    # without:                    Zubenelgenubi,2015/03/06,9:40:16,47d41.5m,0.0,1.9m,3.05,25,1010,6,270d,14d42m,-56d32m
    # db_sights = """\
    #             Zubenelgenubi,2015/03/06,9:40:16,47d41.5m,0.0,1.9m,3.05,25,1010,6,270d,14d42m,-56d32m
    #             Dubhe,2015/03/06,9:44:10,14d26.1m,0.0,1.9m,3.05,25,1010,6,270d,14d42m,-56d32m
    #             Arcturus,2015/03/06,9:49:53,50d59.8m,0.0,1.9m,3.05,25,1010,6,270d,14d42m,-56d32m
    #             """
    # 06-Mar-2015 -- complete set of all six sights -- minus Acamar:
    #  Acamar,2015/03/06,22:49:53,25d56.8m,0.0,2.1m,2.44,25,1010,6,270d,14d40m,-57d44m
    # Could not get this sight to work -- had to manually add its data from Hipparchos DB to PyEphem, but
    # the sight I got out was a few degrees off.
    # db_sights = """\
    #         Zubenelgenubi,2015/03/06,9:40:16,47d41.5m,0.0,1.9m,3.05,25,1010,6,270d,14d42m,-56d32m
    #         Dubhe,2015/03/06,9:44:10,14d26.1m,0.0,1.9m,3.05,25,1010,6,270d,14d42m,-56d32m
    #         Arcturus,2015/03/06,9:49:53,50d59.8m,0.0,1.9m,3.05,25,1010,6,270d,14d42m,-56d32m
    #         Venus,2015/03/06,22:05:53,28d46.1,0.0,2.1m,2.44,25,1010,6,270d,14d40m,-57d44m
    #         Sirius,2015/03/06,22:10:07,51d43.6m,0.0,2.1m,2.44,25,1010,6,270d,14d40m,-57d44m
    #         """
    # 07-Mar-2015
    # db_sights = """\
    #         Kochab,2015/03/07,9:48:11,28d8.1m,0.0,1.8m,3.05,25,1010,6,270d,14d31m,-58d43m
    #         Antares,2015/03/07,9:55:42,49d58.3m,0.0,1.8m,3.05,25,1010,6,270d,14d31m,-58d43m
    #         MoonUL,2015/03/07,10:05:11,14d29.0m,0.0,1.8m,3.05,25,1010,6,270d,14d31m,-58d44m
    #         """
    # 08-Mar-2015
    #db_sights = """\
    #            Dubhe,2015/03/08,9:52:36,13d59.9m,0.0,2.3m,3.05,25,1010,6,270d,14d08m,-60d48m
    #            MoonLL,2015/03/08,10:01:13,26d19.2m,0.0,2.3m,3.05,25,1010,6,270d,14d08m,-60d48m
    #            """

    # http://msi.nga.mil/MSISiteContent/StaticFiles/NAV_PUBS/APN/Chapt-20.pdf
    # db_sights = """\
    #             Kochab,1995/05/17,6:07:43,47d19.1m,0.0,-2.1m,14.63,25,1010,6,90d,39d,-157d10m
    #             Spica,1995/05/17,6:11:26,32d34.8m,0.0,-2.1m,14.63,25,1010,6,90d,39d,-157d8m
    #             """
    # db_sights = """\
    #         Arcturus,2015/02/23,07:54:42,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
    #         Saturn,2015/02/23,8:04:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
    #         Saturn,2015/02/23,8:10:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
    #         Izar,2015/02/23,8:10:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
    #         Kochab,2015/02/23,8:10:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
    #         Vega,2015/02/23,8:10:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
    #         """
    # db_sights = """\
    #         Arcturus,2015/02/23,07:54:42,52d43.6m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
    #         Saturn,2015/02/23,8:04:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
    #         """

    sights = db_sights_preprocess(db_sights)



    ll_lat = []
    ll_lon = []

    # test the prediction routine -- predicts actual sextant sightings from an assumed location & heading
    est_Hs_out = Hs_predict(sights, sights[0].DRv, sights[0].DRbearing, sights[0].latA.rad, sights[0].lonA.rad)

    print(est_Hs_out)

    ## change the sights to be theoretically perfect
    sights = generate_sights_positional(sights)

    # run simulation -- will automatically loop to add artificial noise
    positional_fix_vs_sextant_error(sights)





    # now get the GHA/DEC values and store them, so we don't have to keep looking them up
    ghadec_vals = get_GHADEC_arrays(sights)
    # get the timedeltas and store them, so we don't have to keep looking them up
    td = get_timedeltas(sights)
    # get the observed altitudes
    Ho_obs = get_Ho_altitudes(sights)

    # start with MCMC simulation
    parrange = np.zeros((2, 4))
    parrange[0, :] = np.array([sights[0].latA.rad-5/180*np.pi, sights[0].lonA.rad-5/180*np.pi, 0.0, 200/180.0*np.pi])
    parrange[1, :] = np.array([sights[0].latA.rad + 5 / 180 * np.pi, sights[0].lonA.rad + 5/ 180 * np.pi, 14, 360/180.0*np.pi])

    # prepare MCMC walkers using emcee

    ndim, nwalkers = 4, 44

    # create walkers in random positions within the hypercube of feasible points
    p0 = [np.random.rand(ndim) * (parrange[1, :] - parrange[0, :]) + parrange[0, :] for i in range(nwalkers)]

    # coerce any values outside of the bounds to the boundary
    for i, p in enumerate(p0):
        lidx = p0[i] < parrange[0, :]
        p0[i][lidx] = parrange[0, lidx]
        uidx = p0[i] > parrange[1, :]
        p0[i][uidx] = parrange[1, uidx]

    # DEPRECATED -- code below mostly now included in subfunction
    # sigma_s = 0.5/60
    # fslambda = lambda x: fsight_logp(x, parrange, ghadec_vals, td, Ho_obs, sigma_s)
    #
    #
    # # Ho = Hc * 180.0 / np.pi  # convert to degrees
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, fslambda)
    # # burnin using emcee
    # [pos, lnprobs, rstate] = sampler.run_mcmc(p0, 500)
    #
    # # sample from the posterior using emcee
    # sampler.reset()
    # [pos, lnprobs, rstate] = sampler.run_mcmc(pos, 2500, thin=2)
    #
    #
    # fig, axes = plt.subplots(7, 1, sharex=True, figsize=(8, 9))
    # axes[0].plot(sampler.chain[:, :, 0].T*180.0/np.pi, color="k", alpha=0.4)
    # # axes[0].yaxis.set_major_locator(MaxNLocator(5))
    # # axes[0].axhline(m_true, color="#888888", lw=2)
    # axes[0].set_ylabel("$D_es$")
    # # axes[0].set_ylim([0.4,3])
    #
    # axes[1].plot(sampler.chain[:, :, 1].T*180.0/np.pi, color="k", alpha=0.4)
    # # axes[1].yaxis.set_major_locator(MaxNLocator(5))
    # # axes[1].axhline(b_true, color="#888888", lw=2)
    # axes[1].set_ylabel("$Lambda$")
    #
    # # compute the final lat/lon from the sampler output
    # chain_lat = np.zeros(sampler.chain[:, :, 1].shape)
    # chain_lon = np.zeros(sampler.chain[:, :, 1].shape)
    # for i in range(len(sampler.chain[:, 1, 1])):
    #     for j in range(len(sampler.chain[1, :, 1])):
    #         phi2, lambda2 = compute_displacement(sampler.chain[i,j,0], sampler.chain[i,j,1], sampler.chain[i,j,3], darray[-1])
    #         chain_lat[i, j] = phi2
    #         chain_lon[i, j] = lambda2
    #
    #
    # axes[2].plot(chain_lat.T*180.0/np.pi, color="k", alpha=0.4)
    # # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # # axes[2].axhline(f_true, color="#888888", lw=2)
    # axes[2].set_ylabel("$LATOUT$")
    # axes[2].set_xlabel("step number")
    #
    # axes[3].plot(chain_lon.T*180.0/np.pi, color="k", alpha=0.4)
    # # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # # axes[2].axhline(f_true, color="#888888", lw=2)
    # axes[3].set_ylabel("$LONOUT$")
    #
    # axes[3].set_xlabel("step number")
    #
    # axes[4].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
    # # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # # axes[2].axhline(f_true, color="#888888", lw=2)
    # axes[4].set_ylabel("$Delta1$")
    # # axes[2].set_ylim([0.0,3.0])
    #
    # axes[5].plot(sampler.chain[:, :, 3].T * 180.0 / np.pi, color="k", alpha=0.4)
    # # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # # axes[2].axhline(f_true, color="#888888", lw=2)
    # axes[5].set_ylabel("$Delta2$")
    # axes[5].set_xlabel("step number")
    #
    # axes[6].plot(sampler.lnprobability.T, color="k", alpha=0.4)
    # # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # # axes[2].axhline(f_true, color="#888888", lw=2)
    # axes[6].set_ylabel("$P$")
    #
    # axes[6].set_xlabel("step number")
    #
    #
    #
    # plt.show()
    #
    # # iterate through the hamiltonian parameters of interest and spit out the estimates/intervals
    #
    #
    # LB, UB = HPD_SIM(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi, 0.05)
    # est_mean = np.mean(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi)
    # est_interval = np.max([est_mean - LB, UB - est_mean])
    # print('----------Parameter Lat_In----------')
    # print('{} = {} +- {}'.format('lat in', nadeg(est_mean / 180 * np.pi), nadeg(est_interval / 180 * np.pi)))
    # print('2x std is {}'.format(nadeg(2.0 * np.std(sampler.chain[:, :, 0].flatten()))))
    #
    # LB, UB = HPD_SIM(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi, 0.05)
    # est_mean = np.mean(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi)
    # est_interval = np.max([est_mean - LB, UB - est_mean])
    # print('----------Parameter Lon_In----------')
    # print('{} = {} +- {}'.format('lon in', nadeg(est_mean / 180 * np.pi), nadeg(est_interval / 180 * np.pi)))
    # print('2x std is {}'.format(nadeg(2.0 * np.std(sampler.chain[:, :, 1].flatten()))))
    #
    # LB, UB = HPD_SIM(sampler.chain[:, :, 2].flatten(), 0.05)
    # est_mean = np.mean(sampler.chain[:, :, 2].flatten())
    # est_interval = np.max([est_mean - LB, UB - est_mean])
    # print('----------Parameter Velocity----------')
    # print('{} = {} +- {}'.format('lon in', est_mean, est_interval ))
    # print('2x std is {}'.format(2.0 * np.std(sampler.chain[:, :, 2].flatten())))
    #
    # LB, UB = HPD_SIM(sampler.chain[:, :, 3].flatten() * 180.0 / np.pi, 0.05)
    # est_mean = np.mean(sampler.chain[:, :, 3].flatten() * 180.0 / np.pi)
    # est_interval = np.max([est_mean - LB, UB - est_mean])
    # print('----------Parameter Bearing----------')
    # print('{} = {} +- {}'.format('bearing', nadeg(est_mean / 180 * np.pi), nadeg(est_interval / 180 * np.pi)))
    # print('2x std is {}'.format(nadeg(2.0 * np.std(sampler.chain[:, :, 3].flatten()))))
    #
    # LB, UB = HPD_SIM(chain_lat.flatten()*180.0/np.pi, 0.05)
    # est_mean = np.mean(chain_lat.flatten()*180.0/np.pi)
    # est_interval = np.max([est_mean - LB, UB - est_mean])
    # print('----------Parameter Lat_Out----------')
    # print('{} = {} +- {}'.format('lat out', nadeg(est_mean/180*np.pi), nadeg(est_interval/180*np.pi)))
    # print('2x std is {}'.format(nadeg(2.0*np.std(chain_lat.flatten()))))
    #
    # LB, UB = HPD_SIM(chain_lon.flatten()*180.0/np.pi, 0.05)
    # est_mean = np.mean(chain_lon.flatten()*180.0/np.pi)
    # est_interval = np.max([est_mean - LB, UB - est_mean])
    # print('----------Parameter Lon_Out----------')
    # print('{} = {} +- {}'.format('lon out', nadeg(est_mean/180*np.pi), nadeg(est_interval/180*np.pi)))
    # print('2x std is {}'.format(nadeg(2.0*np.std(chain_lon.flatten()))))
