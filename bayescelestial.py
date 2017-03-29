import datetime as dt
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Longitude
import ephem
import numpy as np
import math
import scipy.optimize
import emcee
import matplotlib
import matplotlib.pyplot as plt

# class constructor, for future use -- not implemented yet. will be much cleaner.
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
        print('GHA/DEC computed... {:.2f} / {:.2f}'.format(self.GHA.deg, self.DEC.deg))
        # this method will update the corrections to the sextant sighting.
        # it is useful to segment the function this way so that when generating theoretical sights and
        # changing the lat/lon to be artificial values, a single method will update the rest of the Sight
        # object's attributes accordingly
        self.update_sight_GHADEC_and_corrections()

        return

    def update_sight_GHADEC_and_corrections(self):
        # compute corrections for the 'apparent altitude'
        self.dip_correction()
        self.index_correction()
        self.Ha = self.Hs + self.dip_corr + self.index_corr
        print('Ha is updated to: {}'.format(self.Ha))
        # compute the 'main correction'
        self.semidiameter_correction()
        self.get_horizontal_parallax()
        self.parallax_in_altitude_correction()
        self.atmo_correction()
        self.Ho = self.Ha + self.SD_corr + self.parallax_corr + self.atmo_corr
        print('Ho is updated to: {}'.format(self.Ho))
        return

    def get_horizontal_parallax(self):
        if self.body == 'SunLL' or self.body == 'SunUL':
            self.HP = Angle('0.0d') # Angle('0.0024d')
            return
        # placeholder function
        self.HP = Angle('0.0d')
        return

    def parallax_in_altitude_correction(self):
        print('Parallax correction started. Ha is: {} LatA is {}'.format(self.Ha.deg, self.latA.deg))
        self.parallax_corr = self.HP*np.cos(self.Ha.deg/180.0*np.pi)*(1.-(np.sin(self.latA.deg/180.0*np.pi)**2.0)/297.0)
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
                self.SD_corr = Angle('{:.3f}d'.format(sds))
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
            s = ephem.Sun()
        # insert moon here
        # planets
        elif self.body == 'Mars':
            s = ephem.Mars()
        elif self.body == 'Venus':
            s = ephem.Venus()
        elif self.body == 'Jupiter':
            s = ephem.Jupiter()
        elif self.body == 'Saturn':
            s = ephem.Saturn()
        elif self.body == 'Uranus':
            s = ephem.Uranus()
        elif self.body == 'Mercury':
            s = ephem.Mercury()
        # anything else must be a star, and its body can be interpreted via the string argument in ephem.star('')
        else:
            s = ephem.star(self.body)

        obs = ephem.Observer()
        # format strings from datetime object
        date = self.datetime.strftime('%Y/%m/%d')
        time = self.datetime.strftime('%H:%M:%S')
        date_string = date + ' ' + time
        obs.date = date_string

        s.compute(date_string, epoch=date_string)

        deg = ephem.degrees(obs.sidereal_time() - s.g_ra).norm
        ghas = nadeg(deg)
        deg = s.g_dec
        decs = nadeg(deg)
        self.GHA = Angle(ghas, unit=u.deg)
        self.DEC = Angle(decs, unit=u.deg)
        print('GHA and DEC for body {} are {:.3f} / {:.3f}'.format(self.body, self.GHA.deg, self.DEC.deg))
        return

    def est_longitude(self):
        # use a root-finding algorithm to compute the longitude based on an
        # assumed latitude. here, the estimated longitude is to start the root-finding algorithm

        print('Finding Zo for true sextant angle of ' + '{:.3f}'.format(self.Ho) + ' degrees...')
        print('GHA and DEC are {} {}'.format(self.GHA.deg, self.DEC.deg))
        start_H = compute_Hc(self.latA, self.lonA, self.GHA, self.DEC).deg
        lower_H = compute_Hc(self.latA, Angle(self.lonA.deg - 2, unit=u.deg), self.GHA, self.DEC).deg

        upper_H = compute_Hc(self.latA, Angle(self.lonA.deg + 2, unit=u.deg), self.GHA, self.DEC).deg

        print(
            'Starting from value ' + '{:.3f}'.format(start_H) + ' degrees, with lower {:.3f} and upper {:.3f}.'.format(
                lower_H, upper_H))

        fz = lambda x: (compute_Hc(self.latA, Angle(x, unit=u.deg), self.GHA, self.DEC).deg - self.Ho.deg)
        print('fz(a) is {:.3f}, fz(b) is {:.3f}'.format(fz(self.lonA.deg-2),fz(self.lonA.deg+2)))
        # spn_out = scipy.optimize.newton(fz, lonA.deg, maxiter=200)
        sp_out = scipy.optimize.brentq(fz, self.lonA.deg - 2.0, self.lonA.deg + 2.0, maxiter=100)

        # sp_out = scipy.optimize.minimize_scalar(fz, bracket=[lonA.deg-2.0,lonA.deg+2.0], method='brent', tol=1.48e-06)
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
    sigma_s = 1/60.0/180.0*np.pi
    y_theory = Ho_predict(ghadec_vals, td, x[0], x[1], x[2], x[3])
    y_data = Ho_obs

    chi = np.power(y_theory - y_data,2.0)/(2.0*sigma_s**2.0)

    return np.sum(chi)

def prior_logP(params, parrange):
    if np.any(params < parrange[0,:]) or np.any(params > parrange[1,:]):
        return -np.inf
    else:
        bearing_prior = -(params[3]-270.0/180.0*np.pi)**2.0/(2.0*((10.0/180.0*np.pi)**2.0))
        velocity_prior = -(params[2]-5.5)**2.0/(2.0*(4.0**2.0))
        return bearing_prior + velocity_prior

def fsight_logp(x, parrange, ghadec_vals, td, Ho_obs):
    # get the prior probability
    logprior = prior_logP(x, parrange)
    if logprior == -np.inf:
        # if out of bounds, just return -inf now
        return -np.inf
    sigma_s = 0.1/60.0
    y_theory = Ho_predict(ghadec_vals, td, x[0], x[1], x[2], x[3])
    y_data = Ho_obs

    # this form of log likelihood is standard and assumes a known error
    chi = -np.power(y_theory - y_data,2.0)/(2.0*sigma_s*sigma_s)
    logp = np.sum(chi) + logprior

    # this form of log likelihood is a lot more conservative w/ respect to errors
    #Rk = (y_theory - y_data)/sigma_s
    #logp = np.sum(np.log( (1.0 - np.exp(-Rk**2.0 / 2.0))/ (Rk**2.0))) + logprior
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

def generate_sights(sights):
    # this function will generate a set of sights from positional data. its purpose is to provide sightings with a
    # known amount of noise in order to test the analysis rountine's output

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
    noise_mag = 0.5/60.0 # in degrees
    for i in range(len(sights)):
        print('New angle is: {}'.format(nadeg(est_Hs[i]/180*np.pi)))
        sights[i].Hs = Angle(est_Hs[i] + np.random.randn()*noise_mag, unit=u.deg)
        sights[i].update_sight_GHADEC_and_corrections()

    return sights

def print_sights(sights):
    print('----------Sight Database----------')
    for i in range(len(sights)):
        print('SIGHT {:d} --- Body: {}'.format(i, sights[i].body))
        print('Hs: {} Ha: {} Ho: {}'.format(nadeg(sights[i].Hs.rad), nadeg(sights[i].Ha.rad), nadeg(sights[i].Ho.rad)))
        print('')

    return


if __name__ == "__main__":
    # 'database' of sightings for reduction into a single lat/long pair + course speed/heading

    db_sights = """\
    Venus,2015/02/21,20:15:13,21d9.5m,0.0,2.3m,3.05,25,1010,5.5,270d,17d0m,-25d50m
    Sirius,2015/02/21,20:25:27,45d32.5m,0.0,2.3m,3.05,25,1010,5.5,270d,17d0m,-25d51m
    SunLL,2015/02/22,12:11:17,51d07.3m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,12:12:13,51d22.1m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,12:13:08,51d29.8m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,12:13:55,51d39.5m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,12:15:05,51d48.1m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,14:21:17,62d42.3m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    SunLL,2015/02/22,14:22:23,62d34.8m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    SunLL,2015/02/22,14:23:11,62d36.4m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    """

    db_sights = """\
        Venus,2016/02/21,20:15:13,21d9.5m,0.0,2.3m,3.05,25,1010,5.5,270d,17d0m,-25d50m
        Sirius,2016/02/21,20:25:27,45d32.5m,0.0,2.3m,3.05,25,1010,5.5,270d,17d0m,-25d51m
        """
    db_sights = """\
            Arcturus,2015/02/23,07:54:42,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
            Saturn,2015/02/23,8:04:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
            Saturn,2015/02/23,8:10:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
            Izar,2015/02/23,8:10:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
            Kochab,2015/02/23,8:10:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
            Vega,2015/02/23,8:10:34,54d34.8m,0.0,2.3m,3.05,25,1010,5.5,270d,16d25m,-28d47m
            """
    sights = db_sights_preprocess(db_sights)
    print(sights)

    ll_lat = []
    ll_lon = []
    # # preprocess the array of sights into corrected sightings based on ephemeris from pyephem
    # for i in range(len(sights)):
    #
    #     # estimate the location
    #     # function def is: est_longitude(body, date, time, Ha, IE, h, temp, pressure, GHA, DEC, latA, lonA):
    #     sights[i].est_longitude()
    #     print('Estimated longitude is: {:.3f}'.format(sights[i].est_lon))
    #     ll_lat.append(sights[i].datetime.strftime('%Y:%m:%d'))
    #     ll_lon.append(sights[i].est_lon.deg)

    # test the prediction routine -- predicts actual sextant sightings from an assumed location & heading
    est_Hs_out = Hs_predict(sights, sights[0].DRv, sights[0].DRbearing, sights[0].latA.rad, sights[0].lonA.rad)

    print(est_Hs_out)

    # change the sights to be theoretically perfect (plus a small amount of random noise)
    print_sights(sights)
    sights = generate_sights(sights)
    print_sights(sights)

    # now get the GHA/DEC values and store them, so we don't have to keep looking them up
    ghadec_vals = get_GHADEC_arrays(sights)
    # get the timedeltas and store them, so we don't have to keep looking them up
    td = get_timedeltas(sights)
    # get the observed altitudes
    Ho_obs = get_Ho_altitudes(sights)
    x0 = np.array((sights[0].latA.rad, sights[0].lonA.rad, 11.5, 270/180.0*np.pi))
    min_out = scipy.optimize.minimize(fsight, x0, args=(ghadec_vals, td, Ho_obs), method='Nelder-Mead')
    print(min_out)
    print('Lat: {:.3f} Lon: {:.3f} Speed: {:.2f} Bearing: {:.2f}'.format(min_out.x[0]*180.0/np.pi, min_out.x[1]*180.0/np.pi, min_out.x[2], min_out.x[3]*180/np.pi))
    # Compute the final lat/lon estimate
    darray = compute_distances(td, min_out.x[2])
    phi2, lambda2 = compute_displacement(min_out.x[0], min_out.x[1], min_out.x[2], darray)
    print('Phi2: {}  Lambda2: {}'.format(phi2*180/np.pi,lambda2*180/np.pi))


    # start with MCMC simulation
    parrange = np.zeros((2,4))
    parrange[0,:] = np.array([sights[0].latA.rad-5/180*np.pi, sights[0].lonA.rad-5/180*np.pi, 0.0, 200/180.0*np.pi])
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


    fslambda = lambda x: fsight_logp(x, parrange, ghadec_vals, td, Ho_obs)


    # print('Now predict step by step...')
    # DRv = x0[2]
    # darray = compute_distances(td, DRv)
    #
    # # compute the lat/lon pairs for each time
    # lat_in = np.ones(len(sights)) * x0[0]
    # lon_in = np.ones(len(sights)) * x0[1]
    #
    # # compute the displacement of lat/lon along a great circle, given the initial point, bearing, and distance travelled
    # phi2, lambda2 = compute_displacement(lat_in, lon_in, x0[3], darray)  # args should be in radians
    # print('phi 2 and lambda2: {} and {}'.format(phi2*180.0/np.pi,lambda2*180/np.pi))
    # # compute the Hs values for each lat/lon pair and sighting
    # Hc = compute_Hc_fast(phi2, lambda2, ghadec_vals[:, 0] / 180.0 * np.pi,
    #                      ghadec_vals[:, 1] / 180.0 * np.pi)  # everything must be in radians
    # print('HC is {}'.format(Hc*180.0/np.pi))
    #
    #
    # # take the true altitudes and subtracted the correction estimates to get what should be observed on a sextant reading
    # Ho = Hc * 180.0 / np.pi  # convert to degrees
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fslambda)
    # burnin using emcee
    [pos, lnprobs, rstate] = sampler.run_mcmc(p0, 500)

    # sample from the posterior using emcee
    sampler.reset()
    [pos, lnprobs, rstate] = sampler.run_mcmc(pos, 500, thin=2)


    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T*180.0/np.pi, color="k", alpha=0.4)
    # axes[0].yaxis.set_major_locator(MaxNLocator(5))
    # axes[0].axhline(m_true, color="#888888", lw=2)
    axes[0].set_ylabel("$D_es$")
    # axes[0].set_ylim([0.4,3])

    axes[1].plot(sampler.chain[:, :, 1].T*180.0/np.pi, color="k", alpha=0.4)
    # axes[1].yaxis.set_major_locator(MaxNLocator(5))
    # axes[1].axhline(b_true, color="#888888", lw=2)
    axes[1].set_ylabel("$Lambda$")

    # compute the final lat/lon from the sampler output
    chain_lat = np.zeros(sampler.chain[:, :, 1].shape)
    chain_lon = np.zeros(sampler.chain[:, :, 1].shape)
    for i in range(len(sampler.chain[:, 1, 1])):
        for j in range(len(sampler.chain[1, :, 1])):
            phi2, lambda2 = compute_displacement(sampler.chain[i,j,0], sampler.chain[i,j,1], sampler.chain[i,j,3], darray[-1])
            chain_lat[i, j] = phi2
            chain_lon[i, j] = lambda2


    axes[2].plot(chain_lat.T*180.0/np.pi, color="k", alpha=0.4)
    # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # axes[2].axhline(f_true, color="#888888", lw=2)
    axes[2].set_ylabel("$LATOUT$")
    axes[2].set_xlabel("step number")

    axes[3].plot(chain_lon.T*180.0/np.pi, color="k", alpha=0.4)
    # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # axes[2].axhline(f_true, color="#888888", lw=2)
    axes[3].set_ylabel("$LONOUT$")

    axes[3].set_xlabel("step number")

    axes[4].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
    # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # axes[2].axhline(f_true, color="#888888", lw=2)
    axes[4].set_ylabel("$Delta1$")
    # axes[2].set_ylim([0.0,3.0])

    axes[5].plot(sampler.chain[:, :, 3].T * 180.0 / np.pi, color="k", alpha=0.4)
    # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # axes[2].axhline(f_true, color="#888888", lw=2)
    axes[5].set_ylabel("$Delta2$")
    axes[5].set_xlabel("step number")

    axes[6].plot(sampler.lnprobability.T, color="k", alpha=0.4)
    # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # axes[2].axhline(f_true, color="#888888", lw=2)
    axes[6].set_ylabel("$P$")

    axes[6].set_xlabel("step number")



    plt.show()

    # iterate through the hamiltonian parameters of interest and spit out the estimates/intervals


    LB, UB = HPD_SIM(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi, 0.05)
    est_mean = np.mean(sampler.chain[:, :, 0].flatten() * 180.0 / np.pi)
    est_interval = np.max([est_mean - LB, UB - est_mean])
    print('----------Parameter Lat_In----------')
    print('{} = {} +- {}'.format('lat in', nadeg(est_mean / 180 * np.pi), nadeg(est_interval / 180 * np.pi)))
    print('2x std is {}'.format(nadeg(2.0 * np.std(sampler.chain[:, :, 0].flatten()))))

    LB, UB = HPD_SIM(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi, 0.05)
    est_mean = np.mean(sampler.chain[:, :, 1].flatten() * 180.0 / np.pi)
    est_interval = np.max([est_mean - LB, UB - est_mean])
    print('----------Parameter Lon_In----------')
    print('{} = {} +- {}'.format('lon in', nadeg(est_mean / 180 * np.pi), nadeg(est_interval / 180 * np.pi)))
    print('2x std is {}'.format(nadeg(2.0 * np.std(sampler.chain[:, :, 1].flatten()))))

    LB, UB = HPD_SIM(sampler.chain[:, :, 2].flatten(), 0.05)
    est_mean = np.mean(sampler.chain[:, :, 2].flatten())
    est_interval = np.max([est_mean - LB, UB - est_mean])
    print('----------Parameter Velocity----------')
    print('{} = {} +- {}'.format('lon in', est_mean, est_interval ))
    print('2x std is {}'.format(2.0 * np.std(sampler.chain[:, :, 2].flatten())))

    LB, UB = HPD_SIM(sampler.chain[:, :, 3].flatten() * 180.0 / np.pi, 0.05)
    est_mean = np.mean(sampler.chain[:, :, 3].flatten() * 180.0 / np.pi)
    est_interval = np.max([est_mean - LB, UB - est_mean])
    print('----------Parameter Bearing----------')
    print('{} = {} +- {}'.format('bearing', nadeg(est_mean / 180 * np.pi), nadeg(est_interval / 180 * np.pi)))
    print('2x std is {}'.format(nadeg(2.0 * np.std(sampler.chain[:, :, 3].flatten()))))

    LB, UB = HPD_SIM(chain_lat.flatten()*180.0/np.pi, 0.05)
    est_mean = np.mean(chain_lat.flatten()*180.0/np.pi)
    est_interval = np.max([est_mean - LB, UB - est_mean])
    print('----------Parameter Lat_Out----------')
    print('{} = {} +- {}'.format('lat out', nadeg(est_mean/180*np.pi), nadeg(est_interval/180*np.pi)))
    print('2x std is {}'.format(nadeg(2.0*np.std(chain_lat.flatten()))))

    LB, UB = HPD_SIM(chain_lon.flatten()*180.0/np.pi, 0.05)
    est_mean = np.mean(chain_lon.flatten()*180.0/np.pi)
    est_interval = np.max([est_mean - LB, UB - est_mean])
    print('----------Parameter Lon_Out----------')
    print('{} = {} +- {}'.format('lon out', nadeg(est_mean/180*np.pi), nadeg(est_interval/180*np.pi)))
    print('2x std is {}'.format(nadeg(2.0*np.std(chain_lon.flatten()))))