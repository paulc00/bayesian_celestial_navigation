import datetime as dt
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Longitude
import ephem
import numpy as np
import math
import scipy.optimize

# class constructor, for future use -- not implemented yet. will be much cleaner.
class Sight:
    def __init__(self, body, date, time, Hs, WE, IE, height, temperature, pressure, DRv, DRbearing, latA, lonA):
        self.body = body

        # process date and time into a single datetime object
        year, month, day = date.split('/')
        hour, minute, second = time.split(':')
        self.datetime = dt.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
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

        # compute corrections for the 'apparent altitude'
        self.dip_correction()
        self.index_correction()
        self.Ha = self.Hs + self.dip_corr + self.index_corr
        print('Ha is {}'.format(self.Ha))

        # compute the 'main correction'
        self.semidiameter_correction()
        self.get_horizontal_parallax()
        self.parallax_in_altitude_correction()
        self.atmo_correction()
        self.Ho = self.Ha + self.SD_corr + self.parallax_corr + self.atmo_corr

    def get_horizontal_parallax(self):
        if self.body == 'SunLL' or self.body == 'SunUL':
            self.HP = Angle('0.0d') # Angle('0.0024d')
            return
        # placeholder function
        self.HP = Angle('0.0d')
        return

    def parallax_in_altitude_correction(self):
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
        print('GHADEC body is {}'.format(self.body))
        if self.body == 'SunLL' or self.body == 'SunUL':
            s = ephem.Sun()
            obs = ephem.Observer()
            # format strings from datetime object
            date = self.datetime.strftime('%Y/%m/%d')
            time = self.datetime.strftime('%H:%M:%S')
            date_string = date + ' ' + time
            print('Date string is: {}'.format(date_string))
            obs.date = date_string

            s.compute(date_string, epoch=date_string)

            deg = ephem.degrees(obs.sidereal_time() - s.g_ra).norm
            ghas = nadeg(deg)
            deg = s.g_dec
            decs = nadeg(deg)
            self.GHA = Angle(ghas, unit=u.deg)
            self.DEC = Angle(decs, unit=u.deg)
            print('GHA and DEC are {:.3f} / {:.3f}'.format(self.GHA.deg, self.DEC.deg))
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


def nadeg(deg):
    # changes ephem.angel (rad) to the format for the Angle class.
    theminus = ""
    if deg < 0:
        theminus = "-"
    g = int(math.degrees(deg))
    m = (math.degrees(deg) - g) * 60
    gm = "%s%sd%04.1fm" % (theminus, abs(g), abs(m))
    return gm


if __name__ == "__main__":
    # 'database' of sightings for reduction into a single lat/long pair + course speed/heading

    db_sights = """\
    SunLL,2015/02/22,12:11:17,51d07.3m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,12:12:13,51d22.1m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,12:13:08,51d29.8m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,12:13:55,51d39.5m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,12:15:05,51d48.1m,0.0,2.0m,2.44,25,1010,5.5,270d,16d44m,-27d7m
    SunLL,2015/02/22,14:21:17,62d42.3m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    SunLL,2015/02/22,14:22:23,62d34.8m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    SunLL,2015/02/22,14:23:11,62d36.4m,0.0,2.1m,3.05,25,1010,5.5,270d,16d42m,-27d53m
    """

    sights = db_sights_preprocess(db_sights)
    print(sights)

    ll_lat = []
    ll_lon = []
    # preprocess the array of sights into corrected sightings based on ephemeris from pyephem
    for i in range(len(sights)):

        # estimate the location
        # function def is: est_longitude(body, date, time, Ha, IE, h, temp, pressure, GHA, DEC, latA, lonA):
        sights[i].est_longitude()
        print('Estimated longitude is: {:.3f}'.format(sights[i].est_lon))
        ll_lat.append(sights[i].datetime.strftime('%Y:%m:%d'))
        ll_lon.append(sights[i].est_lon.deg)

    # test the prediction routine -- predicts actual sextant sightings from an assumed location & heading
    est_Hs_out = Hs_predict(sights, sights[0].DRv, sights[0].DRbearing, sights[0].latA.rad, sights[0].lonA.rad)

    print(est_Hs_out)
