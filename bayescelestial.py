import datetime as dt
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Longitude
import ephem
import numpy as np
import math


# class constructor, for future use -- not implemented yet. will be much cleaner.
class Sight:
    def __init__(self, body, date, time, Hs, WE, IE, height, pressure, temperature, DRv, DRbearing, latA, lonA):
        self.body = body

        # process date and time into a single datetime object
        year, month, day = date.split('/')
        hour, minute, second = time.split(':')
        self.datetime = dt.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        self.Hs = Angle(Hs)

        self.WE = float(WE)
        self.IE = Angle(IE)
        self.height = float(height) # height in meters
        self.pressure = float(pressure) # pressure in mbar
        self.temperature = float(temperature) # temperature in degrees C
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
        self.dip_corr = None
        self.index_corr = None
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
            self.HP = Angle('0.0024d')
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
            time = self.datetime.strftime('%Y:%m:%d')
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
        Ha = self.Ha.radian
        f = 0.28 * Pmb / (TdegC + 273)
        Ro = -0.0167 / np.tan(np.pi / 180.0 * (Ha + 7.31 / (Ha + 4.4)))
        self.atmo_corr = Angle(Ro * f, unit=u.deg)
        return

    def compute_GHA_DEC(self):
        print('GHADEC body is {}'.format(self.body))
        if self.body == 'SunLL' or self.body == 'SunUL':
            s = ephem.Sun()
            obs = ephem.Observer()
            # format strings from datetime object
            date = self.datetime.strftime('%Y/%m/%d')
            time = self.datetime.strftime('%Y:%m:%d')
            date_string = date + ' ' + time
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

        start_H = compute_Hc(self.latA, self.lonA, GHA, DEC).deg
        lower_H = compute_Hc(latA, Angle(lonA.deg - 2, unit=u.deg), GHA, DEC).deg

        upper_H = compute_Hc(latA, Angle(lonA.deg + 2, unit=u.deg), GHA, DEC).deg

        print(
            'Starting from value ' + '{:.3f}'.format(start_H) + ' degrees, with lower {:.3f} and upper {:.3f}.'.format(
                lower_H, upper_H))

        fz = lambda x: (compute_Hc(latA, Angle(x, unit=u.deg), GHA, DEC).deg - (
        Hs.deg + dip_correction(h).deg + index_correction(IE).deg + atmo_correction(Hs, temp,
                                                                                    pressure).deg + semidiameter_correction(
            body, date, time).deg))
        # print('fz(a) is {:.3f}, fz(b) is {:.3f}'.format(fz(lonA.deg-2),fz(lonA.deg+2)))
        # spn_out = scipy.optimize.newton(fz, lonA.deg, maxiter=200)
        sp_out = scipy.optimize.brentq(fz, lonA.deg - 2.0, lonA.deg + 2.0, maxiter=100)

        # sp_out = scipy.optimize.minimize_scalar(fz, bracket=[lonA.deg-2.0,lonA.deg+2.0], method='brent', tol=1.48e-06)

        return Angle(sp_out, unit=u.deg)

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
    SunLL,2015/02/22,12:11:17,51d07.3m,0.0,2.0m,2.44,25,1010,5.5,180d,16d44m,-27d7m
    SunLL,2015/02/22,12:12:13,51d22.1m,0.0,2.0m,2.44,25,1010,5.5,180d,16d44m,-27d7m
    SunLL,2015/02/22,12:13:08,51d29.8m,0.0,2.0m,2.44,25,1010,5.5,180d,16d44m,-27d7m
    SunLL,2015/02/22,12:13:55,51d39.5m,0.0,2.0m,2.44,25,1010,5.5,180d,16d44m,-27d7m
    SunLL,2015/02/22,12:15:05,51d48.1m,0.0,2.0m,2.44,25,1010,5.5,180d,16d44m,-27d7m
    SunLL,2015/02/22,14:21:17,62d42.3m,0.0,2.1m,3.05,25,1010,5.5,180d,16d42m,-27d53m
    SunLL,2015/02/22,14:22:23,62d34.8m,0.0,2.1m,3.05,25,1010,5.5,180d,16d42m,-27d53m
    SunLL,2015/02/22,14:23:11,62d36.4m,0.0,2.1m,3.05,25,1010,5.5,180d,16d42m,-27d53m
    """

    a = db_sights_preprocess(db_sights)
    print(a)

    ll_lat = []
    ll_lon = []
    # preprocess the array of sights into corrected sightings based on ephemeris from pyephem
    for i in range(len(a)):

        # estimate the location
        # function def is: est_longitude(body, date, time, Ha, IE, h, temp, pressure, GHA, DEC, latA, lonA):
        est_long = est_longitude(body, date, time, Hs, IE, height, temp, pressure, GHAc, DECc, latA, lonA)
        print('Estimated longitude is: {:.3f}'.format(est_long))
        ll_lat.append(time)
        ll_lon.append(est_long.deg)
