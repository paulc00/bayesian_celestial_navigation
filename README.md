# Bayesian Celestial Navigation

This package uses Markov Chain Monte Carlo (MCMC) methods to infer a positional fix using readings from a marine sextant.

I created this project to teach myself basic celestial navigation techniques well enough to solve a set of celestial navigation exercises on a blog I found: [Celestial Navigation Practice Problems](https://celestialnavproblems.wordpress.com)

Traditionally, a line of position (LOP) is calculated from a sextant sighting that has been corrected for various errors (e.g. atmospheric refraction). A single measurement determines a circle on the Earth's surface on which the observer must lie, and over short distances, this circle is simply a line. If two different LOPs from two different celestial objects can be determined, then the observer should lie at their intersection point. This is called a positional fix. If only one measurement can be taken (e.g. during the day, when only the sun is visible) or the measurements are taken more than a few tens of minutes apart, a second kind of fix called a `running fix' can be used to estimate the observer's position. In this scenario, an estimate of the ship's position based on dead reckoning, combined with the available LOPs, can produce a rough estimate of the observer's position.

Mainly, I wanted to get a very quantitative picture of the precision limits of celestial navigation. Specifically:

* Can a series of Sun sights be used with only weak prior information to obtain a good fix of lattitude *and* longitude?
* What does the posterior distribution of a positional fix using two celestial objects look like? Nominally, it should look like a sort of "cross" where the lines of position cross -- I'd like to visualize that.
* Given a pair of stars for such a positional fix, what is the relation between how precise one can measure angle with a sextant and how precisely one's location can be determined from that information?

While writing this code, I found Rodger E. Farley's '[The Armchair Celestial Navigator](http://www.dacust.com/navigation/pdf/ArmchairCelestialNavigator.pdf)' to be a great practical resource for navigating via sextant.

## Getting Started

This is a small project that's for my own exploration, so all of the analysis subfunctions are stored in a single file (bayescelestial.py). The simulated sextant sightings are read in from blocks of strings stored there, as well; this isn't pretty, but it made for quick exploratory analysis.

### Prerequisites

Beyond the standard numpy, scipy, and matplotlib packages, this package requires:

astropy -- does lattitude/longitude computations

pyephem -- performs basic astronomical calculations. Most importantly, this is used to compute the Global Hour Angle (GHA) and Declination (DEC), which are the main quantities used in determining the expected object location in the sky, and other quantities like the semi-diameter of the Moon, which must be taken into account for accurate position fixing.

emcee -- performs Markov Chain Monte Carlo simulations. These are the main interest of the package, as they are used for statistical inference of a position given the sextant sightings and GHA/DEC data from pyephem.

## Results

Position plots: The red dot indicates the true position given by the navigation blog, and the black curves are the contours of the posterior probability distribution at the 68%, 95%, and 99% probability levels, as determined via the MCMC simulation.

A classical positional fix from two star sightings:

![ScreenShot](plots/20150221_1.png)

This is a great example of the statistical uncertainty of a regular fix. Since two stars at different locations in the sky are measured, and each measurement defines an arc on the Earth where the position could be, the two arcs crossing produce an "X" pattern. The overlap gives good resolution of both lattitude and longitude, and this is why this is the main technique for fixing position using a sextant.

A positional fix from 13 lower-limb Sun sightings, taken in quick succession (~35 minutes):

![ScreenShot](plots/20150225_1.png)

During the day, when no stars other than the Sun are visible, it can be difficult to estimate a precise lattitude/longitude fix. A simple technique for navigation would be to use dead reckoning, which is where the last known position fix is updated using the ship's estimated velocity, but this is subject to many errors. During the day, the lattitude can be estimated by looking at the Sun's position, especially around the noon hour. In this plot, a sequence of Sun sightings are taken and reduced into the posterior probability distribution of both lattitude and longitude. You can see from the plot that the uncertainty in lattitude is about 1.6 arcminutes, while the uncertainty in longitude is about 17 arcminutes (a 10x increase). This reflects that comparatively less information can be extracted about the longitude. However, this plot answers one of my original questions: it's clear that there's *some* information there, and so even with a few Sun sights, a rough position fix in both lat/lon can be obtained.

A positional fix from six lower- and upper-limb Sun sightings, taken in two sets separated by 4 hours:

![ScreenShot](plots/20150224_1.png)


