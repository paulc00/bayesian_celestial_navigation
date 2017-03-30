# Bayesian Celestial Navigation

This package uses Markov Chain Monte Carlo methods to infer a positional fix using readings from a marine sextant.

I created this project to teach myself basic celestial navigation techniques well enough to solve a set of celestial navigation exercises on a blog I found: [Celestial Navigation Practice Problems](https://celestialnavproblems.wordpress.com)

Traditionally, a line of position (LOP) is calculated from a sextant sighting that has been corrected for various errors (e.g. atmospheric refraction). A single measurement determines a circle on the Earth's surface on which the observer must lie, and over short distances, this circle is simply a line. If two different LOPs from two different celestial objects can be determined, then the observer should lie at their intersection point. This is called a positional fix. If only one measurement can be taken (e.g. during the day, when only the sun is visible) or the measurements are taken more than a few tens of minutes apart, a second kind of fix called a `running fix' can be used to estimate the observer's position. In this scenario, an estimate of the ship's position based on dead reckoning, combined with the available LOPs, can produce a rough estimate of the observer's position.

Mainly, I wanted to get a very quantitative picture of the precision limits of celestial navigation. Specifically:

* Can a series of Sun sights be used with only weak prior information to obtain a good fix of lattitude *and* longitude?
* What does the posterior distribution of a positional fix using two celestial objects look like? Nominally, it should look like a sort of "cross" where the lines of position cross -- I'd like to visualize that.
* Given a pair of stars for such a positional fix, what is the relation between how precise one can measure angle with a sextant and how precisely one's location can be determined from that information?

While writing this code, I found Rodger E. Farley's '[The Armchair Celestial Navigator](http://www.dacust.com/navigation/pdf/ArmchairCelestialNavigator.pdf)' to be a great practical resource for navigating via sextant.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
