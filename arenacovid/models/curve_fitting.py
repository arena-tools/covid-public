import numpy as np

# Modeling
import pymc3 as pm
import theano


def normal(x):
    """
    Return pdf(x) for standard normal N(0,1).
    """
    return np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)


def hierarchical_normal(name, shape, mu=None, group_mu_variance=5, intra_group_variance=1):
    """
    Credit to Austin Rochford: https://www.austinrochford.com/posts/2018-12-20-sports-precision.html
    """
    if mu is None:
        mu = pm.Normal(f"mu_{name}", 0.0, group_mu_variance)

    delta = pm.StudentT(f"delta_{name}", nu=1, shape=shape)
    sigma = pm.HalfCauchy(f"sigma_{name}", intra_group_variance)
    return pm.Deterministic(name, mu + delta * sigma)


class HierarchicalCurveFitter:
    def __init__(self, mu_lower_bound: float = 10, mu_upper_bound: float = 80, p_upper_bound: float=1000, progressbar=False):
        """Hierarchical model for Curve Fitting (Normal Distribution)
        This class fits a Hierarchical Model of Normal Distribution curves
        against observed daily deaths per million of population in separates states,
        counties, countries, etc. 
        Args:
            mu_lower_bound (float / np.array): Lower Bound for the mu parameter, which is
                            the time since 0.3 deaths / million until "peak" death rate.
                            per Million deaths. Applies to all groups. 
            mu_upper_bound (float / np.array): Upper Bound for the mu parameter. Applies to all groups.
            p_upper_bound (float / np.array): upper bound for p, the scale/height of peak
            progressbar (bool): display progress bar while sampling
        """
        self.p_upper_bound = p_upper_bound
        self.mu_lower_bound = mu_lower_bound
        self.mu_upper_bound = mu_upper_bound
        self.progressbar = progressbar

    def fit(self, y, ids, times):
        """ Fit the given daily deaths / million, group ids and times.
        Args:
            y (np.array): The target observations - deaths / million of population
            ids (np.array, int): The group id's for every observation
            time (array): The times for every obsevation
        Returns:
            self
        """
        self.k = len(np.unique(ids))
        self.ids = theano.shared(ids)
        self.times = theano.shared(times)
        self.Y = theano.shared(y)

        with pm.Model() as self.model:
            # http://www.stat.columbia.edu/~gelman/research/published/taumain.pdf
            p_variation = pm.Uniform("p_variation", 0, 1000)
            mu_variation = pm.Uniform("mu_variation", 0, 1000)
            sigma_variation = pm.Uniform("sigma_variation", 0, 1000)

            # Shared Priors on height (p), timing (mu) and acceleration (sigma) of time series.
            shared_p = hierarchical_normal("shared_p", intra_group_variance=p_variation, shape=self.k)
            shared_mu = hierarchical_normal("shared_mu", mu=19, intra_group_variance=mu_variation, shape=self.k)
            shared_sigma = hierarchical_normal("shared_sigma", mu=3, intra_group_variance=sigma_variation, shape=self.k)

            p_err = pm.HalfCauchy("p_err", 25)
            mu_err = pm.HalfCauchy("mu_err", 25)
            sigma_err = pm.HalfCauchy("sigma_err", 25)

            # Cap deaths/million at 500
            p = pm.Bound(pm.Lognormal, upper=self.p_upper_bound)("p", shared_p, sigma=p_err, shape=self.k)
            mu = pm.Bound(pm.Lognormal, upper=self.mu_upper_bound, lower=self.mu_lower_bound)(
                "mu", mu=shared_mu, sigma=mu_err, shape=self.k
            )
            sigma = pm.Lognormal("sigma", shared_sigma, sigma=sigma_err, shape=self.k)

            yhat = pm.Deterministic("yhat", p[self.ids] * (normal((self.times - mu[self.ids]) / sigma[self.ids])))

            pm.Poisson("likelihood", mu=yhat, observed=self.Y)

            self.trace = pm.sample(progressbar=self.progressbar)

        return self

    def sample(self, ids, times):
        """ Sample from Posterior to predict deaths/million
        Args:
            ids (np.array, int): The group id's for every observation
            time (array): The times for every obsevation
        Returns:
            pymc3 trace
        """
        self.ids.set_value(ids)
        self.times.set_value(times)
        return pm.sample_posterior_predictive(
            self.trace, var_names=["yhat"], model=self.model, samples=1000, progressbar=self.progressbar
        )

    def predict(self, ids, times, return_std=False, alpha=0.05):
        """ Predict deaths/million given group id's and observation times
        Args:
            ids (np.array, int): The group id's for every observation
            time (array): The times for every obsevation
        Returns:
            yhat (np.array): predicted deaths/million
            OR 
            yhat, ylower, yupper (np.array): predicted deaths/million
        """
        ppc = self.sample(ids, times)
        yhat = np.quantile(ppc["yhat"].T, 0.5, axis=1)
        if return_std:
            return yhat, np.quantile(ppc["yhat"].T, alpha, axis=1), np.quantile(ppc["yhat"].T, alpha, axis=1)
        else:
            return yhat
