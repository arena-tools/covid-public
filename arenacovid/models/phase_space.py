import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Grid of Parameters for model outputs:
    taus: array of available default smoothing lengths
    b_defaults: array of available default deceleration rates (to
            be used when growth rate has recently accelerated)
"""
taus = np.arange(1, 11)
b_defaults = [-0.06, -0.05, -0.04]


class PhaseFitter:
    def __init__(
        self, tau=1, transform=lambda x: np.log(1 + x), inv_transform=lambda x: np.exp(x) - 1, population=1, b_default=-0.06
    ):
        """ Phase Space Fitter: Model transformed observations x and xdot 
        Position:
            x_t = transform(observations...) 
        Velocity:
            xdot_t = d (x_t) / dt  
        Fit with rolling formula: xdot_t = a_t + b_t * x_t
        Args:
            tau (float): Half-life exponential smoothing length for observations. 
                        As tau->0 only most recent data/trends are used.
            transform (Callable): transorm function for observations
            inv_transform (Callable): inverse transorm function for observations
            population (int): population
            b_default (float): default value for the deceleration rate / slope of
                                xdot(t) = a + b*x(t), where x are the transformed 
                                observations. 
        """
        self.transform = transform
        self.inv_transform = inv_transform
        self.population = 1
        self.tau = tau
        self.imax = 70
        self.b_default = b_default

    def fit(self, y: pd.Series):
        """ Fit the given observation series
        Args:
            y (pd.Series, float): Sorted sequential observations. In this case
                        cumulative COVID-19 cases
        Returns:
            self
        """
        assert y.index.name == "date"
        for tau in range(0, 7):
            tau_smoothing_length = self.tau + tau

            # Take the log of observations - default log
            self.x = y.apply(self.transform)

            # First Derivative of Transformated observations
            # d log (y) / dt
            self.xdot = self.x.diff(1)
            self.x = self.x.loc[~self.xdot.isnull()]
            self.xdot = self.xdot.loc[~self.xdot.isnull()]
            self.xdot_smooth = self.xdot.ewm(halflife=tau_smoothing_length).mean()
            self.x_smooth = self.x.ewm(halflife=tau_smoothing_length).mean()

            # Rolling Slope in State space x, xdot
            self.bvals = (
                (
                    self.xdot.ewm(halflife=tau_smoothing_length).mean().diff(1)
                    / self.x.ewm(halflife=tau_smoothing_length).mean().diff(1)
                )
                .dropna()
                .ewm(halflife=tau_smoothing_length)
                .mean()
            )

            # If valid decelaration rate (<0), exit
            if self.bvals.iloc[-1] < self.b_default:
                break

        # Most recent level and slope in state space
        self.b = min(self.bvals.iloc[-1], self.b_default)
        self.a = self.xdot.iloc[-1] - self.b * self.x.iloc[-1]
        return self

    @property
    def size(self):
        """ Eventual Size of the infected / observed population """
        return np.exp(-self.a / self.b)

    def predict(self, steps=1):
        """ Predict the number of observed cases 
        Args:
            steps (int): Number of steps in the future to predict (from most
                        recent observation)
        Returns:
            pd.Series of predictions
        """
        date = self.xdot.index.values[-1]
        dates = pd.date_range(date, date + pd.Timedelta("1d") * (steps - 1), freq="1d")
        x = pd.Series(np.zeros(steps), index=dates)
        xdot = pd.Series(np.zeros(steps), index=dates)
        x.loc[date] = self.x.iloc[-1]
        xdot.loc[date] = self.xdot.iloc[-1]
        last_i = date
        for i in x.index[1:]:
            xdot.loc[i] = self.a + self.b * x.loc[last_i]
            x.loc[i] = x.loc[last_i] + xdot.loc[last_i]
            last_i = i
        return self.inv_transform(x)

    def plot(self):
        """ Plot the fit and observations
        Args:
            None
        Returns:
            matplotlib ax
        """
        xhat = np.linspace(self.x.max(), np.log(self.size), 10)
        fig = plt.figure(figsize=(18, 3))
        plt.plot(self.x, self.xdot, marker=">", label="observed")
        plt.plot(self.x_smooth, self.xdot_smooth, label="state")
        plt.plot(xhat, self.a + self.b * xhat, marker="+", label="projected path")
        plt.legend()
        plt.ylabel("xdot")
        plt.xlabel("x")
        plt.ylim(0, 1)
        plt.show()
