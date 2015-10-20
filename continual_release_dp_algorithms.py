from __future__ import division
__author__ = 'yedtoss'
"""
Contains algorithms that can be used to play
a game in a differentially private manner or not
"""

import numpy as np
from scipy.stats import laplace
import settings
import utility


class BinaryMechanism(object):
    """
    This class implements the binary mechanism (Algorithm 2) presented in (Private and Continual Release of Statistics)
    https://eprint.iacr.org/2010/076.pdf
    """

    def __init__(self, horizon=settings.MAX_HORIZON, epsilon=settings.EPSILON, delta=settings.DELTA):
        """
        This method is a constructor for the BinaryMechanism class.
        It is automatically called when declaring an instance of this class.
        horizon: integer > 1, This is the horizon T of the experiment or the total number of yes/no answers.
                 It is Required. Default to settings.MAX_HORIZON
        epsilon: real > 0, This is the targeted overall privacy epsilon.
                 It is required. Default to settings.EPSILON
        delta: real > 0, 1-delta represents the probability of making a mistake.
                 Not required and not used by the algorithm. Default to settings.DELTA
        """
        self.horizon = horizon
        self.alpha = np.zeros((np.ceil(np.log2(self.horizon)),))  # Will contain true p-sums
        self.alpha_hat = np.zeros((len(self.alpha),))  # Will contain noisy p-sums
        self.epsilon = epsilon
        self.delta = delta
        self.epsilon_prime = self.epsilon/np.log(self.horizon)  # This is the scale of the noise of the Laplace dist
        self.t = 0  # This  contains the current time step
        self.num_observations = 0  # Contains the actual number of observed statistics
        self.distribution = laplace(loc=0, scale=1./self.epsilon_prime) # Laplace distribution of the noise to add
        # To delete just to check adding reward at each step or at logarithm number of steps
        self.rewards = []

    def observe(self, reward, time_step=1, is_observed=True):
        """
        This method is expected to be called every time a new observation has come.
        For example, every time a new answer (yes/no) is given.

        @param reward: real, 0 or 1. The reward is the value of the observation. It can be 1 or 0.

        @param time_step: Not used, not required. It is supposed to contain the current time step. But the algorithm
                   currently keeps track of that internally.

        @param is_observed:  Boolean, True or False. Indicate whether or not the statistic is
                                                     a real observation or a fake one.
                      Default to True.

        @Return nothing
        """

        # Update the time step
        self.t += 1

        # Update the number of observations
        if is_observed:
            self.num_observations += 1

        # After expressing t in binary form, i is the minimum digit not equal to 0
        i = int(np.log2((self.t & ((~self.t) + 1))))

        # Update each relevant p-sums with the newly received reward
        self.alpha[i] = reward
        for j in xrange(i):
            self.alpha[i] += self.alpha[j]

        # Initialize the noisy p-sums
        for j in xrange(i-1):
            self.alpha[j] = 0
            self.alpha_hat[j] = 0

        # Update the noisy p-sums by adding Laplace noise
        self.alpha_hat[i] = self.alpha[i] + self.distribution.rvs()

    def sum(self):
        """
        This method return the sum of the observations in a differentially private manner
        @return the sum
        """
        total = 0.  # Will contain the sum

        # Loop trough the relevant p-sum to return the current sum
        for j in xrange(len(self.alpha_hat)):
            if bool((self.t >> j) & 1):
                total += self.alpha_hat[j]

        return total

    def mean(self):
        """
        This method return the mean of the observations in a differentially private manner

        @return the mean
        """
        return self.sum()/self.num_observations


class LogarithmMechanism(object):
    """
    This class implements the Logarithm mechanism (Algorithm 3) presented in (Private and Continual Release of Statistics)
    https://eprint.iacr.org/2010/076.pdf
    """

    def __init__(self, horizon=settings.MAX_HORIZON, epsilon=settings.EPSILON, delta=settings.DELTA):
        """
        This method is a constructor for the LogarithmMechanism class.
        It is automatically called when declaring an instance of this class.
        horizon: integer > 1, This is the horizon T of the experiment or the total number of yes/no answers.
                 It is not Required and not used by the algorithm. Default to settings.MAX_HORIZON
        epsilon: real > 0, This is the targeted overall privacy epsilon.
                 It is required. Default to settings.EPSILON
        delta: real > 0, 1-delta represents the probability of making a mistake.
                 Not required and not used by the algorithm. Default to settings.DELTA
        """
        self.epsilon = epsilon
        self.delta = delta
        self.t = 0  # Contain the time step t
        self.num_observations = 0  # Contains the actual number of observed statistics
        self.beta = 0.  # Contain the current noisy sum
        self.distribution = laplace(loc=0, scale=1./self.epsilon)  # Laplace distribution of the noise to add

    def observe(self, reward, time_step=1, is_observed=True):
        """
        This method is expected to be called every time a new observation has come.
        For example, every time a new answer (yes/no) is given.

        @param reward: real, 0 or 1. The reward is the value of the observation. It can be 1 or 0.

        @param time_step: Not used, not required. It is supposed to contain the current time step. But the algorithm
                   currently keeps track of that internally.

        @param is_observed:  Boolean, True or False. Indicate whether or not the statistic is
                                                     a real observation or a fake one.
                      Default to True.

        @Return nothing
        """

        self.beta += reward  # Update the current sum
        self.t += 1  # Update the current time step

        # Update the number of observations
        if is_observed:
            self.num_observations += 1

        # If the time step is a power of 2, add the laplace noise to the current sum
        if utility.is_power2(self.t):
            self.beta += self.distribution.rvs()

    def sum(self):
        """
        This method return the sum of the observations in a differentially private manner
        @return the sum
        """
        return self.beta

    def mean(self):
        """
        This method return the mean of the observations in a differentially private manner

        @return the mean
        """
        return self.sum()/self.num_observations


class HybridMechanism(object):
    """
    This class implements the Hybrid mechanism (Algorithm 4) presented in (Private and Continual Release of Statistics)
    https://eprint.iacr.org/2010/076.pdf.
    It is a slightly improved version from Algorithm 4.
    """

    def __init__(self, horizon=settings.MAX_HORIZON, epsilon=settings.EPSILON, delta=settings.DELTA):
        """
        This method is a constructor for the HybridMechanism class.
        It is automatically called when declaring an instance of this class.
        horizon: integer > 1, This is the horizon T of the experiment or the total number of yes/no answers.
                 It is not Required and not used by the algorithm. Default to settings.MAX_HORIZON
        epsilon: real > 0, This is the targeted overall privacy epsilon.
                 It is required. Default to settings.EPSILON
        delta: real > 0, 1-delta represents the probability of making a mistake.
                 Not required and not used by the algorithm. Default to settings.DELTA
        """
        self.epsilon = epsilon
        self.delta = delta
        self.t = 0  # Contain the time step t
        self.num_observations = 0  # Contains the actual number of observed statistics

        # This contains the binary mechanism used by this algo. We used epsilon and not epsilon/2.
        self.bm = BinaryMechanism(horizon, epsilon, delta)

        self.lm = LogarithmMechanism(horizon, epsilon, delta)  # This contains the logarithm mechanism used by this algo
        self.T = 1  # Will contain the horizon to be passed to the binary mechanism

        # These contain resp the mean and number of observations of the logarithm mechanisms
        # TODO removed this and used the method sum() instead; from the logarithm mechanism
        self.logarithm_mechanism_mean = 0.
        self.logarithm_mechanism_observations = 0

    def observe(self, reward, time_step=1, is_observed=True):
        """
        This method is expected to be called every time a new observation has come.
        For example, every time a new answer (yes/no) is given.

        @param reward: real, 0 or 1. The reward is the value of the observation. It can be 1 or 0.

        @param time_step: Not used, not required. It is supposed to contain the current time step. But the algorithm
                   currently keeps track of that internally.

        @param is_observed:  Boolean, True or False. Indicate whether or not the statistic is
                                                     a real observation or a fake one.
                      Default to True.

        @Return nothing
        """

        self.lm.observe(reward, time_step)  # Update the logarithm mechanism
        self.t += 1  # Update the current time step

        # Update the number of observations
        if is_observed:
            self.num_observations += 1

        if self.t > 1:

            # If t is a power of 2, create a new instance of the binary mechanism and update the mean of the logarithm
            if utility.is_power2(self.t):
                self.T = self.t
                # We used epsilon and not epsilon/2.
                self.bm = BinaryMechanism(self.T, self.epsilon, self.delta)
                self.logarithm_mechanism_mean = self.lm.mean()
                self.logarithm_mechanism_observations = self.lm.num_observations

            # If t is not a power of 2, update the binary mechanism
            else:
                self.bm.observe(reward, time_step)

    def sum(self):
        """
        This method return the sum of the observations in a differentially private manner

        @return the sum
        """
        # If t is a power of 2, use only the logarithm mechanism
        if utility.is_power2(self.t):
            return self.lm.sum()
        # If t is not a power of 2, use both the logarithm mechanism and the binary mechanism
        else:
            return self.lm.sum() + self.bm.sum()

    def mean(self):
        """
        This method return the mean of the observations in a differentially private manner

        @return the mean
        """

        # If t is a power of 2, use only the logarithm mechanism
        if utility.is_power2(self.t):
            return self.lm.mean()

        # If t is not a power of 2, use both the logarithm mechanism and the binary mechanism
        else:
            return (self.logarithm_mechanism_mean*self.logarithm_mechanism_observations + self.bm.mean()*self.bm
                    .num_observations)/self.num_observations



