from __future__ import division
__author__ = 'yedtoss'

# Import needed libraries
import settings
from continual_release_dp_algorithms import BinaryMechanism
from continual_release_dp_algorithms import HybridMechanism
from scipy.stats import bernoulli

"""
This contains simple test about how to use the DP algorithms
"""


if __name__ == "__main__":

    # Generate an experiment where each user answers a question randomly according to Bernoulli distribution
    # Print the true sum and means. Then, print the sum and mean returned by the Binary and Hybrid Mechanism

    num_users = settings.MAX_HORIZON  # Total number of users participating to the experiment
    distribution = bernoulli(p=0.5)  # Bernoulli distribution

    binaryMechanism = BinaryMechanism(horizon=num_users, epsilon=settings.EPSILON)  # Instantiate the binary mechanism
    hybridMechanism = HybridMechanism(epsilon=settings.EPSILON)  # Instantiate the hybrid mechanism
    true_sum = 0.

    for user in xrange(num_users):
        # Generate answer according to bernoulli
        answer = distribution.rvs()

        true_sum += answer  # Update the true sum
        binaryMechanism.observe(reward=answer)  # Update the binary mechanism with the new answer
        hybridMechanism.observe(reward=answer)  # Update the hybrid mechanism with the new answer


    # print results
    print('The true sum is ', true_sum)
    print('The sum returned by the binaryMechanism is', binaryMechanism.sum())
    print('The sum returned by the hybridMechanism is', hybridMechanism.sum())
    print('The privacy achieved by both mechanism is epsilon = ', settings.EPSILON)
    print('The total number of observations is ', num_users)







