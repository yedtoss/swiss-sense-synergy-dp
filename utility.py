from __future__ import division
__author__ = 'yedtoss'
import numpy as np
from scipy.special import zeta

"""
Contains utility functions
"""


def is_power2(num):
    """
    Whether or not an integer is a power of 2
    @param num >0 the integer
    @return True if a power of 2, False otherwise
    If num is 0, we return False instead of True
    """
    return num != 0 and ((num & (num - 1)) == 0)
