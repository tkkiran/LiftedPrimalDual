import numpy as np


def l2_norm(x):
  return np.sqrt((np.array(x)**2).sum())


def inner_prod_sum(g_i, x_i, axis=1):
  """
  sum over i of Inner product of g_i and x_i
  """
  return (g_i*x_i).sum(axis=axis)