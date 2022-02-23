import importlib
import numpy as np


def MirrorProx_optimize(
  prob,
  K=None, z_0=None, stepsize=None,
  strongly_monotone=False, balance=False,
  log_freq=None, log_prefix=None, log_init=True, print_freq=None,):
  """
  Algorithm: Mirror-Prox for smooth convex-concave minmax problems

  [Nem]: Nemirovski, 
  PROX-METHOD WITH RATE OF CONVERGENCE O(1/T) FOR VARIATIONAL 
  INEQUALITIES WITH LIPSCHITZ CONTINUOUS MONOTONE OPERATORS 
  AND SMOOTH CONVEX-CONCAVE SADDLE POINT PROBLEMS

  Minimize:
  f(x) = \\min_x \\max_y g(x, y)
  Maximize:
  h(y) = \\min_x g(x, y)

  Solved as a convex-concave smooth-minimax problem

  Args:
    prob: FiniteMinimaxProb
    K: None or int
    z_0: None or np.array([m])
    stepsize: None or float
    strongly_monotone: None or bool
    balance: None or bool
    log_freq: None or int
    log_prefix: None or str
    log_init: None or bool
    print_freq: None of int

  Returns:
    output: dict

  """
  if z_0 is None:
    x_0, y_0 = np.zeros([prob.dx]), np.zeros([prob.dy])
  else:
    x_0, y_0 = np.array(z_0[0]), np.array(z_0[1])
  x_k, y_k = x_0, y_0

  if log_prefix is None:
    log_prefix = ''

  if stepsize is None:
    if not strongly_monotone:
      stepsize_x = 1./2/prob.L # L2 norm 
    elif not balance:
      stepsize_x = 1./4/prob.L # L2 norm Paul Tseng 1995, Mokhtari et al., 2020, Azizian et al., 2020
    else:
      kappa_balanced = prob.Lx/prob.mux + prob.Lxy/((prob.mux*prob.muy)**0.5) + prob.Ly/prob.muy
      stepsize_x = 1./4/kappa_balanced
  else:
    stepsize_x = stepsize
  stepsize_y = stepsize_x

  k_list = []

  xk_list = []
  yk_list = []
  xk_avg_list = []
  yk_avg_list = []
  txk_list = []
  tyk_list = []
  txk_avg_list = []
  tyk_avg_list = []

  k = 0

  xk_avg, yk_avg = np.zeros_like(x_k), np.zeros_like(y_k)
  txk_avg, tyk_avg = np.zeros_like(x_k), np.zeros_like(y_k)

  metric_dict, metrics_string = prob.metrics(x_k, y_k)  
  output = {}
  for key, value in metric_dict.items():
    output[key] = []
    output['avg_{}'.format(key)] = []
    output['t_{}'.format(key)] = []
    output['tavg_{}'.format(key)] = []

  grad_scale_x = prob.mux if balance else 1.0
  grad_scale_y = prob.muy if balance else 1.0

  while k < K:
    z_k = (x_k, y_k)
    tx_kplus1 = prob.proj_x(x_k - stepsize_x*prob.grad(z_k)[0]/grad_scale_x)
    ty_kplus1 = prob.proj_y(y_k - stepsize_y*prob.grad(z_k)[1]/grad_scale_y)

    tz_kplus1 = (tx_kplus1, ty_kplus1)
    x_kplus1 = prob.proj_x(x_k - stepsize_x*prob.grad(tz_kplus1)[0]/grad_scale_x)
    y_kplus1 = prob.proj_y(y_k - stepsize_y*prob.grad(tz_kplus1)[1]/grad_scale_y)

    xk_avg, yk_avg = (xk_avg*((k)/(k+1)) + x_k/(k+1)), (yk_avg*((k)/(k+1)) + y_k/(k+1))
    txk_avg, tyk_avg = (txk_avg*(k/(k+1)) + tx_kplus1/(k+1)), (tyk_avg*(k/(k+1)) + ty_kplus1/(k+1))

    if log_freq is not None and ((log_init and k==0) or k%log_freq == 0):
      k_list.append(k+1)

      xk_list.append(x_k)
      yk_list.append(y_k)
      xk_avg_list.append(xk_avg)
      yk_avg_list.append(yk_avg)
      txk_list.append(tx_kplus1)
      tyk_list.append(ty_kplus1)
      txk_avg_list.append(txk_avg)
      tyk_avg_list.append(tyk_avg)

      metric_dict, metrics_string = prob.metrics(x_k, y_k)
      avg_metric_dict, avg_metrics_string = prob.metrics(xk_avg, yk_avg)
      tmetric_dict, tmetrics_string = prob.metrics(tx_kplus1, ty_kplus1)
      avg_tmetric_dict, avg_tmetrics_string = prob.metrics(txk_avg, tyk_avg)
    
      for key, value in metric_dict.items():
        output[key].append(value)
        output['avg_{}'.format(key)].append(avg_metric_dict[key])
        output['t_{}'.format(key)].append(tmetric_dict[key])
        output['tavg_{}'.format(key)].append(avg_tmetric_dict[key])

      if print_freq is not None and (k == 0 or k%print_freq == 0):
        log_string = (
          '{}k={},x_k={},y_k={};{};avg:{};t{};tavg:{}'.format(
          log_prefix, k, x_k[:][:3], y_k[:][:3], metrics_string, avg_metrics_string,
          tmetrics_string, avg_tmetrics_string))
        print('{}'.format(log_string))

    x_k, y_k = x_kplus1, y_kplus1
    k += 1

  output['k_list'] = k_list
  output['xk_list'] = xk_list
  output['yk_list'] = yk_list
  output['xk_avg_list'] = xk_avg_list
  output['yk_avg_list'] = yk_avg_list
  output['txk_list'] = txk_list
  output['tyk_list'] = tyk_list
  output['txk_avg_list'] = txk_avg_list
  output['tyk_avg_list'] = tyk_avg_list

  return output


def RelLipMirrorProxSM_optimize(
  prob,
  K=None, z_0=None, stepsize=None,
  log_freq=None, log_prefix=None, log_init=True, print_freq=None,):
  """
  Algorithm: Mirror-Prox for smooth convex-concave minmax problems

  [Algorithm 3, CST]: Cohen, Sidford, Tian 
  Relative Lipschitzness in Extragradient Methods and a Direct Recipe for Acceleration

  Minimize:
  f(x) = \\min_x \\max_y g(x, y)
  Maximize:
  h(y) = \\min_x g(x, y)

  Solved as a strongly-convex--strongly-concave smooth-minimax problem

  Args:
    prob: MinimaxProb
    K: None or int
    z_0: None or np.array([m])
    stepsize: None or float
    log_freq: None or int
    log_prefix: None or str
    log_init: None or bool
    print_freq: None of int

  Returns:
    output: dict

  """
  if z_0 is None:
    x_0, y_0 = np.zeros([prob.dx]), np.zeros([prob.dy])
  else:
    x_0, y_0 = np.array(z_0[0]), np.array(z_0[1])
  x_k, y_k = x_0, y_0

  if log_prefix is None:
    log_prefix = ''

  if stepsize is None:
    kappa_rellip = prob.Ly/prob.mux + prob.Lxy/((prob.mux*prob.muy)**0.5) + prob.Ly/prob.muy
    stepsize_x = 1./kappa_rellip 
  else:
    stepsize_x = stepsize
  stepsize_y = stepsize_x

  k_list = []

  xk_list = []
  yk_list = []
  xk_avg_list = []
  yk_avg_list = []
  txk_list = []
  tyk_list = []
  txk_avg_list = []
  tyk_avg_list = []

  k = 0

  xk_avg, yk_avg = np.zeros_like(x_k), np.zeros_like(y_k)
  txk_avg, tyk_avg = np.zeros_like(x_k), np.zeros_like(y_k)

  metric_dict, metrics_string = prob.metrics(x_k, y_k)  
  output = {}
  for key, value in metric_dict.items():
    output[key] = []
    output['avg_{}'.format(key)] = []
    output['t_{}'.format(key)] = []
    output['tavg_{}'.format(key)] = []

  while k < K:
    z_k = (x_k, y_k)
    tx_kplus1 = prob.proj_x(x_k - stepsize_x*prob.grad(z_k)[0]/prob.mux)
    ty_kplus1 = prob.proj_y(y_k - stepsize_y*prob.grad(z_k)[1]/prob.mux)

    tz_kplus1 = (tx_kplus1, ty_kplus1)
    x_kplus1 = prob.proj_x((x_k + stepsize_x*tx_kplus1 - stepsize_x*prob.grad(tz_kplus1)[0]/prob.mux)/(1+stepsize_x))
    y_kplus1 = prob.proj_y((y_k + stepsize_y*ty_kplus1 - stepsize_y*prob.grad(tz_kplus1)[1]/prob.muy)/(1+stepsize_y))

    xk_avg, yk_avg = (xk_avg*((k)/(k+1)) + x_k/(k+1)), (yk_avg*((k)/(k+1)) + y_k/(k+1))
    txk_avg, tyk_avg = (txk_avg*(k/(k+1)) + tx_kplus1/(k+1)), (tyk_avg*(k/(k+1)) + ty_kplus1/(k+1))

    if log_freq is not None and ((log_init and k==0) or k%log_freq == 0):
      k_list.append(k+1)

      xk_list.append(x_k)
      yk_list.append(y_k)
      xk_avg_list.append(xk_avg)
      yk_avg_list.append(yk_avg)
      txk_list.append(tx_kplus1)
      tyk_list.append(ty_kplus1)
      txk_avg_list.append(txk_avg)
      tyk_avg_list.append(tyk_avg)

      metric_dict, metrics_string = prob.metrics(x_k, y_k)
      avg_metric_dict, avg_metrics_string = prob.metrics(xk_avg, yk_avg)
      tmetric_dict, tmetrics_string = prob.metrics(tx_kplus1, ty_kplus1)
      avg_tmetric_dict, avg_tmetrics_string = prob.metrics(txk_avg, tyk_avg)
    
      for key, value in metric_dict.items():
        output[key].append(value)
        output['avg_{}'.format(key)].append(avg_metric_dict[key])
        output['t_{}'.format(key)].append(tmetric_dict[key])
        output['tavg_{}'.format(key)].append(avg_tmetric_dict[key])

      if print_freq is not None and (k == 0 or k%print_freq == 0):
        log_string = (
          '{}k={},x_k={},y_k={};{};avg:{};t{};tavg:{}'.format(
          log_prefix, k, x_k[:][:3], y_k[:][:3], metrics_string, avg_metrics_string,
          tmetrics_string, avg_tmetrics_string))
        print('{}'.format(log_string))

    x_k, y_k = x_kplus1, y_kplus1
    k += 1

  output['k_list'] = k_list
  output['xk_list'] = xk_list
  output['yk_list'] = yk_list
  output['xk_avg_list'] = xk_avg_list
  output['yk_avg_list'] = yk_avg_list
  output['txk_list'] = txk_list
  output['tyk_list'] = tyk_list
  output['txk_avg_list'] = txk_avg_list
  output['tyk_avg_list'] = tyk_avg_list

  return output