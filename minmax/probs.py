import importlib
import numpy as np
import scipy

from . import utils

importlib.reload(utils)


class SmoothMinimaxProb(object):
  """
  Smooth Minimax:
  \\min_x \\max_y g(x, y)

  g is L-smooth
  sigma_x-strongly convex in x
  sigma_y-strongly concave in y
  """
  def __init__(self):
    pass
  
  def metrics(self, x, y):
    metrics_dict = {}
    z = (x, y)

    if self.primal_func is not None and self.dual_func is not None:
      p_func = self.primal_func(x, y=y)
      d_func = self.dual_func(y, x=x, func_lb=p_func)
      gap = p_func - d_func
      try:
        assert gap >= 0 or np.isclose(gap, 0.)
      except:
        print('p_func={}, d_func={},gap={}'.format(p_func, d_func, gap))
        raise ValueError('Gap is negative!')
      metrics_dict['gap'] = gap
    
    grad_x, grad_y = self.grad(z)
    if self._proj_x is not None:
      x = z[0]
      stepsize_x = 1/self.L/10
      grad_x = (x - self.proj_x(x - stepsize_x*grad_x))/stepsize_x
    if self._proj_y is not None:
      y = z[1]
      stepsize_y = 1/self.L/10
      grad_y = (y - self.proj_y(y - stepsize_y*grad_y))/stepsize_y

    metrics_dict['grad_norm'] = ((grad_x**2).sum() + (grad_y**2).sum())**0.5
    metrics_dict['func'] = self.func(z)

    metrics_string = 'gap={:2.2g},|grad|={:2.2g},func={:.2g}'.format(
      metrics_dict['gap'], metrics_dict['grad_norm'], metrics_dict['func']
    )

    return metrics_dict, metrics_string


class QuadraticSeparableMinimaxProb(SmoothMinimaxProb):
  """
  Minimize x \\in \\reals^dx
  and
  Maximize y \\in \\reals^dy:
  :
  g(x, y) = f(x) + <y, Ax> - h(y)

  f(x) = 0.5 <x, Bx> + <b, x>
  h(y) = 0.5 <y, Cy> + <c, y>

  primal(x) = \\max_{y} g(x, y)
  dual(x) = \\min_{x} g(x, y)

  Attributes:
    dx: dimension of the x variable
    dy: dimension of the y variable

    A: np.array([dy, dx])
    B: np.array([dx, dx])
    b: np.array([dx])
    C: np.array([dy, dy])
    c: np.array([dy])
  """
  def __init__(
    self, A, B=None, C=None, b=None, c=None, 
    Dx=None, Dy=None,
    proj_x=None, proj_y=None, dual_func=None,
    ):
    """
    Args:
      A: np.array([dy, dx])
      B: None or np.array([dx, dx])
      b: None or np.array([dx])
      C: None or np.array([dy, dy])
      c: None or np.array([dy])
    """
    self.A = np.array(A)
    self.dy, self.dx = self.A.shape

    if B is not None:
      self.B = np.array(B)
    else:
      self.B = np.zeros([self.dx, self.dx])
    self.B = (self.B + self.B.T)/2

    if b is not None:
      self.b = np.array(b)
    else:
      self.b = np.zeros([self.dx])

    if C is not None:
      self.C = np.array(C)
    else:
      self.C = np.zeros([self.dy, self.dy])
    self.C = (self.C + self.C.T)/2

    if c is not None:
      self.c = np.array(c)
    else:
      self.c = np.zeros([self.dy])

    self._proj_x = proj_x
    self.proj_x = lambda x: x if proj_x is None else proj_x(x)
    self._proj_y = proj_y
    self.proj_y = lambda x: x if proj_y is None else proj_y(x)

    # Lxx =  - Lipschitz constant of grad_x w.r.t x
    # muxx = - Strong convexity constant of grad_x w.r.t x
    # Lxy =  - Lipschitz constant of grad_x w.r.t y
    # muxy = - "Strong convexity" constant of grad_x w.r.t y
    # Lyx =  - Lipschitz constant of grad_y w.r.t x
    # muyx = - "Strong concavity" constant of grad_y w.r.t x
    # Lyy = - Lipschitz constant of grad_y w.r.t y
    # muyy = - Strong concavity constant of grad_y w.r.t y
    eigvalsxx = np.linalg.eigvalsh(self.B)
    self.Lxx, self.muxx = eigvalsxx[-1], eigvalsxx[0]
    self.Lx, self.mux = self.Lxx, self.muxx
    eigvalsxy = scipy.linalg.svdvals(self.A)
    self.Lxy, self.muxy = eigvalsxy[0], eigvalsxy[-1]
    self.Lyx, self.muyx = self.Lxy, self.muxy
    eigvalsyy = np.linalg.eigvalsh(self.C)
    self.Lyy, self.muyy = eigvalsyy[-1], eigvalsyy[0]
    self.Ly, self.muy = self.Lyy, self.muyy

    self.L = max(self.Lxx, self.Lxy, self.Lyy)

    self.dual_func = self._dual_func if dual_func is None else dual_func

    self.counter = 0

    self.z_opt = None

  def f(self, x):
    """
    Computes the function value
    f(x) = 0.5 <x, Bx> + <b, x>
  
    Args:
      x: np.array([dx])

    Returns:
      f(x): real function value
    """
    _f = 0.2*utils.inner_prod_sum(self.B.dot(x), x, axis=-1)
    _f += utils.inner_prod_sum(self.b, x, axis=-1)

    return _f

  def h(self, y):
    """
    Computes the function value
    h(y) = 0.5 <y, Cy> + <c, y>
  
    Args:
      y: np.array([dy])

    Returns:
      h(y): real function value
    """
    _h = 0.5*utils.inner_prod_sum(self.C.dot(y), y, axis=-1)
    _h += utils.inner_prod_sum(self.c, y, axis=-1)

    return _h

  def func(self, z):
    """
    Computes the function value
    g(x, y) = f(x) + <y, Ax> - h(y)

    f(x) = 0.5 <x, Bx> + <b, x>
    h(y) = 0.5 <y, Cy> + <c, y>
  
    Args:
      z:(x,y) (np.array([dx]), np.array([dy]))

    Returns:
      g(z): real function value
    """
    x, y = z
    return self.f(x) + utils.inner_prod_sum(y, self.A.dot(x), axis=-1) - self.h(y)

  def grad_f(self, x):
    """
    Computes the gradient of the function
    f(x) = 0.5 <x, Bx> + <b, x>

    grad_x = grad_x f(x)

    Returns:
      grad f(x):grad_x gradient
    """
    grad_x = self.B.dot(x) + self.b

    return grad_x

  def grad_h(self, y):
    """
    Computes the gradient of the function
    h(y) = 0.5 <y, Cy> + <c, y>

    grad_y = grad_y h(y)

    Returns:
      grad h(y):grad_y gradient
    """
    grad_y = self.C.dot(y) + self.c

    return grad_y

  def grad_A(self, z):
    """
    Computes the gradient of the function
    <y, Ax>

    (grad_x, grad_y)
    grad_x = grad_x <y, Ax>
    grad_y = grad_y <y, Ax>

    Returns:
      grad g(x, y):(grad_x, grad_y) gradient
    """
    x, y = z

    grad_x = self.A.T.dot(y)
    grad_y = self.A.dot(x)

    return (grad_x, grad_y)

  def grad(self, z):
    """
    Computes the gradient of the function
    g(x, y) = f(x) + <y, Ax> - h(y)

    f(x) = 0.5 <x, Bx> + <b, x>
    h(y) = 0.5 <y, Cy> + <c, y>
    
    (grad_x, -grad_y)
    grad_x = grad_x g(x, y)
    grad_y = -grad_y g(x, y)

    Returns:
      grad g(x, y):(grad_x, -grad_y) gradient
    """
    x, y = z

    _grad_A = self.grad_A(z)
    grad_x = self.grad_f(x) + _grad_A[0]
    grad_y = _grad_A[1] - self.grad_h(y)

    return (grad_x, -grad_y)

  def primal_func(self, x, y=None):
    """
    Computes the function value
    f_max(x) = \max_y g(x, y) 
    
    g(x, y) = f(x) + <y, Ax> - h(y)

    f(x) = 0.5 <x, Bx> + <b, x>
    h(y) = 0.5 <y, Cy> + <c, y>

    Args:
      x: np.array([dx])

    Returns:
      f_max(x): real function value
    """
    matrix = self.C
    vector = self.A.dot(x) - self.c
    try:
      y_max = scipy.linalg.solve(matrix, vector, check_finite=True, assume_a='sym')
    except scipy.linalg.LinAlgError:
      y_max = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]
  
    return self.func((x, y_max))

  def _dual_func(self, y, x=None, func_lb=None):
    """
    Computes the function value
    h_min(y) = \min_x g(x, y) 
    
    g(x, y) = f(x) + <y, Ax> - h(y)

    f(x) = 0.5 <x, Bx> + <b, x>
    h(y) = 0.5 <y, Cy> + <c, y>

    Args:
      y: np.array([dy])

    Returns:
      h_min(y): real function value
    """
    matrix = self.B
    vector = -self.A.T.dot(y) - self.b
    try:
      x_min = scipy.linalg.solve(matrix, vector, check_finite=True, assume_a='sym')
    except scipy.linalg.LinAlgError:
      x_min = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]
    return self.func((x_min, y))

  def dist_squared(self, z):
    if self.mux != 0.0 and self.muy != 0.0:
      if self.z_opt is None:
        matrix = np.block([[self.B, self.A.T], [-self.A, self.C]])
        vector = np.block([-self.b,-self.c])
        try:
          self.z_opt = scipy.linalg.solve(matrix, vector, check_finite=True, assume_a='gen')
        except scipy.linalg.LinAlgError:
          self.z_opt = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]

      distance_squared = ((self.z_opt - np.block([z[0], z[1]]))**2).sum()
    else:
      distance_squared = np.inf
    
    return distance_squared

  def metrics(self, x, y):
    metrics_dict, metrics_string = super().metrics(x, y)

    if self.mux != 0.0 and self.muy != 0.0:
      metrics_dict['dist_sq'] = self.dist_squared((x, y))
      dist_sq = metrics_dict['dist_sq']
      metrics_string = (
        f'{metrics_string},dist_sq={dist_sq:.2g}')

    return metrics_dict, metrics_string

