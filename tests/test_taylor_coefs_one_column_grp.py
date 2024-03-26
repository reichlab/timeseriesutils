import pytest
import pandas as pd
import numpy as np
from timeseriesutils.featurize import taylor_coefs_one_column_grp


def mean_fn(t, a, b, phi=10, d=0):
  '''
  mean function, or its derivative, used for simulating data
  the function is mu(t) = a * cos(t/phi) + b * sin(t/phi)
  the first derivative is mu'(t) = -a / phi * sin(t/phi) + b / phi * cos(t/phi)
  the second derivative is mu''(t) = -a / phi**2 * cos(t/phi) - b / phi**2 * sin(t/phi)
  
  Parameters
  ----------
  t: numpy array of time points at which to evaluate the function
  a, b: coefficients for cos and sin terms
  phi: scaling factor applied to t
  d: integer derivative to evaluate: 0, 1, or 2
  
  Returns
  -------
  numpy array of same length as t with function values
  '''
  if d == 0:
    return a * np.cos(t / phi) + b * np.sin(t / phi)
  elif d == 1:
    return -a / phi * np.sin(t / phi) + b / phi * np.cos(t / phi)
  elif d == 2:
    return -a / (phi**2) * np.cos(t / phi) - b / (phi**2) * np.sin(t / phi)


def one_grp_ex(a=1, b=0.2):
  '''
  Generate example data for one "group", e.g. one location
  
  Parameters
  ----------
  a, b: cos and sin coefficients
  
  Returns
  -------
  Pandas data frame with columns `t` with a time index and
  `y` with response variable values
  '''
  rng = np.random.default_rng(12345)
  t = np.arange(50)
  return pd.DataFrame({
    't': t,
    'y': rng.normal(mean_fn(t, a=a, b=b), scale=0.1)
  })


def test_taylor_coefs_one_column_grp_trailing():
  df = taylor_coefs_one_column_grp(
    data=one_grp_ex(),
    c='y',
    taylor_degree=2,
    window_size=14,
    window_align='trailing',
    ew_span=None,
    fill_edges=False)
  
  mean = mean_fn(t=df['t'], a=1, b=0.2, d=0)
  mean_d1 = mean_fn(t=df['t'], a=1, b=0.2, d=1)
  mean_d2 = mean_fn(t=df['t'], a=1, b=0.2, d=2)
  
  new_cols = [f'y_taylor_d2_c{c}_w14t_sNone' for c in range(3)]
  
  assert list(df.columns) == ['t', 'y'] + new_cols
  assert np.all(np.isnan(df[new_cols].iloc[:14]))
  assert np.all(~np.isnan(df[new_cols].iloc[14:]))
  assert df[new_cols[0]].iloc[14:].values == pytest.approx(mean[14:], abs = 0.2)
  assert df[new_cols[1]].iloc[14:].values == pytest.approx(mean_d1[14:], abs = 0.1)
  assert df[new_cols[2]].iloc[14:].values == pytest.approx(mean_d2[14:], abs = 0.02)
