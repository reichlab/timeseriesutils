import warnings

import numpy as np
import pandas as pd

from timeseriesutils.featurize import taylor_coefs_one_column_grp, windowed_taylor_coefs


def one_grp_ex(location, a=1, b=0.2, phi=10):
  '''
  Generate example data for one "group", e.g. one location

  Parameters
  ----------
  location: value used to tag all rows with a group identifier
  a, b: coefficients for cos and sin terms
  phi: scaling factor applied to t

  Returns
  -------
  Pandas data frame with columns `location`, `t`, and `y`
  '''
  rng = np.random.default_rng(12345)
  t = np.arange(50)
  y = rng.normal(a * np.cos(t / phi) + b * np.sin(t / phi), scale=0.1)
  return pd.DataFrame({'location': location, 't': t, 'y': y})


def test_windowed_taylor_coefs_group_columns_matches_ungrouped():
  '''
  windowed_taylor_coefs with group_columns set should give the same
  per-group results as calling taylor_coefs_one_column_grp directly on
  each group's data, even when the groups' rows are interleaved rather
  than contiguous in the input data frame. This also guards against the
  pandas groupby().apply() deprecation around inclusion of grouping
  columns in the values passed to the applied function.
  '''
  df_a = one_grp_ex(location='a', a=1, b=0.2)
  df_b = one_grp_ex(location='b', a=-2, b=0.5)

  # interleave rows from the two groups instead of stacking them
  # contiguously, to exercise index alignment after the groupby
  df = pd.concat([df_a, df_b]).sort_values('t', kind='stable') \
    .reset_index(drop=True)

  with warnings.catch_warnings():
    warnings.simplefilter('error')
    grouped, feature_names = windowed_taylor_coefs(
      data=df.copy(),
      columns='y',
      group_columns=['location'],
      taylor_degree=2,
      window_size=14,
      window_align='trailing',
      ew_span=None,
      fill_edges=False)

  assert set(grouped['location']) == {'a', 'b'}
  assert grouped.shape[0] == df.shape[0]

  expected_a = taylor_coefs_one_column_grp(
    data=df_a.copy(), c='y', taylor_degree=2, window_size=14,
    window_align='trailing', ew_span=None, fill_edges=False)
  expected_b = taylor_coefs_one_column_grp(
    data=df_b.copy(), c='y', taylor_degree=2, window_size=14,
    window_align='trailing', ew_span=None, fill_edges=False)

  for location, expected in [('a', expected_a), ('b', expected_b)]:
    actual = grouped[grouped['location'] == location] \
      .sort_values('t').reset_index(drop=True)
    expected = expected.sort_values('t').reset_index(drop=True)
    pd.testing.assert_frame_equal(
      actual[feature_names], expected[feature_names])
    # every row should keep its own group's `y` value, not a value that
    # leaked in from another group's rows
    np.testing.assert_array_equal(actual['y'].values, expected['y'].values)
