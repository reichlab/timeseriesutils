import warnings

import numpy as np
import pandas as pd

from timeseriesutils.featurize import domfreq, domfreq_one_column_grp


def one_grp_ex(location, freq):
  '''
  Generate example sinusoidal data for one "group", e.g. one location

  Parameters
  ----------
  location: value used to tag all rows with a group identifier
  freq: dominant frequency of the generated signal

  Returns
  -------
  Pandas data frame with columns `location`, `t`, and `y`
  '''
  t = np.arange(60)
  y = np.sin(2 * np.pi * freq * t)
  return pd.DataFrame({'location': location, 't': t, 'y': y})


def test_domfreq_group_columns_matches_ungrouped():
  '''
  domfreq with group_columns set should give the same per-group results
  as calling domfreq_one_column_grp directly on each group's data, even
  when the groups' rows are interleaved rather than contiguous in the
  input data frame. This also guards against the pandas
  groupby().apply() deprecation around inclusion of grouping columns in
  the values passed to the applied function.
  '''
  df_a = one_grp_ex(location='a', freq=0.1)
  df_b = one_grp_ex(location='b', freq=0.25)

  # interleave rows from the two groups instead of stacking them
  # contiguously, to exercise index alignment after the groupby
  df = pd.concat([df_a, df_b]).sort_values('t', kind='stable') \
    .reset_index(drop=True)

  with warnings.catch_warnings():
    warnings.simplefilter('error')
    grouped, feature_names = domfreq(
      data=df.copy(),
      columns='y',
      group_columns=['location'],
      window_size=11,
      window_align='centered',
      n_domfreq=2)

  assert set(grouped['location']) == {'a', 'b'}
  assert grouped.shape[0] == df.shape[0]

  expected_a = domfreq_one_column_grp(
    data=df_a.copy(), c='y', window_size=11, window_align='centered',
    n_domfreq=2, fs=1.0, detrend='linear')
  expected_b = domfreq_one_column_grp(
    data=df_b.copy(), c='y', window_size=11, window_align='centered',
    n_domfreq=2, fs=1.0, detrend='linear')

  for location, expected in [('a', expected_a), ('b', expected_b)]:
    actual = grouped[grouped['location'] == location] \
      .sort_values('t').reset_index(drop=True)
    expected = expected.sort_values('t').reset_index(drop=True)
    pd.testing.assert_frame_equal(
      actual[feature_names], expected[feature_names])
    # every row should keep its own group's `y` value, not a value that
    # leaked in from another group's rows
    np.testing.assert_array_equal(actual['y'].values, expected['y'].values)
