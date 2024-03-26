from itertools import product
import math
import numpy as np
import pandas as pd
from scipy.signal import periodogram

def featurize_data(data, group_columns=None, features = []):
    """
    Augment a data frame with new columns containing values of features
    calculated based on existing columns.

    Parameters
    ----------
    data: data frame
        It has columns identifying groups of observations (e.g., locations with
        separate series per location) and a column with the target variable to
        summarize. This data frame needs to be sorted by the grouping and time
        variables in ascending order.
    group_columns: list of strings or `None`
        Names of columns in the data frame to group by. Grouping is done before
        calculating features.
    features: list of dictionaries
        List of features to calculate. Each dictionary should have a `fun` key
        with feature function name and an `args` key with parameter name and
        values of this function. Available feature functions are
        'rollmean', 'lag', and 'windowed_taylor_coefs'.
    
    Returns
    -------
    data: data frame
        The original `data` with new columns containing feature values and/or
        forecasting target values
    feature_names: list of strings
        List of feature column names that were created
    """
    # calculate features based on given parameters
    # and collect a list of feature names
    feature_names = list()
    for feature in features:
        args = feature['args']
        args['group_columns'] = group_columns
        fun = feature['fun']
        if type(fun) == str:
            fun = eval(fun)
        data, feature_names = data.pipe(fun,
                                        feature_names=feature_names,
                                        **args)
    
    return data, feature_names


def df_to_train_test_matrices(data, feature_names, target_name):
    """
    Convert a data frame containing columns of feature values and a prediction
    target to matrices `x_train_val` and `x_T` containing feature values and
    `y_train_val` with prediction targets. These are suitable for use in
    machine learning algorithms estimating a relationship between the features
    and the prediction target.
    
    Parameters
    ----------
    data: data frame
        The original `data` with new columns containing feature values and/or
        forecasting target values
    feature_names: list of strings
        List of feature column names
    target_name: string
        Forecast target column name
    
    Returns
    -------
    x_train_val: 3D tensor with shape (L, T, P)
        L is the number of location l and T is the number of time point t for which
        the full feature vector x_{l,t}, possibly including lagged covariate values,
        and the response y_{l,t}, corresponding to the target variable at time t+h,
        could be calculated. P is number of features.
        Each row is a vector x_{l,t} = [x_{l,t,1},...,x_{l,t,P}] of features for some pair
        (l, t) in the training set.
    y_train_val: 2D tensor with with shape (L, T)
        Each value is a forecast target variable value in the training set.
        y_{l, t} = z_{l, 1, t+h}
    x_T: 3D tensor with shape (L, T = 1, P)
        Each value is test set feature for each location at forecast date.
    """
    # extract the largest date
    T = max(data['date'])
    
    # create x_T using data with date = forecast_date (T)
    data_T = data.loc[data['date']== T,:]
    
    # x_T is (L, 1, P)
    x_T = np.expand_dims(data_T[feature_names].values, -2)
    
    # take out nans in data
    train_val = data.dropna()
    
    # reformat selected features
    x_train_val = train_val \
        .pivot(index = 'location', columns = 'date', values = feature_names) \
        .to_numpy()
    # shape is (L, T, P)
    x_train_val = x_train_val \
        .reshape((x_train_val.shape[0],
                    x_train_val.shape[1]//len(feature_names),
                    len(feature_names)),
                 order='F')
    
    # shape is (L, T, P)
    y_train_val = train_val \
        .pivot(index = 'location', columns = 'date', values = target_name) \
        .to_numpy()
    
    # convert everything to tensor
    x_train_val = tf.constant(x_train_val.astype('float64'))
    y_train_val = tf.constant(y_train_val.astype('float64'))
    x_T = tf.constant(x_T.astype('float64'))
    
    return x_train_val, y_train_val, x_T


def rollmean(data, columns, group_columns=None, feature_names=None,
                window_size=7, min_periods=None):
    """
    Calculate moving average of specified variables and store results in new
    columns
    
    Parameters
    ----------
    data: data frame
        It has columns identifying groups of observations (e.g., locations with
        separate series per location) and a column with the target variable to
        summarize. This data frame needs to be sorted by the grouping and time
        variables in ascending order.
    columns: string or list of strings
        Name of column(s) in the data frame with the variable(s) to featurize
    group_columns: list of strings or None
        Names of columns in the data frame to group by.
    feature_names: list of strings or None
        Running list of feature column names
    window_size: integer or list of integers
        Size of the sliding window over which we calculate moving average
    min_periods: integer or None
        Minimum number of observations in window required to have a value;
        otherwise, result is `np.nan`.
    
    Returns
    -------
    data: data frame
        Original data frame with an additional column for the moving average,
        named `f'{c}_roll_mean_{window_size}'` where `c` is an element of
        `columns`
    feature_names: list of strings
        Running list of feature column names, updated with the new column name
    """
    if group_columns is not None and len(group_columns) > 0:
        grouped_data = data.groupby(group_columns, as_index=False)
    else:
        grouped_data = data
    
    if feature_names is None:
        feature_names = [];
    
    if not isinstance(columns, list):
        columns = [columns]
    
    if not isinstance(window_size, list):
        window_size = [window_size]
    
    for c, w in product(columns, window_size):
        column_name = f'{c}_rollmean_w{str(w)}'
        feature_names.append(column_name)
        data[column_name] = grouped_data.rolling(w, min_periods=min_periods)[c] \
            .mean() \
            .values
    
    return data, feature_names


def lag(data, columns, group_columns=None, feature_names=None,
                  window_size=1, lags=None):
    """
    Calculate lagged values of target variables and store results in new columns
    
    Parameters
    ----------
    data: data frame
        It has columns identifying groups of observations (e.g., locations with
        separate series per location) and a column with the response variable to
        summarize. This data frame needs to be sorted by the grouping and time
        variables in ascending order.
    columns: string or list of strings
        Name of column(s) in the data frame with the variable(s) to featurize
    group_columns: list of strings or None
        Names of columns in the data frame to group by.
    feature_names: list of strings or None
        Running list of feature column names
    window_size: integer
        Time window to calculate lagged values for. All lags from 1 to
        `window_size` are calculated. Ignored if `lags` is not `None`.
    lags: list of integers
        List of lags to use.
    
    Returns
    -------
    data: data frame
        Original data frame with additional columns for lagged values, named
        `f'{c}_lag{lag}'` where `c` is an element of `columns` and `lag` is an
        element of `lags` if `lags` was provided or `[1, ..., window_size]`
        otherwise.
    feature_names: list of strings
        Running list of feature column names, updated with the new column names
    """
    if group_columns is not None and len(group_columns) > 0:
        grouped_data = data.groupby(group_columns)
    else:
        grouped_data = data
    
    if feature_names is None:
        feature_names = [];
    
    if not isinstance(columns, list):
        columns = [columns]
    
    if lags is None:
        lags = [l for l in range(1, window_size + 1)]
    
    for c, lag in product(columns, lags):
        feat_name = f'{c}_lag{str(lag)}'
        data[feat_name] = grouped_data[c].shift(lag)
        feature_names.append(feat_name)
    
    return data, feature_names


def horizon_targets(data, columns, group_columns=None, feature_names=None,
                    horizons=1, layout='long'):
    """
    Calculate lagged values of target variables and store results in new columns
    
    Parameters
    ----------
    data: data frame
        It has columns identifying groups of observations (e.g., locations with
        separate series per location) and a column with the response variable to
        summarize. This data frame needs to be sorted by the grouping and time
        variables in ascending order.
    columns: string or list of strings
        Name of column(s) in the data frame with the variable(s) to featurize
    group_columns: list of strings or None
        Names of columns in the data frame to group by.
    feature_names: list of strings or None
        Running list of feature column names
    horizons: list of integers
        List of look-ahead forecast horizons to use.
    layout: string
        Orientation of results: 'long' (the default) results in a data frame
        with columns `horizon` and `f'{c}_target'` for each column `c` in
        `columns`, and number of rows equal to
        `len(horizons) * (number of rows in input data)`. A 'wide'
        orientation augments the input `data` with new columns named
        `f'{c}_target{h}'` for each column `c` in `columns` and horizon `h` in
        `horizons`.
    
    Returns
    -------
    data: data frame
        Original data frame with additional columns for leading values and
        possibly for the horizon; see documentation for the `layout` argument.
    feature_names: list of strings
        Running list of feature column names, updated with new features names.
        If the `layout` is `'long'`, there is one new feature: `'horizon'`.
        If the `layout` is `'wide'`, there are no new features.
    """
    if group_columns is not None and len(group_columns) > 0:
        grouped_data = data.groupby(group_columns)
    else:
        grouped_data = data
    
    if feature_names is None:
        feature_names = [];
    
    if not isinstance(horizons, list):
        horizons = [horizons]
    
    if not isinstance(columns, list):
        columns = [columns]
    
    if layout == 'long':
        def add_one_horizon(h):
            data_h = data.copy()
            for c in columns:
                data_h[f'{c}_target'] = grouped_data[c].shift(-h)
                data_h['horizon'] = h
            return(data_h)
        
        data = pd.concat([add_one_horizon(h) for h in horizons], axis=0)
        feature_names = feature_names + ['horizon']
    else:
        for c, h in product(columns, horizons):
            data[f'{c}_target{str(h)}'] = grouped_data[c].shift(-h)
    
    return data, feature_names


def taylor_coefs_one_column_grp(data,
                                c,
                                taylor_degree=1,
                                window_size=21,
                                window_align='centered',
                                ew_span=None,
                                fill_edges=True):
    '''
    Estimate the parameters of a Taylor polynomial fit to a rolling
    trailing window, with the coefficients in consecutive windows
    updated according to a Taylor process with noise. This function is for
    internal use only, and expects the input `data` to have only one group,
    and works for a single column `c`, `window_size`, `window_span`, and
    `ew_span`.
    
    Parameters
    ----------
    data: a pandas data frame
        Data fram with data for one group unit (e.g., one location)
    c: string
        Name of the column in the data frame with the variable to featurize.
    taylor_degree: integer
        degree of the Taylor polynomial
    window_size: integer
        Time window to use for calculating Taylor polynomials.
    window_align: string
        alignment of window; either 'centered' or 'trailing'
    ew_span: integer or `None`
        Span for exponential weighting of observations. No weighting is done if
        `ew_span` is `None`.
    fill_edges: boolean
        Indicator of whether to compute estimates for windows with incomplete
        data at the beginning and/or end of the time series. If False, Taylor
        coefficient estimates are `nan` in windows with incomplete data. If
        True, the partially available data in the window will be used to compute
        estimates; this may result in unstable behavior.
    
    Returns
    -------
    data: data frame
        A copy of the data frame with additional columns containing estimated
        Taylor polynomial coefficients. New column names are of the form
        `f'{c}_taylor{d}_w{window_size}_a{window_align}_s{ew_span}'` for each
        degree `d` in `0, ..., taylor_degree`
    '''
    result = data.copy()
    if window_align == 'centered':
        half_window = (window_size - 1) // 2
        window_lags = np.arange(-half_window, half_window + 1)
    elif window_align == 'trailing':
        window_lags = np.arange(-window_size, 0) + 1
    
    shift_varnames = []
    for l in window_lags:
        if l < 0:
            shift_varname = c + '_m' + str(abs(l))
        else:
            shift_varname = c + '_p' + str(abs(l))
        
        shift_varnames.append(shift_varname)
        result[shift_varname] = result[[c]].shift(-l)
    
    taylor_X = np.concatenate(
        [np.ones((window_size, 1))] + \
            [np.expand_dims((1 / math.factorial(d)) * window_lags**d, -1) \
                for d in range(1, taylor_degree + 1)],
        axis = 1
    )
    
    y = result[shift_varnames].values.astype('float64')
    y = np.transpose(y)
    
    if ew_span is not None:
        # calculate exponential weighted observation weights
        ew_alpha = 2. / (ew_span + .1)
        obs_weights = ew_alpha * (1 - ew_alpha)**np.abs(window_lags)
        obs_weights = obs_weights / np.sum(obs_weights)
        
        # update X and y to incorporate weights
        W = np.diag(np.sqrt(obs_weights))
        taylor_X = np.matmul(W, taylor_X)
        y = np.matmul(W, y)
    
    # compute coefficient estimates at indices where y is non-missing
    
    beta_hat = np.full(shape=(taylor_X.shape[1], y.shape[1]),
                       fill_value = np.nan)
    
    if window_align == 'centered':
        # fit to sub-window with fully observed data
        beta_hat[:, half_window:-(half_window + 1)] = np.linalg.lstsq(
            taylor_X,
            y[:, half_window:-(half_window + 1)],
            rcond=None)[0]
        
        # clean up beginning and end, where there was not enough data
        if fill_edges:
            for i in range(half_window):
                beta_hat[:, i] = np.linalg.lstsq(taylor_X[(half_window - i):, :],
                                                y[(half_window - i):, i],
                                                rcond=None)[0]
                beta_hat[:, -(i+1)] = np.linalg.lstsq(taylor_X[:(half_window + i + 1), :],
                                                    y[:(half_window + i + 1), -(i + 1)],
                                                    rcond=None)[0]
    elif window_align == 'trailing':
        # fit to sub-window with fully observed data
        beta_hat[:, window_size:] = np.linalg.lstsq(
            taylor_X,
            y[:, window_size:],
            rcond=None)[0]
        
        # clean up beginning and end, where there was not enough data
        if fill_edges:
            for i in range(window_size):
                beta_hat[:, i] = np.linalg.lstsq(taylor_X[(window_size - i):, :],
                                                y[(window_size - i):, i],
                                                rcond=None)[0]
    
    for d in range(taylor_degree + 1):
        fname = f'{c}_taylor_d{str(taylor_degree)}_c{str(d)}_w{str(window_size)}{window_align[0]}' + \
            f'_s{str(ew_span)}'
        result[fname] = beta_hat[d, :]
    
    result = result.drop(shift_varnames, axis=1)
    
    return result


def windowed_taylor_coefs(data,
                          columns,
                          group_columns=None,
                          feature_names=None,
                          taylor_degree=1,
                          window_size=21,
                          window_align='centered',
                          ew_span=None,
                          fill_edges=True):
    '''
    Estimate the parameters of a Taylor polynomial fit to a rolling
    trailing window, with the coefficients in consecutive windows
    updated according to a Taylor process with noise.
    
    Parameters
    ----------
    data: a pandas data frame
        Data fram with data for one group unit (e.g., one location)
    columns: string or list of strings
        Name of column(s) in the data frame with the variable(s) to featurize
    group_columns: list of strings
        Names of columns in the data frame to group by.
    feature_names: list of strings
        Running list of feature column names
    taylor_degree: integer
        degree of the Taylor polynomial
    window_size: list of integers
        Size of time windows used for calculating Taylor coefficients
    window_align: string
        alignment of window; either 'centered' or 'trailing'
    ew_span: integer or `None`
        Span for exponential weighting of observations. No weighting is done if
        `ew_span` is `None`.
    fill_edges: boolean
        Indicator of whether to compute estimates for windows with incomplete
        data at the beginning and/or end of the time series. If False, Taylor
        coefficient estimates are `nan` in windows with incomplete data. If
        True, the partially available data in the window will be used to compute
        estimates; this may result in unstable behavior.
    
    Returns
    -------
    data: data frame
        A copy of the data frame with additional columns containing estimated
        Taylor polynomial coefficients. New column names are of the form
        `f'{c}_taylor_d{str(d)}_w{str(window_size)}{window_align[0]}' +
            f'_s{str(ew_span)` for each degree d in 0, ..., taylor_degree
    '''
    if not isinstance(columns, list):
        columns = [columns]
    
    if not isinstance(window_size, list):
        window_size = [window_size]
    
    if not isinstance(window_align, list):
        window_align = [window_align]
    
    if not isinstance(ew_span, list):
        ew_span = [ew_span]
    
    if feature_names is None:
        feature_names = [];
    
    for (c, w, a, s) in product(columns, window_size, window_align, ew_span):
        if group_columns == list() or group_columns is None:
            data = taylor_coefs_one_column_grp(data,
                                                c=c,
                                                taylor_degree=taylor_degree,
                                                window_size=w,
                                                window_align=a,
                                                ew_span=s,
                                                fill_edges=fill_edges)
        else:
            data = data.groupby(group_columns, as_index=False) \
                .apply(taylor_coefs_one_column_grp,
                        c=c,
                        taylor_degree=taylor_degree,
                        window_size=w,
                        window_align=a,
                        ew_span=s,
                        fill_edges=fill_edges) \
                .reset_index(drop=True)
        
        feat_names = [
            f'{c}_taylor_d{str(taylor_degree)}_c{str(d)}_w{str(w) + a[0]}_s{str(s)}'
            for d in range(taylor_degree + 1)]
        feature_names = feature_names + feat_names
    
    return data, feature_names


def domfreq_one_window(x, c, w, a, n_domfreq = 5, fs = 1.0, detrend = 'linear'):
    '''
    Calculate features based on power spectral density:
    dominant frequencies and power at dominant frequencies
    
    Parameters
    ----------
    x: numeric vector
        data values in one window
    c: string
        Name of the column in the data frame with the variable to featurize.
    w: integer
        Window size
    a: string
        Window alignment
    n_domfreq: integer
        The number of dominant frequencies to calculate. Defaults to 5.
    fs: float
        Sampling frequency of the x time series. See scipy.signal.periodogram.
        Defaults to 1.0.
    detrend: string or function
        Specifies how to detrend each segment. See scipy.signal.periodogram.
        Defaults to 'linear'.
    '''
    freq, psd = periodogram(x[c].values, fs = fs, detrend = detrend)
    
    domfreq_inds = np.argpartition(-psd, n_domfreq)[:n_domfreq]
    domfreq_pows = psd[domfreq_inds]
    domfreq_pows_order = np.argsort(-domfreq_pows)
    domfreq_pows = domfreq_pows[domfreq_pows_order]
    
    domfreq_inds = domfreq_inds[domfreq_pows_order]
    domfreqs = freq[domfreq_inds]
    
    return {f'{c}_domfreq{str(i+1)}_w{str(w)}{a[0]}': domfreqs[i] for i in range(n_domfreq)} | \
        {f'{c}_domfreq{str(i+1)}_logpow_w{str(w)}{a[0]}': np.log(domfreq_pows[i]) for i in range(n_domfreq)}


def domfreq_one_column_grp(data,
                           c,
                           window_size,
                           window_align,
                           n_domfreq,
                           fs,
                           detrend):
    '''
    Returns
    -------
    data: data frame
        A copy of the data frame with additional columns containing estimated
        dominant frequency features. New column names are of the form
        `f'{c}_domfreq{str(i)}_w{str(window_size)}{window_align[0]}' and
        `f'{c}_domfreq{str(i)}_logpow_w{str(window_size)}{window_align[0]}'
        for each index i in 1, ..., n_domfreq
    '''
    if window_align == 'centered':
        center = True
        hw = window_size // 2
        ext_data = pd.concat((
            data.iloc[:hw, :],
            data,
            data.iloc[-hw:, :]
        ), axis = 0)
    elif window_align == 'trailing':
        center = False
        ext_data = pd.concat((
            data.iloc[:window_size, :],
            data
        ), axis = 0)
    else:
        raise ValueError("window_align must be 'centered' or 'trailing'")
    
    feats_df = pd.DataFrame.from_records([
        domfreq_one_window(x=df_, c=c, w=window_size, a=window_align,
                           n_domfreq=n_domfreq, fs=fs, detrend=detrend) \
            for df_ in ext_data.rolling(window_size, center=center)])
    
    if window_align == 'centered':
        feats_df = feats_df.iloc[hw:(-hw), :]
    elif window_align == 'trailing':
        feats_df = feats_df.iloc[window_size:, :]
    
    return data.join(feats_df.set_index(keys=data.index))


def domfreq(data,
            columns,
            group_columns=None,
            feature_names=None,
            window_size=21,
            window_align='centered',
            n_domfreq = 5,
            fs = 1.0,
            detrend = 'linear'):
    '''
    Dominant frequencies and power at dominant frequencies of a signal.
    
    Parameters
    ----------
    data: a pandas data frame
        Data fram with data for one group unit (e.g., one location)
    columns: string or list of strings
        Name of column(s) in the data frame with the variable(s) to featurize
    group_columns: list of strings
        Names of columns in the data frame to group by.
    feature_names: list of strings
        Running list of feature column names
    window_size: list of integers
        Size of time windows used for calculating power spectral density
    window_align: string
        alignment of window; either 'centered' or 'trailing'
    n_domfreq: integer
        The number of dominant frequencies to calculate. Defaults to 5.
    fs: float
        Sampling frequency of the x time series. See scipy.signal.periodogram.
        Defaults to 1.0.
    detrend: string or function
        Specifies how to detrend each segment. See scipy.signal.periodogram.
        Defaults to 'linear'.
    
    Returns
    -------
    data: data frame
        A copy of the data frame with additional columns containing estimated
        dominant frequency features. New column names are of the form
        `f'{c}_domfreq{str(i)}_w{str(window_size)}{window_align[0]}' and
        `f'{c}_domfreq{str(i)}_logpow_w{str(window_size)}{window_align[0]}'
        for each index i in 1, ..., n_domfreq
    '''
    if not isinstance(columns, list):
        columns = [columns]
    
    if not isinstance(window_size, list):
        window_size = [window_size]
    
    if not isinstance(window_align, list):
        window_align = [window_align]
    
    if feature_names is None:
        feature_names = [];
    
    for (c, w, a) in product(columns, window_size, window_align):
        if group_columns == list() or group_columns is None:
            data = domfreq_one_column_grp(data,
                                          c=c,
                                          window_size=w,
                                          window_align=a,
                                          n_domfreq=n_domfreq,
                                          fs=fs,
                                          detrend=detrend)
        else:
            data = data.groupby(group_columns, as_index=False) \
                .apply(domfreq_one_column_grp,
                       c=c,
                       window_size=w,
                       window_align=a,
                       n_domfreq=n_domfreq,
                       fs=fs,
                       detrend=detrend) \
                .reset_index(drop=True)
        
        feat_names = [f'{c}_domfreq{str(i+1)}_w{str(w)}{a[0]}' \
                for i in range(n_domfreq)] + \
            [f'{c}_domfreq{str(i+1)}_logpow_w{str(w)}{a[0]}' \
                for i in range(n_domfreq)]
        feature_names = feature_names + feat_names
    
    return data, feature_names
