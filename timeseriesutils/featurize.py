import math
from tokenize import group
import numpy as np
import pandas as pd


def featurize_data(data, group_vars=[], features = []):
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
    group_vars: list of strings
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
        args['group_vars'] = group_vars
        data, feature_names = data.pipe(eval(feature['fun']),
                                        feature_names=feature_names,
                                        **args)
    
    # # create a column for h horizon ahead target for observed values.
    # # for each location, this column has h nans in the end.
    # # the last nan is for forecast date.
    # if h is not None:
    #     assert target_var in data.columns
    #     target_name = f'{target_var}_lead_{str(h)}'
    #     data[target_name] = data.groupby('location')[target_var].shift(-h)
    # else:
    #     target_name = None
    
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
    x_train_val = train_val.pivot(index = 'location', columns = 'date', values = feature_names).to_numpy()
    # shape is (L, T, P)
    x_train_val = x_train_val.reshape((x_train_val.shape[0], x_train_val.shape[1]//len(feature_names), len(feature_names)),order='F')
    
    # shape is (L, T, P)
    y_train_val = train_val.pivot(index = 'location', columns = 'date', values = target_name).to_numpy()
    
    # convert everything to tensor
    x_train_val = tf.constant(x_train_val.astype('float64'))
    y_train_val = tf.constant(y_train_val.astype('float64'))
    x_T = tf.constant(x_T.astype('float64'))
    
    return x_train_val, y_train_val, x_T


def rollmean(data, target_var, group_vars=[], feature_names=[],
                 window_size = 7):
    """
    Calculate moving average of target variable and store result in a new column
    
    Parameters
    ----------
    data: data frame
        It has columns identifying groups of observations (e.g., locations with
        separate series per location) and a column with the target variable to
        summarize. This data frame needs to be sorted by the grouping and time
        variables in ascending order.
    target_var: string or list of strings
        Name of column(s) in the data frame with the target variable(s)
    group_vars: list of strings
        Names of columns in the data frame to group by.
    feature_names: list of strings
        Running list of feature column names
    window_size: integer
        Size of the sliding window over which we calculate moving average
    
    Returns
    -------
    data: data frame
        Original data frame with an additional column for the moving average,
        named `target_var + '_roll_mean_' + window_size`
    feature_names: list of strings
        Running list of feature column names, updated with the new column name
    """
    if group_vars is not None and len(group_vars) > 0:
        grouped_data = data.groupby(group_vars, as_index=False)
    else:
        grouped_data = data
    
    if not isinstance(target_var, list):
        target_var = [target_var]
    
    if not isinstance(window_size, list):
        window_size = [window_size]
    
    for tv in target_var:
        for w in window_size:
            column_name = f'{tv}_rollmean_w{str(w)}'
            feature_names.append(column_name)
            data[column_name] = grouped_data.rolling(w)[tv] \
                .mean() \
                .values
    
    return data, feature_names


def lag(data, target_var, group_vars=[], feature_names=[],
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
    target_var: string or list of strings
        Name of column(s) in the data frame with the target variable(s)
    group_vars: list of strings
        Names of columns in the data frame to group by.
    feature_names: list of strings
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
        `target_var + '_lag' + lag`
    feature_names: list of strings
        Running list of feature column names, updated with the new column names
    """
    if group_vars is not None and len(group_vars) > 0:
        grouped_data = data.groupby(group_vars)
    else:
        grouped_data = data
    
    if not isinstance(target_var, list):
        target_var = [target_var]
    
    if lags is None:
        lags = [l for l in range(1, window_size + 1)]
    
    for tv in target_var:
        for lag in lags:
            feat_name = f'{tv}_lag{str(lag)}'
            data[feat_name] = grouped_data[tv].shift(lag)
            feature_names.append(feat_name)
    
    return data, feature_names


def windowed_taylor_coefs_one_grp(data,
                                  target_var,
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
    target_var: string
        Name of the column in the data frame with the forecast target variable.
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
        `target_var + '_taylor_' + d` for each degree d in 0, ..., taylor_degree
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
            shift_varname = target_var + '_m' + str(abs(l))
        else:
            shift_varname = target_var + '_p' + str(abs(l))
        
        shift_varnames.append(shift_varname)
        result[shift_varname] = result[[target_var]].shift(-l)
    
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
    
    beta_hat = np.linalg.lstsq(taylor_X, y, rcond=None)[0]
    
    # clean up beginning and end, where there was not enough data
    # fit to sub-window with fully observed data
    if fill_edges:
        if window_align == 'centered':
            for i in range(half_window):
                beta_hat[:, i] = np.linalg.lstsq(taylor_X[(half_window - i):, :],
                                                y[(half_window - i):, i],
                                                rcond=None)[0]
                beta_hat[:, -(i+1)] = np.linalg.lstsq(taylor_X[:(half_window + i + 1), :],
                                                    y[:(half_window + i + 1), -(i + 1)],
                                                    rcond=None)[0]
        elif window_align == 'trailing':
            for i in range(window_size):
                beta_hat[:, i] = np.linalg.lstsq(taylor_X[(window_size - i):, :],
                                                y[(window_size - i):, i],
                                                rcond=None)[0]
    
    for d in range(taylor_degree + 1):
        result[f'{target_var}_taylor_d{str(d)}_w{str(window_size)}'] = beta_hat[d, :]
    
    result = result.drop(shift_varnames, axis=1)
    
    return result


def windowed_taylor_coefs(data,
                          target_var,
                          group_vars=None,
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
    target_var: string or list of strings
        Name of column(s) in the data frame with the target variable(s)
    group_vars: list of strings
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
        `target_var + '_taylor_' + d` for each degree d in 0, ..., taylor_degree
    '''
    if not isinstance(target_var, list):
        target_var = [target_var]
    
    if not isinstance(window_size, list):
        window_size = [window_size]
    
    if feature_names is None:
        feature_names = [];
    
    for tv in target_var:
        for w in window_size:
            if group_vars == list() or group_vars is None:
                data = windowed_taylor_coefs_one_grp(data,
                                                     target_var=tv,
                                                     taylor_degree=taylor_degree,
                                                     window_size=w,
                                                     window_align=window_align,
                                                     ew_span=ew_span,
                                                     fill_edges=fill_edges)
            else:
                data = data.groupby(group_vars, as_index=False) \
                    .apply(windowed_taylor_coefs_one_grp,
                           target_var=tv,
                           taylor_degree=taylor_degree,
                           window_size=w,
                           window_align=window_align,
                           ew_span=ew_span,
                           fill_edges=fill_edges) \
                    .reset_index(drop=True)
            
            feat_names = [f'{tv}_taylor_d{str(d)}_w{str(w)}' \
                            for d in range(taylor_degree + 1)]
            feature_names = feature_names + feat_names
    
    return data, feature_names

