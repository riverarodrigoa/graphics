import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
from scipy import stats


def calculate_partial_correlation(input_df):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables,
    controlling for all other remaining variables

    Parameters
    ----------
    input_df : array-like, shape (n, p)
        Array with the different variables. Each column is taken as a variable.

    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of input_df[:, i] and input_df[:, j]
        controlling for all other remaining variables.
    """
    partial_corr_matrix = np.zeros((input_df.shape[1], input_df.shape[1]))
    for i, column1 in enumerate(input_df):
        for j, column2 in enumerate(input_df):
            control_variables = np.delete(np.arange(input_df.shape[1]), [i, j])
            if i == j:
                partial_corr_matrix[i, j] = 1
                continue
            data_control_variable = input_df.iloc[:, control_variables]
            data_column1 = input_df[column1].values
            data_column2 = input_df[column2].values
            fit1 = linear_model.LinearRegression(fit_intercept=True)
            fit2 = linear_model.LinearRegression(fit_intercept=True)
            fit1.fit(data_control_variable, data_column1)
            fit2.fit(data_control_variable, data_column2)
            residual1 = data_column1 - (np.dot(data_control_variable, fit1.coef_) + fit1.intercept_)
            residual2 = data_column2 - (np.dot(data_control_variable, fit2.coef_) + fit2.intercept_)
            partial_corr_matrix[i, j] = stats.spearmanr(residual1, residual2)[0]  # pearsonr
    return pd.DataFrame(partial_corr_matrix, columns=input_df.columns, index=input_df.columns)


def get_period(x, ix, rate=0.3, offset=0, samples=50):
    n = len(ix)  # 100
    n_val = int(len(ix) * rate)  # 100*0.3= 30 len test set
    imin = offset  # 0
    imax = n - offset - n_val  # 100-0-30=70
    n_sample = (imax - imin) / samples  # (70-0)/2=35

    start = int(imin + x * n_sample)  # 0+0*35= 0
    end = int(start + n_val)  # 0+35=35

    return ix[start], ix[end]


def split_dataset(data, train_test_ratio, offset_data, sample, n_samples):
    start, end = get_period(x=sample, ix=data.index, rate=train_test_ratio, offset=offset_data, samples=n_samples)  # Test set
    test_set = data.loc[start:end, :]
    train_set = data.loc[(data.index < start) | (data.index > end), :]
    return train_set, test_set


# Mean hourly
def msd_hourly(y_true, y_pred):
    yy_true = [np.mean(y_true[i:i+60]) for i in range(0, y_true.shape[0], 60)]
    yy_pred = [np.mean(y_pred[i:i+60]) for i in range(0, y_pred.shape[0], 60)]
    error = mse(yy_true, yy_pred)
    return error


def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)


def get_reconstructed_ts(data, cv_tr, cv_te, i,yvar='CH4d_ppm',ix='Index',n_samples=50):
    train, test = split_dataset(data, train_test_ratio=0.3, offset_data=0, sample=i, n_samples=n_samples)
    yvars = [yvar]
    y = data.loc[:, yvars]
    y_train = train.loc[:, yvars]
    y_test = test.loc[:, yvars]

    y_tr = pd.DataFrame(cv_tr.loc[:, str(i+1)].values, index=y_train.index, columns=[str(i+1)])
    y_te = pd.DataFrame(cv_te.loc[:, str(i+1)].values, index=y_test.index, columns=[str(i+1)])

    y_all = y_tr.append(y_te)
    y_all.sort_values(by=ix, inplace=True)
    y_all['REF'] = y.values
    y_all.columns = ['Model', 'REF']
    return y_all, y_train, y_test, y_tr, y_te


def mean_by_time(x, time_str):
    return x.groupby(pd.Grouper( freq=time_str)).mean()


def h2o_mole_fraction(rh, t, p):
    a1 = (rh / 100) * np.exp(13.7 - 5120 / (t + 273.15))
    mf = 100 * (a1 / ((p / 100000) - a1))
    return mf


def data_over_percentile(d, col_name='CH4_dry', percentile=0.9, labeled=False):
    qq = d.quantile(q=percentile, axis=0)
    if labeled:
        d_p = d.copy()
        d_p['Binary'] = d.loc[:, col_name] > qq.loc[col_name]
        d_p['Binary'] = d_p['Binary'].astype(int)
    else:
        d_p = d[d.loc[:, col_name] > qq.loc[col_name]]
        d_p.dropna(inplace=True)
    return d_p


# Detect spikes (sd over background)
def find_spikes(x, alpha, C_unf, n):  # Detect spikes in a window
    # n = 1
    sigma = np.std(x)
    # C_unf = x[0]
    threshold = C_unf + alpha * sigma + np.sqrt(n) * sigma
    spike = []
    # spike.append(0)
    for i in range(0, len(x)):
        if x[i] >= threshold:
            spike.append(1)
            n += 1
        else:
            spike.append(0)
            C_unf = x[i]
            n = 0
        threshold = C_unf + alpha * sigma + np.sqrt(n) * sigma
    return spike, C_unf, n


def find_spikes_onts(X, window, alpha):  # Detect spikes in a TS
    spike = pd.DataFrame(columns=['spikes'])
    for i in range(0, len(X), window):
        start = i
        end = start + window
        x = X[start:end]
        if i == 0:
            C_unf = x[0]
            n = 0
            spike['spikes'], C_unf_last, n_last = find_spikes(x, alpha=alpha, C_unf=C_unf, n=n)
            # print(C_unf_last, "\t", n_last)
        else:
            spike_t = pd.DataFrame(columns=['spikes'])
            spike_t['spikes'], C_unf_last, n_last = find_spikes(x, alpha=alpha, C_unf=C_unf_last, n=n_last)
            # print(C_unf_last, "\t", n_last)
            spike = spike.append(spike_t, ignore_index=True)
    return spike


def detect_spikes(x, window, alpha, backwards):  # Do the detection FWD & BCKWD
    spike_f = find_spikes_onts(x, window, alpha)
    if backwards:
        X_b = list(reversed(x))
        spike_b = find_spikes_onts(X_b, window, alpha)
        spike = pd.DataFrame(columns=['F', 'B', 'SS'])
        spike['F'] = spike_f.loc[:, 'spikes'].values
        spike['B'] = list(reversed(spike_b.loc[:, 'spikes'].values))
        spike.loc[spike['F'] != spike['B'], 'SS'] = 1
        ix = spike[spike['F'] == spike['B']].index
        spike.loc[ix, 'SS'] = spike.loc[ix, 'F']
    else:
        spike = pd.DataFrame(columns=['SS'])
        spike['SS'] = spike_f.loc[:, 'spikes'].values
    return spike


def spike_detection_it(data, variable, window, alpha, backwards):
    detection = dict()
    it = len(alpha)
    for i in range(0, it):
        if i == 0:
            data_ns = data
        else:
            data_ns = ddata[ddata.loc[:, 'Binary'] == 0].copy()
            del ddata, spike

        X_f = data_ns.loc[:, variable].copy()
        spike = detect_spikes(X_f, window[i], alpha[i], backwards)
        ddata = pd.DataFrame(columns=[variable, 'Binary'])
        ddata[variable] = data_ns.loc[:, variable]
        ddata.iloc[:, 1] = spike.loc[:, 'SS'].values
        detection[i] = ddata

    data_f = data.copy()
    for i in range(0, it):
        ix = detection[i][detection[i].loc[:, 'Binary'] == 1].index
        data_f.loc[ix, 'Binary'] = 1

    return ddata, detection, data_f


def remove_baseline(x, variable, binary_ix, interpolation_method='pad', inverted=False):
    y = pd.DataFrame(columns=['Raw','Binary','Baseline'])
    y['Raw'] = x.loc[:,variable].copy()
    y['Baseline'] = x.copy()
    y['Binary'] = binary_ix
    ix = y[y['Binary'] == 1].index
    y.loc[ix,'Baseline'] = np.nan
    y['Interpolation'] = y.loc[:,'Baseline'].interpolate(method=interpolation_method)
    if inverted:
        y['Corrected'] = y['Interpolation'] - y['Raw']
    else:
        y['Corrected'] = y['Raw'] - y['Interpolation']
    return y.loc[:,['Interpolation', 'Corrected']]


def spike_change(time_ix,spike_ix):
    lst_change = []
    value_old = 0.0
    for ix, v in enumerate(spike_ix):
        if value_old != v:
            lst_change.append(time_ix[ix])
        value_old = v
    return lst_change


def align_ts(x, start, end, var1, var2):
    xpre = x.loc[:start, :]
    xx = x.loc[start:end, :]
    xaft = x.loc[end:, :]

    # Correct shift
    if xx.empty:
        return x
    else:
        m1 = xx.loc[:, [var1, var2]].idxmax()
        shift = m1[0] - m1[1]
        if shift < pd.Timedelta(0):
            shift = m1[1] - m1[0]
            xx.loc[:, var2] = xx.loc[:, var2].shift(-(shift // 5).seconds)
        else:
            xx.loc[:, var2] = xx.loc[:, var2].shift((shift // 5).seconds)

        # Reconstruct df
        xf = xpre.append(xx)
        xf = xf.append(xaft)
        return xf
