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


def get_reconstructed_ts(data, CV_tr, CV_te, i):
    TRAIN, TEST = split_dataset(data, train_test_ratio=0.3, offset_data=0, sample=i, n_samples=50)
    yvars = ['CH4d_ppm']
    y       = data.loc[:, yvars]
    y_train = TRAIN.loc[:, yvars]
    y_test  = TEST.loc[:, yvars]

    y_tr = pd.DataFrame(CV_tr.loc[:,str(i+1)].values, index= y_train.index, columns=[str(i+1)])
    y_te = pd.DataFrame(CV_te.loc[:,str(i+1)].values, index= y_test.index, columns=[str(i+1)])

    y_all = y_tr.append(y_te)
    y_all.sort_values(by='Index',inplace=True)
    y_all['REF'] = y.values
    y_all.columns = ['Model', 'REF']
    return y_all, y_train, y_test, y_tr, y_te


def mean_by_time(x, time_str):
    return x.groupby(pd.Grouper( freq=time_str)).mean()
