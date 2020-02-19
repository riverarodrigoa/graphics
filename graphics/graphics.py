# -*- coding: utf-8 -*-
from __future__ import print_function, division, with_statement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import seaborn as sns
import string
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
from sklearn.metrics import r2_score as r2


def plot_comp_all_vars(da, vars_comp, start=None, end=None, qq=(0.0, 1.0), sec=None, ylabs=None, legend_labs=None,
                       ylims=None, mask_date=None, vline=None, file_name=None, figsize=(30, 23),
                       alpha=1.0, fontsize=16, interplotspace=(None, None), comp_in_subplot=False,
                       reverse=(), k_ticks=None, style=None, grid_plot=True, marker_size=4):
    sns.set(font_scale=1.3)
    sns.set_style("ticks")
    if start is None:
        start = da.index[0]
    if end is None:
        end = da.index[-1]

    if file_name is None:
        save = False
    else:
        save = True

    if sec is None:
        keys = None
    else:
        keys = list(sec.keys())

    if style is None:
        style = '.'

    da = da.loc[start:end, :]
    d = da.copy()
    if k_ticks is not None:
        t_keys = list(k_ticks.keys())
        for i in t_keys:
            d.loc[:, i] = d.loc[:, i] / k_ticks[i]
    dqq = d.quantile(q=qq, axis=0)
    cmap = 'Set2'
    n = len(ylabs)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n + 1))
    if comp_in_subplot:
        c = len(max(vars_comp))
        cmp_colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, c))

    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(nrows=n, sharex=True, figsize=figsize, squeeze=False)
        colors2 = plt.cm.get_cmap(cmap)
        for i in range(0, n):
            if i == keys: # Secondary axes
                ax[i, 0] = d.loc[:, vars_comp[i][0]].plot(ax=ax[i, 0],
                                                          style=style, grid=grid_plot,
                                                          # xticks=xticks.to_list(),
                                                          rot=0, ms=marker_size, alpha=alpha)
                for j in range(1, len(sec[keys[0]])):
                    ax[i, 0] = d.loc[:, vars_comp[i][j]].plot(ax=ax[i, 0],
                                                              secondary_y=True,
                                                              style=style, grid=grid_plot,
                                                              # xticks=xticks.to_list(),
                                                              rot=0, ms=marker_size, alpha=alpha)
            else:
                ax[i, 0] = d.loc[:, vars_comp[i]].plot(ax=ax[i, 0],
                                                       style=style,
                                                       cmap=colors2, grid=grid_plot,
                                                       # xticks=xticks.to_list(),
                                                       rot=0, ms=marker_size, alpha=alpha)

            if comp_in_subplot:
                for k, color in enumerate(cmp_colors):
                    ax[i, 0].lines[k].set_color(color)
            else:
                if i == 0:
                    ax[i, 0].lines[0].set_color('r')
                else:
                    if i != 1:
                        ax[i, 0].lines[0].set_color(colors[i])

            if legend_labs is not None:
                ax[i, 0].legend(legend_labs[i], markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax[i, 0].legend(markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))

            if ylims is not None:
                if i in ylims.keys():
                    a, b = ylims[i]
                else:
                    a = dqq.loc[:, vars_comp[i]].values.min()
                    b = dqq.loc[:, vars_comp[i]].values.max()

            else:
                a = dqq.loc[:, vars_comp[i]].values.min()
                b = dqq.loc[:, vars_comp[i]].values.max()

            ax[i, 0].set_ylim(a, b)

            ax[i, 0].set_xlabel('')
            ax[i, 0].set_ylabel(ylabs[i], fontdict={'size': fontsize})
            ax[i, 0].yaxis.set_major_locator(plt.MaxNLocator(5), )
            ax[i, 0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)
            ax[i, 0].yaxis.set_tick_params(labelsize=fontsize)
            ax[i, 0].xaxis.set_tick_params(labelsize=fontsize)
            locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']
            formatter.zero_formats = [''] + formatter.formats[:-1]
            formatter.zero_formats[2] = '%d-%b'
            formatter.offset_formats = ['', '', '', '', '%d %b %Y', '%d %b %Y %H:%M']

            ax[i, 0].xaxis.set_major_locator(locator)
            ax[i, 0].xaxis.set_major_formatter(formatter)
            ax[i, 0].xaxis.set_minor_locator(mdates.DayLocator())
            ax[i, 0].tick_params(which='minor', length=4, color='k')

            ax[i, 0].spines['left'].set_linewidth(2)
            ax[i, 0].spines['left'].set_color('gray')
            ax[i, 0].spines['bottom'].set_linewidth(2)
            ax[i, 0].spines['bottom'].set_color('gray')

            if len(reverse) != 0:
                if reverse[i] in vars_comp[i]:
                    ax[i, 0].invert_yaxis()

            if vline is not None:
                for k in vline:
                    ax[i, 0].axvline(x=k, color='r', linestyle='--')

            if i != n - 1:
                ax[i, 0].set_xticklabels('')

            plt.xticks(ha='center')

            fig.align_ylabels(ax[:, 0])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=interplotspace[0], hspace=interplotspace[1])
    if save:
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0.1, dpi=200)


def norm(x, type_norm=1, stats=None):
    # assert isinstance(x, pd.DataFrame), "[ERROR]: X must be a pandas DataFrame."
    if not isinstance(stats, pd.DataFrame):
        stats = x.describe().transpose()

    if type_norm == 1:
        return (x - stats['mean']) / stats['std']
    elif type_norm == 2:
        return 2 * (x - stats['min']) / (stats['max'] - stats['min']) - 1
    else:
        return x


def msd_hourly(y_true, y_pred):
    yy_true = [np.mean(y_true[i:i+60]) for i in range(0, y_true.shape[0], 60)]
    yy_pred = [np.mean(y_pred[i:i+60]) for i in range(0, y_pred.shape[0], 60)]
    error = mse(yy_true, yy_pred)
    return error


# Save dataset
def save_dataset(name, df, var_names):
    data = {}
    df['Time'] = df.index
    var_names += ['Time']
    for x in var_names:
        data[x] = df.loc[:, x].values

    np.save(name + '.npy', data, allow_pickle=True, fix_imports=True)
    return None


def plot_ts_residuals3(df_data, ytrain_true, ytrain_model, ytest_true, ytest_model, start=None, end=None,
                       style='.', size=(30, 23), fontsize=14, axs=None, ms=3):
    sns.set(font_scale=1.3)
    sns.set_style("ticks")
    val_ix = [ytest_true.index[0], ytest_true.index[-1]]
    start = df_data.index[0] if start is None else start
    end = df_data.index[-1] if end is None else end
    df_data = df_data.loc[start:end, :]
    leg = [['Reference', 'Model']]

    a = msd_hourly(ytrain_true.values, ytrain_model)
    b = msd_hourly(ytest_true.values, ytest_model)

    text_rmse_train = r'$MSD_{TRAIN}: %.3E $' % (a,)
    text_rmse_test = r'$MSD_{TEST}: %.3E $' % (b,)
    props = dict(boxstyle='round', alpha=0.5)
    colors = plt.cm.get_cmap('Set2')

    locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ['%y', '%b',  # ticks are mostly months
                         '%d',  # ticks are mostly days
                         '%H:%M',  # hrs
                         '%H:%M',  # min
                         '%S.%f', ]  # secs
    formatter.zero_formats = [''] + formatter.formats[:-1]
    formatter.zero_formats[2] = '%d-%b'
    formatter.offset_formats = ['',
                                '%Y',
                                '%Y',
                                '%d %b %Y',
                                '%d %b %Y',
                                '%d %b %Y %H:%M', ]
    with plt.style.context('seaborn-whitegrid'):
        if axs is not None:
            ax1 = axs[0]
        else:
            fig, ax1 = plt.subplots(nrows=1, sharex=True, figsize=size, squeeze=False)

        ax1 = df_data.loc[:, ['REF', 'Model']].plot(ax=ax1, style=style, cmap=colors, grid=True, rot=0, ms=ms)
        ax1.lines[0].set_color('r')
        ax1.lines[1].set_color('b')
        ax1.legend(leg[0], markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('')
        ax1.set_ylabel("$\mathrm{CH_{4}}$ [ppm]", fontdict={'size': fontsize})
        ax1.axvspan(val_ix[0], val_ix[1], facecolor='blue', alpha=0.15)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['left'].set_color('gray')
        ax1.spines['bottom'].set_linewidth(2)
        ax1.spines['bottom'].set_color('gray')
        locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = ['%y',  # ticks are mostly years
                             '%b',  # ticks are mostly months
                             '%d',  # ticks are mostly days
                             '%H:%M',  # hrs
                             '%H:%M',  # min
                             '%S.%f', ]  # secs
        formatter.zero_formats = [''] + formatter.formats[:-1]
        formatter.zero_formats[2] = '%d-%b'
        formatter.offset_formats = ['',
                                    '',
                                    '',
                                    '',
                                    '%b %Y',
                                    '%d %b %Y %H:%M', ]

        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.tick_params(which='minor', length=4, color='k')
        plt.xticks(ha='center')

        ax1.text(0.01, 0.98, text_rmse_train, transform=ax1.transAxes, fontsize=15, verticalalignment='top', bbox=props)
        ax1.text(0.25, 0.98, text_rmse_test , transform=ax1.transAxes, fontsize=15, verticalalignment='top', bbox=props)

    if axs is None:
        fig.align_ylabels(ax1)

    if axs is not None:
        return ax1


def plot_ts_residuals4(df_data, ytrain_true, ytrain_model, ytest_true, ytest_model, start=None, end=None,
                       style='.', size=(30, 23), fontsize=14, ms=3):
    sns.set(font_scale=1.3)
    sns.set_style("ticks")
    val_ix = [ytest_true.index[0], ytest_true.index[-1]]
    start = df_data.index[0] if start is None else start
    end = df_data.index[-1] if end is None else end
    df_data = df_data.loc[start:end, :]
    leg = [['Reference', 'Model']]

    a = msd_hourly(ytrain_true.values, ytrain_model)
    b = msd_hourly(ytest_true.values, ytest_model)

    text_rmse_train = '$\mathrm{MSD_{TRAIN}}$: '+'{:4f}'.format(a)
    text_rmse_test  = '$\mathrm{MSD_{TEST }}$: '+'{:4f}'.format(b)
    props = dict(boxstyle='round', alpha=0.5)
    colors = plt.cm.get_cmap('Set2')

    locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ['%y',  # ticks are mostly years
                         '%b',  # ticks are mostly months
                         '%d',  # ticks are mostly days
                         '%H:%M',  # hrs
                         '%H:%M',  # min
                         '%S.%f', ]  # secs
    formatter.zero_formats = [''] + formatter.formats[:-1]
    formatter.zero_formats[2] = '%d-%b'
    formatter.offset_formats = ['',
                                '%Y',
                                '%Y',
                                '%d %b %Y',
                                '%d %b %Y',
                                '%d %b %Y %H:%M', ]
    with plt.style.context('seaborn-whitegrid'):
        fig, ax1 = plt.subplots(nrows=1, figsize=size, squeeze=True)
        ax1 = df_data.loc[:, ['REF', 'Model']].plot(ax=ax1, style=style, cmap=colors, grid=True, rot=0, ms=ms)
        ax1.lines[0].set_color('r')
        ax1.lines[1].set_color('b')
        ax1.legend(leg[0], markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('')
        ax1.set_ylabel("$\mathrm{CH_{4}}$ [ppm]", fontdict={'size': fontsize})
        ax1.axvspan(val_ix[0], val_ix[1], facecolor='blue', alpha=0.15)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['left'].set_color('gray')
        ax1.spines['bottom'].set_linewidth(2)
        ax1.spines['bottom'].set_color('gray')
        locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = ['%y',  # ticks are mostly years
                             '%b',  # ticks are mostly months
                             '%d',  # ticks are mostly days
                             '%H:%M',  # hrs
                             '%H:%M',  # min
                             '%S.%f', ]  # secs
        formatter.zero_formats = [''] + formatter.formats[:-1]
        formatter.zero_formats[2] = '%d-%b'
        formatter.offset_formats = ['',
                                    '',
                                    '',
                                    '',
                                    '%b %Y',
                                    '%d %b %Y %H:%M', ]

        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.tick_params(which='minor', length=4, color='k')
        plt.xticks(ha='center')

        ax1.text(0.01, 0.98, text_rmse_train, transform=ax1.transAxes, fontsize=15, verticalalignment='top', bbox=props)
        ax1.text(0.25, 0.98, text_rmse_test , transform=ax1.transAxes, fontsize=15, verticalalignment='top', bbox=props)


def plot_response(d, xvars, yvars, xlabl, ylabl, figsize, fontsize=16, file_name=None):
    if file_name is None:
        save = False
    else:
        save = True
    n = len(xvars)

    def make_reg(ds, x_var, y_var):
        dd = ds.loc[:, [x_var] + [y_var]]
        dd.dropna(inplace=True)
        reg = linear_model.LinearRegression()
        reg.fit(dd.loc[:, x_var].values.reshape(len(dd), 1), dd.loc[:, y_var].values.reshape(len(dd), 1))
        dd['y_pred'] = reg.predict(dd.loc[:, x_var].values.reshape(len(dd), 1))
        m_slope = reg.coef_[0][0]
        r_2_score = r2(dd.loc[:, y_var].values.reshape(len(dd), 1), dd.loc[:, 'y_pred'].values.reshape(len(dd), 1))
        num_obs = len(dd)
        return dd, m_slope, r_2_score, num_obs

    def format_axs(ax_f, ylabs, xticks, reverse, fontsize, locator=(3, 1)):
        ax_f.set_xlabel('')
        ax_f.set_ylabel(ylabs, fontdict={'size': fontsize})
        if locator is None:
            ax_f.yaxis.set_major_locator(plt.AutoLocator())
            ax_f.yaxis.set_minor_locator(plt.AutoLocator())
        else:
            ax_f.yaxis.set_major_locator(plt.MultipleLocator(locator[0]))
            ax_f.yaxis.set_minor_locator(plt.MultipleLocator(locator[1]))
        ax_f.yaxis.set_tick_params(labelsize=fontsize)
        ax_f.xaxis.set_tick_params(labelsize=fontsize)
        locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']
        formatter.zero_formats = [''] + formatter.formats[:-1]
        formatter.zero_formats[2] = '%d-%b'
        formatter.offset_formats = ['', '', '', '%d %b %Y', '%d %b %Y', '%d %b %Y %H:%M']

        ax_f.xaxis.set_major_locator(locator)
        ax_f.xaxis.set_major_formatter(formatter)
        ax_f.xaxis.set_minor_locator(mdates.DayLocator())
        ax_f.tick_params(which='minor', length=4, color='k')
        ax_f.spines['left'].set_linewidth(2)
        ax_f.spines['left'].set_color('gray')
        ax_f.spines['bottom'].set_linewidth(2)
        ax_f.spines['bottom'].set_color('gray')
        ax_f.spines['right'].set_linewidth(0.5)
        ax_f.spines['right'].set_color('gray')
        ax_f.spines['top'].set_linewidth(0.5)
        ax_f.spines['top'].set_color('gray')

        if reverse:
            ax_f.invert_yaxis()

        if not xticks:
            ax_f.set_xticklabels('')
        return ax_f

    with plt.style.context('seaborn-whitegrid'):
        sns.set(font_scale=1.3)
        sns.set_style("ticks")

        x = int(np.ceil(np.sqrt(n)))
        y = int(np.ceil(n/x))

        fig, ax = plt.subplots(nrows=y+2, ncols=x, sharey=True, figsize=figsize, squeeze=False)
        gs0 = ax[0, 0].get_gridspec()
        gs1 = ax[1, 0].get_gridspec()
        for a in ax[0, :]:
            a.remove()

        for a in ax[1, :]:
            a.remove()

        ax0 = fig.add_subplot(gs0[0, :])
        ax1 = fig.add_subplot(gs0[1, :])

        ax0 = d.loc[:, yvars[0]].plot(ax=ax0, style='.', grid=True, rot=0, ms=4)
        ax0.set_ylabel(ylabl[0], fontdict={'size': fontsize})
        ax0.lines[0].set_color('r')
        ax0.legend(['Picarro CRDS'], markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
        ax0 = format_axs(ax0, ylabs=ylabl[0], xticks=False, reverse=False, fontsize=fontsize,
                         locator=(3, 1))

        da = d.loc[:, xvars]
        ax1 = da.plot(ax=ax1, style='.', grid=True, rot=0, ms=4)
        cmp_colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, n))

        for k, color in enumerate(cmp_colors):
            ax1.lines[k].set_color(color)

        ax1.legend(xlabl, markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
        ax1 = format_axs(ax1, ylabs='Voltage [V]', xticks=True, reverse=False, fontsize=fontsize,
                         locator=None)
        count = 0

        for i in range(2, y+2):
            for j in range(0, x):
                if count < n:
                    dd_r, m, r_2, n_r = make_reg(d, xvars[count], yvars[count])
                    ax[i, j] = dd_r.plot(ax=ax[i, j], grid=True, style='-', x=xvars[count], y='y_pred')
                    ax[i, j] = dd_r.plot(ax=ax[i, j], grid=True, style='.', ms=5, x=xvars[count], y=yvars[count])
                    ax[i, j].lines[0].set_color('r')
                    ax[i, j].lines[1].set_color('b')
                    ax[i, j].legend(['$m$: {:3.3f}'.format(m) + '\n $\mathrm{R^{2}}$: ' + '{:1.3f}'.format(r_2) + '\n # obs: {:d}'.format(n_r)],
                                    markerscale=3, prop={'size': fontsize}, loc='best', frameon=True, fancybox=True)
                    ax[i, j].set_xlabel(xlabl[count], fontdict={'size': fontsize})
                    ax[i, j].set_ylabel(ylabl[count], fontdict={'size': fontsize})
                    ax[i, j].yaxis.set_major_locator(plt.MultipleLocator(3))
                    ax[i, j].xaxis.set_major_locator(plt.AutoLocator())
                    ax[i, j].yaxis.set_minor_locator(plt.MultipleLocator(1))
                    ax[i, j].xaxis.set_minor_locator(plt.AutoLocator())
                    ax[i, j].spines['left'].set_linewidth(2)
                    ax[i, j].spines['left'].set_color('gray')
                    ax[i, j].spines['bottom'].set_linewidth(2)
                    ax[i, j].spines['bottom'].set_color('gray')
                    ax[i, j].spines['right'].set_linewidth(0.5)
                    ax[i, j].spines['right'].set_color('gray')
                    ax[i, j].spines['top'].set_linewidth(0.5)
                    ax[i, j].spines['top'].set_color('gray')
                else:
                    ax[i, j].axis('off')
                count += 1

        plt.xticks(ha='center')
        fig.align_ylabels([ax0, ax1, ax[2, 0]])
    if save:
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0.1, dpi=200)

########## OLD Functions ################

# def df_to_plot(folder, case, path, df_data, i=1):
#     file = case + '_' + str(i) + '.nc'
#     data_path = path + 'results/' + folder + '/' + case + '/' + file
#     rfile = Dataset(data_path, 'r')
#     nn = np.array(rfile.variables['opt_outputs_ann'][:])
#     da = np.array(rfile.variables['all_data'][:])
#     val = np.array(rfile.variables['validation_index'][:], dtype=np.uint32)
#     train_rmse = rfile.variables['opt_training_rmse'][:][0]
#     test_rmse = rfile.variables['opt_validation_rmse'][:][0]
#     res = pd.DataFrame(data=da, columns=['CH4'])
#     res['MLP'] = nn
#     res['Time'] = df_data.index[:]
#     res.set_index('Time', inplace=True)
#     return res, val, train_rmse, test_rmse


# def plot_ts_residuals(df_data, val, a, b, start=None, end=None, resolution="D", size=(30, 23), diff=True):
#     val_ix = [df_data.index[val[0]], df_data.index[val[-1]]]
#     if start is None:
#         start = df_data.index[0]
#     if end is None:
#         end = df_data.index[-1]
#
#     df_data = df_data.loc[start:end, :]
#
#     xticks = pd.date_range(df_data.index[0], df_data.index[len(df_data) - 1], freq=resolution, normalize=False)
#
#     if diff:
#         df_data.loc[:, 'RESIDUAL'] = df_data.loc[:, 'CH4'] - df_data.loc[:, 'MLP']
#
#     c = [['CH4', 'MLP'], ['RESIDUAL']]  # D.columns
#     leg = [['Reference', 'Model'], ['Residual']]
#     n = len(c)  # len(D.columns)
#     colors = plt.cm.get_cmap('Dark2')(np.linspace(0, 1, n))
#     text_rmse_train = r'$RMSE_{TRAIN} [ppm]: %.4f $' % (a,)
#     text_rmse_test = r'$RMSE_{TEST} [ppm]: %.4f $' % (b,)
#     props = dict(boxstyle='round', alpha=0.5)
#     with plt.style.context('seaborn-whitegrid'):
#         fig, ax = plt.subplots(nrows=n, sharex=True, figsize=size)
#
#         for i in range(0, n):
#             ax[i] = df_data.loc[:, c[i]].plot(ax=ax[i], style='.', grid=True, xticks=xticks.to_list(), rot=60, ms=2)
#             ax[i].lines[0].set_color(colors[i])
#             ax[i].legend(leg[i], markerscale=5, loc='upper right', prop={'size': 14}, bbox_to_anchor=(1, 1.0))
#             ax[i].set_xlabel('')
#             # ax[i].axvspan(D.index[0], D.index[-1], facecolor='white')
#             ax[i].axvspan(val_ix[0], val_ix[1], facecolor='blue', alpha=0.15)
#             if i == n - 1:
#                 if resolution == "D":
#                     ax[i].set_xticklabels(xticks.strftime('%b-%d').tolist(), horizontalalignment='center', fontsize=14)
#                 # else:
#                 #     ax[i].set_xticklabels(xticks.strftime('%b-%d %H:%M').tolist(), horizontalalignment='center', fontsize=14);
#             else:
#                 ax[i].set_xticklabels('')
#
#         ax[0].text(0.01, 0.98, text_rmse_train, transform=ax[0].transAxes, fontsize=15, verticalalignment='top', bbox=props)
#         ax[0].text(0.30, 0.98, text_rmse_test, transform=ax[0].transAxes, fontsize=15, verticalalignment='top', bbox=props)
#
#     return fig


# def scatter_ts(folder, case, i, size=(30, 23)):
#     n = len(i)
#     colors = plt.cm.get_cmap('Dark2')(np.linspace(0, 1, n))
#
#     with plt.style.context('seaborn-whitegrid'):
#         fig, ax = plt.subplots(ncols=n, sharey=True, figsize=size)
#         for j in range(n):
#             D, val, train, test = df_to_plot(folder, case, i[j])
#             ax[j] = D.loc[:, ['CH4d_ppm', 'MLP']].plot(x='CH4d_ppm', y='MLP', style='.', ax=ax[j], grid=True)
#             ax[j].lines[0].set_color(colors[j])
#             # ax[i].legend(markerscale=5, loc='upper right', prop={'size': 14}, bbox_to_anchor=(1, 1.0))
#     return fig


# def comp_study_plot(path, folder, studies, name_studies=None, size=(20, 10)):
#     y_labs = ['RMSE (ppm)', 'BIAS (ppm)', '$\sigma / \sigma_{DATA}$ (%)', r'$\rho$ (%)']
#
#     if name_studies is None:
#         name_studies = studies
#
#     RMSE = pd.DataFrame(columns=studies)
#     BIAS = pd.DataFrame(columns=studies)
#     SIGMA = pd.DataFrame(columns=studies)
#     CORR = pd.DataFrame(columns=studies)
#     for study in studies:
#         for test_set in range(1, 51):
#             D, val_ix, rmse_train, rmse_test = df_to_plot(folder, study, i=test_set, path=path)
#             DD = D.iloc[val_ix, :]
#             bias = (DD.loc[:, 'CH4'] - DD.loc[:, 'MLP']).mean()
#
#             RMSE.loc[test_set, study] = rmse_test
#             BIAS.loc[test_set, study] = bias
#             SIGMA.loc[test_set, study] = (DD.std()[1] / DD.std()[0]) * 100
#             CORR.loc[test_set, study] = DD.corr().loc['CH4', 'MLP'] * 100
#
#     RMSE.columns = name_studies
#     BIAS.columns = name_studies
#     SIGMA.columns = name_studies
#     CORR.columns = name_studies
#     # n = len(studies)
#     # colors = plt.cm.get_cmap('Dark2')(np.linspace(0, 1, n))
#
#     with plt.style.context('seaborn-whitegrid'):
#         fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=size, squeeze=False)
#
#         ax[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.6)
#         ax[2, 0].axhline(y=100, color='k', linestyle='--', alpha=0.6)
#         ax[3, 0].axhline(y=100, color='k', linestyle='--', alpha=0.6)
#
#         ax[0, 0] = sns.boxplot(ax=ax[0, 0], width=0.3, data=RMSE)
#         ax[1, 0] = sns.boxplot(ax=ax[1, 0], width=0.3, data=BIAS)
#         ax[2, 0] = sns.boxplot(ax=ax[2, 0], width=0.3, data=SIGMA)
#         ax[3, 0] = sns.boxplot(ax=ax[3, 0], width=0.3, data=CORR)
#
#         for i in range(0, 4):
#             ax[i, 0].set_ylabel(y_labs[i])
#             ax[i, 0].yaxis.set_major_locator(plt.MaxNLocator(5), )
#             y_labels = ax[i, 0].get_yticks()
#             ax[i, 0].set_yticklabels(y_labels, fontsize=14)
#             if i in [2, 3]:
#                 ax[i, 0].yaxis.set_major_formatter(ticker.PercentFormatter())
#             else:
#                 ax[i, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
#
#     return fig
#
#
# def plot_ts_residuals2(df_data, val, a, b, start=None, end=None, resolution="D", size=(30, 23), fontsize=14):
#     sns.set(font_scale=1.3)
#     val_ix = [df_data.index[val[0]], df_data.index[val[-1]]]
#     if start is None:
#         start = df_data.index[0]
#     if end is None:
#         end = df_data.index[-1]
#
#     df_data = df_data.loc[start:end, :]
#     xticks = pd.date_range(df_data.index[0], df_data.index[len(df_data) - 1], freq=resolution, normalize=False)
#     df_data.loc[:, 'RESIDUAL'] = df_data.loc[:, 'CH4'] - df_data.loc[:, 'MLP']
#     # c = [['CH4', 'MLP'], ['RESIDUAL']]  # D.columns
#     leg = [['Reference', 'Model'], ['Residual']]
#     text_rmse_train = r'$MSD_{TRAIN} [ppm]: %.4f $' % (a * a,)
#     text_rmse_test = r'$MSD_{TEST} [ppm]: %.4f $' % (b * b,)
#     props = dict(boxstyle='round', alpha=0.5)
#     colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, 3))
#     colors2 = plt.cm.get_cmap('Set2')
#     with plt.style.context('seaborn-whitegrid'):
#         fig, ax = plt.subplots(nrows=2, sharex=True, figsize=size)
#         ax[0] = df_data.loc[:, ['CH4', 'MLP']].plot(ax=ax[0], style='.', cmap=colors2, grid=True, xticks=xticks.to_list(), rot=0, ms=3)
#         ax[0].lines[0].set_color('r')
#         ax[0].lines[1].set_color('b')
#         ax[0].legend(leg[0], markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
#         ax[0].set_xlabel('')
#         ax[0].set_ylabel('$CH_{4}$ [ppm]', fontdict={'size': fontsize})
#         ax[0].axvspan(val_ix[0], val_ix[1], facecolor='blue', alpha=0.15)
#
#         ax[1] = df_data.loc[:, ['RESIDUAL']].plot(ax=ax[1], style='.', cmap=colors2, grid=True, xticks=xticks.to_list(), rot=0, ms=3)
#         ax[1].lines[0].set_color(colors[2])
#         ax[1].legend(leg[1], markerscale=5, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': fontsize})
#         ax[1].set_xlabel('Date')
#         ax[1].set_ylabel('Residual [ppm]', fontdict={'size': fontsize})
#         ax[1].axvspan(val_ix[0], val_ix[1], facecolor='blue', alpha=0.15)
#         if "D" in resolution:
#             ax[1].set_xticklabels(xticks.strftime('%b-%d').tolist(), horizontalalignment='center', fontsize=fontsize)
#         else:
#             ax[1].set_xticklabels(xticks.strftime('%b-%d %H:%M').tolist(), horizontalalignment='center', fontsize=fontsize)
#
#         ax[0].text(0.01, 0.98, text_rmse_train, transform=ax[0].transAxes, fontsize=15, verticalalignment='top', bbox=props)
#         ax[0].text(0.30, 0.98, text_rmse_test, transform=ax[0].transAxes, fontsize=15, verticalalignment='top', bbox=props)
#     fig.align_ylabels(ax[:])
#
#     for n, a in enumerate(ax):
#         a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes, size=16, weight='bold')
#     return fig
