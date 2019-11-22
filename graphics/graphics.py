# -*- coding: utf-8 -*-
from __future__ import print_function, division, with_statement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def plot_comp_all_vars(da, vars_comp, start=None, end=None, qq=(0.0, 1.0), sec=None, ylabs=None, legend_labs=None,
                       ylims=None, mask_date=None, vline=None, resolution="D", file_name=None, figsize=(30, 23),
                       alpha=1.0, fontsize=14, interplotspace=(None, None), comp_in_subplot=False):
    sns.set(font_scale=1.3)
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
        keys = sec.keys()[0]

    da = da.loc[start:end, :]
    d = da.copy()
    dqq = d.quantile(q=qq, axis=0)

    xticks = pd.date_range(d.index[0], d.index[len(d) - 1], freq=resolution, normalize=False)  #
    if mask_date is not None:
        xticks = xticks[(xticks < mask_date[0]) | (xticks > mask_date[1])]
    n = len(ylabs)
    # c = d.columns
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n + 1))
    if comp_in_subplot:
        c = len(max(vars_comp))
        cmp_colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, c))
    # colors2 = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(c)))
    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(nrows=n, sharex=True, figsize=figsize, squeeze=False)

        for i in range(0, n):
            colors2 = plt.cm.get_cmap('Set2')
            if i == keys:
                ax[i, 0] = d.loc[:, vars_comp[i][0]].plot(ax=ax[i, 0], style='.', grid=True, xticks=xticks.to_list(), rot=0, ms=3, alpha=alpha)
                for j in range(1, len(sec[keys[0]])):
                    ax[i, 0] = d.loc[:, vars_comp[i][j]].plot(ax=ax[i, 0], secondary_y=True, style='.', grid=True, xticks=xticks.to_list(), rot=0, ms=3, alpha=alpha)
            else:
                ax[i, 0] = d.loc[:, vars_comp[i]].plot(ax=ax[i, 0], style='.', cmap=colors2, grid=True, xticks=xticks.to_list(), rot=0, ms=2, alpha=alpha)

            if comp_in_subplot:
                [ax[i, 0].lines[k].set_color(color) for k, color in enumerate(cmp_colors)]

            else:
                if i == 0:
                    ax[i, 0].lines[0].set_color('r')
                else:
                    if i != 1:
                        ax[i, 0].lines[0].set_color(colors[i + 1])

            if legend_labs is not None:
                ax[i, 0].legend(legend_labs[i], markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax[i, 0].legend(markerscale=5, prop={'size': fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))

            ax[i, 0].set_xlabel('')
            ax[i, 0].set_ylabel(ylabs[i], fontdict={'size': fontsize})
            ax[i, 0].yaxis.set_major_locator(plt.MaxNLocator(5), )
            ax[i, 0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)
            # ax[i, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
            ax[i, 0].yaxis.set_tick_params(labelsize=fontsize)
            ax[i, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator())

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
            if vline is not None:
                for k in vline:
                    ax[i, 0].axvline(x=k, color='r', linestyle='--')
            if i == n - 1:
                if "H" in resolution:
                    ax[i, 0].set_xticklabels(xticks.strftime('%d-%b %H:%M').tolist(), horizontalalignment='center', fontsize=fontsize)
                else:
                    ax[i, 0].set_xticklabels(xticks.strftime('%d-%b').tolist(), horizontalalignment='center', fontsize=fontsize)
                ax[i, 0].tick_params(axis='x', pad=10)
            else:
                ax[i, 0].set_xticklabels('')

            fig.align_ylabels(ax[:, 0])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=interplotspace[0], hspace=interplotspace[1])
    if save:
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0.1, dpi=200)
