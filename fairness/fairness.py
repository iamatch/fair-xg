import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from copy import deepcopy
from matplotlib import cm
from matplotlib.colors import rgb2hex
from statsmodels.stats.proportion import proportions_ztest

from .metrics import MetricsEvaluator
from app.streamlit.static import *

GREENS = cm.get_cmap('Greens')
REDS = cm.get_cmap('Reds')


class FairnessEvaluator:

    def __init__(self, df, target, preds):
        """
        :param df: (pandas.DataFrame)
        :param target: (str) name of the target column
        :param preds: (list) names of the predictions column
        """

        if isinstance(preds, str):
            preds = [preds]

        df['-'] = '-'

        self.df = df
        self.preds = preds
        self.target = target

        self.me = MetricsEvaluator()

        self.sensitives_grouped = {}

        self.thresholds = np.arange(0.1, 1, 0.1)
        self.threshold = 0.5

        self.alpha = 0.05

        self.fair_summary = {}
        self.metrics_summary = {}

    def _find_threshold_idx(self, threshold, verbose=False):
        diff = np.abs(self.thresholds - threshold)
        arg_min = np.argmin(diff)
        if verbose and min(diff) > 0.01:
            print(f'Closest threshold is {self.thresholds[arg_min]}')
        return arg_min

    def _get_metric(self, v, threshold='default'):
        if threshold is None:
            return v['mean']
        elif threshold == 'default':
            return v[threshold]
        else:
            threshold_idx = self._find_threshold_idx(threshold)
            return v['details'][threshold_idx]

    def _set_unique_sensitive(self, sensitives):
        for sensitive in sensitives:
            self.sensitives_grouped[sensitive] = []
            for s in self.df[sensitive].unique():
                self.sensitives_grouped[sensitive].append(s)
        self.sensitives_grouped['-'] = ['-']

    def _compute_sensitive_metrics(self, pred, metric_name=TPR_METRIC):

        metrics_by_sensitive = {}

        for sensitive, ss in self.sensitives_grouped.items():

            metrics_by_s = {}

            for s in ss:

                if sensitive == '-':
                    df_s = self.df
                else:
                    df_s = self.df[self.df[sensitive] == s]

                y_pred_quant = df_s[pred]
                y = df_s[self.target]

                metrics = []

                for i, t in enumerate(self.thresholds):
                    y_pred = np.greater(y_pred_quant, t).astype(int)
                    metric = self.me.compute_metric(y, y_pred, metric_name=metric_name, alpha=self.alpha)
                    metric.update({'t': t})
                    metrics.append(metric)

                d_metrics = {'details': metrics}
                if metrics:
                    d_mean = {k: round(float(np.mean([d[k] for d in metrics if d[k] == d[k]])), 3) for k in metrics[0]}
                    d_metrics.update({'mean': d_mean})

                y_pred = np.greater(y_pred_quant, self.threshold).astype(int)
                d_default = self.me.compute_metric(y, y_pred, metric_name=metric_name, alpha=self.alpha)
                d_default.update({'t': self.threshold})
                d_metrics.update({'default': d_default})

                metrics_by_s[s] = d_metrics

            metrics_by_sensitive[sensitive] = metrics_by_s

        return metrics_by_sensitive

    def _compare_metrics(self, metrics_1, metrics_2):
        d_stats = {}
        details = []
        for i, (metric_1, metric_2) in enumerate(zip(metrics_1, metrics_2)):
            count = [metric_1['n_true'], metric_2['n_true']]
            nobs = [metric_1['n'], metric_2['n']]
            if nobs[0] and nobs[1]:
                stat, p_value = proportions_ztest(count, nobs)
            else:
                stat, p_value = np.nan, 1
            diff = round(metric_1['value'] - metric_2['value'], 3)
            fair = np.int(p_value > self.alpha)
            d = {
                'stat': round(stat, 3),
                'p_value': p_value,
                'diff': diff,
                'diff_percent': diff / metric_1['value'],
                'alpha': self.alpha,
                'fair': fair,
                't': metric_1['t'],
            }
            d = self.me.round_d(d, 3)
            if i == 0:
                d_stats['default'] = d
            else:
                details.append(d)

        d_stats['details'] = details
        if details:
            d_mean = {k: round(float(np.mean([d[k] for d in details if d[k] == d[k]])), 3) for k in details[0]}
            d_stats.update({'mean': d_mean})

        return d_stats

    def _compare_sensitive_metrics(self, pred, metric_name=TPR_METRIC):
        stats_by_ss = {}

        metrics_by_sensitive = self.metrics_summary[pred][metric_name]

        for sensitive, metrics_by_s in metrics_by_sensitive.items():

            stats_by_ss[sensitive] = {}

            for s1, s2 in itertools.permutations(metrics_by_s, 2):

                metrics_1 = [self._get_metric(metrics_by_s[s1])] + [self._get_metric(metrics_by_s[s1], i) for i in self.thresholds]
                metrics_2 = [self._get_metric(metrics_by_s[s2])] + [self._get_metric(metrics_by_s[s2], i) for i in self.thresholds]

                stats_by_ss[sensitive][(s1, s2)] = self._compare_metrics(metrics_1, metrics_2)

            for s in metrics_by_s:

                metrics_1 = [self._get_metric(metrics_by_s[s])] + [self._get_metric(metrics_by_s[s], i) for i in self.thresholds]
                metrics_2 = [self._get_metric(metrics_by_sensitive['-']['-'])] + [self._get_metric(metrics_by_sensitive['-']['-'], i) for i in self.thresholds]

                stats_by_ss[sensitive][(s, '-')] = self._compare_metrics(metrics_1, metrics_2)
                stats_by_ss[sensitive][('-', s)] = self._compare_metrics(metrics_2, metrics_1)

        return stats_by_ss

    def _generate_fig_axes(self, n_rows, n_columns=None, sharex='row'):
        if n_columns is None:
            n_columns = len(self.sensitives_grouped)
        fig, axes_list = plt.subplots(n_rows, n_columns, sharex=sharex)
        fig.set_figheight(4 * n_rows)
        fig.set_figwidth(8 * n_columns)

        if n_rows == 1:
            axes_list = [axes_list]
        if len(self.sensitives_grouped) == 1 or n_columns == 1:
            axes_list = [[ax] for ax in axes_list]

        return fig, axes_list

    @staticmethod
    def _color_green(val):
        color = rgb2hex(GREENS(val))
        return 'background-color: %s' % color

    @staticmethod
    def _color_red(val):
        color = rgb2hex(REDS(val))
        return 'background-color: %s' % color

    @staticmethod
    def _color(val):
        if val >= 0:
            color = rgb2hex(GREENS(abs(val)))
        else:
            color = rgb2hex(REDS(abs(val)))
        return 'background-color: %s' % color

    @staticmethod
    def _color_opposite(val):
        if val <= 0:
            color = rgb2hex(GREENS(abs(val)))
        else:
            color = rgb2hex(REDS(abs(val)))
        return 'background-color: %s' % color

    ###################
    # Compute metrics #
    ###################

    def fit(self, sensitives, metric_names=None, threshold=0.5, alpha=0.05):
        """
        :param sensitives: (list of str) names of the sensitive columns
        :param metric_names: (list of str) metric names among ['acc', 'tnr', 'tpr']
        :param threshold: (float) prediction threshold
        :param alpha: (float) alpha used to compute proportions ztest
        """
        if isinstance(sensitives, str):
            sensitives = [sensitives]

        if metric_names is None:
            metric_names = METRICS

        elif isinstance(metric_names, str):
            metric_names = [metric_names]

        if threshold is not None:
            self.threshold = threshold

        self.alpha = alpha
        self._set_unique_sensitive(sensitives)

        for pred in self.preds:
            self.metrics_summary[pred] = {}
            self.fair_summary[pred] = {}
            for metric_name in metric_names:
                self.metrics_summary[pred][metric_name] = self._compute_sensitive_metrics(pred, metric_name)
                self.fair_summary[pred][metric_name] = self._compare_sensitive_metrics(pred, metric_name)

    ##################
    # Metrics tables #
    ##################

    def get_metrics_table(self, pred=None, sensitives=None, metric_names=None, threshold=None):

        if threshold is not None:
            self._find_threshold_idx(threshold, verbose=True)

        if sensitives is None:
            sensitives = self.sensitives_grouped.keys()

        if pred is None:
            pred = self.preds[0]

        default_metric_names = sorted(self.metrics_summary[pred])

        if metric_names is None:
            metric_names = default_metric_names
        else:
            metric_names = [m for m in metric_names if m in default_metric_names]

        if not metric_names:
            return

        metrics_tables = []

        for metric_name in metric_names:
            metric_table_s = []
            for sensitive in sensitives:
                metric_table = pd.DataFrame({k: {k2: v2 for k2, v2 in self._get_metric(v, threshold).items()
                                                 if k2 in ['value', 'ci']}
                                             for k, v in self.metrics_summary[pred][metric_name][sensitive].items()}).T
                metric_table.columns = [(metric_name, c) for c in metric_table.columns]
                metric_table.index = [(sensitive, i) for i in metric_table.index]
                metric_table_s.append(metric_table)
            metrics_tables.append(pd.concat(metric_table_s))

        metrics_tables = pd.concat(metrics_tables, axis=1).sort_index()
        metrics_tables.index = pd.MultiIndex.from_tuples(metrics_tables.index)
        metrics_tables.columns = pd.MultiIndex.from_tuples(metrics_tables.columns)

        metrics_tables_style = metrics_tables.style
        subset = [c for c in metrics_tables.columns if c[1] == 'value']
        metrics_tables_style = metrics_tables_style.applymap(self._color_green, subset=subset)
        subset = [c for c in metrics_tables.columns if c[1] == 'ci']
        metrics_tables_style = metrics_tables_style.applymap(self._color_red, subset=subset)
        return metrics_tables_style.format("{:.3f}", subset)

    def get_bias_cross_table(self, pred=None, alpha=None, sensitive=None, metric_names=None, threshold=None):

        if threshold is not None:
            self._find_threshold_idx(threshold, verbose=True)

        if pred is None:
            pred = self.preds[0]

        if sensitive is None:
            sensitive = list(self.sensitives_grouped)[0]

        if alpha is None:
            alpha = self.alpha

        default_metric_names = sorted(self.metrics_summary[pred])

        if metric_names is None:
            metric_names = default_metric_names
        else:
            metric_names = [m for m in metric_names if m in default_metric_names]

        if not metric_names:
            return

        values = deepcopy(self.sensitives_grouped[sensitive])
        values.append('-')

        bias_tables = []
        for metric_name in metric_names:
            summary = self.fair_summary[pred][metric_name][sensitive]
            bias_table = np.empty((len(values), len(values)), dtype=object)

            for s1, s2 in summary:
                i = values.index(s1)
                j = values.index(s2)
                v = summary[(s1, s2)]
                bias_table[i, j] = (self._get_metric(v, threshold)['diff'],
                                    self._get_metric(v, threshold)['p_value'])

            bias_table = pd.DataFrame(bias_table, columns=values, index=values)
            bias_table.columns = [(metric_name, c) for c in bias_table.columns]
            bias_tables.append(bias_table)

        bias_tables = pd.concat(bias_tables, axis=1)
        bias_tables.columns = pd.MultiIndex.from_tuples(bias_tables.columns)

        def _color_red_or_green(val):
            if val:
                if val[1] > alpha:
                    colors = GREENS
                else:
                    colors = REDS
                color = rgb2hex(colors(0.5))
                return 'background-color: %s' % color

        return bias_tables.style.applymap(_color_red_or_green)

    def get_bias_table(self, pred=None, threshold='default'):

        if threshold is not None and threshold != 'default':
            self._find_threshold_idx(threshold, verbose=True)

        if pred is None:
            pred = self.preds[0]

        bias_table = pd.DataFrame(
            {k: {k2: round(self._get_metric(v2['-']['-'], threshold)['value'], 3) for k2, v2 in v.items()}
             for k, v in self.metrics_summary.items()})
        bias_table = bias_table[[pred]]
        bias_table = bias_table.rename(columns={pred: ('-', 'value')})
        bias_table.columns = pd.MultiIndex.from_tuples(bias_table.columns)

        default_metric_names = sorted(self.metrics_summary[pred])

        bias_stats = {}
        for metric_name in default_metric_names:
            bias_stats[metric_name] = {}
            metrics = self.fair_summary[pred][metric_name]
            for sensitive, metrics_by_sensitive in metrics.items():
                if sensitive != '-':
                    metrics_by_sensitive = {k2[1]: self._get_metric(v2, threshold)
                                            for k2, v2 in metrics_by_sensitive.items() if k2[0] == '-'}
                    fair = np.mean([1 - m['fair'] for m in metrics_by_sensitive.values()])
                    max_diff = max([abs(m['diff']) for m in metrics_by_sensitive.values()])
                    s_not_fair = [k for k, v in metrics_by_sensitive.items() if v['fair'] < 0.5]
                    mean_diff = np.mean([abs(m['diff_percent']) * (1 - m['fair']) for m in metrics_by_sensitive.values()])
                    bias_stats[metric_name][(sensitive, 'unfair')] = fair
                    bias_stats[metric_name][(sensitive, 'max_diff')] = max_diff
                    bias_stats[metric_name][(sensitive, 'mean_diff')] = mean_diff
                    bias_stats[metric_name][(sensitive, 'mods_unfair')] = s_not_fair

        bias_stats = pd.DataFrame(bias_stats).T

        bias_table = bias_table.merge(bias_stats, left_index=True, right_index=True)

        bias_table_style = bias_table.style
        bias_table_style = bias_table_style.applymap(self._color_green, subset=[('-', 'value')])
        subset = [c for c in bias_table.columns if c[1] == 'unfair']
        bias_table_style = bias_table_style.applymap(self._color_red, subset=subset)
        subset = [c for c in bias_table.columns if c[1] != 'mods_unfair']
        return bias_table_style.format("{:.3f}", subset)

    def compare_bias_table(self, preds=None, threshold=None):

        if threshold is not None:
            self._find_threshold_idx(threshold, verbose=True)

        if preds is None:
            preds = self.preds[:2]

        bias_table_1 = self.get_bias_table(pred=preds[0], threshold=threshold)
        bias_table_2 = self.get_bias_table(pred=preds[1], threshold=threshold)

        cc = [c for c in bias_table_1.columns if c[1] != 'mods_unfair']

        bias_table = bias_table_1.data[cc] - bias_table_2.data[cc]

        bias_table_style = bias_table.style
        subset = [c for c in bias_table.columns if c[1] in ['value']]
        bias_table_style = bias_table_style.applymap(self._color, subset=subset)
        subset = [c for c in bias_table.columns if c[1] in ['unfair']]
        bias_table_style = bias_table_style.applymap(self._color_opposite, subset=subset)
        subset = [c for c in bias_table.columns if c[1] != 'mods_unfair']
        return bias_table_style.format("{:.3f}", subset)

    #########
    # Plots #
    #########

    def plot_metrics_curves(self, pred=None, opacity=0.2):

        if pred is None:
            pred = self.preds[0]

        metric_names = sorted(self.metrics_summary[pred])

        if not metric_names:
            return

        _, axes_list = self._generate_fig_axes(len(metric_names))

        for metric_name, axes in zip(metric_names, axes_list):
            for (sensitive, metrics), ax in zip(self.metrics_summary[pred][metric_name].items(), axes):
                for s, metric in metrics.items():
                    # label = ' '.join([str(sensitive), '=', str(s)])
                    label = str(s)
                    x = [m['t'] for m in metric['details']]
                    y = [m['value'] for m in metric['details']]
                    y1 = [m['value'] - m['ci'] for m in metric['details']]
                    y2 = [m['value'] + m['ci'] for m in metric['details']]
                    ax.plot(x, y, 'o', linestyle='-', label=label)
                    ax.fill_between(x, y1, y2, alpha=opacity)
                ax.set_title(sensitive + ' ' + metric_name)
        for axes in axes_list[-1]:
            axes.set_xlabel('Threshold')
        for axes in axes_list[0]:
            axes.legend()
        plt.show()

    def plot_distributions(self, pred=None, threshold=None):

        if threshold is None:
            threshold = 0.5

        if pred is None:
            pred = self.preds[0]

        bins = np.arange(0, 1.05, 0.05)

        conditions = [(None, None, pred),
                      (self.target, 0, pred),
                      (self.target, 1, pred),
                      (None, None, self.target),
                      (pred, 0, self.target),
                      (pred, 1, self.target)]

        _, axes_list = self._generate_fig_axes(len(conditions))

        for cond, axes in zip(conditions, axes_list):
            if cond[0]:
                sub_df = self.df[(self.df[cond[0]] > threshold).astype(int) == cond[1]]
            else:
                sub_df = self.df
            for sensitive, ax in zip(self.sensitives_grouped, axes):
                for s in self.sensitives_grouped[sensitive]:
                    # label = ' '.join([str(sensitive), '=', str(s)])
                    label = str(s)
                    ax.hist(sub_df[sub_df[sensitive] == s][cond[2]].astype(float),
                            density=True, alpha=0.5, label=label, bins=bins)
                title = 'Distribution by ' + str(sensitive)
                if cond[0]:
                    title = ' '.join([title, '|', str(cond[0]), '=', str(cond[1])])
                title = cond[2] + ' ' + title
                ax.set_title(title)
        for axes in axes_list[0]:
            axes.legend()
        plt.show()

    def get_threshold_curves(self, pred=None):

        if pred is None:
            pred = self.preds[0]

        metric_names = sorted(self.metrics_summary[pred])

        if not metric_names:
            return

        figs = {}
        for metric_name in metric_names:
            figs[metric_name] = {}
            for (sensitive, metrics) in self.metrics_summary[pred][metric_name].items():
                figs[metric_name][sensitive] = {}
                for s, metric in metrics.items():
                    # label = ' '.join([str(sensitive), '=', str(s)])
                    label = str(s)
                    x = [m['t'] for m in metric['details']]
                    y = [m['value'] for m in metric['details']]
                    y1 = [m['value'] - m['ci'] for m in metric['details']]
                    y2 = [m['value'] + m['ci'] for m in metric['details']]
                    figs[metric_name][sensitive][s] = (x, y, y1, y2, label)
        return figs

    def get_threshold_figs(self, preds=None, sensitives=None, threshold=0.5):

        if preds is None:
            preds = self.preds[:2]

        if sensitives is None:
            sensitives = self.sensitives_grouped.keys()

        figs = {}
        curves_list = [self.get_threshold_curves(pred) for pred in preds]
        for metric_name in sorted(curves_list[0]):
            figs[metric_name] = []
            fig = go.Figure()
            fig.add_traces([
                go.Scatter(
                    x=[threshold, threshold],
                    y=[0, 1],
                    line=dict(color="rgb(246,51,102)", dash='dash'),
                    mode='lines',
                    showlegend=False
                )])
            y_min = 1
            y_max = 0
            for sensitive in sensitives + ['-']:
                for i, curves in enumerate(curves_list):
                    for s, color_ in zip(curves[metric_name][sensitive], COLOR_PALETE):
                        if sensitive == '-':
                            color_ = "rgb(246,51,102)"
                        x, y, y_lower, y_upper, name = curves[metric_name][sensitive][s]
                        y_min = min(y_min, min(y_lower))
                        y_max = max(y_max, max(y_upper))
                        if name == '-':
                            name = 'Dataset'
                        dash = None
                        fig.add_traces([
                            go.Scatter(
                                x=x,
                                y=y,
                                line=dict(color=color_, dash=dash),
                                mode='lines+markers',
                                showlegend=True,
                                name=name,
                            ),
                            go.Scatter(
                                x=x + x[::-1],  # x, then x reversed
                                y=y_upper + y_lower[::-1],  # upper, then lower reversed
                                fill='toself',
                                fillcolor=f"rgba{color_[3:-1]}, {0.4})",
                                line=dict(color=f"rgba{color_[3:-1]}, {0})"),
                                hoverinfo="skip",
                                name=name,
                                showlegend=False
                            )
                        ])
            fig.update_yaxes(ticklabelposition="inside top")
            fig.update_xaxes(title="Thresholds")
            fig.update_layout(margin=go.layout.Margin(t=0, b=0, l=0, r=0), width=800, height=400,
                              legend=dict(orientation="h"),
                              )
            fig.update_layout(yaxis_range=[y_min, y_max], xaxis_range=[0.05, 0.95])
            figs[metric_name].append(fig)
        return figs

    def unpack_metrics(self, r):
        metrics_r_norm = r.items()
        metrics_r_norm = sorted(metrics_r_norm, key=lambda x: x[0])
        metrics, r_norm = zip(*metrics_r_norm)
        metrics = list(metrics)
        r_norm = list(r_norm)
        metrics.append(metrics[0])
        r_norm.append(r_norm[0])
        return metrics, r_norm

    def plot_radar(self, metric_preds, metric_preds_2, names, colors):

        fig = go.Figure()

        # TODO fix radar plot bug when threshold = 1
        warning = False
        metrics_warning = []
        for i, (r, r2, name, color) in enumerate(list(zip(metric_preds, metric_preds_2, names, colors))):
            if r is not None:

                metrics, r_norm = self.unpack_metrics(r)
                metrics_2, r_norm_2 = self.unpack_metrics(r2)

                metrics_bias = []
                metrics_warning = []
                for r_norm_, metric in zip(r_norm_2, metrics_2):
                    if r_norm_ < 1:
                        warning = True
                        metrics_bias.append('⚠️ ' + metric)
                        metrics_warning.append(metric)
                    else:
                        metrics_bias.append('✅ ' + metric)
                traces = [go.Scatterpolar(
                    r=r_norm,
                    theta=metrics_bias,
                    mode='markers+lines',
                    hoverinfo='text',
                    text=r_norm,
                    fill='toself',
                    fillcolor=f"rgba{color[3:-1]}, {0.4})",
                    marker=dict(
                        size=8,
                        color=color
                    ),
                    name=name
                )]
                fig.add_traces(traces)
        fig.update_layout(
            polar=dict(
                bgcolor='#f0f2f6',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.02],
                ),
                angularaxis=dict(tickfont=dict(size=13), rotation=90),),
            showlegend=False,
            margin=go.layout.Margin(t=25, b=25, l=25, r=25),
            dragmode=False,
        )
        fig.update_xaxes(fixedrange=True)
        return fig, warning, metrics_warning
