import pandas as pd
import streamlit as st

import app.streamlit.utils.fairness as fu

from app.streamlit.static import *
from fairness.fairness import FairnessEvaluator


def fairness_analysis(df):

    # Main parameters
    target_columns, preds_columns, sensitives_columns = fu.get_target_preds_sensitves_columns(df)

    # Other parameters
    with st.beta_expander('Parameters', True):
        col1, col2, col3 = st.beta_columns(3)

        target = col1.selectbox('Target', target_columns)
        preds_options = [col for col in preds_columns if col != target]
        pred = col2.selectbox('Prediction', preds_options)
        preds = [pred]
        sensitives_options = [col for col in sensitives_columns if col not in [target] + preds]
        sensitive = col3.selectbox('Sensitive', sensitives_options)
        sensitives = [sensitive]

        # Load fairness evaluator
        fe = FairnessEvaluator(df=df, target=target, preds=preds)

        col1, col2 = st.beta_columns(2)
        # fair_metric = col1.selectbox('Fairness Metric', list(FAIRNESS_METRIC.keys()),
        #                              format_func=lambda d: FAIRNESS_METRIC[d])
        fair_metric = 'unfair'
        threshold = col1.slider('Prediction Threshold', min_value=0., max_value=1., value=0.5, step=0.05)
        alpha = col2.slider('Risk', min_value=0., max_value=.2, value=0.05)

    # Fit fairness metrics
    if 1 <= len(preds) <= 2:
        fe.fit(sensitives=sensitives, threshold=threshold, alpha=alpha)

        ###############
        # RADAR PLOTS #
        ###############
        with st.beta_expander('Fairness Radar', True):

            # Get data for radar plot
            data_preds = [fe.get_bias_table(pred=pred).data for pred in fe.preds]

            # Display metrics radar plot
            metric_preds = [data[('-', 'value')] for data in data_preds]
            metric_preds_2 = [1 - data[(sensitives[0], fair_metric)] for data in data_preds]
            fig, warning, metrics_warning = fe.plot_radar(metric_preds, metric_preds_2, fe.preds, ["rgb(246,51,102)", RGB_GREY])
            fu.plot_fig(st, fig)
            if warning:
                st.warning("Some bias have been detected")

        ###############
        # CURVE PLOTS #
        ###############

        # TODO Add option to scale curves
        with st.beta_expander('Analysis'):
            metrics_sorted = sorted(METRICS, key=lambda x: (x not in metrics_warning, x))
            metric_name = st.selectbox("Fairness Metric", metrics_sorted, format_func=lambda x: '⚠️ ' + x if x in metrics_warning else '✅ ' + x)
            figs = fe.get_threshold_figs(fe.preds, sensitives, threshold)
            # Plot figs for each sensitive variable
            for fig in figs[metric_name]:
                fu.plot_fig(st, fig)


def write():

    ################
    # LOAD DATASET #
    ################

    fake_df = False
    dataset = st.selectbox('Dataset', list(DATASET_DICT.keys()), format_func=lambda d: DATASET_DICT.get(d, d))

    # My CSV File
    if dataset == '':
        file = st.file_uploader('CSV File')
        # Read csv uploaded file
        if file is not None:
            file.seek(0)
            df = pd.read_csv(file)
            fairness_analysis(df)
        # Generate fake data frame
        else:
            fake_df = True
            df = fu.generate_fake_df()
    # Example
    else:
        df = pd.read_csv(f'data/{dataset}.csv').fillna(0)
        del df['random_xg']

    ###################
    # DATASET PREVIEW #
    ###################

    with st.beta_expander('Dataset Preview', True):
        fu.display_df(df)

    #####################
    # FAIRNESS ANALYSIS #
    #####################

    if not fake_df:
        fairness_analysis(df)
