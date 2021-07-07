import pandas as pd
import streamlit as st

import app.streamlit.utils.fairness as fu

from app.streamlit.static import *
from fairness.fairness import FairnessEvaluator


def fairness_analysis(df):

    # Main parameters
    target_columns, preds_columns, sensitives_columns = fu.get_target_preds_sensitves_columns(df)
    col1, col2, col3 = st.beta_columns(3)

    target = col1.selectbox('Target', target_columns)
    preds_options = [col for col in preds_columns if col != target]
    preds = col2.multiselect('Prediction', preds_options, [preds_options[0]])
    sensitives_options = [col for col in sensitives_columns if col not in [target] + preds]
    sensitives = col3.multiselect('Sensitive', sensitives_options, [sensitives_options[0]])

    # Load fairness evaluator
    fe = FairnessEvaluator(df=df, target=target, preds=preds)

    # Other parameters
    with st.beta_expander('Other parameters'):
        col1, col2, col3 = st.beta_columns(3)
        fair_metric = col1.selectbox('Fairness Metric', list(FAIRNESS_METRIC.keys()),
                                     format_func=lambda d: FAIRNESS_METRIC[d])
        threshold = col2.slider('Prediction Threshold', min_value=0., max_value=1., value=0.5, step=0.05)
        alpha = col3.slider('Risk', min_value=0., max_value=.2, value=0.05)

    # Fit fairness metrics
    if 1 <= len(preds) <= 2:
        fe.fit(sensitives=sensitives, threshold=threshold, alpha=alpha)

        ###############
        # RADAR PLOTS #
        ###############

        cols = st.beta_columns(1 + len(sensitives))

        # Get data for radar plot
        data_preds = [fe.get_bias_table(pred=pred).data for pred in fe.preds]

        # Display metrics radar plot
        metric_preds = [data[('-', 'value')] for data in data_preds]
        cols[0].markdown("<h3 style='text-align: center;'>%s</h3>" % 'Metrics', unsafe_allow_html=True)
        fig = fe.plot_radar(metric_preds, fe.preds, [COLOR_PALETE[0], RGB_GREY])
        fu.plot_fig(cols[0], fig)

        # Display fairness indicators radar plots
        for col, s, color in zip(cols[1:], sensitives, COLOR_PALETE[1:]):
            metric_preds = [1 - data[(s, fair_metric)] for data in data_preds]
            title = s.title() + ' Fairness'
            col.markdown("<h3 style='text-align: center;'>%s</h3>" % title, unsafe_allow_html=True)
            fig = fe.plot_radar(metric_preds, fe.preds, [color, RGB_GREY], marker_bool=True)
            fu.plot_fig(col, fig)
            bias_metrics = [x[0] for x in metric_preds[0].items() if x[1] < 1]
            if bias_metrics:
                col.error(f'Model is not fair regarding **{s.title()}**.')
            else:
                col.success(f'Model is fair regarding **{s.title()}**.')

        ###############
        # CURVE PLOTS #
        ###############

        with st.beta_expander('Thresholds curves'):
            figs = fe.get_threshold_figs(fe.preds, sensitives)
            # Plot figs for each metric name
            for metric_name in figs:
                st.markdown("<p style='text-align: center;'>%s</p>" % metric_name, unsafe_allow_html=True)
                # Plot figs for each sensitive variable
                cols = st.beta_columns(1 + len(sensitives))
                for col, fig in zip(cols, figs[metric_name]):
                    fu.plot_fig(col, fig)


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

    ###################
    # DATASET PREVIEW #
    ###################

    fu.display_df(df)

    #####################
    # FAIRNESS ANALYSIS #
    #####################

    if not fake_df:
        fairness_analysis(df)
