import numpy as np
import pandas as pd
import streamlit as st


def generate_fake_df():
    size = 5
    df = pd.DataFrame()
    df['target_example'] = np.random.choice([0, 1], size)
    df['prediction_example'] = np.random.rand(size)
    df['sensitive_example'] = np.random.choice(['s1', 's2', 's3', 's4'], size)
    return df


def get_target_preds_sensitves_columns(df):
    columns = df.columns
    uniques = [df[c].nunique() for c in columns if c != '-']

    preds_columns = df.select_dtypes(include=['float64', 'int64', 'bool']).columns
    target_columns = [columns[i] for i, u in enumerate(uniques) if u == 2 and columns[i] in preds_columns]
    sensitives_columns = [columns[i] for i, u in enumerate(uniques) if u < 20]

    return target_columns, preds_columns, sensitives_columns


###################
# STREAMLIT UTILS #
###################

def display_df(df):
    st.dataframe(df, height=150)
    st.markdown(f"**{len(df)}** rows")


def plot_fig(st_object, fig):
    st_object.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
