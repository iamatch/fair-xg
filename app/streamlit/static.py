import plotly.express as px

# Page names
PAGE_FAIRNESS = 'XG Fairness'
PAGE_IAMATCH = 'About us'

SORTED_PAGES = [PAGE_FAIRNESS, PAGE_IAMATCH]

# Map ./data file name to display name
DATASET_DICT = {
    'example': 'StatsBomb XG',
    '': 'My CSV File'
}

# Fairness metric to display name
FAIRNESS_METRIC = {
    'unfair': 'Count',
    'mean_diff': 'Mean Difference'
}

# Metrics
ACC_METRIC = 'acc'
TNR_METRIC = 'tnr'
TPR_METRIC = 'tpr'
NPV_METRIC = 'npv'
PPV_METRIC = 'ppv'
METRICS = [ACC_METRIC, TNR_METRIC, TPR_METRIC, NPV_METRIC, PPV_METRIC]

# Colors
RGB_GREY = 'rgb(39,39,39)'
RGB_GREY_2ND = 'rgb(160,160,160)'
COLOR_PALETE = [RGB_GREY, RGB_GREY_2ND] + px.colors.qualitative.Set2
