import plotly.express as px

# Page names
PAGE_FAIRNESS = 'fairness'

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

# Colors
RGB_GREY = 'rgb(173,173,173)'
BG_COLOR = '#f0f2f6'
COLOR_PALETE = px.colors.qualitative.Set2
