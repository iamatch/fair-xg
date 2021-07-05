from app.streamlit.pages import fairness
from app.streamlit.static import *

PAGES = {PAGE_FAIRNESS: fairness}

page = PAGES[PAGE_FAIRNESS]
page.write()
