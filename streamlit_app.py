import streamlit as st

from app.streamlit.pages import fairness, iamatch
from app.streamlit.static import *

PAGES = {
    PAGE_FAIRNESS: fairness,
    PAGE_IAMATCH: iamatch,
}


def main():
    st.set_page_config(page_title=f"{PAGE_FAIRNESS} | IAmatch", page_icon=":soccer:", initial_sidebar_state="auto")
    st.sidebar.title("IAmatch")
    page_name = st.sidebar.radio("", SORTED_PAGES)
    page = PAGES[page_name]
    st.markdown("<h1 style='text-align: center;'>%s</h1>" % page_name, unsafe_allow_html=True)
    page.write()


if __name__ == "__main__":
    main()
