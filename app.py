import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris

st.set_page_config(page_title="EDA Dashboard", layout="wide")

px.defaults.template = "plotly_white"

MAX_POINTS = 8000  # avoid freezes

@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(
        iris.target, iris.target_names
    )
    return df


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


@st.cache_data
def apply_filter(df, col, values):
    return df[df[col].isin(values)]


@st.cache_data
def compute_corr(df, cols):
    return df[list(cols)].corr()


def main():
    st.title(" Interactive EDA Dashboard")

    st.sidebar.header("Controls")

    data_source = st.sidebar.radio(
        "Choose Data Source",
        ["Sample Iris Dataset", "Upload CSV"]
    )

    if data_source == "Sample Iris Dataset":
        df = load_iris_data()
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.warning("Upload a CSV file to continue.")
            return
        df = load_csv(uploaded)

    if df.empty:
        st.error("Dataset is empty.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns available for plotting.")
        return

    if cat_cols:
        filter_col = st.sidebar.selectbox("Filter column", cat_cols)
        selected_vals = st.sidebar.multiselect(
            f"Select {filter_col}",
            df[filter_col].unique(),
            default=list(df[filter_col].unique())
        )
        df_filtered = apply_filter(df, filter_col, tuple(selected_vals))
    else:
        df_filtered = df

    if df_filtered.empty:
        st.warning("No data after filtering.")
        return

    st.markdown(" Data Overview")

    with st.expander("Show first 10 rows"):
        st.dataframe(df_filtered.head(10), use_container_width=True)

    with st.expander("Show statistical summary"):
        st.write(df_filtered.describe())

    plot_df = df_filtered
    if len(plot_df) > MAX_POINTS:
        plot_df = plot_df.iloc[:MAX_POINTS]

    x_col = st.selectbox("X-axis", numeric_cols)
    y_col = st.selectbox(
        "Y-axis",
        numeric_cols,
        index=1 if len(numeric_cols) > 1 else 0
    )

    st.markdown(" Visualizations")

    tab1, tab2, tab3 = st.tabs(
        ["Distributions", "Relationships", "Correlation"]
    )

    #TAB 1: DISTRIBUTIONS
    with tab1:
        if st.checkbox("Show Histogram"):
            fig = px.histogram(plot_df, x=x_col)
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Box Plot"):
            fig = px.box(plot_df, y=y_col)
            st.plotly_chart(fig, use_container_width=True)

    #TAB 2: RELATIONSHIPS
    with tab2:
        if st.checkbox("Show Scatter Plot"):
            fig = px.scatter(plot_df, x=x_col, y=y_col)
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Line Chart"):
            fig = px.line(
                plot_df.sort_values(x_col),
                x=x_col,
                y=y_col
            )
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Bar Chart"):
            fig = px.bar(plot_df, x=x_col, y=y_col)
            st.plotly_chart(fig, use_container_width=True)

    #TAB 3: CORRELATION
    with tab3:
        if st.checkbox("Show Correlation Heatmap"):
            corr = compute_corr(df_filtered, tuple(numeric_cols))
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption("Optimized EDA Dashboard • Cached • Safe • Stable")


if __name__ == "__main__":
    main()
