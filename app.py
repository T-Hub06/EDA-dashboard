import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris

#sample dataset 
@st.cache_data  # Caches data to avoid reloading on every interaction
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df


def main():
    st.title("Interactive EDA Dashboard")
    st.markdown("Explore your dataset with interactive visualizations. Upload your own CSV or use the sample Iris dataset.")
    
    st.sidebar.header("Controls")
    
    data_option = st.sidebar.radio("Choose Data Source", ["Sample Iris Dataset", "Upload CSV"])
    
    df = None
    if data_option == "Sample Iris Dataset":
        df = load_data()
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return
        else:
            st.warning("Please upload a CSV file or select the sample dataset.")
            return
    
    if df is None or df.empty:
        st.error("No data available. Please select a data source.")
        return
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))
    
    st.subheader("Data Summary")
    st.write(df.describe()) 
    
    st.sidebar.subheader("Filters")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        selected_cat = st.sidebar.selectbox("Select a categorical column for filtering", categorical_cols)
        unique_vals = df[selected_cat].unique()
        selected_vals = st.sidebar.multiselect(f"Select {selected_cat} values", unique_vals, default=unique_vals)
        df_filtered = df[df[selected_cat].isin(selected_vals)]
    else:
        df_filtered = df
    
    # Histogram
    st.subheader("Histogram")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select a numeric column for histogram", numeric_cols)
        color_col = None
        if categorical_cols:
            color_col = st.selectbox("Select a column to color by (optional)", [None] + categorical_cols)
        fig_hist = px.histogram(df_filtered, x=selected_col, color=color_col, 
                                title=f"Histogram of {selected_col}")
        st.plotly_chart(fig_hist)  
    else:
        st.info("No numeric columns available for histogram.")
    
    # Scatter Plot
    st.subheader("Scatter Plot")
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis", numeric_cols, index=0)
        y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        color_col = None
        if categorical_cols:
            color_col = st.selectbox("Select a column to color by (optional)", [None] + categorical_cols, key="scatter_color")
        fig_scatter = px.scatter(df_filtered, x=x_col, y=y_col, color=color_col,
                                 title=f"Scatter Plot: {x_col} vs {y_col}")
        st.plotly_chart(fig_scatter)
    else:
        st.info("At least two numeric columns are required for scatter plot.")
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if len(numeric_cols) > 1:
        corr = df_filtered[numeric_cols].corr()
        fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig_heatmap.update_layout(title="Correlation Matrix")
        st.plotly_chart(fig_heatmap)
    else:
        st.info("At least two numeric columns are required for correlation heatmap.")
    
    # Box Plot
    st.subheader("Box Plot")
    if numeric_cols:
        selected_col = st.selectbox("Select a numeric column for box plot", numeric_cols, key="box_col")
        color_col = None
        if categorical_cols:
            color_col = st.selectbox("Select a column to color by (optional)", [None] + categorical_cols, key="box_color")
        fig_box = px.box(df_filtered, x=color_col, y=selected_col, 
                         title=f"Box Plot of {selected_col}" + (f" by {color_col}" if color_col else ""))
        st.plotly_chart(fig_box)
    else:
        st.info("No numeric columns available for box plot.")
    
    # Bar Chart
    st.subheader("Bar Chart")
    if categorical_cols:
        selected_cat = st.selectbox("Select a categorical column for bar chart", categorical_cols, key="bar_cat")
        # Count occurrences of each category
        value_counts = df_filtered[selected_cat].value_counts().reset_index()
        value_counts.columns = [selected_cat, 'count']
        fig_bar = px.bar(value_counts, x=selected_cat, y='count', 
                         title=f"Bar Chart of {selected_cat} Counts")
        st.plotly_chart(fig_bar)
    else:
        st.info("No categorical columns available for bar chart.")
    
    # Line Chart
    st.subheader("Line Chart")
    if len(numeric_cols) >= 1:
        x_col = st.selectbox("X-axis (numeric or index)", ["Index"] + numeric_cols, key="line_x")
        y_col = st.selectbox("Y-axis", numeric_cols, key="line_y")
        color_col = None
        if categorical_cols:
            color_col = st.selectbox("Select a column to color by (optional)", [None] + categorical_cols, key="line_color")
        # If x_col is "Index", use df_filtered.index; else use the column
        x_data = df_filtered.index if x_col == "Index" else df_filtered[x_col]
        fig_line = px.line(df_filtered, x=x_data, y=y_col, color=color_col,
                           title=f"Line Chart: {y_col} vs {x_col}")
        st.plotly_chart(fig_line)
    else:
        st.info("At least one numeric column is required for line chart.")

if __name__ == "__main__":
    main()
