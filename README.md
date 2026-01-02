# Interactive EDA Dashboard

An interactive Exploratory Data Analysis (EDA) dashboard built using Streamlit to quickly explore CSV datasets with performance-safe visualizations.

## Features
- Upload custom CSV files or use the built-in Iris dataset
- Dataset preview (first 10 rows)
- Statistical summary of numerical columns
- Interactive charts:
  - Histogram
  - Box Plot
  - Scatter Plot
  - Line Chart
  - Bar Chart
  - Correlation Heatmap
- Sidebar-based column selection and chart controls

## Performance Notes
- Large datasets are automatically sampled (up to 8,000 rows) to prevent browser freezing.
- Charts are rendered , only when enabled , using checkboxes.
- Correlation heatmap is computed on demand due to its high cost.
- Data preview and summary are placed inside expandable sections to reduce initial load time.

## Tech Stack
- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py

#Live App: 
https://eda-dashboard-b5swpmhvhlup6hxeqsti3a.streamlit.app/