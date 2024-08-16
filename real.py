import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px

# Load data from CSV file
@st.cache_data
def load_data():
    return pd.read_csv('rates.csv')

data = load_data()

# Convert Date column to datetime with specified format
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Sort data by Date
data = data.sort_values('Date')

# Define X and Y variables
st.sidebar.header('Select Variables')
y_value = st.sidebar.selectbox('Select Y Value', data.columns)
x_values = st.sidebar.multiselect('Select X Values (Max 3)', data.columns)

X = data[x_values]
Y = data[y_value]

# Split data into training and testing sets based on Date
train_size = int(0.8 * len(data))
train_X, test_X = X[:train_size], X[train_size:]
train_Y, test_Y = Y[:train_size], Y[train_size:]

# Model Training and Evaluation
def train_and_evaluate_model(model):
    model.fit(train_X, train_Y)
    predictions = model.predict(test_X)
    mae = mean_absolute_error(test_Y, predictions)
    r_squared = model.score(test_X, test_Y)
    mse = mean_squared_error(test_Y, predictions)
    return mae, r_squared, mse

linear_reg = LinearRegression()
rf_reg = RandomForestRegressor()
gb_reg = GradientBoostingRegressor()
svm_reg = SVR()

models = {'Linear Regression': linear_reg, 'Random Forest': rf_reg, 'Gradient Boosting': gb_reg, 'Support Vector Machine': svm_reg}

results = {}
for name, model in models.items():
    mae, r_squared, mse = train_and_evaluate_model(model)
    results[name] = {'Mean Absolute Error': mae, 'R-squared': r_squared, 'Mean Squared Error': mse}

# Display Results
st.title('Regression Output')
for name, metrics in results.items():
    st.header(name)
    st.write(metrics)

# Plot Graphs
st.title('Plot Graphs')
selected_columns = st.multiselect('Select Columns to Plot Over Time (Max 3)', data.columns)

if selected_columns:
    fig = go.Figure()
    for column in selected_columns:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name=column))
    fig.update_layout(xaxis_title='Date', yaxis_title='Value', title='Time Series Plot')
    st.plotly_chart(fig)
