import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file into a DataFrame
@st.cache_data
def load_data():
    # Replace 'your_data.csv' with the path to your CSV file
    ml_df = pd.read_csv('suzyo.csv')
    scaler = StandardScaler()
    ml_df['money_supply_scaled'] = scaler.fit_transform(ml_df[['money_supply']])
    return ml_df

# Function to run the OLS regression 5yr
def run_ols_regression_5yr(ml_df):
    # Define the dependent and independent variables
    X_VARIABLES = ml_df.drop(columns=['Date', 'Yield_Spread_5yr_1yr', 'Yield_Spread_5yr_91_days','money_supply'])
    Y_VARIABLE = ml_df['Yield_Spread_5yr_1yr']
    
    # Add a constant to the independent variables
    X_VARIABLES = sm.add_constant(X_VARIABLES)
    
    # Fit the OLS model
    model = sm.OLS(Y_VARIABLE, X_VARIABLES).fit()
    
    return model

# Function to run the OLS regression 5yr
def run_ols_regression_91days(ml_df):
    # Define the dependent and independent variables
    X_VARIABLES = ml_df.drop(columns=['Date', 'Yield_Spread_5yr_1yr', 'Yield_Spread_5yr_91_days', 'money_supply'])
    Y_VARIABLE = ml_df['Yield_Spread_5yr_91_days']
    
    # Add a constant to the independent variables
    X_VARIABLES = sm.add_constant(X_VARIABLES)
    
    # Fit the OLS model
    model = sm.OLS(Y_VARIABLE, X_VARIABLES).fit()
    
    return model

# Function to run Random Forest regression
def run_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rf, mse, r2, y_pred

# Function to run Gradient Boosting regression
def run_gradient_boosting(X_train, X_test, y_train, y_test):
    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return gb, mse, r2, y_pred

# Streamlit app
def main():
    st.title("OLS Regression Model")
    
    # Load data
    ml_df = load_data()
    
    # Display the data
    st.write("Data Overview")
    st.write(ml_df.head())
    
    if st.button(label='Download data'):
        st.write('loading')
        ml_df.to_csv()

    # Run OLS regression
    model = run_ols_regression_5yr(ml_df)
    
    # Display OLS model summary
    st.header("Scenario 1")
    st.header("Yield_Spread_5yr_1yr - OLS Model Summary")
    st.write(model.summary())
    st.divider()


    st.header("Scenario 2")
    st.header("Yield_Spread_5yr_91_days - OLS Model Summary")
    # Run OLS regression
    model = run_ols_regression_91days(ml_df)
    
    # Display OLS model summary
    st.write(model.summary())
    st.divider()

# Prepare data for machine learning models
    X = ml_df.drop(columns=['Date', 'Yield_Spread_5yr_1yr', 'Yield_Spread_5yr_91_days', 'money_supply'])
    y = ml_df['Yield_Spread_5yr_91_days']
    
    # Split the data into training and testing sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Run Random Forest regression
    rf_model, rf_mse, rf_r2, rf_pred = run_random_forest(X_train, X_test, y_train, y_test)
    
    # Display Random Forest results
    st.header("Random Forest Regression Results")
    st.metric(label='Mean Squared Error', value=rf_mse)
    st.metric(label='R-squared', value=rf_r2)
    
    # Run Gradient Boosting regression
    gb_model, gb_mse, gb_r2, gb_pred = run_gradient_boosting(X_train, X_test, y_train, y_test)
    
    # Display Gradient Boosting results
    st.header("Gradient Boosting Regression Results")
    st.metric(label='Mean Squared Error', value=gb_mse)
    st.metric(label='R-squared', value=gb_r2)
    
    
    # Display predictions
    st.header("Predictions")
    predictions_ml_df = pd.DataFrame({
        'Actual': y_test,
        'OLS_Predicted': model.predict(sm.add_constant(X_test)),
        'Random_Forest_Predicted': rf_pred,
        'Gradient_Boosting_Predicted': gb_pred
    })
    st.write(predictions_ml_df)

    if gb_r2 < rf_r2:
        st.write('Random forest model is the best model')
    else:
        st.write('Gradient boosting is the best model')

    


if __name__ == "__main__":
    main()
