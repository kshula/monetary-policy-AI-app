import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


@st.cache_data
def load_data():
    # Load data
    df = pd.read_csv('rates.csv', parse_dates=['Date'], dayfirst=True)
    
    return df

@st.cache_data
def load_ml_data():
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



def main():
    df = load_data()

    # Sidebar
    page = st.sidebar.radio("Select a page", ["Home", "Yield Curve", "Bonds & Treasury Bill", "Interest Rate", "Statistics", "MPC", "Regression"])

    # Home page
    # Home page
    if page == "Home":
        st.title("Yield spread and Yield curve Analysis")

        # App Features
        st.header("App Features:")
        st.markdown("""
        - **Yield Curve Analysis:** Explore the yield curve and analyze the trends over time.
        - **Bonds & Treasury Bill:** Visualize data related to bonds and treasury bills.
        - **Interest Rate Trends:** Track lending rate, inflation, lending margin, interbank rate, and base rate.
        - **Statistics:** View statistical summaries and correlation heatmaps.
        - **MPC Prediction:** Predict Monetary Policy Committee (MPC) decisions using machine learning.

        Explore different pages from the sidebar navigation.
        """)
        
        # Instructions on Use
        st.header("Instructions on Use:")
        st.markdown("""
        1. **Yield Curve Page:** Use the slider on your left to select a specific date and explore the yield curve. 
        2. **Bonds & Treasury Bill Page:** Visualize bond and treasury bill data over time.
        3. **Interest Rate Page:** Track various interest rates and trends.
        4. **Statistics Page:** View statistical summaries and correlation heatmaps.
        5. **MPC Page:** Predict MPC decisions based on user input.

        Adjust parameters using sliders and explore different visualizations on each page.

        """)

        # Footer
        st.markdown("---")
        st.markdown("© 2023 Yield spread and Yield curve Analysis.")

# Yield Curve page
    elif page == "Yield Curve":
        st.title("Yield Curve")

        # Bar chart for Yield Curve
        df_long = pd.melt(df, id_vars=['Date'], value_vars=df.columns[1:10], var_name='Tenor', value_name='Rate')
        fig_yield_curve = px.bar(df_long, x='Tenor', y='Rate', color='Tenor', animation_frame='Date',
                                labels={'Rate': 'Rate', 'Tenor': 'Tenor'})
        st.plotly_chart(fig_yield_curve)
        st.divider()  # 👈 Draws a horizontal rule
        st.text('Adjust slider to view yield curve over time.')
        # Line chart for '10_year_minus_91_days' against time
        fig_10_vs_91 = px.line(df, x='Date', y='10_year_minus_91_days', labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_10_vs_91)
        st.divider()  # 👈 Draws a horizontal rule
        st.text('The differnce between 10 year and 91 days shows direction of economy loosely.')
        # Line chart for '10_year_minus_3_year' against time
        fig_10_vs_3 = px.line(df, x='Date', y='10_year_minus_3_year', labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_10_vs_3)
        st.divider()  # 👈 Draws a horizontal rule
        st.text('The difference between 10 year and 3 year bond indicates trajectory of economy')


    # Bonds & Treasury Bill page
    elif page == "Bonds & Treasury Bill":
        st.title("Bonds & Treasury Bill")
        
        # Line chart for Treasury Bills (91_days to 364_days) against time
        fig_treasury_bills = px.line(df, x='Date', y=df.columns[1:5], labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_treasury_bills)
        st.divider()  # 👈 Draws a horizontal rule
        # Line chart for 3_year to 15_year columns against time
        fig_bonds = px.line(df, x='Date', y=df.columns[5:10], labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_bonds)

    # Interest Rate page
    elif page == "Interest Rate":
        st.title("Interest Rate")
        
        
        # Line chart for lending rate, inflation, lending_margin, interbank_rate, and base rate
        fig_interest_rate = px.line(df, x='Date', y=df.columns[10:15], labels={'value': 'Rate', 'variable': 'Interest Rate'})
        st.plotly_chart(fig_interest_rate)

    
    # Statistics page
    elif page == "Statistics":
        st.title("Descriptive Statistics")
        
        # Drop 'mc' column
        df_no_mpc = df.drop(['mpc', 'Date'], axis=1)
        
        # Display statistical summary of columns
        st.write(df_no_mpc.describe())
        st.divider()  # 👈 Draws a horizontal rule
        # Correlation heatmap
        
        
    # MPC page
    elif page == "MPC":
        st.title("MPC Prediction")
        
        # Encode the 'mpc' column using LabelEncoder
        le = LabelEncoder()
        df['mpc_encoded'] = le.fit_transform(df['mpc'])

        # Define X and Y variables for classification
        Y_classification = df['mpc_encoded']

        # Use columns 10-14 for training
        features_for_training = df.columns[11:15]
        X_classification = df[features_for_training]

        # Train the Logistic Regression model
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_classification, Y_classification)

        # Train the Random Forest Classifier model
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_classification, Y_classification)

        # Train the Gradient Boosting Classifier model
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_classification, Y_classification)

        
        # User-adjustable parameters using only columns 10-14
        inflation = st.slider("Inflation", min_value=df['inflation'].min(), max_value=df['inflation'].max())
        reserve = st.slider("Reserve", min_value=df['reserve'].min(), max_value=df['reserve'].max())

        # Create a DataFrame with user-input features for prediction
        user_input = pd.DataFrame({
            'lending_rate': df['lending_rate'].iloc[-1],
            'inflation': [inflation],
            'lending_margin': df['lending_margin'].iloc[-1],  # Use the last available value for lending_margin
            'interbank_rate': df['interbank_rate'].iloc[-1],  # Use the last available value for interbank_rate
            'reserve': [reserve]
        })

        # Select only columns 10-14 for user slider in predictions
        features_for_prediction = df.columns[11:15]
        user_input_slider = user_input[features_for_prediction]

        # Predict using the trained models
        lr_prediction = le.inverse_transform(lr_model.predict(user_input_slider))[0]
        rf_prediction = le.inverse_transform(rf_model.predict(user_input_slider))[0]
        gb_prediction = le.inverse_transform(gb_model.predict(user_input_slider))[0]

       
        # Display predictions
        st.write("Logistic Regression Prediction:", lr_prediction)
        st.write("Random Forest Classifier Prediction:", rf_prediction)
        st.write("Gradient Boosting Classifier Prediction:", gb_prediction)

 
        # Regression page
    elif page == "Regression":
        st.title("Yield Spread Regression Analysis")
        ml_df = load_ml_data()
       
        # Descriptive Statistics
        st.header("Descriptive Statistics")
        st.write(ml_df.describe())
        st.divider()  # 👈 Draws a horizontal rule

        
        # Line chart for '10_year_minus_1yr' against time
        st.header("Line chart for '5_year_minus_91_days' against time")
        fig_Yield_Spread_5yr_1yr = px.bar(ml_df, x='Date', y='Yield_Spread_5yr_1yr', labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_Yield_Spread_5yr_1yr)
        st.divider()  # 👈 Draws a horizontal rule

        st.header("Line chart for '5_year_minus_91_days' against time")
        fig_Yield_Spread_5yr_91_days = px.bar(ml_df, x='Date', y='Yield_Spread_5yr_91_days', labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_Yield_Spread_5yr_91_days)
        st.divider()  # 👈 Draws a horizontal rule

        # Display the data
        st.write("Data Overview")
        st.write(ml_df.head())

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
