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
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import het_white, het_white
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.tsa.vector_ar.vecm as vecm
from statsmodels.tsa.vector_ar.vecm import VECM

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
    ml_df = ml_df.drop(columns=['Yield_Spread_5yr_91_days', 'Date','money_supply']) 
    return ml_df


# Function to run the OLS regression 5yr
def run_ols_regression_5yr(ml_df):
    # Define the dependent and independent variables
    X_VARIABLES = ml_df.drop(columns=['Yield_Spread_5yr_1yr'])
    Y_VARIABLE = ml_df['Yield_Spread_5yr_1yr']
    
    # Add a constant to the independent variables
    X_VARIABLES = sm.add_constant(X_VARIABLES)
    
    # Fit the OLS model
    model = sm.OLS(Y_VARIABLE, X_VARIABLES).fit()
    
    return model

# Function to run the OLS regression 5yr
def run_ols_regression_91days(ml_df):
    # Define the dependent and independent variables
    X_VARIABLES = ml_df.drop(columns=['Yield_Spread_5yr_1yr'])
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

# Step 2: Perform Unit Root Test (ADF Test)
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return result[0], result[1], result[4]['5%']

# Step 3: Perform Cointegration Test
def cointegration_test(y, x):
    result = coint(y, x)
    return result[0], result[1], result[2][1]  # t-statistic, p-value, critical value (5%)

# Step 4: Perform Multicollinearity Test (VIF)
def multicollinearity_test(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Step 5: Perform Heteroskedasticity Test (White's Test)
def heteroskedasticity_test(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    white_test = het_white(model.resid, model.model.exog)
    return white_test[0], white_test[1]  # LM statistic and its p-value

# Step 6: Perform Autocorrelation Test (Durbin-Watson Test)
def autocorrelation_test(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    dw_stat = sm.stats.durbin_watson(model.resid)
    return dw_stat

# Step 2: Stationarity Test (ADF Test)
def adf_test(series):
    result = adfuller(series)
    return {'ADF Statistic': result[0], 'p-value': result[1]}

# Step 3: Cointegration Test (Johansen Test)
def johansen_test(df):
    # Dropping the 'Date' column and converting the dataframe to a numpy array
    data = df
    # Performing the Johansen cointegration test
    result = vecm.coint_johansen(data, det_order=0, k_ar_diff=1)
    return result

# Step 4: Multicollinearity Check (VIF)
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data['feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

# Step 5: Heteroskedasticity Test (White's Test)
def white_test(df, y_var):
    X = sm.add_constant(df)
    y = df[y_var]
    model = sm.OLS(y, X).fit()
    white_test_result = het_white(model.resid, model.model.exog)
    return white_test_result

# Step 6: Autocorrelation Test (Durbin-Watson)
def durbin_watson_test(df, y_var):
    X = sm.add_constant(df)
    y = df[y_var]
    model = sm.OLS(y, X).fit()
    dw_stat = durbin_watson(model.resid)
    return dw_stat

def apply_regularization(df, target_var):
    # Separate predictors and target variable
    X = df.drop(columns=[target_var])
    y = df[target_var]

    # Add a constant to the predictors (intercept)
    X = sm.add_constant(X)

    # Apply Elastic Net regularization as an example
    model = sm.OLS(y, X).fit_regularized(method='elastic_net', alpha=1.0, L1_wt=0.5)  # L1_wt=0.5 is a mix of L1 and L2

    # Extract model coefficients
    coefficients = pd.Series(model.params, index=X.columns)

    # Manually calculate metrics
    fitted_values = model.predict(X)
    residuals = y - fitted_values
    ssr = (residuals ** 2).sum()  # Sum of Squared Residuals
    sst = ((y - y.mean()) ** 2).sum()  # Total Sum of Squares
    r_squared = 1 - ssr / sst

    # Display results
    st.write("Coefficients:")
    st.write(coefficients)
    st.write("\nR-squared:", r_squared)

    # Return the model and important metrics
    return model, coefficients, r_squared

def vecm_model(df):
    # Fit a VECM model for cointegration analysis
    vecm = VECM(df, k_ar_diff=1, coint_rank=1)
    vecm_fit = vecm.fit()
    return vecm_fit

def main():
    df = load_data()

    # Sidebar
    page = st.sidebar.radio("Select a page", ["Home", "Yield Curve", "Bonds & Treasury Bill", "Interest Rate", "Statistics", "MPC", "Regression", "correlation","diagnostic tests", "New diagnostic tests"])

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
        df = load_data()
       
        # Descriptive Statistics
        st.header("Descriptive Statistics")
        st.write(df.describe())
        st.divider()  # 👈 Draws a horizontal rule

        
        # Line chart for '10_year_minus_1yr' against time
        st.header("Line chart for '5_year_minus_1year' against time")
        fig_Yield_Spread_5yr_1yr = px.bar(df, x='Date', y='Yield_Spread_5yr_1yr', labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_Yield_Spread_5yr_1yr)
        st.divider()  # 👈 Draws a horizontal rule

        st.header("Line chart for '5_year_minus_91_days' against time")
        fig_Yield_Spread_5yr_91_days = px.bar(df, x='Date', y='Yield_Spread_5yr_91_days', labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_Yield_Spread_5yr_91_days)
        st.divider()  # 👈 Draws a horizontal rule

        # Display the data
        st.write("Data Overview")
        ml_df = load_ml_data()
        st.write(ml_df.head())

        # Run OLS regression
        model = run_ols_regression_5yr(ml_df)
        
        # Display OLS model summary
        st.header("Scenario 1")
        st.header("Yield_Spread_5yr_1yr - OLS Model Summary")
        st.write(model.summary())
        st.divider()
        
        # Display OLS model summary
        st.write(model.summary())
        st.divider()

    # Prepare data for machine learning models
        X = ml_df.drop(columns=[ 'Yield_Spread_5yr_1yr'])
        y = ml_df['Yield_Spread_5yr_1yr']
        
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

    elif page == "correlation":
        # Step 2: Display the Data
        ml_df = load_ml_data()
        st.write("Data Preview:", ml_df)

        # Step 3: Compute the Correlation Matrix
        correlation_matrix = ml_df.corr()

        # Step 4: Create the Plotly Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis'))

        fig.update_layout(title='Correlation Heatmap', xaxis_title='Variables', yaxis_title='Variables')

        # Step 5: Display the Heatmap in Streamlit
        st.plotly_chart(fig)

    elif page == "diagnostic tests":
        st.header("Diagnostic tests 1")
        ml_df = load_ml_data()
        # Stationarity Test
        st.write("The ADF test checks whether each series is stationary. A p-value < 0.05 suggests the series is stationary.")
        st.subheader("Stationarity Test (ADF Test)")
        for column in ml_df.columns:
            if column != 'Date':
                adf_result = adf_test(ml_df[column])
                st.write(f"{column}: ADF Statistic = {adf_result['ADF Statistic']}, p-value = {adf_result['p-value']}")

        # Cointegration Test
        st.subheader("Cointegration Test (Johansen Test)")
        st.write("The Johansen test checks for long-term relationships among the variables. If trace and max-eigen statistics are greater than their critical values, it suggests cointegration.")
        johansen_result = johansen_test(ml_df)
        st.write("Trace Statistic:", johansen_result.lr1)
        st.write("Critical Values (Trace):", johansen_result.cvt)
        st.write("Max-Eigen Statistic:", johansen_result.lr2)
        st.write("Critical Values (Max-Eigen):", johansen_result.cvm)

        # Multicollinearity Test
        st.subheader("Multicollinearity Test (VIF)")
        st.write("VIF assesses multicollinearity. A VIF > 10 suggests high multicollinearity, which can distort regression results.")
        vif_result = calculate_vif(ml_df)
        st.write(vif_result)

        # Heteroskedasticity Test
        st.subheader("Heteroskedasticity Test (White's Test)")
        st.write("White's test checks for heteroskedasticity, where variance of errors is not constant. A low p-value indicates heteroskedasticity.")
        white_result = white_test(ml_df, 'Yield_Spread_5yr_1yr')
        st.write(f"Test Statistic = {white_result[0]}, p-value = {white_result[1]}")

        # Autocorrelation Test
        st.subheader("Autocorrelation Test (Durbin-Watson)")
        st.write("Durbin-Watson tests for autocorrelation in residuals. Values close to 2 indicate no autocorrelation.")
        dw_stat = durbin_watson_test(ml_df, 'Yield_Spread_5yr_1yr')
        st.write(f"Durbin-Watson Statistic = {dw_stat}")

    elif page == "New diagnostic tests":
        ml_df = load_ml_data()

    # Lag Variable Addition
    st.subheader("Adding Lagged Variables")
    for column in ml_df.columns:
        if column != 'Date':
            ml_df[f"{column}_lag1"] = ml_df[column].shift(12)
    ml_df.dropna(inplace=True)

    # Stationarity Test
    st.subheader("Stationarity Test (ADF Test)")
    for column in ml_df.columns:
        if column != 'Date':
            adf_result = adf_test(ml_df[column])
            st.write(f"{column}: ADF Statistic = {adf_result['ADF Statistic']}, p-value = {adf_result['p-value']}")

    # Cointegration Test
    st.subheader("Cointegration Test (Johansen Test)")
    johansen_result = johansen_test(ml_df)
    st.write("Trace Statistic:", johansen_result.lr1)
    st.write("Critical Values (Trace):", johansen_result.cvt)
    st.write("Max-Eigen Statistic:", johansen_result.lr2)
    st.write("Critical Values (Max-Eigen):", johansen_result.cvm)

    # Multicollinearity Test with Regularization
    st.subheader("Multicollinearity Test with Regularization")
    vif_result = calculate_vif(ml_df)
    st.write(vif_result)
    st.write("Applying regularization to handle multicollinearity.")
    st.write("Regularized Model Summary:")
    reg_model = apply_regularization(ml_df, 'Yield_Spread_5yr_1yr')

    # Heteroskedasticity Test with Robust Standard Errors
    st.subheader("Heteroskedasticity Test (White's Test)")
    white_result = white_test(ml_df, 'Yield_Spread_5yr_1yr')
    st.write(f"Test Statistic = {white_result[0]}, p-value = {white_result[1]}")
    st.write("Using robust standard errors to correct for heteroskedasticity.")
    model = sm.OLS(ml_df['Yield_Spread_5yr_1yr'], sm.add_constant(ml_df.drop(columns=['Yield_Spread_5yr_1yr']))).fit(cov_type='HC3')
    st.write(model.summary())

    # Autocorrelation Test
    st.subheader("Autocorrelation Test (Durbin-Watson)")
    dw_stat = durbin_watson_test(ml_df, 'Yield_Spread_5yr_1yr')
    st.write(f"Durbin-Watson Statistic = {dw_stat}")

    # Fitting a VECM Model
    st.subheader("VECM Model for Cointegration")
    vecm_fit = vecm_model(ml_df)
    st.write("VECM Model Summary:")
    st.write(vecm_fit.summary())

if __name__ == "__main__":
    main()
