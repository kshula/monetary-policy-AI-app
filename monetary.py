import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
from PIL import Image

@st.cache_data
def load_data():
    # Load data
    df = pd.read_csv('rates.csv', parse_dates=['Date'], dayfirst=True)
    
    return df

def main():
    df = load_data()

    # Sidebar
    page = st.sidebar.selectbox("Select a page", ["Home", "Yield Curve", "Bonds & Treasury Bill", "Interest Rate", "Statistics", "MPC", "Regression"])

    # Home page
    # Home page
    if page == "Home":
        st.title("Monetary Policy AI App")
        st.subheader("Welcome to the Monetary Policy AI App!")
        
        
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

        # About the Creator
        st.header("About the Creator:")
        st.markdown("""
        This app was created by Kampamba Shula.

        **Contact Information:**
        - Email: kampambashula@gmail.com
        - LinkedIn: https://www.linkedin.com/in/kampamba-shula-03946633/
        - GitHub: https://github.com/kshula
        """)

        # Disclaimer
        st.header("Disclaimer:")
        st.markdown("""
        This app is for educational and informational purposes only. The predictions made by the app are based on historical data and should not be considered as financial advice.
        """)

        # Footer
        st.markdown("---")
        st.markdown("© 2023 Monetary Policy AI App. All rights reserved.")

    # Yield Curve page
# Yield Curve page
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

        # Bar chart for last values in columns 1-9
        fig_last_values = px.bar(x=df.columns[1:10], y=df.iloc[-1, 1:10], labels={'value': 'Rate', 'variable': 'Tenor'})
        st.plotly_chart(fig_last_values)
        st.text('This is latest yield curve, a flat or inverted curve indicates a recession')


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
        st.title("Statistics")
        
        # Drop 'mc' column
        df_no_mpc = df.drop('mpc', axis=1)
        
        # Display statistical summary of columns
        st.write(df_no_mpc.describe())
        st.divider()  # 👈 Draws a horizontal rule
        # Correlation heatmap
        corr_matrix = df_no_mpc.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    # MPC page
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
        st.title("Regression Analysis")
        
        
        # Define X and Y variables for regression
        Y = df['base_rate']
        X = df.drop(['Date', 'base_rate', 'mpc'], axis=1)
        
        # Split the data based on dates
        train_size = int(0.8 * len(df))
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
        
        # Separate X and Y for training and testing
        X_train, Y_train = train_data.drop(['Date', 'base_rate', 'mpc'], axis=1), train_data['base_rate']
        X_test, Y_test = test_data.drop(['Date', 'base_rate', 'mpc'], axis=1), test_data['base_rate']
        
        # Create a linear regression model
        model = LinearRegression()
        
        # Fit the model to the training data
        model.fit(X_train, Y_train)
        
        # Predict the target variable on the test set
        Y_pred = model.predict(X_test)
        
        # Display regression results
        st.write(f'Coefficients: {model.coef_}')
        st.write(f'Intercept: {model.intercept_}')
        st.write(f'Mean Squared Error: {mean_squared_error(Y_test, Y_pred)}')
        st.write(f'R-squared: {r2_score(Y_test, Y_pred)}')

if __name__ == "__main__":
    main()
