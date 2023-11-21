import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title='GDP of World',  layout='wide', page_icon=':global:')

# Load your dataset
df = pd.read_csv('world_development_data_imputed.csv')

df['Year'] = df['Year'].astype(int)

# Calculate the distribution of regions
region_distribution = df['Region'].value_counts()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Data Visualization", "Linear Regression", "Random Forest Regressor"])

# Display the selected page
if page == "Home":
    st.header("Global Socio-Economic & Demographic Insights")
    st.image('https://i.imgur.com/kv1SBoB.png',  use_column_width=True)
    st.write("Analyzing global socio-economic and demographic trends reveals a complex tapestry of interconnected factors shaping the world's development. Key indicators, including GDP, education levels, and healthcare access, showcase the diverse trajectories of nations. Demographic insights, such as population growth, life expectancy, and migration patterns, underscore the dynamic nature of global societies. Examining these trends over time and across regions provides critical insights into economic development, social well-being, and disparities among nations. The interplay between socio-economic and demographic variables unveils intricate relationships, highlighting the need for comprehensive strategies to address global challenges. Through rigorous data collection and analysis, researchers gain a nuanced understanding of the forces influencing societies worldwide, contributing to informed policy decisions and sustainable development goals.")
    
    st.image('https://i.imgur.com/x29Oppp.png',  use_column_width=True)
    

elif page == "Data Exploration":
    # Section 1: GDP Growth in 2000 - Top Countries
    st.header(' GDP  of  Top Countries')

    # Filter data for the year 2000
    df_2000 = df[df['Year'] == 2000]

    # Display a sample of the DataFrame
    st.subheader("Sample of DataFrame")
    st.write(df)

    # Columns to drop
    columns_to_drop = ['PopTotal', 'PopDens', 'PopGrowth%', 'UrbanPopGrowth%']

    # Drop columns
    df.drop(columns=columns_to_drop, inplace=True)

    # Streamlit App
    st.title('DataFrame After Dropping Columns')

    # Display the modified DataFrame
    st.dataframe(df)

elif page == "Data Visualization":
    st.header("Data Visualization")
    st.write(f"### GDP  of the  Year - Top Countries")

    # Filter data for the selected year
    selected_year = st.sidebar.selectbox("Select a Year", sorted(df['Year'].unique()))
    df_selected_year = df[df['Year'] == selected_year]

    # Set up the matplotlib figure for GDP Growth
    fig_gdp_growth, ax_gdp_growth = plt.subplots(figsize=(20, 40))

    # Create a bar chart using Seaborn
    sns.barplot(x='GDP', y='Country', data=df_selected_year, palette='viridis', ax=ax_gdp_growth)

    # Customize the plot
    ax_gdp_growth.set_xlabel('GDP  Percentage')
    ax_gdp_growth.set_ylabel('Country')
    ax_gdp_growth.set_title(f'GDP in {selected_year} - Top Countries')

    # Display the plot in Streamlit
    st.pyplot(fig_gdp_growth)

    st.write(f"### GDP in {selected_year} - Top Countries")

    # Set up the matplotlib figure for GDP
    fig_gdp, ax_gdp = plt.subplots(figsize=(20, 40))

    # Create a bar chart using Seaborn
    sns.barplot(x='GDP', y='Country', data=df_selected_year, palette='viridis', ax=ax_gdp)

    # Customize the plot
    ax_gdp.set_xlabel('GDP')
    ax_gdp.set_ylabel('Country')
    ax_gdp.set_title(f'GDP in {selected_year} - Top Countries')

    # Display the plot in Streamlit
    st.pyplot(fig_gdp)

    st.write(f"### Distribution of Regions in {selected_year}")

    # Set up the matplotlib figure for the pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))

    # Create a pie chart
    ax_pie.pie(region_distribution, labels=region_distribution.index, autopct='%1.1f%%', startangle=90)

    # Customize the plot
    ax_pie.set_title(f'Distribution of Regions in {selected_year}')

    # Display the pie chart in Streamlit
    st.pyplot(fig_pie)

    # Section 4: Correlation Heatmap - Selected Year
    st.subheader(f"Correlation Heatmap - {selected_year}")

    # Select relevant numerical columns for correlation analysis
    numerical_columns = ['SurfAreaSqKm', 'GDP', 'GDPGrowth%', 'GNIAtlas',
                         'Imports%GDP', 'IndValAdd%GDP', 'InflConsPric%', 'LifeExpBirth', 'MerchTrade%GDP',
                         'MobileSubs/100', 'MortRateU5', 'NetMigr']

    # Calculate the correlation matrix
    correlation_matrix = df_selected_year[numerical_columns].corr()

    # Set up the matplotlib figure for the heatmap
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 10))

    # Create a heatmap using Seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_heatmap)

    # Customize the plot
    ax_heatmap.set_title(f'Correlation Heatmap - {selected_year}')

    # Display the heatmap in Streamlit
    st.pyplot(fig_heatmap)

elif page == "Linear Regression":
    # Section 2: Linear Regression Predictions
    st.header(' Linear Regression Predictions')

    # Selecting features and target variable
    features = ['SurfAreaSqKm', 'LifeExpBirth', 'GDPGrowth%', 'AdolFertRate', 'AgriValAdd%GDP', 'Exports%GDP',
                'GNIAtlas', 'Imports%GDP', 'IndValAdd%GDP', 'InflConsPric%', 'MerchTrade%GDP', 'MobileSubs/100',
                'MortRateU5', 'NetMigr']
    target = 'GDP'

    # Dropping rows with missing values for simplicity
    df_ml = df.dropna(subset=features + [target])

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_ml[features], df_ml[target], test_size=0.2, random_state=42)

    # Creating and training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions on the test set
    predictions = model.predict(X_test)

    # Calculating R-squared for actual and predicted values
    r2_actual = r2_score(y_test, y_test)  # Should be 1.0 for actual values
    r2_predicted = r2_score(y_test, predictions)

    # Display model performance metrics
    st.subheader('Model Performance Metrics')
    st.write(f'Actual R-squared: {r2_actual}')
    st.write(f'Predicted R-squared: {r2_predicted}')

    # Section 3: Scatter Plot of Actual vs. Predicted GDP
    st.subheader('Scatter Plot of Actual vs. Predicted GDP')

    # Plotting predicted vs. actual values
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, predictions)
    ax3.set_xlabel('Actual GDP')
    ax3.set_ylabel('Predicted GDP')
    ax3.set_title('Actual vs. Predicted GDP')

    # Display the plot in the Streamlit app
    st.pyplot(fig3)

elif page == "Random Forest Regressor":
    # Section 3: Random Forest Regressor
    st.header(' Random Forest Regressor')

    # Selecting features and target variable for Random Forest Regressor
    features_rf = ['SurfAreaSqKm', 'LifeExpBirth', 'AdolFertRate', 'AgriValAdd%GDP', 'Exports%GDP',
                   'GNIAtlas', 'Imports%GDP', 'IndValAdd%GDP', 'InflConsPric%', 'MerchTrade%GDP', 'MobileSubs/100',
                   'MortRateU5', 'NetMigr']
    target_rf = 'GDP'

    # Dropping rows with missing values for simplicity
    df_rf = df.dropna(subset=features_rf + [target_rf])

    # Splitting the data into training and testing sets
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(df_rf[features_rf], df_rf[target_rf],
                                                                    test_size=0.2, random_state=42)

    # Creating and training the Random Forest Regressor model
    model_rf = RandomForestRegressor()
    model_rf.fit(X_train_rf, y_train_rf)

    # Making predictions on the test set
    predictions_rf = model_rf.predict(X_test_rf)

    # Calculating R-squared for Random Forest Regressor
    r2_rf = r2_score(y_test_rf, predictions_rf)
    mse_rf = mean_squared_error(y_test_rf, predictions_rf)

    # Display model evaluation metrics
    st.subheader('Model Evaluation Metrics for Random Forest Regressor')
    st.write(f'Mean Squared Error: {mse_rf}')
    st.write(f'R-squared: {r2_rf}')

    # Plotting predicted vs. actual values
    st.subheader('Actual vs. Predicted GDP for Random Forest Regressor')
    fig_rf, ax_rf = plt.subplots()
    ax_rf.scatter(y_test_rf, predictions_rf)
    ax_rf.set_xlabel('Actual GDP')
    ax_rf.set_ylabel('Predicted GDP')
    ax_rf.set_title('Actual vs. Predicted GDP for Random Forest Regressor')

    # Display the plot in the Streamlit app
    st.pyplot(fig_rf)




