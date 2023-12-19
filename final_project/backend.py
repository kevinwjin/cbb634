# Server backend using Streamlit for CBB 634 final project
import streamlit as st # Highly recommended for data visualization dashboards
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Immigration amid Japan's Demographic Crash", page_icon="🇯🇵") # Set page title and favicon

st.title("🇯🇵 Immigration amid Japan's Demographic Crash")

st.header("Part 1: 📉 Demographic crash")

st.subheader("1a: 👶🏻 How does Japan's fertility rate compare to peer nations?")

# Load data for total fertility rates by year and country
fertility_rates = pd.read_csv("datasets/1_demographics/total_fertility_rates_by_year.csv")
fertility_rates.drop(columns=["Country Code", "Indicator Name", "Indicator Code"], inplace=True) # Drop unnecessary columns
fertility_rates = fertility_rates.transpose() # Transpose the dataframe for easier manipulation
fertility_rates.rename(columns=fertility_rates.iloc[0], inplace=True) # Rename columns to country names
fertility_rates.drop(fertility_rates.index[0], inplace = True) # Drop the first row
fertility_rates.insert(0, 'Year', fertility_rates.index) # Add a column for the year
fertility_rates = fertility_rates.reset_index() # Reset the index
fertility_rates.drop(columns=["index"], inplace=True) # Drop the old index
fertility_rates.drop(fertility_rates.tail(1).index, inplace=True) # Drop the last row of 2022, which is a row of NaNs

# Plot fertility rate for Japan only over 1960-2021
fertility_japan = px.line(fertility_rates, x="Year", y="Japan", title="Total Fertility Rate in Japan")
fertility_japan.update_traces(line_color='crimson') # Change color of line
fertility_japan.update_xaxes(tickmode='linear', tick0=1960, dtick=5, tickangle=45) # Only show x-axis labels every 5 years and rotate them 45 degrees
fertility_japan.update_yaxes(range=[0, 8]) # Set y-axis range to 0-8
fertility_japan.update_layout(title_text='Total Fertility Rate of Japan (1960 - 2021)', xaxis_title='Year', yaxis_title='Total Fertility Rate') # Plot and axis titles
fertility_japan.add_hline(y=2.1, line_dash="dash", line_color="black", annotation_text="Global Replacement Fertility Rate", annotation_position="top right") # Indicate global average fertility rate of 2.1 with horizontal line

# Plot fertility rates for Japan and peer nations over 1960-2021
fertility_peer_nations = px.line(fertility_rates, x="Year", y=["Japan", "China", "Korea, Rep.", "India", "United States", "Germany", "France", "United Kingdom", "Italy", "Canada", "Australia", "Spain", "Russian Federation"], title="Total Fertility Rates of Japan and Selected Nations")
fertility_peer_nations.update_xaxes(tickmode='linear', tick0=1960, dtick=5, tickangle=45) # Only show x-axis labels every 5 years and rotate them 45 degrees
fertility_peer_nations.update_traces(line=dict(color='crimson'), selector=dict(name='Japan')) # Change the color of the Japan line
fertility_peer_nations.update_yaxes(range=[0, 8]) # Set y-axis range to 0-8
fertility_peer_nations.update_layout(title_text='Total Fertility Rates of Japan and Selected Nations (1960 - 2021)', xaxis_title='Year', yaxis_title='Total Fertility Rate') # Plot and axis titles
fertility_peer_nations.update_layout(legend_title_text='Country') # Move the legend to the top right corner
fertility_peer_nations.add_hline(y=2.1, line_dash="dash", line_color="black", annotation_text="Global Replacement Fertility Rate", annotation_position="top right") # Indicate global average fertility rate of 2.1 with horizontal line

# Create tabs for the two plots
japan_only, japan_with_peer_nations = st.tabs(["Japan", "Japan with Peer Nations"])
with japan_only:
    st.plotly_chart(fertility_japan, use_container_width=True)
with japan_with_peer_nations:
    st.plotly_chart(fertility_peer_nations, use_container_width=True)

st.subheader("1b: 💸 What is causing falling fertility rates?")

# Load wage data for Japan
wages = pd.read_csv("datasets/1_demographics/average_wages.csv")
wages_japan = wages[wages['LOCATION'] == "JPN"] # Take only Japan rows
wages_japan = wages_japan[['TIME', 'Value']] # Grab year and value columns
wages_japan.rename(columns={"TIME": "Year", "Value": "Average Wage (USD)"}, inplace=True) # Rename columns
wages_japan.reset_index(drop=True, inplace=True) # Reset the index

# Grab fertility rates for Japan from 1991 - 2022
fertility_rates_japan = fertility_rates[["Year", "Japan"]].apply(pd.to_numeric) # Convert columns to numeric
fertility_rates_japan.rename(columns={"Japan": "Total Fertility Rate"}, inplace=True) # Rename column
fertility_rates_japan = fertility_rates_japan[fertility_rates_japan['Year'] >= 1991] # Take only rows from 1991 onwards

# Combine the two dataframes
wages_japan = wages_japan.join(fertility_rates_japan.set_index('Year'), on='Year') # Join the two dataframes on the year column

# Plot average wages against fertility rates for Japan over 1991 - 2022
wages_japan_plot = px.scatter(wages_japan, x="Average Wage (USD)", y="Total Fertility Rate", trendline="ols", trendline_color_override="red") # Plot scatter plot with linear trendline
wages_japan_plot.update_layout(title_text='Average Wage vs. Total Fertility Rate in Japan (1991 - 2022)', xaxis_title='Average Wage (USD)', yaxis_title='Total Fertility Rate') # Plot and axis titles

# Repeat for peer nations (United States, Canada, South Korea, United Kingdom, Germany, France) using multiple subplots
from plotly.subplots import make_subplots

# Grab fertility rates for peer nations from 1991 - 2022
fertility_rates_peers = fertility_rates[["Year", "United States", "Canada", "Korea, Rep.", "United Kingdom", "Germany", "France"]].apply(pd.to_numeric) # Convert columns to numeric
fertility_rates_peers.rename(columns={"United States": "United States", "Canada": "Canada", "Korea, Rep.": "South Korea", "United Kingdom": "United Kingdom", "Germany": "Germany", "France": "France"}, inplace=True) # Rename columns
fertility_rates_peers = fertility_rates_peers[fertility_rates_peers['Year'] >= 1991] # Take only rows from 1991 onwards

# Combine fertility rates with average wages for peer nations
wages_peers = wages[wages['LOCATION'].isin(["USA", "CAN", "KOR", "GBR", "DEU", "FRA"])] # Take only peer nations
wages_peers = wages_peers[['LOCATION', 'TIME', 'Value']] # Grab country, year, and value columns
wages_peers.rename(columns={"TIME": "Year", "Value": "Average Wage (USD)"}, inplace=True) # Rename columns
wages_peers.reset_index(drop=True, inplace=True) # Reset the index
wages_peers = wages_peers[wages_peers['Year'] >= 1991] # Take only rows from 1991 onwards
wages_peers = wages_peers.pivot(index='Year', columns='LOCATION', values='Average Wage (USD)') # Pivot the dataframe to make country codes into columns
wages_peers.reset_index(inplace=True) # Reset the index
wages_peers = wages_peers.apply(pd.to_numeric) # Convert columns to numeric
wages_peers = wages_peers.join(fertility_rates_peers.set_index('Year'), on='Year') # Join the two dataframes on the year column

# Plot average wages against fertility rates for peer nations over 1991 - 2022
wages_peers_plot = make_subplots(rows=2, cols=3, subplot_titles=("United States", "Canada", "South Korea", "United Kingdom", "Germany", "France"), x_title="Average Wage (USD)", y_title="Total Fertility Rate", ) # Create subplots
wages_peers_plot.add_trace(px.scatter(wages_peers, x="USA", y="United States", trendline="ols", trendline_color_override="red").data[0], row=1, col=1) 
wages_peers_plot.add_trace(px.scatter(wages_peers, x="CAN", y="Canada", trendline="ols", trendline_color_override="red").data[0], row=1, col=2)
wages_peers_plot.add_trace(px.scatter(wages_peers, x="KOR", y="South Korea", trendline="ols", trendline_color_override="red").data[0], row=1, col=3)
wages_peers_plot.add_trace(px.scatter(wages_peers, x="GBR", y="United Kingdom", trendline="ols", trendline_color_override="red").data[0], row=2, col=1)
wages_peers_plot.add_trace(px.scatter(wages_peers, x="DEU", y="Germany", trendline="ols", trendline_color_override="red").data[0], row=2, col=2)
wages_peers_plot.add_trace(px.scatter(wages_peers, x="FRA", y="France", trendline="ols", trendline_color_override="red").data[0], row=2, col=3)
wages_peers_plot.update_layout(title_text='Average Wage vs. Total Fertility Rate in Peer Nations (1991 - 2022)') # Plot title

fertility_japan_tab, fertility_peer_nations_tab = st.tabs(["Japan", "Peer Nations"])
with fertility_japan_tab:
    st.plotly_chart(wages_japan_plot, use_container_width=True)
with fertility_peer_nations_tab:
    st.plotly_chart(wages_peers_plot, use_container_width=True)

st.header("Part 2: ✈️ Immigration")

st.subheader("2a: 👷🏻‍♀️ What kinds of people are coming to Japan?")
# Load visa data for Japan
visas = pd.read_csv("datasets/2_immigration/visa_number_in_japan.csv")

# Plot total visas issued by Japan from 2006-2017
total = visas[visas["Country"] == "total"] # Pull country totals
visas = visas[visas["Country"] != "total"].reset_index(drop=True) # Delete non-total rows
visas_japan = px.bar(total, x="Year", y="Number of issued")
visas_japan.update_layout(title="Total visas issued by Japan from 2006-2017")
st.plotly_chart(visas_japan, use_container_width=True)

# Plot visas issued for top ten countries of issuance except China from 2006-2017
top_countries = visas[visas['Country'] != 'total'].groupby('Country')['Number of issued'].sum().nlargest(10).index
top_countries_data = visas[(visas['Country'].isin(top_countries)) & (visas['Country'] != 'total')]
top_countries_data = top_countries_data[top_countries_data['Country'] != 'China']
visas_top_ten_no_china = px.line(top_countries_data, x='Year', y='Number of issued', color='Country')
visas_top_ten_no_china.update_layout(title='Number of Visas Issued for Top Ten Countries of Issuance, excluding China (2006-2017)',
                  xaxis_title='Year',
                  yaxis_title='Number of Visas Issued')

# Plot visas issued for top ten countries of issuance (with China) from 2006-2017
top_countries = visas[visas['Country'] != 'total'].groupby('Country')['Number of issued'].sum().nlargest(10).index
top_countries_data = visas[(visas['Country'].isin(top_countries)) & (visas['Country'] != 'total')]

visas_top_ten = px.line(top_countries_data, x='Year', y='Number of issued', color='Country')
visas_top_ten.update_layout(title='Number of Visas Issued Over Time for Top Ten Countries of Issuance (2006-2017)',
                  xaxis_title='Year',
                  yaxis_title='Number of Visas Issued')

visas_top_ten_tab, visas_no_china_tab = st.tabs(["With China", "Without China"])
with visas_top_ten_tab:
    st.plotly_chart(visas_top_ten, use_container_width=True)
with visas_no_china_tab:
    st.plotly_chart(visas_top_ten_no_china, use_container_width=True)

st.subheader("Tangent: 📊 Predicting the country of future immigrants with Random Forests")

# Attempt to predict the number of visas issued for each country in 2018 using Random Forests
num_estimators = st.number_input("Specify the number of tree estimators", min_value=10, max_value=1000, value=100, step=1, format="%d")


# API endpoint for Random Forests classifier
def random_forests_classifier(num_estimators):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    # Load the dataset
    visas = pd.read_csv("datasets/2_immigration/visa_number_in_japan.csv")

    # Preprocess the data
    X = visas.drop(columns=["Country"])
    X.fillna(0, inplace=True) # Replace NaN with 0
    y = visas["Country"]

    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    progress_text = f"Fitting model with {num_estimators} estimators..."
    progress_bar = st.progress(0, text=progress_text)
    progress_bar.progress(25, text=progress_text)
    # Create and train the Random Forest classifier
    n_estimators = num_estimators  # Number of trees in the forest
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    progress_text = "Making predictions..."
    progress_bar.progress(40, text=progress_text)
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    progress_text = "Evaluating model..."
    progress_bar.progress(50, text=progress_text)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    progress_text = "Generating a prediction..."
    progress_bar.progress(75, text=progress_text)
    # Make a prediction
    prediction = model.predict(X_test[0].reshape(1, -1))
    # Print the prediction
    st.write("Predicted country:", prediction[0])
    
    progress_text = "Plotting feature importance..."
    progress_bar.progress(100, text=progress_text)
    # Plot feature importances of the Random Forest classifier
    import plotly.graph_objects as go

    # Get feature importances
    importances = model.feature_importances_

    # Get feature names
    feature_names = X.columns

    # Sort feature importances in descending order
    indices = importances.argsort()[::-1]

    feature_importance = go.Figure(data=[go.Bar(x=feature_names[indices], y=importances[indices])])
    feature_importance.update_layout(
        title="Feature importance of the Random Forest classifier",
        xaxis_title="Features",
        yaxis_title="Importance",
        xaxis_tickangle=-90
    )
    st.plotly_chart(feature_importance, use_container_width=True)
    progress_text = "Model fitting complete."
    progress_bar.progress(100, text=progress_text)


random_forests_classifier(num_estimators)

st.subheader("2b: 🗼 Where are they going?")

# Load administrative boundary data for Japan's prefectures
import geopandas as gpd

japan = gpd.read_file("datasets/2_immigration/prefectures.geojson")
prefectures_english = [
    "Hokkaido", "Aomori", "Iwate", "Miyagi", "Akita", "Yamagata", "Fukushima", "Ibaraki", "Tochigi", 
    "Gunma", "Saitama", "Chiba", "Tokyo", "Kanagawa", "Niigata", "Toyama", "Ishikawa", "Fukui", 
    "Yamanashi", "Nagano", "Gifu", "Shizuoka", "Aichi", "Mie", "Shiga", "Kyoto", "Osaka", "Hyogo", 
    "Nara", "Wakayama", "Tottori", "Shimane", "Okayama", "Hiroshima", "Yamaguchi", "Tokushima", 
    "Kagawa", "Ehime", "Kochi", "Fukuoka", "Saga", "Nagasaki", "Kumamoto", "Oita", "Miyazaki", 
    "Kagoshima", "Okinawa"
]
japan['P'] = prefectures_english
japan.rename(columns={'P': 'Prefecture'}, inplace=True)

# Load foreign resident data
foreigners = pd.read_csv("datasets/2_immigration/foreign_residents.csv")
foreigners = foreigners.drop(0) # Drop first row

# Merge two dataframes
japan_foreigners = japan.merge(foreigners, on='Prefecture') # Add foreign resident data to prefecture geodataframe
japan_foreigners[japan_foreigners.columns[2:]] = japan_foreigners[japan_foreigners.columns[2:]].replace(',','', regex=True) # Strip commas from numbers 
japan_foreigners[japan_foreigners.columns[2:]] = japan_foreigners[japan_foreigners.columns[2:]].apply(pd.to_numeric)

# Plot map of Japan prefectures colored by number of foreign residents
foreigners_map = px.choropleth_mapbox(japan_foreigners, geojson=japan_foreigners.geometry, locations=japan_foreigners.index, color="Total", hover_name="Prefecture", hover_data="Total", mapbox_style="carto-positron", zoom=3, center = {"lat": 37.0902, "lon": 138.7129}, opacity=0.5, labels={'Total':'Number of Foreign Residents'})
foreigners_map.update_layout(title_text='Number of Foreign Residents in Japan by Prefecture (2023)')
with st.spinner("Generating map..."):
    st.plotly_chart(foreigners_map, use_container_width=True)
