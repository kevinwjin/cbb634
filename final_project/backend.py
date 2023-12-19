# Server backend using Streamlit for CBB 634 final project
import streamlit as st # Highly recommended for data visualization dashboards
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Immigration amid Japan's Demographic Crash", page_icon="🇯🇵")
st.title("Immigration amid Japan's Demographic Crash 🇯🇵")

st.header("Introduction")
st.markdown("""
In July 2023, the Japanese government released data indicating that in 2022, the number of people in all 47 prefectures of the country fell for the first time since the government began tracking the data in 1968.[^1] Last year marked the 14th consecutive year that Japan's population has been falling and for the first time included Okinawa prefecture, which has historically had a high birthrate.[^2]

Being a relatively new and unprecedented phenomenon, the consequences of population decline are chiefly theoretical; however, their beginnings are already being felt by Japanese society. The main detriment of a middle- or top-heavy population pyramid is a rise in the dependency ratio, or heavier reliance by the older generations on the younger generations and greater economic pressure on the workforce as a result.[^3] Greater risk of recession and decreased prosperity follow, accompanied by less innovation, worsened culture, declining military strength, strained mental health resources, and so on.[^4] As such, with a population already skewing older, Japan faces grim economic prospects. The Japanese government has invested a tremendous amount of resources into addressing the issue, yet heated debate continues over the causes of Japan's population decline and the potential solution of immigration and its implications.[^5]

This report will present Japan's ongoing demographic crash in graphic detail and attempt to identify causes. It will also spotlight immigration and provide spatial insights into the state of immigration in Japan.

[^1]: https://english.kyodonews.net/news/2023/07/c6b8e75dc7a9-japanese-population-falls-in-all-47-prefectures-for-1st-time.html
[^2]: https://www.bloomberg.com/news/articles/2023-07-26/japanese-population-falls-in-all-47-prefectures-for-first-time
[^3]: https://www.youtube.com/watch?v=LBudghsdByQ
[^4]: https://en.wikipedia.org/w/index.php?title=Population_decline#Possible_consequences
[^5]: https://www.economist.com/asia/2023/12/16/how-to-entice-japanese-couples-to-have-babies
""")

st.header("Part 1: Demographic crash 📉")

st.subheader("1a: How does Japan's fertility rate compare to other nations?")

# Load data for total fertility rates by year and country
fertility_rates = pd.read_csv("datasets/1_demographics/total_fertility_rates_by_year.csv")
fertility_rates.drop(columns=["Country Code", "Indicator Name", "Indicator Code"], inplace=True) # Drop unnecessary columns
fertility_rates = fertility_rates.transpose() # Transpose the dataframe for easier manipulation
fertility_rates.rename(columns=fertility_rates.iloc[0], inplace=True) # Rename columns to country names
fertility_rates.drop(fertility_rates.index[0], inplace = True) # Drop the first row
fertility_rates.insert(0, 'Year', fertility_rates.index) # Add a column for the year
fertility_rates = fertility_rates.reset_index() # Reset the index
fertility_rates.head()
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

japan_only, japan_with_peer_nations = st.tabs(["Japan only", "Japan with peer nations"])
with japan_only:
    st.plotly_chart(fertility_japan, use_container_width=True)
with japan_with_peer_nations:
    st.plotly_chart(fertility_peer_nations, use_container_width=True)

st.subheader("1b: What is causing falling fertility rates?")


st.header("Part 2: Immigration ✈️")

st.subheader("2a: What kinds of people are coming to Japan?")

st.subheader("2b: Where are they going?")

st.header("Conclusion")
