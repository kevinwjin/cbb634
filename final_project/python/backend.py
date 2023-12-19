# Server backend using Streamlit for CBB 634 final project
import streamlit as st # Highly recommended for data visualization dashboards
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Immigration amid Japan's Demographic Crash")

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

st.header("Part 1: Demographic crash")

st.subheader("1a: How does Japan's fertility rate compare to other nations?")

st.subheader("1b: What is causing falling fertility rates?")


st.header("Part 2: Immigration")

st.subheader("2a: What kinds of people are coming to Japan?")

st.subheader("2b: Where are they going?")

st.header("Conclusion")
