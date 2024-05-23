#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install streamlit pandas geopandas matplotlib plotly


# In[ ]:


import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px

# Load data
data = pd.read_csv('hi.csv')
gdf = gpd.read_file('NUTS3.shp')

# Merge data
data['Region'] = data['Region'].replace({'Dublin and Mid-East': 'Mid-East', 'Midland': 'Midlands'})
merged = gdf.merge(data, left_on='NUTS_NAME', right_on='Region')

# Sidebar for year selection
year = st.sidebar.slider('Select Year', min_value=int(data['Year'].min()), max_value=int(data['Year'].max()), step=1)

# Filter data for the selected year
year_data = merged[merged['Year'] == year]

# Map
fig = px.choropleth(year_data, geojson=year_data.geometry, locations=year_data.index, color="Value",
                    hover_name="Region", hover_data=["Statistic Label"],
                    title=f"Distribution of Agriculture Statistics in {year}")
fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig)

# Pie chart for statistic labels greater than 5% of total
stat_label_data = year_data.groupby('Statistic Label')['Value'].sum().reset_index()
total_value = stat_label_data['Value'].sum()
stat_label_data = stat_label_data[stat_label_data['Value'] > 0.05 * total_value]

fig2 = px.pie(stat_label_data, values='Value', names='Statistic Label',
              title='Proportion of Each Statistic Label (greater than 5% of total)')
st.plotly_chart(fig2)

# Line chart for selected statistic label
stat_label = st.sidebar.selectbox('Select Statistic Label', data['Statistic Label'].unique())
stat_label_years = data[data['Statistic Label'] == stat_label]

fig3, ax = plt.subplots()
for region in stat_label_years['Region'].unique():
    region_data = stat_label_years[stat_label_years['Region'] == region]
    ax.plot(region_data['Year'], region_data['Value'], label=region)

ax.set_title(f'Trend of {stat_label} Over Years')
ax.set_xlabel('Year')
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig3)

