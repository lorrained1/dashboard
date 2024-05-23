#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit


# In[2]:


pip install pydeck


# In[3]:


pip install streamlit folium streamlit-folium


# In[4]:


import pandas as pd


# In[5]:


data_filtered = pd.read_csv("hi.csv")


# In[12]:


data_filtered.head()


# In[8]:


print(folium.__version__)


# In[9]:


import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
import branca.colormap as cm
import streamlit as st
import plotly.express as px
from streamlit_folium import folium_static


# In[16]:


total_data = pd.read_csv("hi.csv")


total_data['Region'] = total_data['Region'].replace({'Dublin and Mid-East': 'Mid-East', 'Midland': 'Midlands'})

# Ensure the column names match for merging
print(gdf.columns)  # Check column names in the shapefile
print(total_data.columns)  # Check column names in the dataset
# Simplify geometries to reduce memory usage
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)

# Merge the dataset with the shapefile
merged = gdf.merge(total_data, left_on='NUTS3NAME', right_on='Region')


# In[ ]:


# Sum all "Statistic Labels" by year for each region
total_data = data_filtered.groupby(['Year', 'Region', 'Statistic Label'])['VALUE'].sum().reset_index()

# Ensure the column names match for merging
total_data['Region'] = total_data['Region'].replace({'Dublin and Mid-East': 'Mid-East', 'Midland': 'Midlands'})

# Simplify geometries to reduce memory usage
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)

# Cache the GeoDataFrame for faster processing
geojson_data = gdf.to_json()

def create_folium_map(year):
    # Filter the data for the selected year
    data_for_year = total_data[total_data['Year'] == year]
    
    # Merge the dataset with the GeoJSON file
    merged = gpd.read_file(geojson_data)
    merged = merged.merge(data_for_year, left_on='NUTS3NAME', right_on='Region')

    m = folium.Map(location=[53.1424, -7.6921], zoom_start=6)

    # Define a colormap
    min_value = merged['VALUE'].min()
    max_value = merged['VALUE'].max()
    
    # Ensure we have valid range
    if min_value == max_value:
        min_value -= 1  # Adjust min_value to avoid having identical min and max

    colormap = cm.linear.YlOrRd_09.scale(min_value, max_value)
    
    # Add the colormap to the map
    colormap.add_to(m)

    folium.GeoJson(
        merged,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['VALUE']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        },
        tooltip=GeoJsonTooltip(
            fields=['NUTS3NAME', 'VALUE'],
            aliases=['Region:', 'Value:'],
            localize=True
        )
    ).add_to(m)
    
    return m

# Streamlit layout
st.set_page_config(page_title="Agriculture Data Visualization", layout="wide")

st.title('Agriculture Data Visualization')
st.write('A web application for visualizing agriculture data by region.')

# Dropdown for year selection
selected_year = st.selectbox('Select Year', total_data['Year'].unique())

# Generate and display folium map
st.subheader('Agriculture Data Map')
folium_map = create_folium_map(selected_year)
folium_static(folium_map)

# Dropdowns for region and statistic label selection
selected_region = st.selectbox('Select Region', total_data['Region'].unique())
selected_statistic_label = st.selectbox('Select Statistic Label', total_data['Statistic Label'].unique())

# Data for the selected region and statistic label
data_for_line_chart = total_data[(total_data['Region'] == selected_region) & (total_data['Statistic Label'] == selected_statistic_label)]

# Line chart
st.subheader(f'Value Over Time for {selected_region} - {selected_statistic_label}')
line_fig = px.line(data_for_line_chart, x='Year', y='VALUE', title=f'Value Over Time for {selected_region} - {selected_statistic_label}')
st.plotly_chart(line_fig)

# Data for the pie chart
data_for_pie_chart = total_data[total_data['Year'] == selected_year]

# Pie chart
st.subheader(f'Agriculture Value Distribution by Region for {selected_year}')
pie_fig = px.pie(data_for_pie_chart, names='Region', values='VALUE', title=f'Agriculture Value Distribution by Region for {selected_year}')
st.plotly_chart(pie_fig)


# In[ ]:




