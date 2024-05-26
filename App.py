#!/usr/bin/env python
# coding: utf-8

# In[7]:


#pip install panel hvplot holoviews geopandas scikit-learn matplotlib


# In[8]:


import pandas as pd
import panel as pn
import hvplot.pandas
import folium
from folium import Choropleth, GeoJsonTooltip
import geopandas as gpd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dashboard_data.csv'
agri_2010_df = pd.read_csv(file_path)

# Convert 'Year' to datetime and ensure 'Adjusted Value' is float
agri_2010_df['Year'] = pd.to_datetime(agri_2010_df['Year'], format='%Y').dt.year
agri_2010_df['Adjusted Value'] = pd.to_numeric(agri_2010_df['Adjusted Value'], errors='coerce')

# Load the shapefile
gdf = gpd.read_file('NUTS3.shp')

# Simplify geometries to reduce memory usage
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)

agri_2010_df['Region'] = agri_2010_df['Region'].replace({'Dublin and Mid-East': 'Mid-East', 'Midland': 'Midlands'})

# Ensure all necessary columns are of appropriate types
agri_2010_df['Year'] = agri_2010_df['Year'].astype(int)  # Convert Year to integer
agri_2010_df['Adjusted Value'] = agri_2010_df['Adjusted Value'].astype(float)

# Create selection widgets for the statistic labels and year
statistic_selector_1 = pn.widgets.Select(name='Statistic Label 1', options=list(agri_2010_df['Statistic Label'].unique()))
statistic_selector_2 = pn.widgets.Select(name='Statistic Label 2', options=list(agri_2010_df['Statistic Label'].unique()))
year_selector = pn.widgets.Select(name='Year', options=list(agri_2010_df['Year'].unique()))
map_statistic_selector = pn.widgets.RadioButtonGroup(name='Map Statistic', options=['Statistic Label 1', 'Statistic Label 2'], button_type='success')

# Create a plot based on the selected statistics
@pn.depends(statistic_selector_1, statistic_selector_2)
def create_plot(statistic_label_1, statistic_label_2):
    filtered_data_1 = agri_2010_df[(agri_2010_df['Statistic Label'] == statistic_label_1) & (agri_2010_df['Region'] == 'State')]
    filtered_data_2 = agri_2010_df[(agri_2010_df['Statistic Label'] == statistic_label_2) & (agri_2010_df['Region'] == 'State')]
    
    if filtered_data_1.empty or filtered_data_2.empty:
        return hv.Curve([]).opts(title='No data available for the selected statistic labels in State')
    
    plot_1 = filtered_data_1.hvplot.line(x='Year', y='Adjusted Value', label=statistic_label_1)
    plot_2 = filtered_data_2.hvplot.line(x='Year', y='Adjusted Value', label=statistic_label_2)
    
    combined_plot = (plot_1 * plot_2).opts(
        title='Time Series Comparison', ylabel='Value (Euro Million)', xlabel='Year',
        legend_position='right', legend_opts={'location': 'center', 'title_text_font_size': '10pt'}
    )
    return combined_plot

# Aggregate data and train the model
def train_model(statistic_label):
    filtered_data = agri_2010_df[agri_2010_df['Statistic Label'] == statistic_label]
    aggregated_df = filtered_data.groupby('Year')['Adjusted Value'].sum().reset_index()

    X = aggregated_df[['Year']]
    y = aggregated_df['Adjusted Value']

    model = LinearRegression()
    model.fit(X, y)
    
    return model, aggregated_df

# Create a plot based on the selected statistics and include predictions
@pn.depends(statistic_selector_1, statistic_selector_2)
def create_predict(statistic_label_1, statistic_label_2):
    model_1, aggregated_df_1 = train_model(statistic_label_1)
    model_2, aggregated_df_2 = train_model(statistic_label_2)
    
    predictions_1 = model_1.predict(aggregated_df_1[['Year']])
    predictions_2 = model_2.predict(aggregated_df_2[['Year']])
    
    aggregated_df_1['Predicted Value'] = predictions_1
    aggregated_df_2['Predicted Value'] = predictions_2
    
    year_2024 = np.array([[2024]])
    predicted_value_2024_1 = model_1.predict(year_2024)[0]
    predicted_value_2024_2 = model_2.predict(year_2024)[0]
    
    plt.figure(figsize=(6, 3))
    plt.scatter(aggregated_df_1['Year'], aggregated_df_1['Adjusted Value'], color='blue', label=f'Actual Data - {statistic_label_1}')
    plt.plot(aggregated_df_1['Year'], predictions_1, color='red', label=f'Regression Line - {statistic_label_1}')
    plt.scatter(2024, predicted_value_2024_1, color='green', label=f'Predicted 2024 - {statistic_label_1}', s=50)
    
    plt.scatter(aggregated_df_2['Year'], aggregated_df_2['Adjusted Value'], color='orange', label=f'Actual Data - {statistic_label_2}')
    plt.plot(aggregated_df_2['Year'], predictions_2, color='purple', label=f'Regression Line - {statistic_label_2}')
    plt.scatter(2024, predicted_value_2024_2, color='brown', label=f'Predicted 2024 - {statistic_label_2}', s=50)
    
    plt.annotate(f'Predicted 2024: {predicted_value_2024_1:.2f}', xy=(2024, predicted_value_2024_1), 
                 xytext=(2024, predicted_value_2024_1 + 10),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Predicted 2024: {predicted_value_2024_2:.2f}', xy=(2024, predicted_value_2024_2), 
                 xytext=(2024, predicted_value_2024_2 + 10),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('Year')
    plt.ylabel('Adjusted Value')
    plt.title(f'Linear Regression Model and Prediction for 2024 - {statistic_label_1} vs {statistic_label_2}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    
    fig_pane = pn.pane.Matplotlib(plt.gcf(), tight=True)
    plt.close()
    
    return fig_pane

# Create a table based on the selected statistics and year
@pn.depends(statistic_selector_1, statistic_selector_2, year_selector)
def create_table(statistic_label_1, statistic_label_2, year):
    filtered_data_1 = agri_2010_df[(agri_2010_df['Statistic Label'] == statistic_label_1) & 
                                 (agri_2010_df['Year'] == year)]
    filtered_data_2 = agri_2010_df[(agri_2010_df['Statistic Label'] == statistic_label_2) & 
                                 (agri_2010_df['Year'] == year)]
    
    table_1 = filtered_data_1[['Region', 'Adjusted Value']].rename(columns={'Adjusted Value': f'{statistic_label_1} Value'})
    table_2 = filtered_data_2[['Region', 'Adjusted Value']].rename(columns={'Adjusted Value': f'{statistic_label_2} Value'})
    
    combined_table = table_1.merge(table_2, on='Region', how='outer').fillna(0)
    table = combined_table.hvplot.table(width=400)
    return table



# In[ ]:


# Create a bar chart based on the selected statistics and year
@pn.depends(statistic_selector_1, statistic_selector_2, year_selector)
def create_bar_chart(statistic_label_1, statistic_label_2, year):
    filtered_data_1 = agri_2010_df[(agri_2010_df['Statistic Label'] == statistic_label_1) & 
                                 (agri_2010_df['Year'] == year)]
    filtered_data_2 = agri_2010_df[(agri_2010_df['Statistic Label'] == statistic_label_2) & 
                                 (agri_2010_df['Year'] == year)]
    
    if filtered_data_1.empty or filtered_data_2.empty:
        return hv.Bars([]).opts(title=f'No data available for the selected statistic labels in {year}')
    
    bar_chart_1 = filtered_data_1.hvplot.bar(x='Region', y='Adjusted Value', label=statistic_label_1, rot=90)
    bar_chart_2 = filtered_data_2.hvplot.bar(x='Region', y='Adjusted Value', label=statistic_label_2, rot=90)
    
    combined_bar_chart = (bar_chart_1 * bar_chart_2).opts(
        title=f'{statistic_label_1} vs {statistic_label_2} in {year}',
        ylabel='Value (Euro Million)',
        xlabel='Region',
        legend_position='right'
    )
    return combined_bar_chart

# Create a folium map based on the selected statistic and year
@pn.depends(map_statistic_selector, year_selector)
def create_map(selected_statistic, year):
    statistic_label = statistic_selector_1.value if selected_statistic == 'Statistic Label 1' else statistic_selector_2.value
    data_filtered = agri_2010_df[(agri_2010_df['Statistic Label'] == statistic_label) & 
                                 (agri_2010_df['Year'] == year)]
    
    if data_filtered.empty:
        m = folium.Map(location=[53.1424, -7.6921], zoom_start=6)
        return pn.pane.HTML(m._repr_html_(), sizing_mode='stretch_both')
    
    # Merge the dataset with the shapefile
    merged = gdf.merge(data_filtered, left_on='NUTS3NAME', right_on='Region')
    
    # Ensure all columns are serializable
    merged['Adjusted Value'] = merged['Adjusted Value'].astype(float)
    
    m = folium.Map(location=[53.1424, -7.6921], zoom_start=6)
    
    # Add Choropleth to the folium map
    Choropleth(
        geo_data=merged,
        name='choropleth',
        data=merged,
        columns=['NUTS3NAME', 'Adjusted Value'],
        key_on='feature.properties.NUTS3NAME',
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'{statistic_label} Value (Euro Million)'
    ).add_to(m)
    
    # Add GeoJson tooltips
    folium.GeoJson(
        merged,
        style_function=lambda x: {
            'fillColor': '#ffffff',
            'color': 'black',
            'fillOpacity': 0,
            'weight': 0.1
        },
        tooltip=GeoJsonTooltip(
            fields=['NUTS3NAME', 'Adjusted Value'],
            aliases=['Region:', 'Value:'],
            localize=True
        )
    ).add_to(m)
    
    return pn.pane.HTML(m._repr_html_(), sizing_mode='stretch_both')

# Compare the performance of the two statistics
@pn.depends(statistic_selector_1, statistic_selector_2)
def compare_performance(statistic_label_1, statistic_label_2):
    model_1, aggregated_df_1 = train_model(statistic_label_1)
    model_2, aggregated_df_2 = train_model(statistic_label_2)
    
    last_year = max(agri_2010_df['Year'])
    value_1 = aggregated_df_1.loc[aggregated_df_1['Year'] == last_year, 'Adjusted Value'].values[0]
    value_2 = aggregated_df_2.loc[aggregated_df_2['Year'] == last_year, 'Adjusted Value'].values[0]
    
    better_performing = statistic_label_1 if value_1 > value_2 else statistic_label_2
    
    return pn.pane.Markdown(f"### Better Performing Statistic Label: {better_performing}\n"
                            f"**{statistic_label_1}** in {last_year}: {value_1:.2f} Euro Million\n"
                            f"**{statistic_label_2}** in {last_year}: {value_2:.2f} Euro Million")

# Create a dashboard layout
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown("# Ireland Livestock Product Statistics Dashboard")),
    pn.Row(statistic_selector_1, statistic_selector_2, year_selector),
    pn.Row(create_plot),
    pn.Row(create_table, create_bar_chart),
    pn.Row(map_statistic_selector, create_map),
    pn.Row(create_predict),
    pn.Row(compare_performance)
)

# Display the dashboard
dashboard.servable()


# In[ ]:




