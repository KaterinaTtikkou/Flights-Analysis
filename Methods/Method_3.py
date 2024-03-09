import os
import requests
from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.graph_objects as go

    def plot_flights_from_airport(self, airport_code, internal=False):
        # Filter routes based on the specified airport as the source airport
        airport_routes = self.routes_df[self.routes_df['Source airport'] == airport_code]

        # Optionally, filter internal flights
        if internal:
            airport_country = self.airports_df.loc[self.airports_df['IATA'] == airport_code, 'Country'].iloc[0]
            airport_routes = airport_routes[airport_routes['Destination airport'].isin(
                self.airports_df[self.airports_df['Country'] == airport_country]['IATA']
            )]

            # Set title for internal flights
            title = f'Internal flights from {airport_code} Airport'
        else:
            # Set title for all flights
            title = f'All flights from {airport_code} Airport'

        # Create Plotly trace for flight paths
        flight_paths = []
        for i, row in airport_routes.iterrows():
            origin = self.airports_df.loc[self.airports_df['IATA'] == row['Source airport']].iloc[0]
            destination = self.airports_df.loc[self.airports_df['IATA'] == row['Destination airport']].iloc[0]

            flight_paths.append(
                go.Scattergeo(
                    locationmode='ISO-3',
                    lon=[float(origin['Longitude']), float(destination['Longitude'])],
                    lat=[float(origin['Latitude']), float(destination['Latitude'])],
                    mode='lines',
                    line=dict(width=0.5, color='red'),
                    opacity=0.8,
                    name=row['Destination airport']
                )
            )

        # Create Plotly trace for the specified airport marker
        airport_marker = go.Scattergeo(
            locationmode='ISO-3',
            lon=[float(self.airports_df.loc[self.airports_df['IATA'] == airport_code, 'Longitude'].iloc[0])],
            lat=[float(self.airports_df.loc[self.airports_df['IATA'] == airport_code, 'Latitude'].iloc[0])],
            mode='markers',
            marker=dict(size=10, color='blue'),
            text=f'{airport_code} Airport',
            name=f'{airport_code} Airport'
        )

        # Set layout
        layout = go.Layout(
            title=title,
            showlegend=True,
            geo=dict(
                scope='world',
                showland=True,
                landcolor='rgb(229, 229, 229)',
                countrycolor='rgb(255, 255, 255)',
                coastlinecolor='rgb(255, 255, 255)',
                projection_type='natural earth',
            )
        )

        # Create the figure and plot
        fig = go.Figure(data=[*flight_paths, airport_marker], layout=layout)
        fig.show()