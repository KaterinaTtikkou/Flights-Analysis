    def plot_flights_from_country(self, country, internal=False):
        # Filter routes based on the country
        country_routes = self.routes_df[self.routes_df['Source airport'].isin(self.airports_df[self.airports_df['Country'] == country]['IATA'])]

        # Optionally, filter internal flights
        if internal:
            country_airports = self.airports_df[self.airports_df['Country'] == country]['IATA']
            country_routes = country_routes[country_routes['Destination airport'].isin(country_airports)]

        # Create Plotly traces for flight paths
        flight_paths = []
        for i, row in country_routes.iterrows():
            origin = self.airports_df.loc[self.airports_df['IATA'] == row['Source airport']].iloc[0]
            destination = self.airports_df.loc[self.airports_df['IATA'] == row['Destination airport']].iloc[0]

            flight_paths.append(
                go.Scattergeo(
                    locationmode='ISO-3',
                    lon=[float(origin['Longitude']), float(destination['Longitude'])],
                    lat=[float(origin['Latitude']), float(destination['Latitude'])],
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    opacity=0.7,
                    name=f"{origin['IATA']} to {destination['IATA']}"
                )
            )

        # Set layout for the map
        layout = go.Layout(
            title='Internal flights' if internal else f"All flights from {country}",
            showlegend=True,
            geo=dict(
                scope='world',
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
        )

        # Create the figure and add the traces
        fig = go.Figure(data=flight_paths, layout=layout)
        fig.show()

