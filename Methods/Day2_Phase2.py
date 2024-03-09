    def plot_flights_from_country(self, country, internal=False, short_haul_cutoff=1000):
        # Filter routes by country
        country_airports = self.airports_df[self.airports_df['Country'] == country]['IATA']
        relevant_routes = self.routes_df[self.routes_df['Source airport'].isin(country_airports)]

        if internal:
            relevant_routes = relevant_routes[relevant_routes['Destination airport'].isin(country_airports)]

        flight_paths = []
        total_short_haul_distance = 0  # Initialize total distance for short-haul flights

        # Define emissions factors (example values, adjust according to credible sources)
        flight_emission_per_km = 246  # grams of CO2 per km for flights
        train_emission_per_km = 35    # grams of CO2 per km for trains

        # Containers for the legend names
        added_short_haul_legend = False
        added_long_haul_legend = False

        for _, row in relevant_routes.iterrows():
            origin = self.airports_df[self.airports_df['IATA'] == row['Source airport']].iloc[0]
            destination = self.airports_df[self.airports_df['IATA'] == row['Destination airport']].iloc[0]
            distance = self.haversine_distance(origin['Longitude'], origin['Latitude'], destination['Longitude'], destination['Latitude'])

            is_short_haul = distance <= short_haul_cutoff
            color = 'green' if is_short_haul else 'red'
            if is_short_haul:
                total_short_haul_distance += distance
                if not added_short_haul_legend:
                    name = 'Short-Haul Flight'
                    added_short_haul_legend = True
                else:
                    name = None
            else:
                if not added_long_haul_legend:
                    name = 'Long-Haul Flight'
                    added_long_haul_legend = True
                else:
                    name = None

            flight_paths.append(go.Scattergeo(
                locationmode='ISO-3',
                lon=[origin['Longitude'], destination['Longitude']],
                lat=[origin['Latitude'], destination['Latitude']],
                mode='lines',
                line=dict(width=2, color=color),
                opacity=0.7,
                name=name
            ))

        # Calculate emissions and reductions
        total_flight_emissions = total_short_haul_distance * flight_emission_per_km / 1e6  # Convert to tons
        potential_train_emissions = total_short_haul_distance * train_emission_per_km / 1e6  # Convert to tons
        potential_emission_reduction = total_flight_emissions - potential_train_emissions


        # Annotations for total short-haul distance and potential emission reductions
        annotations = [
            go.layout.Annotation(
                text=f"Total short-haul distance: {total_short_haul_distance:.2f} km",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.01,
                y=1.10,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.9)"
            ),
            go.layout.Annotation(
                text=f"Potential CO2 reduction by replacing flights with trains: {potential_emission_reduction:.2f} tons",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.01,
                y=1.05,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.9)"
            )
        ]

        # Create the figure with the flight paths added
        fig = go.Figure(data=flight_paths)

        # Update the layout for the figure
        fig.update_layout(
            title=f"Flight Routes from {country} (Short-Haul vs Long-Haul)",
            showlegend=True,
            geo=dict(
                scope='world',
                projection_type='equirectangular',
                showland=True,
                landcolor='rgb(243,243,243)',
                countrycolor='rgb(204,204,204)'
            ),
            annotations=annotations,
            margin=dict(t=120, l=0, r=0, b=0)  # Adjust the top margin to ensure annotations are visible
        )

        # Show the figure
        fig.show()
