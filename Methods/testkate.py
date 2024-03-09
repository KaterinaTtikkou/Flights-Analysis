def calculate_distances(self):
    distances = {}

    # Create GeoDataFrame for airports
    gdf_airports = gpd.GeoDataFrame(
        self.airports_df,
        geometry=gpd.points_from_xy(self.airports_df['Longitude'], self.airports_df['Latitude'])
    )

    # Iterate over routes
    for index, row in self.routes_df.iterrows():
        source_airport_id = row['Source airport ID']
        dest_airport_id = row['Destination airport ID']

        source_airport = gdf_airports[gdf_airports['Airport ID'] == source_airport_id]
        dest_airport = gdf_airports[gdf_airports['Airport ID'] == dest_airport_id]

        # Check if the airports exist
        if not source_airport.empty and not dest_airport.empty:
            source_point = source_airport['geometry'].iloc[0]
            dest_point = dest_airport['geometry'].iloc[0]

            # Check if the geometries are valid before calculating distance
            assert not source_point.is_empty, f"Source geometry is empty for airport ID {source_airport_id}"
            assert not dest_point.is_empty, f"Destination geometry is empty for airport ID {dest_airport_id}"

            distance = source_point.distance(dest_point).km
            distances[(source_airport_id, dest_airport_id)] = distance

    return distances