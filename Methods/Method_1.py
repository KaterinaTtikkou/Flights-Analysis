    def plot_airports_by_country(self, country_name):
        # Filter airports by the specified country
        country_airports = self.airports_df[self.airports_df['Country'] == country_name]

        # Check if any airports found
        if country_airports.empty:
            print("Error: Country not found or no airports available for this country.")
            return

        # Convert DataFrame to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            country_airports, geometry=gpd.points_from_xy(country_airports.Longitude, country_airports.Latitude))

        # Load world map
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        country_map = world[world.name == country_name]
        
        # Plotting
        fig, ax = plt.subplots()
        country_map.plot(ax=ax, color='white', edgecolor='black')
        gdf.plot(ax=ax, color='red', markersize=10)
        plt.show()
        

