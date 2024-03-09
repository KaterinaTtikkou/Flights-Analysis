    def add_distance_column(self):

    # Merge routes_df with airports_df to get source airport coordinates
        routes_with_source_coords = self.routes_df.merge(
            self.airports_df[['Airport ID', 'Latitude', 'Longitude']],
            left_on='Source airport ID',
            right_on='Airport ID',
            how='left'
        ).rename(columns={'Latitude': 'Source Latitude', 'Longitude': 'Source Longitude'})

    # Merge routes_with_source_coords with airports_df to get destination airport coordinates
        routes_with_all_coords = routes_with_source_coords.merge(
            self.airports_df[['Airport ID', 'Latitude', 'Longitude']],
            left_on='Destination airport ID',
            right_on='Airport ID',
            how='left'
        ).rename(columns={'Latitude': 'Destination Latitude', 'Longitude': 'Destination Longitude'})

    # Calculate distances and add as a new column
        self.routes_df['distance'] = routes_with_all_coords.apply(
            lambda row: self.haversine_distance(
                row['Source Latitude'], row['Source Longitude'],
                row['Destination Latitude'], row['Destination Longitude']
            ),
            axis=1
        )


    def distance_analysis(self, bins=30, show_mean=True, show_median=True):

        # Filter out None values (if any)
        valid_distances = self.routes_df['distance'].dropna()

        # Create a histogram of flight distances
        plt.figure(figsize=(10, 6))
        plt.hist(valid_distances, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

        # Add vertical lines for mean and median distances
        if show_mean:
            mean_val = valid_distances.mean()
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
        if show_median:
            median_val = valid_distances.median()
            plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.2f}')

        # Set labels and title
        plt.xlabel('Flight Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Flight Distances')

        # Show plot
        plt.legend()
        plt.show()