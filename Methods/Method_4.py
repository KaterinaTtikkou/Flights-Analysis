import matplotlib.pyplot as plt

    def top_models(self, countries=None, top_n=10):
        """
        Plot the N most used airplane models by number of routes.

        Parameters:
        - countries (str or list, optional): The country or list of countries to filter the routes by.
          If not provided (default), it considers all routes.
        - top_n (int, optional): Number of top airplane models to plot. Default is 10.

        Returns:
        - Plot of the N most used airplane models by number of routes.
        - None
        """
        # Merge dataframes
        merged_df1 = pd.merge(self.routes_df, self.airports_df, left_on='Source airport ID', right_on='Airport ID')
        merged_df2 = pd.merge(merged_df1, self.airplanes_df, how="left", left_on='Equipment', right_on='IATA code')

        # Filter routes by selected country(s)
        routes_subset = (
            merged_df2[merged_df2['Country'].isin(countries)] if isinstance(countries, list) else
            merged_df2[merged_df2['Country'] == countries] if countries else
            merged_df2
        )

        # Group by airplane model and count number of routes
        model_counts = routes_subset.groupby('Name_y').size().reset_index(name='Count')

        # Sort by count in descending order
        model_counts = model_counts.sort_values('Count', ascending=False)

        # Select the top N airplane models
        top_models = model_counts.head(top_n)

        # Plot the N most used airplane models
        if not top_models.empty:
            plt.figure(figsize=(12, 6))
            top_models.plot(x='Name_y', y='Count', kind='bar', color='skyblue')
            plt.xlabel('Airplane Model')
            plt.ylabel('Number of Routes')
            title = f'Top {top_n} Airplane Models by Number of Routes'
            title += f' for {countries}' if countries else ' for all countries'
            plt.title(title)
            plt.show()
        else:
            print(f'No data available for the specified conditions: countries={countries}, top_n={top_n}')