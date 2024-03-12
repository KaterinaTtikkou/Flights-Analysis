    
    
    def aircrafts(self):
        """
        Retrieve a list of unique aircraft names based on the merged data of routes and airplanes.

        Returns:
            numpy.ndarray: An array containing unique aircraft names.

        Example:
            your_instance = YourClassName()
            unique_aircraft_names = your_instance.aircrafts()
            print(unique_aircraft_names)
        """
        aircraft_models = pd.merge(self.routes_df, self.airplanes_df, left_on="Model", right_on="IATA code", how='left')
        unique_aircraft_names = aircraft_models['Name'].unique()
        return unique_aircraft_names