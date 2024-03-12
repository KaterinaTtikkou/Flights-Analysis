   
    def airports(self):
        """
        Retrieve a list of unique airport names based on the 'IATA' codes in the airports DataFrame.

        Returns:
            numpy.ndarray: An array containing unique airport names.

        Example:
            your_instance = YourClassName()
            unique_airport_names = your_instance.airports()
            print(unique_airport_names)
        """
        airport_names = self.airports_df['IATA'].unique()
        return airport_names
    