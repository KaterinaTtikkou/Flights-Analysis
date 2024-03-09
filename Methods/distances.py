     def calculate_distances(self):
        distances = {}

        for index, row in self.routes_df.iterrows():
            source_airport_id = row['Source airport ID']
            dest_airport_id = row['Destination airport ID']

            source_coords = self.get_coordinates(source_airport_id)
            dest_coords = self.get_coordinates(dest_airport_id)

            if source_coords is not None and dest_coords is not None:
                distance = self.haversine_distance(*source_coords, *dest_coords)
                distances[(source_airport_id, dest_airport_id)] = distance

        return distances


    def haversine_distance(self, lat1, lon1, lat2, lon2):
        if None in [lat1, lon1, lat2, lon2]:
            return 0  # Return a default distance in case of None values

        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2  # Fix here
        c = 2 * atan2(sqrt(max(0, a)), sqrt(max(0, 1 - a)))  # Ensure values >= 0
        distance = R * c
        return distance
    
    def get_coordinates(self, airport_id):
        row = self.airports_df[self.airports_df['Airport ID'] == airport_id]
        if not row.empty:
            latitude = row['Latitude'].values[0]
            longitude = row['Longitude'].values[0]
            return latitude, longitude
        else:
            return None, None