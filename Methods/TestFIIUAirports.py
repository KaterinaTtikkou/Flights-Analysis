class TestFIIUAirports(unittest.TestCase):

    def setUp(self):
        self.fiiu_instance = FIIU()

    def test_distance_between_same_airport(self):
        distance = self.fiiu_instance.haversine_distance(1, 1, 1, 1)
        self.assertEqual(distance, 0)

    def test_distance_between_different_airports(self):
        airport_data = pd.DataFrame({
            'Airport ID': [1, 2, 3],
            'Latitude': [-6.081690, -5.207080, -5.826790],
            'Longitude': [145.391998, 145.789001, 144.296005]
        })
        self.fiiu_instance.airports_df = airport_data
        lat1, lon1 = airport_data.loc[airport_data['Airport ID'] == 1, ['Latitude', 'Longitude']].values[0]
        lat2, lon2 = airport_data.loc[airport_data['Airport ID'] == 2, ['Latitude', 'Longitude']].values[0]
        distance = self.fiiu_instance.haversine_distance(lat1, lon1, lat2, lon2)
        expected_distance = 106.71389671030464
        self.assertAlmostEqual(distance, expected_distance, places=2)

    def test_distance_between_airports_in_different_continents(self):
        airport_data = pd.DataFrame({
            'Airport ID': [1, 2, 3],
            'Latitude': [-6.081690, -5.207080, -5.826790],
            'Longitude': [145.391998, 145.789001, 144.296005]
        })
        self.fiiu_instance.airports_df = airport_data
        lat1, lon1 = airport_data.loc[airport_data['Airport ID'] == 1, ['Latitude', 'Longitude']].values[0]
        lat3, lon3 = airport_data.loc[airport_data['Airport ID'] == 3, ['Latitude', 'Longitude']].values[0]
        distance = self.fiiu_instance.haversine_distance(lat1, lon1, lat3, lon3)
        expected_distance = 124.48103955643961
        self.assertAlmostEqual(distance, expected_distance, places=2)

if __name__ == '__main__':
    unittest.main(argv=[''], defaultTest='TestFIIUAirports', verbosity=2, exit=False)