    
    
    def aircraft(self):
        aircraft_models = pd.merge(self.routes_df, self.airplanes_df, left_on="Model", right_on="IATA code", how='left')
        unique_aircraft_names = aircraft_models['Name'].unique()
        return unique_aircraft_names