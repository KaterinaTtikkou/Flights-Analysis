import os
import pandas as pd
from zipfile import ZipFile
import requests
from io import BytesIO
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from langchain_openai import OpenA, ChatOpenAI
import langchain
from IPython.display import Markdown

class FIIU:
    
    def __init__(self, zip_url='https://gitlab.com/adpro9641208/group_03/-/raw/main/flight_data.zip'):
        """
        Downloads, extracts, and processes flight data from the given zip URL.

        Parameters:
        - zip_url (str): The URL of the flight data zip file.
        """
        # Set the destination folder for the extracted data in the main directory
        self.destination_folder = 'downloads'

        # Ensure the destination folder exists
        os.makedirs(self.destination_folder, exist_ok=True)

        # Define the path to the downloaded zip file
        zip_file_path = os.path.join(self.destination_folder, 'flight_data.zip')

        # Check if the zip file already exists
        if not os.path.exists(zip_file_path):
            # Download the zip file from the given URL
            response = requests.get(zip_url)

            # Save the zip file in the destination folder
            with open(zip_file_path, 'wb') as zip_file:
                zip_file.write(response.content)

            print("File downloaded successfully.")
        else:
            print("File already exists.")

        # Process the downloaded zip file
        zip_file = ZipFile(zip_file_path)
        zip_file.extractall(self.destination_folder)

        # Assign values to the instance attributes
        self.airlines_df = pd.read_csv(os.path.join(self.destination_folder, 'airlines.csv'))
        self.airplanes_df = pd.read_csv(os.path.join(self.destination_folder, 'airplanes.csv'))
        self.airports_df = pd.read_csv(os.path.join(self.destination_folder, 'airports.csv'))
        self.routes_df = pd.read_csv(os.path.join(self.destination_folder, 'routes.csv'))
        
        # Removing Superflous Columns
        self.airports_df = self.airports_df.drop(['index', 'Timezone', 'Altitude', 'DST', 'Tz database time zone', 'Type', 'Source'], axis=1)
        self.routes_df = self.routes_df.drop(['index', 'Codeshare', 'Stops'], axis=1)
        
        # Cleaning and Type Casting
        self.airports_df = self.airports_df.dropna()
        self.routes_df = self.routes_df.dropna()
        self.airplanes_df = self.airplanes_df.dropna()
        self.airplanes_df = self.airplanes_df[self.airplanes_df['IATA code'] != '\\N']
        self.routes_df = self.routes_df[self.routes_df['Source airport ID'] != '\\N']
        self.routes_df['Source airport ID'] = self.routes_df['Source airport ID'].astype(int)
        self.routes_df['Model'] = self.routes_df['Equipment'].str.split('-')
        self.routes_df = self.routes_df.explode('Model')
        self.routes_df['Source airport ID'] = pd.to_numeric(self.routes_df['Source airport ID'].replace('\\N', np.nan), errors='coerce')
        self.routes_df['Destination airport ID'] = pd.to_numeric(self.routes_df['Destination airport ID'].replace('\\N', np.nan), errors='coerce')
        self.airports_df['Airport ID'] = pd.to_numeric(self.airports_df['Airport ID'].replace('\\N', np.nan), errors='coerce')
        
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
        
    def plot_flights_from_airport(self, airport_code, internal=False):
        # Filter routes based on the specified airport as the source airport
        airport_routes = self.routes_df[self.routes_df['Source airport'] == airport_code]

        # Optionally, filter internal flights
        if internal:
            airport_country = self.airports_df.loc[self.airports_df['IATA'] == airport_code, 'Country'].iloc[0]
            airport_routes = airport_routes[airport_routes['Destination airport'].isin(
                self.airports_df[self.airports_df['Country'] == airport_country]['IATA']
            )]

            # Set title for internal flights
            title = f'Internal flights from {airport_code} Airport'
        else:
            # Set title for all flights
            title = f'All flights from {airport_code} Airport'

        # Create Plotly trace for flight paths
        flight_paths = []
        for i, row in airport_routes.iterrows():
            origin = self.airports_df.loc[self.airports_df['IATA'] == row['Source airport']].iloc[0]
            destination = self.airports_df.loc[self.airports_df['IATA'] == row['Destination airport']].iloc[0]

            flight_paths.append(
                go.Scattergeo(
                    locationmode='ISO-3',
                    lon=[float(origin['Longitude']), float(destination['Longitude'])],
                    lat=[float(origin['Latitude']), float(destination['Latitude'])],
                    mode='lines',
                    line=dict(width=0.5, color='red'),
                    opacity=0.8,
                    name=row['Destination airport']
                )
            )

        # Create Plotly trace for the specified airport marker
        airport_marker = go.Scattergeo(
            locationmode='ISO-3',
            lon=[float(self.airports_df.loc[self.airports_df['IATA'] == airport_code, 'Longitude'].iloc[0])],
            lat=[float(self.airports_df.loc[self.airports_df['IATA'] == airport_code, 'Latitude'].iloc[0])],
            mode='markers',
            marker=dict(size=10, color='blue'),
            text=f'{airport_code} Airport',
            name=f'{airport_code} Airport'
        )

        # Set layout
        layout = go.Layout(
            title=title,
            showlegend=True,
            geo=dict(
                scope='world',
                showland=True,
                landcolor='rgb(229, 229, 229)',
                countrycolor='rgb(255, 255, 255)',
                coastlinecolor='rgb(255, 255, 255)',
                projection_type='natural earth',
            )
        )

        # Create the figure and plot
        fig = go.Figure(data=[*flight_paths, airport_marker], layout=layout)
        fig.show()

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

    def plot_flights_from_country(self, country, internal=False):
        # Filter routes based on the country
        country_routes = self.routes_df[self.routes_df['Source airport'].isin(self.airports_df[self.airports_df['Country'] == country]['IATA'])]

        # Optionally, filter internal flights
        if internal:
            country_airports = self.airports_df[self.airports_df['Country'] == country]['IATA']
            country_routes = country_routes[country_routes['Destination airport'].isin(country_airports)]

        # Create Plotly traces for flight paths
        flight_paths = []
        for i, row in country_routes.iterrows():
            origin = self.airports_df.loc[self.airports_df['IATA'] == row['Source airport']].iloc[0]
            destination = self.airports_df.loc[self.airports_df['IATA'] == row['Destination airport']].iloc[0]

            flight_paths.append(
                go.Scattergeo(
                    locationmode='ISO-3',
                    lon=[float(origin['Longitude']), float(destination['Longitude'])],
                    lat=[float(origin['Latitude']), float(destination['Latitude'])],
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    opacity=0.7,
                    name=f"{origin['IATA']} to {destination['IATA']}"
                )
            )

        # Set layout for the map
        layout = go.Layout(
            title='Internal flights' if internal else f"All flights from {country}",
            showlegend=True,
            geo=dict(
                scope='world',
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
        )

        # Create the figure and add the traces
        fig = go.Figure(data=flight_paths, layout=layout)
        fig.show()
        def plot_flights_from_country(self, country, internal=False):
        # Filter routes based on the country
        country_routes = self.routes_df[self.routes_df['Source airport'].isin(self.airports_df[self.airports_df['Country'] == country]['IATA'])]

        # Optionally, filter internal flights
        if internal:
            country_airports = self.airports_df[self.airports_df['Country'] == country]['IATA']
            country_routes = country_routes[country_routes['Destination airport'].isin(country_airports)]

        # Create Plotly traces for flight paths
        flight_paths = []
        for i, row in country_routes.iterrows():
            origin = self.airports_df.loc[self.airports_df['IATA'] == row['Source airport']].iloc[0]
            destination = self.airports_df.loc[self.airports_df['IATA'] == row['Destination airport']].iloc[0]

            flight_paths.append(
                go.Scattergeo(
                    locationmode='ISO-3',
                    lon=[float(origin['Longitude']), float(destination['Longitude'])],
                    lat=[float(origin['Latitude']), float(destination['Latitude'])],
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    opacity=0.7,
                    name=f"{origin['IATA']} to {destination['IATA']}"
                )
            )

        # Set layout for the map
        layout = go.Layout(
            title='Internal flights' if internal else f"All flights from {country}",
            showlegend=True,
            geo=dict(
                scope='world',
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
        )

        # Create the figure and add the traces
        fig = go.Figure(data=flight_paths, layout=layout)
        fig.show()
    
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
    
    def aircraft_info(self, _aircraft_name):
        """
        Display specifications of a given aircraft using the OpenAI GPT-3.5 Turbo model.

        Parameters:
        - _aircraft_name (str): The name of the aircraft for which specifications are requested.

        Raises:
        - ValueError: If the provided aircraft name is not found in the dataset.
                      Instructs the user to choose a valid aircraft name from the available dataset.
        - ValueError: If the OpenAI API key is not found in the environment variable.
                      Instructs the user to set the 'OPENAI_API_KEY' environment variable with their API key.

        Note:
        - Ensure that the OpenAI API key is set in the 'OPENAI_API_KEY' environment variable.
        - The generated specifications are displayed in Markdown format.
        
        Example usage:
        your_instance = FIIU()
        your_instance.aircraft_info("Boeing 747")
        """
        # Check if the aircraft name is in the list of aircrafts
        aircrafts = self.aircrafts()
        if _aircraft_name not in self.aircrafts_data:
            raise ValueError(f"Aircraft '{_aircraft_name}' not found. Please choose a valid aircraft name from the dataset. Available aircraft models:\n{list(self.aircrafts_data.keys())}")

        # Fetch your OpenAI API key from the environment variable
        api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable with your API key.")
        else:
            # Initialize the OpenAI language model
            llm = ChatOpenAI(api_key=api_key, temperature=0.9)

            # Generate a table of specifications in Markdown using LLM
            specifications_prompt = f"Provide specifications table for {_aircraft_name}."
            result = llm.invoke(specifications_prompt)
            specifications_content = result.content

            # Display the generated table of specifications in Markdown
            display(Markdown(specifications_content))
       
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
    
    
    def airport_info(self, _airport_name):
        """
        Display specifications of a given airport using the OpenAI GPT-3.5 Turbo model.

        Parameters:
        - _airport_name (str): The name of the airport for which specifications are requested.

        Raises:
        - ValueError: If the OpenAI API key is not found in the environment variable.
                      Instructs the user to set the 'OPENAI_API_KEY' environment variable with their API key.

        Note:
        - Ensure that the OpenAI API key is set in the 'OPENAI_API_KEY' environment variable.
        - The generated specifications are displayed in Markdown format.

        Example usage:
        >>> your_instance = FIIU()
        >>> your_instance.airport_info("John F. Kennedy International Airport")
        """
        # Fetch your OpenAI API key from the environment variable
        api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable with your API key.")
        else:
            # Initialize the OpenAI language model
            llm = ChatOpenAI(api_key=api_key, temperature=0.9)

            # Check if the airport name is in the list of airports
            airports = self.airports()
            if _airport_name not in airports:
                print(f"Airport information not available for '{_airport_name}'.")
            else:
                # Generate a table of specifications in Markdown using LLM
                specifications_prompt = f"Provide specifications table for {_airport_name}."
                result = llm.invoke(specifications_prompt)
                specifications_content = result.content

                # Display the generated table of specifications in Markdown
                display(Markdown(specifications_content))

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


