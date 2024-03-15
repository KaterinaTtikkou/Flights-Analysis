import os
from zipfile import ZipFile
from io import BytesIO
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import requests
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from langchain_openai import OpenAI, ChatOpenAI
import langchain
from IPython.display import Markdown


class FIIU:
    """
    Flight Information and Analysis Utility class.

    This class provides methods for downloading,
    processing, and analyzing flight data.
    It includes functionalities such as calculating distances,
    plotting flight paths, and analyzing aircraft and airport information.
    """

    def __init__(
        self,
        zip_url="https://gitlab.com/adpro9641208/group_03/-/raw/main/flight_data.zip",
    ):
        """
        Downloads, extracts, and processes flight data from the given zip URL.

        Parameters:
        - zip_url (str): The URL of the flight data zip file.
        """
        # Set destination folder for extracted data in main directory
        self.destination_folder = "downloads"

        # Ensure the destination folder exists
        os.makedirs(self.destination_folder, exist_ok=True)

        # Define the path to the downloaded zip file
        zip_file_path = os.path.join(self.destination_folder, "flight_data.zip")

        # Check if the zip file already exists
        if not os.path.exists(zip_file_path):
            # Download the zip file from the given URL
            response = requests.get(zip_url)

            # Save the zip file in the destination folder
            with open(zip_file_path, "wb") as zip_file:
                zip_file.write(response.content)

            print("File downloaded successfully.")
        else:
            print("File already exists.")

        # Process the downloaded zip file
        zip_file = ZipFile(zip_file_path)
        zip_file.extractall(self.destination_folder)

        # Assign values to the instance attributes
        self.airlines_df = pd.read_csv(
            os.path.join(self.destination_folder, "airlines.csv")
        )
        self.airplanes_df = pd.read_csv(
            os.path.join(self.destination_folder, "airplanes.csv")
        )
        self.airports_df = pd.read_csv(
            os.path.join(self.destination_folder, "airports.csv")
        )
        self.routes_df = pd.read_csv(
            os.path.join(self.destination_folder, "routes.csv")
        )

        # Removing Superflous Columns
        self.airports_df = self.airports_df.drop(
            [
                "index",
                "Timezone",
                "Altitude",
                "DST",
                "Tz database time zone",
                "Type",
                "Source",
            ],
            axis=1,
        )
        self.routes_df = self.routes_df.drop(["index", "Codeshare", "Stops"], axis=1)

        # Cleaning and Type Casting
        self.airports_df = self.airports_df.dropna()
        self.routes_df = self.routes_df.dropna()
        self.airplanes_df = self.airplanes_df.dropna()
        self.airplanes_df = self.airplanes_df[self.airplanes_df["IATA code"] != "\\N"]
        self.routes_df = self.routes_df[self.routes_df["Source airport ID"] != "\\N"]
        self.routes_df["Source airport ID"] = self.routes_df[
            "Source airport ID"
        ].astype(int)
        self.routes_df["Model"] = self.routes_df["Equipment"].str.split("-")
        self.routes_df = self.routes_df.explode("Model")
        self.routes_df["Source airport ID"] = pd.to_numeric(
            self.routes_df["Source airport ID"].replace("\\N", np.nan), errors="coerce"
        )
        self.routes_df["Destination airport ID"] = pd.to_numeric(
            self.routes_df["Destination airport ID"].replace("\\N", np.nan),
            errors="coerce",
        )
        self.airports_df["Airport ID"] = pd.to_numeric(
            self.airports_df["Airport ID"].replace("\\N", np.nan), errors="coerce"
        )

    def calculate_distances(self):
        """
        Calculates distances between source and
        destination airports in the routes DataFrame.

        Returns:
        dict: A dictionary containing pairs of airport IDs as keys and
        their corresponding haversine distances as values.
            The distances are calculated using the haversine_distance method.
        """
        distances = {}

        for index, row in self.routes_df.iterrows():
            source_airport_id = row["Source airport ID"]
            dest_airport_id = row["Destination airport ID"]

            source_coords = self.get_coordinates(source_airport_id)
            dest_coords = self.get_coordinates(dest_airport_id)

            if source_coords is not None and dest_coords is not None:
                distance = self.haversine_distance(*source_coords, *dest_coords)
                distances[(source_airport_id, dest_airport_id)] = distance

        return distances

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculates the haversine distance between
        two sets of latitude and longitude coordinates.

        Parameters:
        - lat1 (float): Latitude of the first point.
        - lon1 (float): Longitude of the first point.
        - lat2 (float): Latitude of the second point.
        - lon2 (float): Longitude of the second point.

        Returns:
        float:
        The haversine distance between the
        two sets of coordinates, in kilometers.
        If any of the input values is None,
        a default distance of 0 is returned.
        """
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
        """
        Retrieves latitude and longitude coordinates
        for a specified airport ID from the airports DataFrame.

        Parameters:
        - airport_id (int): The unique identifier of the airport
        for which coordinates are requested.

        Returns:
        tuple:
        A tuple containing latitude and longitude coordinates
        of the specified airport.
        If the airport ID is not found in the DataFrame,
        returns (None, None).
        """
        row = self.airports_df[self.airports_df["Airport ID"] == airport_id]
        if not row.empty:
            latitude = row["Latitude"].values[0]
            longitude = row["Longitude"].values[0]
            return latitude, longitude
        else:
            return None, None

    def plot_airports_by_country(self, country_name):
        """
        Plots airports in a specified country on a world map.

        Parameters:

        - country_name (str):
          The name of the country for which airports will be plotted.

        Returns:
        None

        This method filters the airports DataFrame
        based on the provided country name,
        converts the data to a GeoDataFrame,
        and plots the airports on a world map.
        If no airports are found for the specified country,
        an error message is printed.
        """
        # Filter airports by the specified country
        country_airports = self.airports_df[self.airports_df["Country"] == country_name]

        # Check if any airports found
        if country_airports.empty:
            print("Error: Country not found or no airports available for this country.")
            return

        # Convert DataFrame to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            country_airports,
            geometry=gpd.points_from_xy(
                country_airports.Longitude, country_airports.Latitude
            ),
        )

        # Load world map
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        country_map = world[world.name == country_name]

        # Plotting
        fig, ax = plt.subplots()
        country_map.plot(ax=ax, color="white", edgecolor="black")
        gdf.plot(ax=ax, color="red", markersize=10)
        plt.show()

    def add_distance_column(self):
        """
        Adds a 'distance' column to the routes DataFrame,
        representing the haversine distance
        between source and destination airports.

        Returns:
        None

        This method merges the routes DataFrame with
        the airports DataFrame twice to obtain
        the coordinates (latitude and longitude)
        for both source and destination airports.
        Then, it calculates the haversine distance between
        these coordinates and adds the result
        as a new 'distance' column to the routes DataFrame.
        """

        # Merge routes_df with airports_df to get source airport coordinates
        routes_with_source_coords = self.routes_df.merge(
            self.airports_df[["Airport ID", "Latitude", "Longitude"]],
            left_on="Source airport ID",
            right_on="Airport ID",
            how="left",
        ).rename(
            columns={"Latitude": "Source Latitude", "Longitude": "Source Longitude"}
        )

        # Merge to get destination airport coordinates
        routes_with_all_coords = routes_with_source_coords.merge(
            self.airports_df[["Airport ID", "Latitude", "Longitude"]],
            left_on="Destination airport ID",
            right_on="Airport ID",
            how="left",
        ).rename(
            columns={
                "Latitude": "Destination Latitude",
                "Longitude": "Destination Longitude",
            }
        )

        # Calculate distances and add as a new column
        self.routes_df["distance"] = routes_with_all_coords.apply(
            lambda row: self.haversine_distance(
                row["Source Latitude"],
                row["Source Longitude"],
                row["Destination Latitude"],
                row["Destination Longitude"],
            ),
            axis=1,
        )

    def distance_analysis(self, bins=30, show_mean=True, show_median=True):
        """
        Analyzes the distribution of flight distances in the routes DataFrame.

        Parameters:
        - bins (int, optional): Number of bins in the histogram. Default is 30.
        - show_mean (bool, optional):
          Whether to display a vertical line for the mean distance.
        Default is True.
        - show_median (bool, optional):
          Whether to display a vertical line for the median distance.
        Default is True.

        Returns:
        None

        This method filters out None values,
        creates a histogram of flight distances,
        and optionally adds vertical lines for the mean
        and median distances.
        The resulting plot provides insights into
        the distribution of flight distances.
        """
        # Filter out None values (if any)
        valid_distances = self.routes_df["distance"].dropna()

        # Create a histogram of flight distances
        plt.figure(figsize=(10, 6))
        plt.hist(
            valid_distances, bins=bins, color="skyblue", edgecolor="black", alpha=0.7
        )

        # Add vertical lines for mean and median distances
        if show_mean:
            mean_val = valid_distances.mean()
            plt.axvline(
                mean_val,
                color="red",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean: {mean_val:.2f}",
            )
        if show_median:
            median_val = valid_distances.median()
            plt.axvline(
                median_val,
                color="green",
                linestyle="dashed",
                linewidth=1,
                label=f"Median: {median_val:.2f}",
            )

        # Set labels and title
        plt.xlabel("Flight Distance")
        plt.ylabel("Frequency")
        plt.title("Distribution of Flight Distances")

        # Show plot
        plt.legend()
        plt.show()

    def plot_flights_from_airport(self, airport_code, internal=False):
        """
        Plots flight paths originating from a specified airport.

        Parameters:
        - airport_code (str): The IATA code of the source airport.
        - internal (bool, optional):
          Whether to filter and plot only internal flights.
        Default is False.

        Returns:
        None

        This method filters routes based on the specified airport
        as the source airport, and optionally filters internal flights.
        It then creates a Plotly trace for flight paths
        and plots the resulting paths on a geographical map.
        """
        # Filter routes based on the specified airport as the source airport
        airport_routes = self.routes_df[
            self.routes_df["Source airport"] == airport_code
        ]

        # Optionally, filter internal flights
        if internal:
            airport_country = self.airports_df.loc[
                self.airports_df["IATA"] == airport_code, "Country"
            ].iloc[0]
            airport_routes = airport_routes[
                airport_routes["Destination airport"].isin(
                    self.airports_df[self.airports_df["Country"] == airport_country][
                        "IATA"
                    ]
                )
            ]

            # Set title for internal flights
            title = f"Internal flights from {airport_code} Airport"
        else:
            # Set title for all flights
            title = f"All flights from {airport_code} Airport"

        # Create Plotly trace for flight paths
        flight_paths = []
        for i, row in airport_routes.iterrows():
            origin = self.airports_df.loc[
                self.airports_df["IATA"] == row["Source airport"]
            ].iloc[0]
            destination = self.airports_df.loc[
                self.airports_df["IATA"] == row["Destination airport"]
            ].iloc[0]

            flight_paths.append(
                go.Scattergeo(
                    locationmode="ISO-3",
                    lon=[float(origin["Longitude"]), float(destination["Longitude"])],
                    lat=[float(origin["Latitude"]), float(destination["Latitude"])],
                    mode="lines",
                    line=dict(width=0.5, color="red"),
                    opacity=0.8,
                    name=row["Destination airport"],
                )
            )

        # Create Plotly trace for the specified airport marker
        airport_marker = go.Scattergeo(
            locationmode="ISO-3",
            lon=[
                float(
                    self.airports_df.loc[
                        self.airports_df["IATA"] == airport_code, "Longitude"
                    ].iloc[0]
                )
            ],
            lat=[
                float(
                    self.airports_df.loc[
                        self.airports_df["IATA"] == airport_code, "Latitude"
                    ].iloc[0]
                )
            ],
            mode="markers",
            marker=dict(size=10, color="blue"),
            text=f"{airport_code} Airport",
            name=f"{airport_code} Airport",
        )

        # Set layout
        layout = go.Layout(
            title=title,
            showlegend=True,
            geo=dict(
                scope="world",
                showland=True,
                landcolor="rgb(229, 229, 229)",
                countrycolor="rgb(255, 255, 255)",
                coastlinecolor="rgb(255, 255, 255)",
                projection_type="natural earth",
            ),
        )

        # Create the figure and plot
        fig = go.Figure(data=[*flight_paths, airport_marker], layout=layout)
        fig.show()

    def top_models(self, countries=None, top_n=10):
        """
        Plot the N most used airplane models by number of routes.

        Parameters:
        - countries (str or list, optional):
          The country or list of countries to filter the routes by.
          If not provided (default), it considers all routes.
        - top_n (int, optional):
          Number of top airplane models to plot. Default is 10.

        Returns:
        - Plot of the N most used airplane models by number of routes.
        - None
        """
        # Merge dataframes
        merged_df1 = pd.merge(
            self.routes_df,
            self.airports_df,
            left_on="Source airport ID",
            right_on="Airport ID",
        )
        merged_df2 = pd.merge(
            merged_df1,
            self.airplanes_df,
            how="left",
            left_on="Equipment",
            right_on="IATA code",
        )

        # Filter routes by selected country(s)
        routes_subset = (
            merged_df2[merged_df2["Country"].isin(countries)]
            if isinstance(countries, list)
            else (
                merged_df2[merged_df2["Country"] == countries]
                if countries
                else merged_df2
            )
        )

        # Group by airplane model and count number of routes
        model_counts = routes_subset.groupby("Name_y").size().reset_index(name="Count")

        # Sort by count in descending order
        model_counts = model_counts.sort_values("Count", ascending=False)

        # Select the top N airplane models
        top_models = model_counts.head(top_n)

        # Plot the N most used airplane models
        if not top_models.empty:
            plt.figure(figsize=(12, 6))
            top_models.plot(x="Name_y", y="Count", kind="bar", color="skyblue")
            plt.xlabel("Airplane Model")
            plt.ylabel("Number of Routes")
            title = f"Top {top_n} Airplane Models by Number of Routes"
            title += f" for {countries}" if countries else " for all countries"
            plt.title(title)
            plt.show()
        else:
            print(
                f"No data available for the specified conditions: countries={countries}, top_n={top_n}"
            )

    def plot_flights_from_country(self, country, internal=False):
        """
        Plots flight paths originating from airports in a specified country,
        distinguishing between internal and all flights.
        The resulting visualization is displayed on a world map.

        Parameters:
        - country (str):
          Name of country for which flight paths will be plotted.
        - internal (bool, optional):
          Whether to filter and plot only internal flights.
          Default is False.

        Returns:
        None

        This method filters routes based on airports in the specified country,
        optionally filters internal flights,
        and creates a Plotly trace for flight paths.
        The resulting paths are displayed on a world map,
        and the title of the plot indicates whether it shows internal
        flights or all flights from the country.
        """
        # Filter routes based on the country
        country_routes = self.routes_df[
            self.routes_df["Source airport"].isin(
                self.airports_df[self.airports_df["Country"] == country]["IATA"]
            )
        ]

        # Optionally, filter internal flights
        if internal:
            country_airports = self.airports_df[self.airports_df["Country"] == country][
                "IATA"
            ]
            country_routes = country_routes[
                country_routes["Destination airport"].isin(country_airports)
            ]

        # Create Plotly traces for flight paths
        flight_paths = []
        for i, row in country_routes.iterrows():
            origin = self.airports_df.loc[
                self.airports_df["IATA"] == row["Source airport"]
            ].iloc[0]
            destination = self.airports_df.loc[
                self.airports_df["IATA"] == row["Destination airport"]
            ].iloc[0]

            flight_paths.append(
                go.Scattergeo(
                    locationmode="ISO-3",
                    lon=[float(origin["Longitude"]), float(destination["Longitude"])],
                    lat=[float(origin["Latitude"]), float(destination["Latitude"])],
                    mode="lines",
                    line=dict(width=2, color="blue"),
                    opacity=0.7,
                    name=f"{origin['IATA']} to {destination['IATA']}",
                )
            )

        # Set layout for the map
        layout = go.Layout(
            title="Internal flights" if internal else f"All flights from {country}",
            showlegend=True,
            geo=dict(
                scope="world",
                projection_type="natural earth",
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
            ),
        )

        # Create the figure and add the traces
        fig = go.Figure(data=flight_paths, layout=layout)
        fig.show()

    def aircraft(self):
        """
        Return a list of unique aircraft names from the merged DataFrame of routes and airplanes.
        """
        aircraft_names = self.airplanes_df["Name"].unique().tolist()
        return aircraft_names

    def aircraft_info(self, _aircraft_name):
        """
        Display specifications of a given aircraft using the OpenAI GPT-3.5 Turbo model.

        Parameters:
            _aircraft_name (str): The name of the aircraft for which specifications are requested.

        Raises:
            ValueError: If the provided aircraft name is not found in the dataset.
                        Instructs the user to choose a valid aircraft name from the available dataset.
            ValueError: If the OpenAI API key is not found in the environment variable.
                        Instructs the user to set the 'OPENAI_API_KEY' environment variable with their API key.

        Note:
            Ensure that the OpenAI API key is set in the 'OPENAI_API_KEY' environment variable.
            The generated specifications are displayed in Markdown format.
        """
        # Check if the aircraft name is in the list of aircrafts
        aircraft_details = self.aircraft()  # Call the aircraft method to get the details

        if _aircraft_name not in aircraft_details:
            raise ValueError(
                f"Aircraft '{_aircraft_name}' not found. Please choose a valid aircraft name from the dataset. Available aircraft models:\n{aircraft_details}"
        )

        # Fetch your OpenAI API key from the environment variable
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable with your API key."
            )
        else:
            # Initialize the OpenAI language model
            llm = ChatOpenAI(api_key=api_key, temperature=0.9)

            # Generate a table of specifications in Markdown using LLM
            specifications_prompt = (
                f"Provide specifications table for {_aircraft_name}."
            )
            result = llm.invoke(specifications_prompt)
            specifications_content = result.content

            # Display the generated table of specifications in Markdown
            display(Markdown(specifications_content))
            
    def airports(self):
        """
        Return unique airport names from the airports DataFrame.
        """
        airport_names = list(set(self.airports_df["Name"]))
        return airport_names


    def airport_info(self, _airport_name):
        """
        Display specifications of a given airport using the OpenAI GPT-3.5 Turbo model.

        Parameters:
            _airport_name (str): The name of the airport for which specifications are requested.

        Raises:
            ValueError: If the OpenAI API key is not found in the environment variable.
                        Instructs the user to set the 'OPENAI_API_KEY' environment variable with their API key.

        Note:
            Ensure that the OpenAI API key is set in the 'OPENAI_API_KEY' environment variable.
            The generated specifications are displayed in Markdown format.
        """
        # Fetch your OpenAI API key from the environment variable
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable with your API key."
            )
        else:
            # Initialize the OpenAI language model
            llm = ChatOpenAI(api_key=api_key, temperature=0.9)

            # Get the list of airport names
            airports = self.airports()
            if _airport_name not in airports:
                print(f"Airport information not available for '{_airport_name}'.")
            else:
                # Generate a table of specifications in Markdown using LLM
                specifications_prompt = (
                    f"Provide specifications table for {_airport_name}."
                )
                result = llm.invoke(specifications_prompt)
                specifications_content = result.content

                # Display the generated table of specifications in Markdown
                display(Markdown(specifications_content))

    def refined_plot_flights_from_country(
        self, country, internal=False, short_haul_cutoff=1000
    ):
        """
        Plots flight paths originating from airports
        in a specified country, distinguishing between
        short-haul and long-haul flights. Additionally,
        provides information on total short-haul distance,
        potential CO2 emissions reduction by replacing flights with trains,
        and visualizes flight paths.

        Parameters:
        - country (str): The name of the country for which
          flight paths will be plotted.
        - internal (bool, optional): Whether to filter
          and plot only internal flights.
          Default is False.
        - short_haul_cutoff (float, optional): Distance threshold
          (in kilometers) for defining short-haul flights.
          Default is 1000.

        Returns:
        None

        This method filters routes based on airports
        in the specified country, calculates distances,
        and creates a Plotly trace for flight paths.
        It distinguishes between short-haul and long-haul flights,
        provides annotations for total short-haul distance
        and potential CO2 emission reductions,
        and visualizes the flight paths on a world map.
        """
        # Filter routes by country
        country_airports = self.airports_df[self.airports_df["Country"] == country][
            "IATA"
        ]
        relevant_routes = self.routes_df[
            self.routes_df["Source airport"].isin(country_airports)
        ]

        if internal:
            relevant_routes = relevant_routes[
                relevant_routes["Destination airport"].isin(country_airports)
            ]

        flight_paths = []
        total_short_haul_distance = (
            0  # Initialize total distance for short-haul flights
        )

        # Define emissions factors
        # (example values, adjust according to credible sources)
        flight_emission_per_km = 246  # grams of CO2 per km for flights
        train_emission_per_km = 35  # grams of CO2 per km for trains

        # Containers for the legend names
        added_short_haul_legend = False
        added_long_haul_legend = False

        for _, row in relevant_routes.iterrows():
            origin = self.airports_df[
                self.airports_df["IATA"] == row["Source airport"]
            ].iloc[0]
            destination = self.airports_df[
                self.airports_df["IATA"] == row["Destination airport"]
            ].iloc[0]
            distance = self.haversine_distance(
                origin["Longitude"],
                origin["Latitude"],
                destination["Longitude"],
                destination["Latitude"],
            )

            is_short_haul = distance <= short_haul_cutoff
            color = "green" if is_short_haul else "red"
            if is_short_haul:
                total_short_haul_distance += distance
                if not added_short_haul_legend:
                    name = "Short-Haul Flight"
                    added_short_haul_legend = True
                else:
                    name = None
            else:
                if not added_long_haul_legend:
                    name = "Long-Haul Flight"
                    added_long_haul_legend = True
                else:
                    name = None

            flight_paths.append(
                go.Scattergeo(
                    locationmode="ISO-3",
                    lon=[origin["Longitude"], destination["Longitude"]],
                    lat=[origin["Latitude"], destination["Latitude"]],
                    mode="lines",
                    line=dict(width=2, color=color),
                    opacity=0.7,
                    name=name,
                )
            )

        # Calculate emissions and reductions
        total_flight_emissions = (
            total_short_haul_distance * flight_emission_per_km / 1e6
        )  # Convert to tons
        potential_train_emissions = (
            total_short_haul_distance * train_emission_per_km / 1e6
        )  # Convert to tons
        potential_emission_reduction = (
            total_flight_emissions - potential_train_emissions
        )

        # Annotations for total short-haul distance
        # Annotations potential emission reductions
        annotations = [
            go.layout.Annotation(
                text=f"Total short-haul distance: {total_short_haul_distance:.2f} km",
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=1.10,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.9)",
            ),
            go.layout.Annotation(
                text=f"Potential CO2 reduction by replacing flights with trains: {potential_emission_reduction:.2f} tons",
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=1.05,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.9)",
            ),
        ]

        # Create the figure with the flight paths added
        fig = go.Figure(data=flight_paths)

        # Update the layout for the figure
        fig.update_layout(
            title=f"Flight Routes from {country} (Short-Haul vs Long-Haul)",
            showlegend=True,
            geo=dict(
                scope="world",
                projection_type="equirectangular",
                showland=True,
                landcolor="rgb(243,243,243)",
                countrycolor="rgb(204,204,204)",
            ),
            annotations=annotations,
            margin=dict(
                t=120, l=0, r=0, b=0
            ),  # Adjust the top margin to ensure annotations are visible
        )

        # Show the figure
        fig.show()
