import os
import pandas as pd
from zipfile import ZipFile
import requests
from io import BytesIO

class FlightDataProcessor:
    def __init__(self, gitlab_url):
        # Download the zip file from the GitLab URL
        response = requests.get(gitlab_url)
    
        zip_file = ZipFile(BytesIO(response.content))

        # Set the destination folder for the extracted data
        destination_folder = os.path.join('group_03', 'downloads1')

        # Ensure the destination folder exists
        os.makedirs(destination_folder, exist_ok=True)

        # Extract the datasets from the zip folder
        zip_file.extractall(destination_folder)

        # Read the datasets into pandas dataframes
        self.airlines_df = pd.read_csv(os.path.join(destination_folder, 'airlines.csv'))
        self.airplanes_df = pd.read_csv(os.path.join(destination_folder, 'airplanes.csv'))
        self.airports_df = pd.read_csv(os.path.join(destination_folder, 'airports.csv'))
        self.routes_df = pd.read_csv(os.path.join(destination_folder, 'routes.csv'))

# GitLab URL for the raw flight_data.zip file
gitlab_url = 'https://gitlab.com/adpro9641208/group_03/-/raw/main/flight_data.zip'

# Create an instance of FlightDataProcessor with the GitLab URL
flight_processor = FlightDataProcessor(gitlab_url)

# Access the dataframes
airlines = flight_processor.airlines_df
airplanes = flight_processor.airplanes_df
airports = flight_processor.airports_df
routes = flight_processor.routes_df
