import os
import pandas as pd
from zipfile import ZipFile
import requests
from io import BytesIO

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
