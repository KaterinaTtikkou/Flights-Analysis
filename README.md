# Group_03

Welcome to Group_03! This README file serves as a guide to understanding the purpose, functionality, and usage of the flight processor class, FIIU. Whether you're a developer contributing to the codebase, a user exploring its features, or simply curious about what this class entails, you'll find essential information here to get started.

**Fun Fact**🌟
Did you know that analyzing flight data with our class FIIU can sometimes feel like being on a turbulent flight? But don't worry, we'll navigate through the data with smooth landings and insightful discoveries! Buckle up and enjoy the data ride!

## Table of Contents

- [Introduction](#introduction)
- [Contacts](#contacts)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Compliance](#compliance)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Group_03 presents the FIIU ✈️, a powerful flight processor class designed to download files from [International Air Transportation Association](https://www.iata.org/) and process the provided datasets. FIIU offers various methods for analyzing airport distances, country flights, aircraft information, and more, making it an indispensable tool for handling diverse aviation-related requests.

FIIU simplifies the process of retrieving and analyzing flight data by seamlessly downloading files from the provided link and parsing them for relevant information. FIIU reads four datasets: airlines, airplanes, airports, and routes, enabling comprehensive analysis of commercial flight data. Whether you're a developer seeking to automate flight data retrieval tasks or an analyst exploring aviation trends, FIIU provides the functionality and flexibility to streamline your workflow.

This README provides an overview of FIIU's features, installation instructions, usage guidelines, and information on contributing to its development. Explore the capabilities of FIIU and unlock new insights into aviation data with ease. 🚀

## Contacts

1. Katerina Ttikkou - 40298 - 40298@novasbe.pt
2. Rita Thaci - 61359 - 61359@novasbe.pt
3. Ayscha Dräger - 57892 - 57892@novasbe.pt
4. Anna Lena Schwarz - 58902 - 58902@novasbe.pt

## Features

List the key features and functionalities

- [__innit__]: Downloads, extracts, and processes flight data from the given zip URL.
- [calculate_distances]: Calculates distances between source and destination airports in the routes DataFrame.
- [haversine_distance]: Calculates the haversine distance between two sets of latitude and longitude coordinates.
- [get_coordinates]: Retrieves latitude and longitude coordinates for a specified airport ID from the airports DataFrame.
- [plot_airports_by_country]: Plots airports in a specified country on a world map.
- [add_distance_column]: Adds a 'distance' column to the routes DataFrame, representing the haversine distance between source and destination airports.
- [distance_analysis]: Analyzes the distribution of flight distances in the routes DataFrame.
- [plot_flights_from_airport]: Plots flight paths originating from a specified airport.
- [top_models]: Plot the N most used airplane models by number of routes.
- [plot_flights_from_country]: Plots flight paths originating from airports in a specified country, distinguishing between internal and all flights. The resulting visualization is displayed on a world map.
- [aircrafts]: Retrieve a list of unique aircraft names based on the merged data of routes and airplanes.
- [aircraft_info]: Display specifications of a given aircraft using the OpenAI GPT-3.5 Turbo model.
- [airports]: Retrieve list of unique airport based on the 'IATA' codes in airports DataFrame.
- [airport_info]: Display specifications of given airport using OpenAI GPT-3.5 Turbo model.
- [refined_plot_flights_from_country]: Plots flight paths originating from airports in a specified country, distinguishing between short-haul and long-haul flights. Additionally, provides information on total short-haul distance, potential CO2 emissions reduction by replacing flights with trains, and visualizes flight paths.

## Installation

To install Group_03 follow these steps:

1. Clone the repository from GitLab
2. Navigate to the project directory
3. Activate the virtual environment

## Dependencies

Before running the project, make sure you have the following dependencies installed:

- Pandas >= A powerful data manipulation library.
- Requests >= HTTP library for making requests.
- Numpy >= For numerical computing.
- Geopandas >= Extends the Pandas library to allow spatial operations on geometric types.
- Matplotlib >= A comprehensive library for creating static, animated, and interactive visualizations in Python.
- Plotly >= A graphing library used for creating interactive plots and dashboards.
- Langchain >= A library/module for natural language processing (NLP).
- IPython >= Provides an architecture for interactive computing.
- Markdown >= For rendering Markdown text in IPython notebooks.

## Usage

Once installed, you can use FIIU by following these steps:

1. **Download and process data**: To download and read the dataset as well as initiate the class, please use the below:

    ```python
    from Class.FIIU import FIIU
    test.FIIU()

2. **Calculate Distances**: Call the `calculate_distances` method to calculate distances between source and destination airports in the routes DataFrame.

    ```python
    fiiu.calculate_distances()
    ```

3. **Print Haversine distane**: To calculate the haversine distance between two sets of coordinates, use the `haversine_distance` method.

    ```python
    fiiu.haversine_distance(lat1, lon1, lat2, lon2)
    ```

4. **Print coordinates**: To retrieves latitude and longitude coordinates for a specified airport ID, use the `get_coordinates` method.

    ```python
    fiiu.get_coordinates(airport_id=2)
    ```

5. **Plot Airports by Country**: Use the `plot_airports_by_country` method to plot airports belonging to a specific country.

    ```python
    fiiu.plot_airports_by_country(country_name='United States')
    ```

6. **Create a distance column**: Use the `add_distance_column` to add a distance column into the dataframe.

    ```python
    add_distance_column()
    ```

7. **Display flight distances**: Analyze the distribution of flight distances using the `distance_analysis` method.

    ```python
    fiiu.distance_analysis(bins=30, show_mean=True, show_median=True)
    ```

8. **Plot flights from an airport**: Plot flight paths originating from a specified airport using the `plot_flights_from_airport` method.

    ```python
    fiiu.plot_flights_from_airport(airport_code="JFK", internal=False)
    ```

9. **Plot the most used airplane models**: Plot the N most used airplane models by number of routes using the `top_models` method.

    ```python
    fiiu.top_models(countries="United States", top_n=10)
    ```

10. **Plot flights from a country**: Plot flight paths originating from airports in a specified country using the `plot_flights_from_country` method.

    ```python
    fiiu.plot_flights_from_country(country="United States", internal=False)
    ```

11. **List of unique aircraft names**: Retrieve a list of unique aircraft names based on the merged data of routes and airplanes using the `aircrafts` method.

    ```python
    aircrafts()
    ```

12. **Display specifications or a given aircraft**: Display specifications of a given aircraft using the OpenAI GPT-3.5 Turbo model with the `aircraft_info` method. Ensure that your OpenAI API key is set in the 'OPENAI_API_KEY' environment variable.

    ```python
    fiiu.aircraft_info(_aircraft_name="Boeing 747")
    ```

13. **Retrieve Unique Airport Names**: Retrieve a list of unique airport names based on the 'IATA' codes in the airports DataFrame using the `airports` method.

    ```python
    fiiu.airports()
    ```

14. **Airport Information**: Plot flight paths originating from airports in a specified country, distinguishing between short-haul and long-haul flights using the `refined_plot_flights_from_country` method.

    ```python
    airport_info(_airport_name='JFK')
    ```

15. **Display short-haul and long-haul flights**: Plot flight paths originating from airports in a specified country, distinguishing between short-haul and long-haul flights using the `refined_plot_flights_from_country` method.

    ```python
    fiiu.refined_plot_flights_from_country(country="United States", internal=False, short_haul_cutoff=1000)
    ```

## Compliance

Our project follows the PEP 8 style guide for Python code. With a Pydantic score of 7.3, we maintain a high level of compliance with the PEP 8 guidelines, ensuring clean and readable code.

We strive to uphold these standards throughout the development process and welcome contributions that align with PEP 8 principles.

## Contributing

We welcome contributions from developers of all skill levels. If you'd like to contribute to Group_03, please follow these guidelines:

- Clone the repository
- Create a new branch (`git checkout -b feature`)
- Make your changes
- Commit your changes (`git commit -am 'Add feature'`)
- Push to the branch (`git push origin feature`)
- Create a new Pull Request

## License

Group_03 is licensed under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. See the [LICENSE](LICENSE) file for details.
