    def aircraft(self):
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
        aircraft_names = self.aircraft()
        
        if _aircraft_name not in aircraft_names:
            raise ValueError(f"Aircraft '{_aircraft_name}' not found. Please choose a valid aircraft name from the dataset. Available aircraft models:\n{list(self.aircrafts.keys())}")

        # Fetch your OpenAI API key from the environment variable
        os.environ['OPENAI_API_KEY'] = "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"
        api_key = os.environ['OPENAI_API_KEY']
        #api_key = os.environ.get('OPENAI_API_KEY')
        
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
        airport_names = self.airports_df['Name'].unique()
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