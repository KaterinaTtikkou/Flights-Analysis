
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

