'''
- [ ] Define a new method called **aircrafts** that receives no arguments and prints only the list of aircraft models (Names)ircraf.
- [ ] Define a new method called **aircraft_info** that receives a string called _aircraft_name_. If the string is **NOT** in the list of aircrafts in the data, it should return an exception and present a way to guide the user into how they could choose a correct aircraft name.
- [ ] The latter method should use an LLM to print out a table of specifications about the aircraft model in Markdown.
- [ ] Define a new method called **airport_info** that does the same but for airports (don't make checks in this method, you are already demonstrating you understood it in the case for aicrafts).

<div class="alert alert-danger">
    <b> Do not include the API KEY in the project. Declare the API KEY as a system variable.</b>
    <br>
    <b> If the API KEY is not working, let me know ASAP. </b>
</div>



'''
    def airports_info(self, _airport_name_):
        
        # Retrieving the list of aircraft models from the aircraft method
        unique_airport_names = self.airport()
        
        if _airport_name_ not in unique_airport_names:
            raise ValueError(f"Airport '{_airport_name_}' not found. Please choose a valid aircraft name from the dataset. Available aircraft models:\n{unique_airport_names_}")
        else:
            pass
        '''
            # Fetch your OpenAI API key from the environment variable
            api_key = os.environ.get('OPENAI_API_KEY')
            
            # Initialize the OpenAI language model
            llm = OpenAI(api_key=api_key, temperature=0.9)
            
            # Create a prompt template for generating aircraft specifications
            prompt_template_airport_info = PromptTable(
                input_variables=['_airport_name_'], 
                template="Print out a table of specifications about {_airport_name_}?"
            )
            
            # Generate and print the specifications using the language model
            print(llm(prompt_template_airport_specs.format(_airport_name_=_airport_name_)))
        '''
'''
# Create a system variable: did that
# pip install openai
# Authentication: Bearer OPENAI_API_KEY
# 

   '''   