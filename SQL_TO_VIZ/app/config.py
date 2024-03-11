from dotenv import load_dotenv
import mysql.connector
from pathlib import Path


# create a config class
class Config:
    # initialized the class
    def __init__(self):
        # Model configuration
        self._model = "gpt-3.5-turbo"
        self._max_tokens = 1024
        self._top_k_value = 0
        self._temperature = 0

        # API configuration
        self._api_host = "localhost"
        self._api_port = 5000
        self._api_debug = False

        # SQL connection
        self.database_username = "root"
        self.database_password = "Thrang2sql"
        self.database_host = "localhost"
        self.database = "sakila"  

        # Load environment variables from .env file
        load_dotenv()

    # return the model attribute
    @property
    def model(self):
        return self._model
    
    # return the max_tokens attribute
    @property
    def max_tokens(self):
        return self._max_tokens
    
    # return the top_k attribute
    @property
    def top_k(self):
        return self._top_k_value
    
    # return the temperature attribute
    @property
    def temperature(self):
        return self._temperature
    
    # return the api_host attribute
    @property
    def api_host(self):
        return self._api_host
    
    # return the api_port attribute
    @property
    def api_port(self):
        return self._api_port
    
    # return the api_debug attribute
    @property
    def api_debug(self):
        return self._api_debug
    
    # return the engine
    @property
    def engine(self):
        return mysql.connector.connect(
            user=self.database_username, 
            password=self.database_password, 
            host=self.database_host, 
            database=self.database
        )

