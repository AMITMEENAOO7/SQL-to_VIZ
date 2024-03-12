from dotenv import load_dotenv
import mysql.connector
from pathlib import Path


# create a config class
class Config:
    # initialized the class
    def __init__(self):
        # Model configuration
        self._model = "gemini-pro"


        # API configuration
        self._api_host = "localhost"
        self._api_port = 5000
        self._api_debug = False

        # SQL connection
        self.database_username = "root"
        self.database_password = "feb182001"
        self.database_host = "localhost"
        self.database = "sakila"  

        # Load environment variables from .env file
        load_dotenv()

    # return the model attribute
    @property
    def model(self):
        return self._model
    

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

