import asyncio
from .config import Config
from .api.app import start_api_server

# main
def main():
    # initialized the config
    config = Config()

    # calling the api server
    start_api_server(host=config.api_host, port=config.api_port)

if __name__=="__main__":
    main()
    