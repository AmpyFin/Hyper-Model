from datetime import datetime
from clients.client import Client

class SystemClient(Client):
    
    def run(self) -> None:
        self.logger.info("System client started")
        # How this will run is essentially this:

        # During Early Hours:
        # Run early hour client

        # During Trading Hours:
        # Run trading and ranking client

        # During After Hours:
        # Run log pushing client, then testing, training, and parameter pushing client


def main():
    client = SystemClient()
    client.run()

if __name__ == "__main__":
    main()
