from abc import ABC, abstractmethod
from clients import setup_client_logger

class Client(ABC):
    """Base client interface that all clients must implement."""
    
    def __init__(self):
        """Initialize the client with its logger."""
        self.logger = setup_client_logger(self.__class__.__name__.lower())
    
    @abstractmethod
    def run(self) -> None:
        """
        Main execution method that must be implemented by all clients.
        This method should contain the primary logic for the client's operation.
        """
        pass
