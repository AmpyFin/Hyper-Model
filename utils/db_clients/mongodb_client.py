# This is a client that will connect to MongoDB and perform operations on the database.

import os
from typing import Optional
from pymongo import MongoClient
import certifi

class MongoDBClient:
    _instance: Optional['MongoDBClient'] = None
    _client: Optional[MongoClient] = None

    def __new__(cls) -> 'MongoDBClient':
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if MongoDBClient._client is None:
            self._initialize_client()

    def _initialize_client(self) -> None:
        # Try to get connection string from environment variable first
        mongo_url = os.getenv('MONGO_URL')
        
        # If not in environment, fall back to config.py
        if not mongo_url:
            try:
                from config import MONGO_URL
                mongo_url = MONGO_URL
            except ImportError:
                raise ValueError("MongoDB connection string not found in environment or config.py")

        # Initialize the MongoDB client with proper SSL certificate handling
        MongoDBClient._client = MongoClient(mongo_url, tlsCAFile=certifi.where())

    def get_client(self) -> MongoClient:
        """Returns the MongoDB client instance."""
        if not MongoDBClient._client:
            self._initialize_client()
        return MongoDBClient._client

    def close(self) -> None:
        """Closes the MongoDB connection."""
        if MongoDBClient._client:
            MongoDBClient._client.close()
            MongoDBClient._client = None

