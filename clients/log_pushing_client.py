from datetime import datetime, timedelta, timezone
import os
from typing import List
from clients.client import Client
from utils.db_clients.mongodb_client import MongoDBClient
from control import log_expiration_days

# Will connect to MongoDB and push logs to a collection - which is stored for 30 days. Each log is pushed daily at the end of trading day. That is when it's called.
# It will have logs, then the date. We will check to see if any log dates are longer than 30 days. If they are, they are deleted from the database.
class LogPushingClient(Client):
    def __init__(self):
        super().__init__()
        self.mongodb = MongoDBClient()
        self.client = self.mongodb.get_client()
        self.logs_db = self.client.logs
        
    def _get_collection_name(self, filename: str) -> str:
        """Convert log filename to collection name with proper underscore format."""
        # Remove .log extension
        name = filename[:-4]
        
        # Handle special cases
        if name == "logpushingclient":
            return "log_pushing_client"
        elif name == "parameterpushingclient":
            return "parameter_pushing_client"
        elif name == "earlyhourclient":
            return "early_hour_client"
        else:
            # For other clients, insert underscore before "client"
            return name.replace("client", "_client")

    def insert_logs_to_mongodb(self) -> None:
        """Step 1: Insert logs into respective collections in MongoDB with current date."""
        self.logger.info("Starting log insertion to MongoDB")
        current_date = datetime.now(timezone.utc)
        
        # Get all logs from the logs directory
        log_files = os.listdir("logs")
        
        for log_file in log_files:
            if not log_file.endswith('.log'):
                continue
                
            # Transform filename to proper collection name
            collection_name = self._get_collection_name(log_file)
            collection = self.logs_db[collection_name]
            
            # Read the log file content
            with open(os.path.join("logs", log_file), 'r') as f:
                log_lines = f.readlines()
            
            # Skip empty files
            if not log_lines:
                continue
                
            # Process each log line
            for log_line in log_lines:
                log_line = log_line.strip()
                if log_line:  # Skip empty lines
                    # Insert each log statement as a separate document
                    collection.insert_one({
                        "date": current_date,
                        "log_statement": log_line
                    })
            
            self.logger.info(f"Inserted logs from {log_file} to MongoDB collection {collection_name}")

    def clean_old_logs_from_mongodb(self) -> None:
        """Step 2: Delete logs older than log_expiration_days from MongoDB."""
        self.logger.info(f"Cleaning logs older than {log_expiration_days} days from MongoDB")
        
        # Calculate cutoff date using timezone-aware datetime
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=log_expiration_days)
        
        # Clean old logs from each collection
        for collection_name in self.logs_db.list_collection_names():
            collection = self.logs_db[collection_name]
            result = collection.delete_many({"date": {"$lt": cutoff_date}})
            self.logger.info(f"Deleted {result.deleted_count} old logs from {collection_name}")

    def clean_local_log_files(self) -> None:
        """Step 3: Delete local log files after successful MongoDB insertion."""
        self.logger.info("Cleaning local log files")
        
        log_files = os.listdir("logs")
        for log_file in log_files:
            if not log_file.endswith('.log'):
                continue
                
            file_path = os.path.join("logs", log_file)
            # Create empty file instead of deleting to maintain logging capability
            with open(file_path, 'w') as f:
                pass
            self.logger.info(f"Cleaned {log_file}")

    def run(self) -> None:
        """Execute the log pushing process in sequence."""
        self.logger.info("Log pushing client started")
        
        try:
            # Step 1: Insert logs to MongoDB
            self.insert_logs_to_mongodb()
            
            # Step 2: Clean old logs from MongoDB
            self.clean_old_logs_from_mongodb()
            
            # Step 3: Clean local log files
            self.clean_local_log_files()
            
            self.logger.info("Log pushing client operations completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during log pushing: {str(e)}")
            raise
        finally:
            self.mongodb.close()

def main():
    client = LogPushingClient()
    client.run()

if __name__ == "__main__":
    main()
