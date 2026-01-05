from pymongo import MongoClient
from typing import Optional, Dict, List
import os
from datetime import datetime
from bson import ObjectId
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MongoDBClient:
    def __init__(self):
        self.uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("DATABASE_NAME", "4_agents_db")
        self.client: Optional[MongoClient] = None
        self.db = None
        self.connect()
    
    def connect(self):
        try:
            if not self.uri:
                print("Warning: MONGODB_URI not found in environment variables. MongoDB features will be disabled.")
                self.client = None
                self.db = None
                return
            
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.admin.command('ping')
            print("Connected to MongoDB successfully")
        except Exception as e:
            print(f"Warning: Error connecting to MongoDB: {e}")
            print("The application will continue without database persistence.")
            self.client = None
            self.db = None
    
    def is_connected(self) -> bool:
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except:
            return False
    
    def save_analysis(self, analysis_data: Dict):
        """Save analysis result to database"""
        if not self.db:
            return None
        
        try:
            collection = self.db.analyses
            analysis_data["created_at"] = datetime.utcnow()
            result = collection.insert_one(analysis_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return None
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Get a specific analysis by ID"""
        if not self.db:
            return None
        
        try:
            collection = self.db.analyses
            analysis = collection.find_one({"_id": ObjectId(analysis_id)})
            if analysis:
                analysis["_id"] = str(analysis["_id"])
            return analysis
        except Exception as e:
            print(f"Error getting analysis: {e}")
            return None
    
    def get_all_analyses(self) -> List[Dict]:
        """Get all analyses"""
        if not self.db:
            return []
        
        try:
            collection = self.db.analyses
            analyses = list(collection.find().sort("created_at", -1).limit(50))
            for analysis in analyses:
                analysis["_id"] = str(analysis["_id"])
            return analyses
        except Exception as e:
            print(f"Error getting analyses: {e}")
            return []

