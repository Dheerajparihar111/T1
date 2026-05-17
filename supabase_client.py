# ============================================================
# supabase_client.py
# Supabase Connection & Data Fetching Module
# Handles all database operations in one place
# ============================================================

import os
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging so we can see what's happening
logger = logging.getLogger(__name__)


# ============================================================
# STEP 1: Initialize Supabase Client (Singleton Pattern)
# We create ONE client instance and reuse it throughout the app
# ============================================================

def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client using credentials from .env file.
    
    Returns:
        Client: Authenticated Supabase client ready to query
        
    Raises:
        ValueError: If environment variables are missing
        Exception: If connection fails
    """
    # Read credentials from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    # Safety check — make sure credentials exist
    if not supabase_url:
        raise ValueError("❌ SUPABASE_URL is missing from .env file")
    if not supabase_key:
        raise ValueError("❌ SUPABASE_ANON_KEY is missing from .env file")

    try:
        # Create the Supabase client
        client: Client = create_client(supabase_url, supabase_key)
        logger.info("✅ Supabase client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Supabase: {str(e)}")
        raise


# Create a single global client instance (initialized once at startup)
try:
    supabase: Client = get_supabase_client()
except Exception as e:
    logger.critical(f"💥 Cannot start app without Supabase connection: {e}")
    raise


# ============================================================
# STEP 2: Fetch Latest Health Record
# Gets the most recently added record from health_data table
# ============================================================

async def fetch_latest_health_data() -> Optional[Dict[str, Any]]:
    """
    Fetches the MOST RECENT health record from Supabase.
    Orders by 'created_at' descending and returns the first row.
    
    Returns:
        dict: A single health record with all columns
        None: If no records found
        
    Raises:
        Exception: If database query fails
    """
    # Read table name from environment (defaults to 'health_data')
    table_name = os.getenv("SUPABASE_TABLE_NAME", "health_data")
    
    try:
        logger.info(f"📡 Fetching latest record from '{table_name}' table...")
        
        # Query Supabase:
        # - Select ALL columns (*)
        # - Order by created_at newest first
        # - Limit to 1 record (the latest)
        response = (
            supabase
            .table(table_name)        # Which table to query
            .select("*")              # Get all columns
            .order("created_at", desc=True)   # Newest first
            .limit(1)                 # Only need the latest one
            .execute()                # Run the query
        )

        # Check if we got any data back
        if not response.data or len(response.data) == 0:
            logger.warning("⚠️ No health records found in the database")
            return None

        # Return the first (latest) record as a dictionary
        latest_record = response.data[0]
        logger.info(f"✅ Successfully fetched record for: {latest_record.get('name', 'Unknown')}")
        return latest_record

    except Exception as e:
        logger.error(f"❌ Database query failed: {str(e)}")
        raise Exception(f"Failed to fetch health data from Supabase: {str(e)}")


# ============================================================
# STEP 3: Fetch Health Record by User Name
# Gets a specific user's latest record by their name
# ============================================================

async def fetch_health_data_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the latest health record for a specific user by name.
    
    Args:
        name: The user's name to search for
        
    Returns:
        dict: The user's health record
        None: If user not found
    """
    table_name = os.getenv("SUPABASE_TABLE_NAME", "health_data")
    
    try:
        logger.info(f"📡 Fetching health data for user: '{name}'...")
        
        response = (
            supabase
            .table(table_name)
            .select("*")
            .eq("name", name)               # Filter by name (eq = equals)
            .order("created_at", desc=True)  # Get most recent entry
            .limit(1)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            logger.warning(f"⚠️ No records found for user: {name}")
            return None

        logger.info(f"✅ Found health record for: {name}")
        return response.data[0]

    except Exception as e:
        logger.error(f"❌ Failed to fetch data for {name}: {str(e)}")
        raise Exception(f"Database error: {str(e)}")


# ============================================================
# STEP 4: Insert New Health Record (Bonus utility function)
# Saves a new health record to Supabase
# ============================================================

async def insert_health_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Inserts a new health record into the Supabase table.
    
    Args:
        data: Dictionary with health data fields
        
    Returns:
        dict: The newly created record
        None: If insertion failed
    """
    table_name = os.getenv("SUPABASE_TABLE_NAME", "health_data")
    
    try:
        logger.info(f"💾 Inserting new health record for: {data.get('name', 'Unknown')}...")
        
        response = (
            supabase
            .table(table_name)
            .insert(data)      # Insert the data dictionary
            .execute()
        )

        if response.data:
            logger.info("✅ Health record inserted successfully")
            return response.data[0]
        return None

    except Exception as e:
        logger.error(f"❌ Failed to insert health data: {str(e)}")
        raise Exception(f"Database insert error: {str(e)}")
