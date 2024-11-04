# write_to_postgres.py - Database Utilities for SLIF
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This script provides functions for interacting with a PostgreSQL database within the Scalable LDA Insights Framework (SLIF).
# It includes utilities for dynamically creating tables, inserting LDA model data, and managing database connections,
# leveraging SQLAlchemy for ORM capabilities.
#
# Functions:
# - Table creation: Defines functions for dynamically creating tables to store model data.
# - Data insertion: Includes methods for inserting large datasets efficiently using Dask and SQLAlchemy.
# - Connection management: Manages database connections and sessions.
#
# Dependencies:
# - Python libraries: os, json, random, hashlib, zipfile, logging, numpy, pandas
# - Database libraries: sqlalchemy, dask.dataframe
#
# Developed with AI assistance.

import os
from json import load
from random import shuffle
import pandas as pd 
import hashlib
import zipfile
import logging
import dask.dataframe as dd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Float, LargeBinary, DateTime, JSON, TEXT
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from .utils import garbage_collection

Base = declarative_base()

# Function to save text data and model to single ZIP file
def save_to_zip(time, top_folder, text_data, text_json, ldamodel, corpus, dictionary, texts_zip_dir):
    # Generate a unique filename based on current timestamp
    timestamp_str = time
    text_zip_filename = f"{timestamp_str}.zip"
    
    # Write the text content and model to a zip file within TEXTS_ZIP_DIR
    zip_path = os.path.join(texts_zip_dir,top_folder)
    zpath = os.path.join(zip_path, text_zip_filename)
    with zipfile.ZipFile(zpath, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"doc_{timestamp_str}.txt", text_data)
        zf.writestr(f"model_{timestamp_str}.pkl", ldamodel)
        zf.writestr(f"dict_{timestamp_str}.pkl", dictionary)
        zf.writestr(f"corpus_{timestamp_str}.pkl", corpus)
        zf.writestr(f"json_{timestamp_str}.pkl", text_json)
    return zpath


def create_dynamic_table_class(table_name):
    """
    Create a new SQLAlchemy model class with a dynamic table name.
    
    Args:
        table_name (str): The name of the table to be created.
        
    Returns:
        A new SQLAlchemy model class linked to the specified table name.
    """
    
    # Ensure that the provided table_name is a string and not empty
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError("Table name must be a non-empty string.")
    
    # Define a class that describes the table structure
    attributes = {
        '__tablename__' : table_name,
        '__table_args__': {'extend_existing': True},
        'time_key' : Column(TEXT, primary_key=True, nullable=False),
        'type' : Column(String),
        'num_workers' : Column(Integer),
        'batch_size' : Column(Integer),
        'num_documents' : Column(Integer),
        'text' : Column(TEXT) , # text is a list of file paths
        'text_json' : Column(LargeBinary),  # text_json is a pickled text_json
        'text_sha256' : Column(String),
        'text_md5' : Column(String),
        'convergence' : Column(Float(precision=32)),
        'perplexity' : Column(Float(precision=32)),
        'coherence' : Column(Float(precision=32)),
        'topics' : Column(Integer),
            
        # alpha_str and beta_str are categorical string representations; store them as strings
        'alpha_str' : Column(String),
        'n_alpha' : Column(Float(precision=32)),

        # beta_str is similar to alpha_str; n_beta is a numerical representation of beta
        'beta_str' : Column(String),
        'n_beta' :  Column(Float(precision=32)),

        # passes, iterations, update_every, eval_every, chunksize and random_state are integers
        'passes' : Column(Integer),
        'iterations' : Column(Integer),
        'update_every' : Column(Integer),
        'eval_every' : Column(Integer),
        'chunksize' : Column(Integer),
        'random_state' : Column(Integer),

        'per_word_topics' : Column(Boolean),
        'show_topics': Column(JSONB),

        # For lda_model, corpus, and dictionary, if they are binary blobs:
        #'lda_model' : Column(LargeBinary),
        #'corpus' : Column(LargeBinary),
        #'dictionary' : Column(LargeBinary)  ,      
        'create_pylda' : Column(Boolean) ,
        'create_pcoa' : Column(Boolean),

        # Enforce datetime type for time fields
        'time' : Column(DateTime),
        'end_time' : Column(DateTime)
    }

    # Create a new class type with a unique name derived from table_name
    dynamic_class = type(f"DynamicModelMetadata_{table_name}", (Base,), attributes)

    return dynamic_class


def create_table_if_not_exists(table_class, database_uri):
    """
    Create a table in the database based on the SQLAlchemy model class if it does not already exist.
    
    Args:
        table_class (class): The SQLAlchemy model class for the target table.
        database_uri (str): The database connection string.
    """
    
    # Create an engine using the provided DATABASE_URI
    engine = create_engine(database_uri, echo=False)
    
    # Use inspector to check if table exists
    inspector = inspect(engine)

    # Check if the table already exists in the database
    if not inspector.has_table(table_class.__tablename__):
        # The table does not exist, attempt to create it
        try:
            # Only create tables for this Base instance
            table_class.metadata.create_all(engine)
            logging.info(f"Table '{table_class.__tablename__}' created successfully.")
        except ProgrammingError as e:
            logging.error(f"An error occurred while creating the table: {e}")
            raise  # Re-raise exception after logging it for further handling or clean exit.
    else:
        logging.info(f"Table '{table_class.__tablename__}' already exists. No action taken.")


# Function to add new model data to metadata postgres table
def add_model_data_to_database(model_data, table_name, database_uri, 
                               num_documents, workers, batchsize, texts_zip_dir):
    """
    Add new model data to the specified table in the database.
    
    Args:
        model_data (dict): The dictionary containing model data.
        table_class (class): The SQLAlchemy model class for the target table.
        database_uri (str): The database connection string.
    """

    # Save large body of text to zip and update model_data reference
    texts_zipped = []

    # Path to the distinct document folder to contain all related documents
    document_dir = os.path.join(texts_zip_dir, model_data['text_md5'])
    document_dir = os.path.join(document_dir, f"number_of_topics-{model_data['topics']}")
    os.makedirs(document_dir, exist_ok=True)

    #for text_list in model_data['text']:
    for text_list in model_data['text']:
        combined_text = ''.join([''.join(sent) for sent in text_list])  # Combine all sentences into one string
        zip_path = save_to_zip(model_data['time_key'], document_dir, combined_text, \
                               model_data['text_json'], model_data['lda_model'], \
                               model_data['corpus'], model_data['dictionary'], texts_zip_dir)
        texts_zipped.append(zip_path)

    # Create an engine using the provided DATABASE_URI
    engine = create_engine(database_uri, echo=False)
    
    # Create a session factory bound to this engine
    Session = sessionmaker(bind=engine)
    
    # Instantiate a session
    session = Session()
    
    try:
        # Use the provided function to get the dynamic table class for table_name
        DynamicModel = create_dynamic_table_class(table_name)

        # Update model_data with additional information if necessary
        model_data['batch_size'] = batchsize
        model_data['num_workers'] = workers
        model_data['num_documents'] = num_documents
        new_model_data = {key: val for key, val in model_data.items() if key not in ['lda_model', 'corpus', 'dictionary']}
        
        # Log type information before insertion
        #logging.info(f"Type of 'create_pylda' before insertion: {type(new_model_data['create_pylda'])}")
        #logging.info(f"Type of 'create_pylda' before insertion: {new_model_data['create_pylda']}")
        #logging.info(f"Type of 'create_pcoa' before insertion: {type(new_model_data['create_pcoa'])}")
        #logging.info(f"Type of 'create_pcoa' before insertion: {new_model_data['create_pcoa']}")

        # Create an instance of the dynamic table class with model_data
        record = DynamicModel(**new_model_data)
        
        # Add the new record to the session and commit it to the database
        session.add(record)
        session.commit()
        
        logging.info("Data added successfully to Postgres Table")
        
    except Exception as e:
        # If there was any exception during insertion, rollback the transaction
        session.rollback()
        
        # Log or print error message here (depending on your logging setup)
        logging.error(f"An error occurred while adding data: {e}")
        
    finally:
        # Close the session whether or not an exception occurred
        if 'session' in locals():
            session.close()