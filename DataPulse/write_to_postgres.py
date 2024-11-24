# write_to_postgres.py - SpectraSync: Dynamic Database Interface for Topic Modeling Data
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This module empowers SpectraSync with robust PostgreSQL utilities for storing and managing topic modeling data.
# It facilitates seamless database interactions, providing dynamic table creation, efficient data insertion, and
# streamlined connection management—all optimized to handle large-scale data with Dask and SQLAlchemy. Designed
# to support SpectraSync’s multi-dimensional topic analysis, this interface ensures data integrity and efficiency.
#
# Functions:
# - Table creation: Dynamic functions to create structured tables for storing model insights and metadata.
# - Data insertion: High-efficiency methods for bulk data insertion, leveraging Dask and SQLAlchemy’s ORM.
# - Connection management: Manages persistent database connections and sessions for stable, high-volume transactions.
#
# Dependencies:
# - Python libraries: os, json, random, hashlib, zipfile, logging, numpy, pandas
# - Database libraries: sqlalchemy, dask.dataframe
#
# Developed with AI assistance to power SpectraSync’s scalable and adaptive data architecture.


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
from sqlalchemy import Column, Integer, String, DateTime, Boolean, LargeBinary, TEXT, JSON, Float, Numeric
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
import pickle
from .utils import garbage_collection

Base = declarative_base()

# Function to save text data and model to single ZIP file
def save_to_zip(time, top_folder, text_data, text_json, ldamodel, corpus, dictionary, texts_zip_dir):
    #print("We are inside save_to_zip")
    # Generate a unique filename based on current timestamp
    timestamp_str = time
    text_zip_filename = f"{timestamp_str}.zip"
    
    text = pickle.loads(text_data)
    # Write the text content and model to a zip file within TEXTS_ZIP_DIR
    zip_path = os.path.join(texts_zip_dir,top_folder)
    #try:
    #    logging.info(f"Attempting to create Zip directory: {zip_path}")
    #    os.makedirs(zip_path, exist_ok=True)
    #    logging.info(f"Zip directory created at: {zip_path}")
    #except Exception as e:
    #    logging.error(f"Failed to create ZIP directory: {e}")

    zpath = os.path.join(zip_path, text_zip_filename)
    with zipfile.ZipFile(zpath, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"doc_{timestamp_str}.txt", text)
        zf.writestr(f"model_{timestamp_str}.pkl", ldamodel)
        zf.writestr(f"dict_{timestamp_str}.pkl", dictionary)
        zf.writestr(f"corpus_{timestamp_str}.pkl", corpus)
        zf.writestr(f"json_{timestamp_str}.pkl", text_json)

    #logging.info(f"Zip file created at: {zip_path}")
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
    
    # Metadata and Identifiers
    'time_key' : Column(TEXT, primary_key=True, nullable=False),
    'type' : Column(String, nullable=False),
    'start_time' : Column(DateTime, nullable=False),
    'end_time' : Column(DateTime, nullable=False),
    'num_workers' : Column(Integer, nullable=False),
    
    # Document and Batch Details
    'batch_size' : Column(Integer, nullable=False),
    'num_word' : Column(Integer, nullable=False),
    'text' : Column(LargeBinary, nullable=False),
    'text_json' : Column(LargeBinary, nullable=False),
    'max_attempts': Column(Integer, nullable=False),
    'top_topics': Column(JSONB, nullable=False),
    'topics_words': Column(JSONB, nullable=False),
    'validation_result': Column(JSONB, nullable=False),
    'text_sha256' : Column(String(64), nullable=False),
    'text_md5' : Column(String(32), nullable=False),
    'text_path' : Column(TEXT, nullable=False),
    'pca_path' : Column(TEXT, nullable=False),
    'pca_gpu_path' : Column(TEXT, nullable=False),
    'pylda_path' : Column(TEXT, nullable=False),


    # Model and Training Parameters
    'topics' : Column(Integer, nullable=False),
    'alpha_str' : Column(String(20), nullable=False),
    'n_alpha' : Column(Numeric(precision=20, scale=15), nullable=False),
    'beta_str' : Column(String(20), nullable=False),
    'n_beta' : Column(Numeric(precision=20, scale=15), nullable=False),
    'passes' : Column(Integer, nullable=False),
    'iterations' : Column(Integer, nullable=False),
    'update_every' : Column(Integer, nullable=False),
    'eval_every' : Column(Integer, nullable=False),
    'chunksize' : Column(Integer, nullable=False),
    'random_state' : Column(Integer, nullable=False),
    'per_word_topics' : Column(Boolean, nullable=False),
    
    # Evaluation Metrics
    'convergence': Column(Numeric(precision=20, scale=10), nullable=False),
    'nll': Column(Numeric(precision=20, scale=10), nullable=False),
    'perplexity': Column(Numeric(precision=20, scale=10), nullable=False),
    'coherence': Column(Numeric(precision=20, scale=10), nullable=False),
    'mean_coherence': Column(Numeric(precision=20, scale=10), nullable=False),
    'median_coherence': Column(Numeric(precision=20, scale=10), nullable=False),
    'mode_coherence': Column(Numeric(precision=20, scale=10), nullable=False),
    'std_coherence': Column(Numeric(precision=20, scale=10), nullable=False),
    'perplexity_threshold': Column(Numeric(precision=20, scale=10), nullable=False),
    
    # Visualization Placeholders
    'create_pylda' : Column(Boolean, nullable=False),
    'create_pcoa' : Column(Boolean, nullable=False),
    'create_pca_gpu': Column(Boolean, nullable=False)
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
def add_model_data_to_database(model_data, phase, table_name, database_uri, 
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
    #print("Attempting to create document directory...")
    document_dir = os.path.join(texts_zip_dir, phase, model_data['text_md5'])
    document_dir = os.path.join(document_dir, f"number_of_topics-{model_data['topics']}")
    #print(f"Document directory path: {document_dir}")
    #print(f"Final document directory path: {document_dir}")

    try:
        os.makedirs(document_dir, exist_ok=True)
        logging.info(f"Directory created at: {document_dir}")
    except Exception as e:
        logging.error(f"Error creating directory {document_dir}: {e}")

    try:
        logging.info(f"model_data['text_md5'] contents: {model_data.get('text_md5')}")
        text = [pickle.loads(model_data['text'])]
        for text_list in text:
            combined_text = ' '.join([''.join(sent) for sent in text_list])  # Combine all sentences into one string

            #logging.info("Calling save_to_zip...")
            zip_path = save_to_zip(model_data['time_key'], document_dir, pickle.dumps(combined_text), \
                                model_data['text_json'], model_data['lda_model'], \
                                model_data['corpus'], model_data['dictionary'], texts_zip_dir)
            
            texts_zipped.append(zip_path)
    except Exception as e:
        logging.error(f"Error during zipping process: {e}")

    # Create an engine using the provided DATABASE_URI
    engine = create_engine(database_uri, echo=False)  # Optional: `echo=True` for detailed SQL logging
    logging.info("Database engine created successfully.")

    # Create a session factory bound to this engine
    Session = sessionmaker(bind=engine)
    session = Session()
    logging.info("Database session created successfully.")
    
    try:
        # Use the provided function to get the dynamic table class for table_name
        DynamicModel = create_dynamic_table_class(table_name)

        # Update model_data with additional information if necessary
        model_data['batch_size'] = batchsize
        model_data['num_workers'] = workers
        model_data['num_documents'] = num_documents
        new_model_data = {key: val for key, val in model_data.items() if key not in ['num_documents','lda_model', 'corpus', 'dictionary']}
        
        # Log type information before insertion
        #logging.info(f"Type of 'create_pylda' before insertion: {type(new_model_data['create_pylda'])}")
        #logging.info(f"Type of 'create_pylda' before insertion: {new_model_data['create_pylda']}")
        #logging.info(f"Type of 'create_pcoa' before insertion: {type(new_model_data['create_pcoa'])}")
        #logging.info(f"Type of 'create_pcoa' before insertion: {new_model_data['create_pcoa']}")

        # Create an instance of the dynamic table class with model_data
        record = DynamicModel(**new_model_data)
        
        # Add the new record to the session and commit it to the database
        session.add(record)
        logging.info("Data added to session successfully.")

        # Commit the session to save changes
        session.commit()
        logging.info("Data committed successfully to the database.")
        
    except Exception as e:
        # Log or print error message here (depending on your logging setup)
        logging.error(f"An error occurred while adding data: {e}")
        # If there was any exception during insertion, rollback the transaction
        session.rollback()
    finally:
        # Close the session whether or not an exception occurred
        session.close()
        logging.info("Database session closed.")

        return zip_path