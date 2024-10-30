#%%

from sqlalchemy import create_engine, Column, String, Integer, Boolean, Float, LargeBinary, DateTime, JSON, TEXT
from sqlalchemy.ext.declarative import declarative_base

# Define the base class for declarative tables
Base = declarative_base()

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
    
    # Define the dynamic class within the function scope

    # Define a class that describes the table structure
    class DynamicModelMetadata(Base):
        __tablename__ = table_name


        time_key = Column(String, primary_key=True, nullable=False)
        type = Column(String)
        num_workers = Column(Integer)
        batch_size = Column(Integer)
        num_documents = Column(Integer)
        text = Column(JSON)  # text is a list of file paths
        text_json = Column(JSON)  # text_json is a list of lists of tokenized sentences
        text_sha256 = Column(String)
        text_md5 = Column(String)
        convergence = Column(Float(precision=32))
        perplexity = Column(Float(precision=32))
        coherence = Column(Float(precision=32))
        topics = Column(Integer)
            
        # alpha_str and beta_str are categorical string representations; store them as strings
        alpha_str = Column(String)
        n_alpha = Column(Float(precision=32))

        # beta_str is similar to alpha_str; n_beta is a numerical representation of beta
        beta_str = Column(String)
        n_beta =  Column(Float(precision=32))

        # passes, iterations, update_every, eval_every, chunksize and random_state are integers
        passes = Column(Integer)
        iterations = Column(Integer)
        update_every = Column(Integer)
        eval_every = Column(Integer)
        chunksize = Column(Integer)
        random_state = Column(Integer)

        per_word_topics = Column(Boolean)
        top_words = Column(TEXT)  # Assuming top_words is a long string or JSON serializable
        # For lda_model, corpus, and dictionary, if they are binary blobs:
        lda_model = Column(LargeBinary)
        corpus = Column(LargeBinary)
        dictionary = Column(LargeBinary)
            
        create_pylda = Column(Boolean) 
        create_pcoa = Column(Boolean)

        # Enforce datetime type for time fields
        time = Column(DateTime)
        end_time = Column(DateTime)

    return DynamicModelMetadata

# Example usage:
table_name_example = "dynamic_model_metadata"
DynamicModelMetadataClass = create_dynamic_table_class(table_name_example)

# Now you can use DynamicModelMetadataClass like any other SQLAlchemy model class.
# For example:

# Create an engine instance with your database URI
engine_example_uri = "postgresql://postgres:admin@localhost:5432/SLIF"
engine_example = create_engine(engine_example_uri)

# Create all tables stored in this metadata (if they don't exist)
Base.metadata.create_all(engine_example)