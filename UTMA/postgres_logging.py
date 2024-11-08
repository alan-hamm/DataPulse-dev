"""
PostgresLoggingHandler - Custom logging handler to store logs in a PostgreSQL database.

This script defines a custom logging handler, `PostgresLoggingHandler`, that uses a PostgreSQL
connection pool to efficiently manage connections for logging purposes. Log entries are stored 
in a specified table in the database, which is created automatically if it does not exist.

Author: Alan Hamm
Date: November 2024
Created with AI-Assistance

Classes:
    PostgresLoggingHandler: Custom logging handler for PostgreSQL that uses a connection pool 
                            and automatically creates a logging table if needed.

Usage:
    Initialize the `PostgresLoggingHandler` with database parameters, a table name, and 
    connection pool settings. Attach the handler to Python's logging system to log messages
    directly to PostgreSQL.

Example:
    db_params = {
        'dbname': 'your_db',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'localhost',
        'port': 5432
    }
    postgres_handler = PostgresLoggingHandler(db_params, table_name="logs", minconn=1, maxconn=5)
    logger = logging.getLogger()
    logger.addHandler(postgres_handler)

Notes:
    - Ensure PostgreSQL server settings allow sufficient concurrent connections as specified by 
      minconn and maxconn parameters in `SimpleConnectionPool`.
    - This script requires the psycopg2 library for PostgreSQL connectivity.
"""

import logging
import psycopg2
from psycopg2 import sql, pool
from datetime import datetime

class PostgresLoggingHandler(logging.Handler):
    pool = None  # Define a class-level pool

    def __init__(self, db_params, table_name="logs", minconn=1, maxconn=5):
        super().__init__()
        self.table_name = table_name
        if not PostgresLoggingHandler.pool:
            try:
                # Initialize the connection pool if it hasn't been created yet
                PostgresLoggingHandler.pool = psycopg2.pool.SimpleConnectionPool(
                    minconn, maxconn, **db_params
                )
            except psycopg2.OperationalError as e:
                print(f"Database connection pool error: {e}")

        # Create the table if it does not exist
        self.create_table()

    def get_connection(self):
        """Retrieve a connection from the pool."""
        if PostgresLoggingHandler.pool:
            try:
                return PostgresLoggingHandler.pool.getconn()
            except psycopg2.OperationalError as e:
                print("Error obtaining connection from pool:", e)
                return None
        else:
            print("Connection pool is not initialized.")
            return None

    def release_connection(self, conn):
        """Release the connection back to the pool."""
        if PostgresLoggingHandler.pool and conn:
            PostgresLoggingHandler.pool.putconn(conn)

    def create_table(self):
        """Create the logs table if it does not exist."""
        conn = self.get_connection()  # Get a connection for table creation
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        id SERIAL PRIMARY KEY,
                        log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        log_level VARCHAR(20),
                        message TEXT,
                        module VARCHAR(100),
                        func_name VARCHAR(100),
                        line_no INTEGER
                    );
                """).format(sql.Identifier(self.table_name)))
                conn.commit()
            except Exception as e:
                print("Error creating table:", e)
            finally:
                cursor.close()
                self.release_connection(conn)  # Release the connection after use

    def emit(self, record):
        conn = self.get_connection()  # Get a connection from the pool
        if not conn:
            return  # Skip logging if no connection is available

        log_time = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        cursor = conn.cursor()
        try:
            cursor.execute(
                sql.SQL("""
                    INSERT INTO {} (log_time, log_level, message, module, func_name, line_no)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """).format(sql.Identifier(self.table_name)),
                (
                    log_time,
                    record.levelname,
                    record.getMessage(),
                    record.module,
                    record.funcName,
                    record.lineno,
                )
            )
            conn.commit()
        except Exception as e:
            print("Failed to log to PostgreSQL:", e)
        finally:
            cursor.close()
            self.release_connection(conn)  # Release the connection back to the pool

    def close(self):
        if PostgresLoggingHandler.pool:
            PostgresLoggingHandler.pool.closeall()  # Close all connections in the pool
        super().close()
