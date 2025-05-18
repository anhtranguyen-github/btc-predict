#!/usr/bin/env python3
# BinanceConsumerWithML.py
# Integration of ML pipeline with BinanceConsumer for real-time predictions

import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf, lit, from_json, to_json, struct, current_timestamp
from pyspark.sql.avro.functions import from_avro
from BinanceMLPipeline import BinanceMLPipeline

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("BinanceConsumerWithML")

def create_spark_session():
    """Create a Spark session configured for the Binance crypto data pipeline"""
    spark = SparkSession.builder \
        .appName("BinanceConsumerWithML") \
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.apache.spark:spark-avro_2.12:3.5.0,"
                "com.datastax.spark:spark-cassandra-connector_2.12:3.4.1") \
        .config("spark.sql.streaming.checkpointLocation", os.environ.get("CHECKPOINT_DIR", "/tmp/checkpoint")) \
        .config("spark.cassandra.connection.host", os.environ.get("ASSET_CASSANDRA_HOST", "binance-cassandra")) \
        .config("spark.cassandra.connection.port", os.environ.get("ASSET_CASSANDRA_PORT", "9042")) \
        .config("spark.cassandra.auth.username", os.environ.get("ASSET_CASSANDRA_USERNAME", "adminadmin")) \
        .config("spark.cassandra.auth.password", os.environ.get("ASSET_CASSANDRA_PASSWORD", "adminadmin")) \
        .getOrCreate()
    
    return spark

def main():
    """Main execution function for the BinanceConsumer with ML integration"""
    # Get configuration from environment variables
    kafka_brokers = os.environ.get("REDPANDA_BROKERS", "binance-redpanda:29092")
    asset_prices_topic = os.environ.get("ASSET_PRICES_TOPIC", "data.asset_prices")
    avro_schema_location = os.environ.get("ASSET_SCHEMA_LOCATION", "./schemas/assets.avsc")
    cassandra_keyspace = os.environ.get("ASSET_CASSANDRA_KEYSPACE", "assets")
    cassandra_table = os.environ.get("ASSET_CASSANDRA_TABLE", "assets")
    ml_checkpoint_dir = os.environ.get("ML_CHECKPOINT_DIR", "/tmp/ml-checkpoints")
    model_output_path = os.environ.get("MODEL_OUTPUT_PATH", "/tmp/models")
    
    # Initialize Spark
    spark = create_spark_session()
    logger.info("Spark session initialized")
    
    # Read Avro schema
    try:
        with open(avro_schema_location, 'r') as f:
            avro_schema_str = f.read()
        logger.info(f"Successfully loaded Avro schema from {avro_schema_location}")
    except Exception as e:
        logger.error(f"Failed to load Avro schema: {e}")
        return
    
    # Define Spark schema for after decoding
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("asset_name", StringType(), True),
        StructField("open", StringType(), True),
        StructField("high", StringType(), True),
        StructField("low", StringType(), True),
        StructField("close", StringType(), True),
        StructField("volume", StringType(), True),
        StructField("quote_volume", StringType(), True),
        StructField("trades", StringType(), True),
        StructField("is_closed", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("close_time", StringType(), True),
        StructField("collected_at", StringType(), True)
    ])
    
    # Read from Kafka
    logger.info(f"Connecting to Kafka at {kafka_brokers}, topic {asset_prices_topic}")
    kafka_stream = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_brokers) \
        .option("subscribe", asset_prices_topic) \
        .option("startingOffsets", "latest") \
        .load()
    
    # Deserialize Avro data
    logger.info("Setting up Avro deserialization")
    value_df = kafka_stream.select(
        from_avro(col("value"), avro_schema_str).alias("data"),
        col("timestamp").alias("kafka_timestamp")
    )
    
    # Extract and transform fields
    logger.info("Extracting fields from Avro data")
    parsed_df = value_df.select("data.*")
    
    # Transform data - convert strings to appropriate types
    logger.info("Transforming data types")
    transformed_df = parsed_df \
        .withColumn("open", col("open").cast("float")) \
        .withColumn("high", col("high").cast("float")) \
        .withColumn("low", col("low").cast("float")) \
        .withColumn("close", col("close").cast("float")) \
        .withColumn("volume", col("volume").cast("float")) \
        .withColumn("quote_volume", col("quote_volume").cast("float")) \
        .withColumn("trades", col("trades").cast("integer")) \
        .withColumn("is_closed", col("is_closed").cast("boolean")) \
        .withColumn("consumed_at", current_timestamp())
    
    # Write transformed data to Cassandra (original consumer functionality)
    logger.info(f"Setting up Cassandra sink to {cassandra_keyspace}.{cassandra_table}")
    cassandra_write_query = transformed_df \
        .writeStream \
        .format("org.apache.spark.sql.cassandra") \
        .option("checkpointLocation", os.path.join(os.environ.get("CHECKPOINT_DIR", "/tmp/checkpoint"), "cassandra")) \
        .option("keyspace", cassandra_keyspace) \
        .option("table", cassandra_table) \
        .outputMode("append") \
        .start()
    
    # Initialize ML Pipeline
    logger.info("Initializing ML Pipeline")
    ml_pipeline = BinanceMLPipeline(
        spark_session=spark,
        cassandra_keyspace=cassandra_keyspace,
        predictions_table="predictions",
        checkpoint_dir=ml_checkpoint_dir,
        model_output_path=model_output_path
    )
    
    # Send the same transformed data to the ML pipeline
    logger.info("Starting ML pipeline integration")
    ml_query = ml_pipeline.integrate_with_spark_stream(transformed_df)
    
    # Wait for termination
    logger.info("All streaming queries started, waiting for termination")
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main() 