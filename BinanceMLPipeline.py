#!/usr/bin/env python3
# BinanceMLPipeline.py
# Real-time ML pipeline integrated with Spark Structured Streaming for cryptocurrency data

import os
import uuid
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf, lit, from_json, to_json, struct, unix_timestamp, current_timestamp
from pyspark.sql.streaming import StreamingQuery
import river
from river import compose, feature_extraction, linear_model, preprocessing, stats, metrics, drift

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("BinanceMLPipeline")

class BinanceMLPipeline:
    """Real-time machine learning pipeline for processing cryptocurrency data from Spark Structured Streaming"""
    
    def __init__(self, 
                spark_session: SparkSession = None,
                cassandra_host: str = None,
                cassandra_port: str = None,
                cassandra_username: str = None, 
                cassandra_password: str = None,
                cassandra_keyspace: str = None,
                predictions_table: str = None,
                checkpoint_dir: str = "/tmp/ml-checkpoints",
                model_output_path: str = "/tmp/models"):
        """
        Initialize the ML Pipeline
        
        Args:
            spark_session: Existing SparkSession (if None, will create a new one)
            cassandra_host: Cassandra host for storing predictions
            cassandra_port: Cassandra port
            cassandra_username: Cassandra username
            cassandra_password: Cassandra password  
            cassandra_keyspace: Cassandra keyspace
            predictions_table: Table to store predictions
            checkpoint_dir: Directory for checkpointing streaming state
            model_output_path: Directory to save trained models
        """
        # Set up environment variables
        self.cassandra_host = cassandra_host or os.environ.get("ASSET_CASSANDRA_HOST", "binance-cassandra")
        self.cassandra_port = cassandra_port or os.environ.get("ASSET_CASSANDRA_PORT", "9042")
        self.cassandra_username = cassandra_username or os.environ.get("ASSET_CASSANDRA_USERNAME", "adminadmin")
        self.cassandra_password = cassandra_password or os.environ.get("ASSET_CASSANDRA_PASSWORD", "adminadmin") 
        self.cassandra_keyspace = cassandra_keyspace or os.environ.get("ASSET_CASSANDRA_KEYSPACE", "assets")
        self.predictions_table = predictions_table or os.environ.get("ASSET_PREDICTIONS_TABLE", "predictions")
        self.checkpoint_dir = checkpoint_dir
        self.model_output_path = model_output_path
        
        # Initialize or use provided SparkSession
        self.spark = spark_session or self._create_spark_session()
        
        # Supported assets for prediction
        self.supported_assets = ['bitcoin', 'ethereum', 'binance-coin']
        
        # Initialize models (one per asset)
        self.models = self._initialize_models()
        
        # Initialize metrics tracker
        self.metrics_tracker = {asset: metrics.MSE() for asset in self.supported_assets}
        
        # Initialize drift detectors
        self.drift_detectors = {asset: drift.ADWIN() for asset in self.supported_assets}
        
        # Initialize feature windows for time series features
        self.feature_windows = {asset: [] for asset in self.supported_assets}
        self.window_size = 10  # Number of prior observations to use for features
        
        # Ensure directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.model_output_path, exist_ok=True)
    
    def _create_spark_session(self) -> SparkSession:
        """Create a new SparkSession configured for ML pipeline"""
        spark = SparkSession.builder \
            .appName("BinanceMLPipeline") \
            .config("spark.sql.streaming.checkpointLocation", self.checkpoint_dir) \
            .config("spark.cassandra.connection.host", self.cassandra_host) \
            .config("spark.cassandra.connection.port", self.cassandra_port) \
            .config("spark.cassandra.auth.username", self.cassandra_username) \
            .config("spark.cassandra.auth.password", self.cassandra_password) \
            .getOrCreate()
        
        # Add necessary JARs for Cassandra connectivity
        return spark
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize online learning models for each asset"""
        models = {}
        
        for asset in self.supported_assets:
            # Define preprocessing pipeline
            preprocessor = compose.Pipeline(
                ('scale', preprocessing.StandardScaler()),
                ('drift_guard', preprocessing.Imputer())
            )
            
            # Create online model - here using Linear Regression with SGD
            base_model = compose.Pipeline(
                ('prep', preprocessor),
                ('regressor', linear_model.LinearRegression(
                    optimizer=river.optim.SGD(lr=0.01),
                    loss=river.optim.losses.Squared()
                ))
            )
            
            models[asset] = base_model
            
        return models
    
    def create_cassandra_predictions_table(self) -> None:
        """Create table for storing predictions if it doesn't exist"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.cassandra_keyspace}.{self.predictions_table} (
            id text,
            asset_name text,
            actual_price float,
            predicted_price float,
            prediction_horizon text,
            prediction_timestamp timestamp,
            features map<text, float>,
            metrics map<text, float>,
            drift_detected boolean,
            PRIMARY KEY (asset_name, prediction_timestamp)
        ) WITH CLUSTERING ORDER BY (prediction_timestamp DESC);
        """
        
        self.spark.sparkContext._jvm.org.apache.spark.sql.cassandra \
            .CassandraConnector \
            .apply(self.spark.sparkContext._jsc.sc().getConf()) \
            .withSessionDo(lambda session: session.execute(create_table_query))
        
        logger.info(f"Created or verified Cassandra predictions table: {self.cassandra_keyspace}.{self.predictions_table}")
    
    def process_batch(self, batch_df: DataFrame, batch_id: int) -> None:
        """Process a micro-batch of data from structured streaming"""
        if batch_df.isEmpty():
            logger.info(f"Batch {batch_id}: Empty batch, skipping")
            return
            
        # Convert streaming DataFrame to Pandas for ML processing
        pandas_df = batch_df.toPandas()
        
        # Process records by asset
        for asset in self.supported_assets:
            # Filter dataframe for current asset
            asset_df = pandas_df[pandas_df['asset_name'] == asset]
            
            if len(asset_df) == 0:
                continue
                
            # Sort by timestamp to ensure proper sequence
            asset_df = asset_df.sort_values('timestamp')
            
            # Process each record for this asset
            for _, row in asset_df.iterrows():
                self._process_record(asset, row)
        
        logger.info(f"Batch {batch_id}: Processed {len(pandas_df)} records")
    
    def _process_record(self, asset: str, row: pd.Series) -> None:
        """Process a single data record for model training and prediction"""
        try:
            # Extract features
            features = self._extract_features(asset, row)
            if features is None:
                # Not enough history to create features
                return
                
            # Get target value (we'll predict next period's closing price)
            target = float(row['close'])
            
            # Make prediction (before updating the model)
            prediction = self.models[asset].predict_one(features)
            
            # Update the model with this example
            self.models[asset].learn_one(features, target)
            
            # Update metrics
            self.metrics_tracker[asset].update(target, prediction)
            current_mse = self.metrics_tracker[asset].get()
            
            # Check for drift
            drift_detected = False
            if self.drift_detectors[asset].update(abs(prediction - target)):
                logger.info(f"Drift detected for {asset}")
                drift_detected = True
            
            # Store prediction
            self._store_prediction(
                asset=asset,
                actual=target,
                predicted=prediction,
                timestamp=row['timestamp'],
                features=features,
                metrics={'mse': current_mse},
                drift_detected=drift_detected
            )
            
            # Update feature window for this asset
            self.feature_windows[asset].append({
                'timestamp': row['timestamp'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'quote_volume': float(row['quote_volume']),
                'trades': int(row['trades'])
            })
            
            # Keep window at fixed size
            if len(self.feature_windows[asset]) > self.window_size:
                self.feature_windows[asset].pop(0)
                
        except Exception as e:
            logger.error(f"Error processing record for {asset}: {e}")
    
    def _extract_features(self, asset: str, row: pd.Series) -> Optional[Dict[str, float]]:
        """Extract features from the current record and historical window"""
        # Add current row data to feature set
        current = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            'quote_volume': float(row['quote_volume']),
            'trades': int(row['trades'])
        }
        
        # Need at least some history for proper features
        window = self.feature_windows[asset]
        if len(window) < 2:  # Need at least 2 data points for delta features
            return None
        
        features = {}
        
        # Basic price features
        features['open'] = current['open']
        features['high'] = current['high']
        features['low'] = current['low']
        features['close'] = current['close']
        features['volume'] = current['volume']
        
        # Add ratio and delta features
        features['high_low_ratio'] = current['high'] / current['low'] if current['low'] > 0 else 1.0
        features['close_open_ratio'] = current['close'] / current['open'] if current['open'] > 0 else 1.0
        
        prev = window[-1]
        features['price_change'] = current['close'] - prev['close']
        features['price_change_pct'] = features['price_change'] / prev['close'] if prev['close'] > 0 else 0.0
        features['volume_change'] = current['volume'] - prev['volume']
        features['volume_change_pct'] = features['volume_change'] / prev['volume'] if prev['volume'] > 0 else 0.0
        
        # Time-based aggregations if we have enough history
        if len(window) >= 5:
            last_5 = window[-5:]
            features['close_5_mean'] = np.mean([x['close'] for x in last_5])
            features['close_5_std'] = np.std([x['close'] for x in last_5])
            features['volume_5_mean'] = np.mean([x['volume'] for x in last_5])
            
        if len(window) >= 10:
            last_10 = window[-10:]
            features['close_10_mean'] = np.mean([x['close'] for x in last_10])
            features['close_10_std'] = np.std([x['close'] for x in last_10])
            features['volume_10_mean'] = np.mean([x['volume'] for x in last_10])
            
        return features
    
    def _store_prediction(self, asset: str, actual: float, predicted: float, 
                         timestamp: str, features: Dict[str, float], 
                         metrics: Dict[str, float], drift_detected: bool) -> None:
        """Store prediction results in Cassandra"""
        prediction_id = str(uuid.uuid4())
        
        # Convert dictionaries to Cassandra-compatible maps
        features_map = {str(k): float(v) for k, v in features.items()}
        metrics_map = {str(k): float(v) for k, v in metrics.items()}
        
        # Format timestamp
        pred_timestamp = datetime.now().isoformat()
        
        # Create a DataFrame with the prediction data
        prediction_df = self.spark.createDataFrame([
            (
                prediction_id, 
                asset, 
                float(actual), 
                float(predicted), 
                "1_period",  # Prediction horizon
                pred_timestamp,
                features_map, 
                metrics_map, 
                drift_detected
            )
        ], ["id", "asset_name", "actual_price", "predicted_price", 
            "prediction_horizon", "prediction_timestamp", 
            "features", "metrics", "drift_detected"])
        
        # Write to Cassandra
        prediction_df.write \
            .format("org.apache.spark.sql.cassandra") \
            .mode("append") \
            .option("keyspace", self.cassandra_keyspace) \
            .option("table", self.predictions_table) \
            .save()
    
    def save_models(self) -> None:
        """Save trained models to disk"""
        for asset, model in self.models.items():
            try:
                # River models can be pickled, but for better interoperability,
                # we'll just save key parameters as JSON
                model_params = {
                    "weights": self.models[asset]['regressor'].weights,
                    "bias": self.models[asset]['regressor'].bias,
                    "stats": {
                        "mse": self.metrics_tracker[asset].get()
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                model_path = os.path.join(self.model_output_path, f"{asset}_model.json")
                with open(model_path, 'w') as f:
                    json.dump(model_params, f)
                    
                logger.info(f"Saved model for {asset} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model for {asset}: {e}")
                
    def integrate_with_spark_stream(self, input_stream: DataFrame) -> StreamingQuery:
        """
        Integrate the ML pipeline with an existing Spark structured stream
        
        Args:
            input_stream: DataFrame stream from Spark Structured Streaming
            
        Returns:
            StreamingQuery object for the running query
        """
        # Ensure prediction table exists
        self.create_cassandra_predictions_table()
        
        # Define foreachBatch function to process micro-batches
        def _process_ml_batch(batch_df, batch_id):
            self.process_batch(batch_df, batch_id)
            # Periodically save models
            if batch_id % 10 == 0:
                self.save_models()
                
        # Start streaming query with our ML pipeline integration
        query = input_stream \
            .writeStream \
            .foreachBatch(_process_ml_batch) \
            .outputMode("update") \
            .option("checkpointLocation", os.path.join(self.checkpoint_dir, "ml-pipeline")) \
            .trigger(processingTime="10 seconds") \
            .start()
            
        return query
    
# Example of integrating this with existing BinanceConsumer
if __name__ == "__main__":
    # This would typically be integrated into your existing BinanceConsumer.py
    
    # Example: Create Spark session
    spark = SparkSession.builder \
        .appName("BinancePricePredictor") \
        .config("spark.sql.shuffle.partitions", 4) \
        .getOrCreate()
        
    # Create ML Pipeline
    ml_pipeline = BinanceMLPipeline(spark_session=spark)
    
    # Here you would get your existing streaming DataFrame
    # from BinanceConsumer after Avro decoding and transformation
    
    # Hypothetical example (replace with actual code from BinanceConsumer):
    """
    # Input stream from Kafka with Avro decoding
    input_stream = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "binance-redpanda:29092") \
        .option("subscribe", "data.asset_prices") \
        .option("startingOffsets", "latest") \
        .load() \
        .select(avro_decode_udf(col("value")).alias("data")) \
        .select(from_json(col("data"), schema).alias("parsed_data")) \
        .select("parsed_data.*") \
        .withColumn("open", col("open").cast("float")) \
        .withColumn("high", col("high").cast("float")) \
        .withColumn("low", col("low").cast("float")) \
        .withColumn("close", col("close").cast("float")) \
        .withColumn("volume", col("volume").cast("float")) \
        .withColumn("quote_volume", col("quote_volume").cast("float")) \
        .withColumn("trades", col("trades").cast("integer")) \
        .withColumn("consumed_at", current_timestamp())
    
    # Pass the transformed stream to the ML pipeline
    query = ml_pipeline.integrate_with_spark_stream(input_stream)
    
    query.awaitTermination()
    """ 