FROM bitnami/spark:latest

USER root

RUN apt-get update && \
    apt-get install -y curl python3-dev python3-pip libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy application files
COPY BinanceMLPipeline.py .
COPY BinanceConsumerWithML.py .
COPY schemas/ ./schemas/
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary JARs for Kafka, Avro, and Cassandra
RUN mkdir -p /opt/bitnami/spark/jars && \
    curl https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.0/spark-sql-kafka-0-10_2.12-3.5.0.jar -o /opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.5.0.jar && \
    curl https://repo1.maven.org/maven2/org/apache/spark/spark-avro_2.12/3.5.0/spark-avro_2.12-3.5.0.jar -o /opt/bitnami/spark/jars/spark-avro_2.12-3.5.0.jar && \
    curl https://repo1.maven.org/maven2/com/datastax/spark/spark-cassandra-connector_2.12/3.4.1/spark-cassandra-connector_2.12-3.4.1.jar -o /opt/bitnami/spark/jars/spark-cassandra-connector_2.12-3.4.1.jar && \
    curl https://repo1.maven.org/maven2/com/datastax/spark/spark-cassandra-connector-driver_2.12/3.4.1/spark-cassandra-connector-driver_2.12-3.4.1.jar -o /opt/bitnami/spark/jars/spark-cassandra-connector-driver_2.12-3.4.1.jar

# Create directories for checkpointing and model storage
RUN mkdir -p /tmp/checkpoint /tmp/ml-checkpoints /models && \
    chmod -R 777 /tmp/checkpoint /tmp/ml-checkpoints /models

# Command to run the application
CMD ["python3", "BinanceConsumerWithML.py"] 