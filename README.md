# Bitcoin Trading Strategy ML System

A production-ready machine learning system for developing, testing, and deploying algorithmic trading strategies for Bitcoin markets. This project transforms raw Bitcoin price data into actionable trading signals using advanced technical indicators and machine learning models.

## Features

- **Modular Architecture**: Clean separation of data loading, feature engineering, model training, and evaluation components
- **Technical Indicators**: Implementation of 20+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Multiple Models**: Support for various ML algorithms (Random Forest, XGBoost, LogisticRegression, etc.)
- **Model Pipelines**: Streamlined preprocessing, feature selection, and model training pipelines
- **Feature Importance Analysis**: Tools to identify the most predictive technical indicators
- **Dimensionality Reduction**: PCA and other techniques to handle high-dimensional feature spaces
- **Performance Metrics**: Comprehensive model evaluation including accuracy, precision, recall, and F1-score
- **Training Progress Tracking**: Real-time monitoring of model training performance and times
- **Modular Design**: Easily extensible for new indicators, models, or evaluation metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bitcoin-trading.git
cd bitcoin-trading

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- catboost
- shap (for feature importance analysis)

## Project Structure

```
bitcoin_trading/
├── data/
│   ├── data_loader.py           # Data loading and preprocessing
│   └── data_loader_examples.py  # Examples for data loading
├── features/
│   ├── technical_indicators.py  # Technical indicator implementations
│   └── feature_importance.py    # Feature importance analysis tools
├── models/
│   ├── model_factory.py         # Model creation and configuration
│   ├── model_evaluator.py       # Model evaluation and comparison
│   └── model_pipeline.py        # ML pipeline construction
├── dim_reduction/
│   └── feature_reducer.py       # Dimensionality reduction techniques
└── main.py                      # Main execution script
```

## Usage

### Training a Model

```python
from bitcoin_trading.data.data_loader import BitcoinDataLoader
from bitcoin_trading.features.technical_indicators import TechnicalIndicators
from bitcoin_trading.models.model_factory import ModelFactory
from bitcoin_trading.models.model_evaluator import ModelEvaluator

# Load data
loader = BitcoinDataLoader('btcusd_1-min_data.csv')
df = loader.load_data()

# Split into train/test sets
df_train, df_test = loader.split_timeseries(df, test_size=0.2)

# Generate technical indicators
indicators = TechnicalIndicators()
df_train = indicators.add_all_indicators(df_train)
df_test = indicators.add_all_indicators(df_test)

# Create target signal
df_train = indicators.create_target_signal(df_train)
df_test = indicators.create_target_signal(df_test)

# Handle missing values
df_train = df_train.dropna()
df_test = df_test.dropna()

# Create and evaluate models
models = ModelFactory.create_models()
evaluator = ModelEvaluator(models, n_fold=5, scoring='accuracy')
results, train_times, _ = evaluator.evaluate(df_train)

# Print results
print(results)
```

### Running a Full Training Pipeline

```bash
python -m bitcoin_trading.train-2.py
```

This script will:
1. Load the Bitcoin price data
2. Process and clean the data
3. Generate technical indicators
4. Create target signals
5. Train multiple models with different preprocessing strategies
6. Evaluate and compare model performance
7. Save the best model for future use

### Analyzing Feature Importance

```python
from bitcoin_trading.features.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(df_train, target_col='signal')
importance_df = analyzer.analyze()
analyzer.plot_importance(importance_df)
```

## Data

The system is designed to work with minute-by-minute Bitcoin price data. The expected format is a CSV file with at least the following columns:

- Timestamp (index)
- Open
- High
- Low
- Close
- Volume

Sample data format:
```
Timestamp,Open,High,Low,Close,Volume
2025-03-01 00:01:00,45000.12,45010.25,44998.77,45005.88,2.35
2025-03-01 00:02:00,45005.88,45020.47,45005.88,45015.72,1.89
...
```

You can obtain cryptocurrency data from various sources:
- [Binance API](https://binance-docs.github.io/apidocs/)
- [Coinbase API](https://developers.coinbase.com/)
- [CryptoCompare](https://www.cryptocompare.com/api/)

## Model Architecture

The system implements several machine learning models for comparison:

- **Random Forest**: Ensemble method using multiple decision trees
- **XGBoost**: Gradient boosting algorithm known for its performance and speed
- **Logistic Regression**: Linear model for binary classification
- **Support Vector Machine (SVM)**: For finding optimal decision boundaries
- **Neural Networks**: Simple feed-forward networks for non-linear patterns

Each model can be combined with various preprocessing strategies:
- Standard scaling
- Min-max normalization
- Robust scaling
- Feature selection
- Feature reduction through PCA

## Model Pipeline

The system implements a flexible machine learning pipeline architecture that supports various preprocessing strategies and model combinations:

```python
from bitcoin_trading.models.model_pipeline import ModelPipeline
from sklearn.ensemble import RandomForestClassifier

# Create a pipeline with standard scaling preprocessing
pipeline = ModelPipeline.create_pipeline(
    RandomForestClassifier(n_estimators=100),
    preprocess_strategy='standard'  # Options: 'none', 'standard', 'minmax', 'robust', 'impute'
)
```

### Preprocessing Options

- **None**: No preprocessing (raw features)
- **Standard**: StandardScaler (zero mean, unit variance)
- **MinMax**: MinMaxScaler (0-1 range)
- **Robust**: RobustScaler (resistant to outliers)
- **Impute**: SimpleImputer + StandardScaler (handles missing values)

### Feature Selection

The pipeline can incorporate feature selection to reduce dimensionality:

```python
# Create a pipeline with feature selection (keep top 20 features)
pipeline = ModelPipeline.create_pipeline(
    model,
    preprocess_strategy='standard',
    feature_select='kbest',
    n_features=20
)
```

### Dimensionality Reduction

PCA can be applied to reduce feature dimensions while preserving variance:

```python
# Create a pipeline with PCA
pipeline = ModelPipeline.create_pipeline(
    model,
    preprocess_strategy='standard',
    feature_select='pca',
    n_components=10
)
```

This modular pipeline approach allows for systematic experimentation with different preprocessing strategies and model combinations to find the optimal configuration for Bitcoin trading signals.

## Results

Performance comparison between baseline models and models trained with all technical indicators:

### Baseline Models Performance

| Model | CV Average | Train | Test | All |
|-------|------------|-------|------|-----|
| LDA   | 0.5158     | 0.5248| 0.5170| 0.5233 |
| KNN   | 0.5143     | 0.7126| 0.5055| 0.7091 |
| TREE  | 0.5116     | 0.9985| 0.5060| 0.9986 |
| NB    | 0.4968     | 0.5119| 0.5689| 0.5225 |
| GBM   | 0.5138     | 0.5704| 0.4723| 0.5564 |
| XGB   | 0.5241     | 0.6125| 0.4861| 0.6023 |
| CAT   | 0.5223     | 0.5758| 0.4703| 0.5654 |
| RF    | 0.5149     | 0.9966| 0.5087| 0.9968 |

### Models with All Technical Indicators

| Model | CV Average | Train | Test | All |
|-------|------------|-------|------|-----|
| LDA   | 0.9194     | 0.9203| 0.9109| 0.9200 |
| KNN   | 0.8184     | 0.9760| 0.8552| 0.9756 |
| TREE  | 0.8671     | 1.0000| 0.8907| 1.0000 |
| NB    | 0.7898     | 0.7906| 0.7884| 0.7917 |
| GBM   | 0.9143     | 0.9177| 0.9102| 0.9159 |
| XGB   | 0.9293     | 0.9470| 0.9253| 0.9437 |
| CAT   | 0.9253     | 0.9363| 0.9215| 0.9335 |
| RF    | 0.9149     | 0.9996| 0.9182| 0.9996 |

The results clearly demonstrate the significant improvement in model performance when using the full set of technical indicators. XGBoost (XGB) and CatBoost (CAT) achieve the best test accuracies at 92.5% and 92.1% respectively, compared to baseline models that perform only slightly better than random (50-57%).

**Key observations:**
- Adding technical indicators improves accuracy by approximately 40 percentage points across all models
- Tree-based models (TREE, RF) tend to overfit on the training data when using just baseline features
- XGBoost achieves the best overall performance with technical indicators
- Simple models like LDA perform surprisingly well with proper feature engineering

## Real-time Trading Capabilities

The system supports real-time trading with the following components:

### Cassandra Integration

The project includes a `CassandraBitcoinDataLoader` that inherits from the standard `BitcoinDataLoader`, allowing seamless transition from historical backtesting to real-time data:

```python
from bitcoin_trading.data.cassandra_data_loader import CassandraBitcoinDataLoader

# Initialize the loader
loader = CassandraBitcoinDataLoader(
    cassandra_hosts=['127.0.0.1'],
    keyspace='bitcoin_trading',
    table='price_data'
)

# Load recent data
df = loader.load_data(start_time=start_time, end_time=end_time)
```

### WebSocket Data Collection

Real-time market data is collected via WebSockets and stored in Cassandra:

```python
from bitcoin_trading.data.real_time_handler import RealTimeHandler

# Initialize the handler
handler = RealTimeHandler(
    cassandra_hosts=['127.0.0.1'],
    keyspace='bitcoin_trading',
    websocket_url='wss://stream.binance.com:9443/ws/btcusdt@kline_1m'
)

# Start collecting data
handler.start()
```

### Live Prediction Pipeline

The system supports making predictions on streaming data:

1. Real-time data is collected from exchange WebSockets
2. Data is processed and stored in Cassandra
3. Technical indicators are calculated on the latest data
4. Trained models make predictions using the real-time features
5. Trading signals are generated based on predictions

This pipeline enables the system to respond to market changes in near real-time, with typical end-to-end latency under 1 second from data receipt to signal generation.

## Configuration

Model parameters can be configured in `models/model_factory.py`. Key parameters include:

- Random Forest: `n_estimators`, `max_depth`, `max_features`
- XGBoost: `n_estimators`, `learning_rate`, `max_depth`
- Neural Network: `hidden_layer_sizes`, `activation`, `solver`

Feature engineering parameters can be adjusted in `features/technical_indicators.py`.

## Logging and Monitoring

Training progress is logged to the console with details on:
- Data preparation steps
- Feature generation
- Model training progress for each algorithm
- Training times
- Evaluation metrics

For more advanced monitoring, the system can be extended to support tools like:
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [MLflow](https://mlflow.org/)
- [Weight & Biases](https://wandb.ai/)

## Contributing

Contributions are welcome! Here are some ways to contribute:

1. Add new technical indicators
2. Implement additional machine learning models
3. Improve performance optimization
4. Add backtesting capabilities
5. Enhance documentation

Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk and is not suitable for every investor. Do not trade with money you cannot afford to lose. 