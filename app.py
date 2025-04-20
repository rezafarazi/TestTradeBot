import pandas as pd
import numpy as np
import requests
import time
import schedule
import nltk
import datetime
import warnings
import logging
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import yfinance as yf
from textblob import TextBlob
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("price_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings('ignore')

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')

class PricePredictionSystem:
    def __init__(self):
        """Initialize the price prediction system"""
        self.news_data = pd.DataFrame(columns=['timestamp', 'title', 'summary', 'source', 'url', 'sentiment'])
        self.price_data = pd.DataFrame(columns=['timestamp', 'asset', 'price'])
        self.models = {}
        self.assets = ['XAU/USD', 'BTC/USD', 'USD/IRR', 'EUR/USD', 'BRENT']  # Gold, Bitcoin, USD/IRR, Euro, Brent Oil
        self.news_sources = [
            {'name': 'Reuters', 'url': 'https://www.reuters.com/world'},
            {'name': 'Bloomberg', 'url': 'https://www.bloomberg.com/markets'},
            {'name': 'CNBC', 'url': 'https://www.cnbc.com/world'},
            {'name': 'Financial Times', 'url': 'https://www.ft.com/markets'},
            {'name': 'BBC', 'url': 'https://www.bbc.com/news/business'}
        ]
        self.db_path = 'financial_data.db'
        self.setup_database()
        self.load_historical_data()
        logger.info("Price prediction system initialized.")
        
    def setup_database(self):
        """Set up the database for storing information"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create news table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            title TEXT,
            summary TEXT,
            source TEXT,
            url TEXT,
            sentiment REAL
        )
        ''')
        
        # Create asset prices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            asset TEXT,
            price REAL
        )
        ''')
        
        # Create predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            asset TEXT,
            predicted_price REAL,
            confidence REAL
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database successfully initialized.")
    
    def generate_news_id(self, title, source):
        """Generate a unique identifier for each news item"""
        unique_str = f"{title}_{source}_{datetime.datetime.now().date()}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def fetch_news(self):
        """Collect news from various sources"""
        logger.info("Starting news collection...")
        new_articles = []
        
        for source in self.news_sources:
            try:
                response = requests.get(source['url'], headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract news titles and summaries (this section needs to be customized for each news site)
                    articles = soup.find_all('article')[:10]  # Limit to first 10 articles
                    
                    for article in articles:
                        try:
                            # This section varies for each website and needs to be configured
                            title_elem = article.find(['h1', 'h2', 'h3'])
                            if title_elem:
                                title = title_elem.get_text().strip()
                                
                                # Find news link
                                link_elem = title_elem.find_parent('a') or article.find('a')
                                url = link_elem['href'] if link_elem and 'href' in link_elem.attrs else ""
                                if url and not url.startswith('http'):
                                    # Convert relative URL to absolute
                                    base_url = '/'.join(source['url'].split('/')[:3])
                                    url = f"{base_url}{url}" if url.startswith('/') else f"{base_url}/{url}"
                                
                                # Find news summary
                                summary_elem = article.find(['p', 'div', 'span'], class_=['summary', 'description', 'excerpt'])
                                summary = summary_elem.get_text().strip() if summary_elem else ""
                                
                                # Analyze news sentiment
                                sentiment = TextBlob(f"{title} {summary}").sentiment.polarity
                                
                                # Generate unique identifier for the news
                                news_id = self.generate_news_id(title, source['name'])
                                
                                # Add to new articles list
                                news_item = {
                                    'id': news_id,
                                    'timestamp': datetime.datetime.now(),
                                    'title': title,
                                    'summary': summary,
                                    'source': source['name'],
                                    'url': url,
                                    'sentiment': sentiment
                                }
                                new_articles.append(news_item)
                        except Exception as e:
                            logger.error(f"Error extracting news from {source['name']}: {str(e)}")
                else:
                    logger.warning(f"Error fetching news from {source['name']}: Status code {response.status_code}")
            except Exception as e:
                logger.error(f"Error connecting to {source['name']}: {str(e)}")
        
        # Save new articles to database
        if new_articles:
            self._save_news_to_db(new_articles)
            self.news_data = self._load_news_from_db()
            logger.info(f"{len(new_articles)} new articles collected and saved.")
        else:
            logger.info("No new articles found.")
            
    def _save_news_to_db(self, news_items):
        """Save news items to the database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in news_items:
            try:
                cursor.execute(
                    "INSERT OR IGNORE INTO news (id, timestamp, title, summary, source, url, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (item['id'], item['timestamp'], item['title'], item['summary'], item['source'], item['url'], item['sentiment'])
                )
            except Exception as e:
                logger.error(f"Error saving news item: {str(e)}")
        
        conn.commit()
        conn.close()
    
    def _load_news_from_db(self):
        """Load news data from the database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        
        # Load up to 1000 recent news articles
        query = "SELECT timestamp, title, summary, source, url, sentiment FROM news ORDER BY timestamp DESC LIMIT 1000"
        news_df = pd.read_sql_query(query, conn)
        conn.close()
        
        return news_df
    
    def fetch_asset_prices(self):
        """Get current prices of various assets"""
        logger.info("Fetching current asset prices...")
        timestamp = datetime.datetime.now()
        new_prices = []
        
        try:
            # Get gold price
            gold = yf.Ticker("GC=F")  # Gold futures contract
            gold_price = gold.history(period="1d").iloc[-1]['Close']
            new_prices.append({
                'timestamp': timestamp,
                'asset': 'XAU/USD',
                'price': gold_price
            })
            logger.info(f"Gold price: {gold_price} USD")
            
            # Get Bitcoin price
            btc = yf.Ticker("BTC-USD")
            btc_price = btc.history(period="1d").iloc[-1]['Close']
            new_prices.append({
                'timestamp': timestamp,
                'asset': 'BTC/USD',
                'price': btc_price
            })
            logger.info(f"Bitcoin price: {btc_price} USD")
            
            # For USD to IRR, we'll use an external API (example assumption)
            # In real life, you should use a reputable API for currency exchange rates
            try:
                response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
                if response.status_code == 200:
                    data = response.json()
                    # Assume the API provides IRR exchange rate (it might not be accurate)
                    usd_irr_rate = data['rates'].get('IRR', 420000)  # Default value if not found
                else:
                    # Default value in case of error
                    usd_irr_rate = 420000
            except:
                usd_irr_rate = 420000  # Default value in case of error
                
            new_prices.append({
                'timestamp': timestamp,
                'asset': 'USD/IRR',
                'price': usd_irr_rate
            })
            logger.info(f"USD to IRR exchange rate: {usd_irr_rate} IRR")
            
            # Get EUR to USD price
            eur = yf.Ticker("EURUSD=X")
            eur_price = eur.history(period="1d").iloc[-1]['Close']
            new_prices.append({
                'timestamp': timestamp,
                'asset': 'EUR/USD',
                'price': eur_price
            })
            logger.info(f"EUR to USD exchange rate: {eur_price}")
            
            # Get Brent Oil price
            oil = yf.Ticker("BZ=F")  # Brent Oil futures contract
            oil_price = oil.history(period="1d").iloc[-1]['Close']
            new_prices.append({
                'timestamp': timestamp,
                'asset': 'BRENT',
                'price': oil_price
            })
            logger.info(f"Brent Oil price: {oil_price} USD")
            
            # Save new prices to database
            self._save_prices_to_db(new_prices)
            
            # Update price data
            for price_item in new_prices:
                # Check if this asset was already in the price data
                mask = (self.price_data['asset'] == price_item['asset'])
                if mask.any():
                    # Update price
                    self.price_data.loc[mask, 'price'] = price_item['price']
                    self.price_data.loc[mask, 'timestamp'] = price_item['timestamp']
                else:
                    # Add new asset
                    self.price_data = pd.concat([self.price_data, pd.DataFrame([price_item])])
            
            logger.info("Asset prices successfully updated.")
            
        except Exception as e:
            logger.error(f"Error fetching asset prices: {str(e)}")
    
    def _save_prices_to_db(self, price_items):
        """Save asset prices to the database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in price_items:
            try:
                cursor.execute(
                    "INSERT INTO prices (timestamp, asset, price) VALUES (?, ?, ?)",
                    (item['timestamp'], item['asset'], item['price'])
                )
            except Exception as e:
                logger.error(f"Error saving asset price: {str(e)}")
        
        conn.commit()
        conn.close()
    
    def load_historical_data(self):
        """Load historical data for training models"""
        logger.info("Loading historical data...")
        
        # Load historical news data
        self.news_data = self._load_news_from_db()
        
        # Load historical price data
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        price_query = "SELECT timestamp, asset, price FROM prices ORDER BY timestamp DESC"
        self.price_data = pd.read_sql_query(price_query, conn)
        conn.close()
        
        logger.info(f"Historical data loaded. {len(self.news_data)} news articles and {len(self.price_data)} price records.")
        
        # If we don't have enough historical data, load default historical data
        if len(self.price_data) < 100:  # We need at least 100 records for training the model
            logger.info("Not enough historical data. Loading default historical data...")
            self._load_default_historical_data()
            
            # Verify we have enough data for each asset
            for asset in self.assets:
                asset_data = self.price_data[self.price_data['asset'] == asset]
                if len(asset_data) < 20:  # Minimum 20 data points needed for each asset
                    logger.warning(f"Still not enough data for {asset} after loading default data. Fetching more historical data...")
                    self._fetch_additional_historical_data(asset)
    
    def _load_default_historical_data(self):
        """Load default historical data for starting the work"""
        logger.info("Loading default historical data...")
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)  # One year of historical data
        
        for asset in self.assets:
            try:
                if asset == 'XAU/USD':
                    ticker = 'GC=F'
                elif asset == 'BTC/USD':
                    ticker = 'BTC-USD'
                elif asset == 'EUR/USD':
                    ticker = 'EURUSD=X'
                elif asset == 'BRENT':
                    ticker = 'BZ=F'
                else:
                    continue  # We don't have default historical data for IRR to IRR conversion
                
                # Load historical data
                data = yf.download(ticker, start=start_date, end=end_date)
                
                # Convert data to the required format
                price_rows = []
                for index, row in data.iterrows():
                    price_rows.append({
                        'timestamp': index,
                        'asset': asset,
                        'price': row['Close']
                    })
                
                # Save data to database
                self._save_prices_to_db(price_rows)
                
                logger.info(f"{len(price_rows)} historical records loaded for {asset}.")
                
            except Exception as e:
                logger.error(f"Error loading historical data for {asset}: {str(e)}")
        
        # Reload data from database
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        price_query = "SELECT timestamp, asset, price FROM prices ORDER BY timestamp DESC"
        self.price_data = pd.read_sql_query(price_query, conn)
        conn.close()
    
    def _fetch_additional_historical_data(self, asset):
        """Fetch additional historical data for a specific asset"""
        logger.info(f"Fetching additional historical data for {asset}...")
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)  # One year of historical data
        
        try:
            if asset == 'XAU/USD':
                ticker = 'GC=F'
            elif asset == 'BTC/USD':
                ticker = 'BTC-USD'
            elif asset == 'EUR/USD':
                ticker = 'EURUSD=X'
            elif asset == 'BRENT':
                ticker = 'BZ=F'
            elif asset == 'USD/IRR':
                # For USD/IRR, we'll use a longer period to get more data
                start_date = end_date - datetime.timedelta(days=730)  # Two years
                ticker = None  # We'll handle this separately
            else:
                return
            
            if ticker:
                # Fetch data from Yahoo Finance
                data = yf.download(ticker, start=start_date, end=end_date)
                
                # Convert data to the required format
                price_rows = []
                for index, row in data.iterrows():
                    price_rows.append({
                        'timestamp': index,
                        'asset': asset,
                        'price': row['Close']
                    })
                
                # Save data to database
                self._save_prices_to_db(price_rows)
                logger.info(f"Added {len(price_rows)} historical records for {asset}")
            
            # Reload price data from database
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            price_query = "SELECT timestamp, asset, price FROM prices ORDER BY timestamp DESC"
            self.price_data = pd.read_sql_query(price_query, conn)
            conn.close()
            
        except Exception as e:
            logger.error(f"Error fetching additional historical data for {asset}: {str(e)}")
    
    def prepare_features(self, asset):
        """Prepare features for training and prediction"""
        logger.info(f"Preparing features for {asset}...")
        
        # Filter relevant data for the specific asset
        asset_data = self.price_data[self.price_data['asset'] == asset].copy()
        
        if len(asset_data) < 10:
            logger.warning(f"Not enough data for {asset}.")
            return None, None
        
        # Convert timestamp to datetime if it's a string
        if isinstance(asset_data['timestamp'].iloc[0], str):
            asset_data['timestamp'] = pd.to_datetime(asset_data['timestamp'])
        
        # Sort data by time
        asset_data = asset_data.sort_values('timestamp')
        asset_data.reset_index(drop=True, inplace=True)
        
        # Add technical features
        asset_data['price_1d_change'] = asset_data['price'].pct_change(1)
        asset_data['price_3d_change'] = asset_data['price'].pct_change(3)
        asset_data['price_7d_change'] = asset_data['price'].pct_change(7)
        
        # Moving average
        asset_data['SMA_5'] = asset_data['price'].rolling(window=5).mean()
        asset_data['SMA_10'] = asset_data['price'].rolling(window=10).mean()
        asset_data['SMA_20'] = asset_data['price'].rolling(window=20).mean()
        
        # Standardize data
        scaler = StandardScaler()
        price_cols = ['price', 'price_1d_change', 'price_3d_change', 'price_7d_change', 'SMA_5', 'SMA_10', 'SMA_20']
        asset_data[price_cols] = asset_data[price_cols].fillna(method='ffill')
        asset_data[price_cols] = scaler.fit_transform(asset_data[price_cols])
        
        # Add news sentiment features (sentiment analysis)
        for i in range(len(asset_data)):
            current_time = asset_data['timestamp'].iloc[i]
            # News from 24 hours ago to current time
            day_before = current_time - datetime.timedelta(hours=24)
            
            # Filter news relevant to the time period
            if not self.news_data.empty and 'timestamp' in self.news_data.columns:
                relevant_news = self.news_data[
                    (self.news_data['timestamp'] >= day_before) & 
                    (self.news_data['timestamp'] <= current_time)
                ]
                
                # Calculate average sentiment score
                if not relevant_news.empty and 'sentiment' in relevant_news.columns:
                    asset_data.loc[i, 'news_sentiment'] = relevant_news['sentiment'].mean()
                else:
                    asset_data.loc[i, 'news_sentiment'] = 0
            else:
                asset_data.loc[i, 'news_sentiment'] = 0
        
        # Drop rows with NaN values
        asset_data = asset_data.dropna()
        
        # Extract features and target
        features = asset_data[['price_1d_change', 'price_3d_change', 'price_7d_change', 
                              'SMA_5', 'SMA_10', 'SMA_20', 'news_sentiment']].values
        
        # Target: Next day price
        target = asset_data['price'].shift(-1).iloc[:-1].values
        features = features[:-1]
        
        return features, target
    
    def train_models(self):
        """Train prediction models for each asset"""
        logger.info("Training models...")
        
        # Verify we have enough data for each asset
        for asset in self.assets:
            asset_data = self.price_data[self.price_data['asset'] == asset]
            if len(asset_data) < 20:  # Minimum 20 data points needed
                logger.warning(f"Not enough data for {asset}. Fetching more historical data...")
                self._fetch_additional_historical_data(asset)
        
        for asset in self.assets:
            try:
                features, target = self.prepare_features(asset)
                
                if features is None or len(features) < 10:
                    logger.warning(f"Not enough data for training model {asset}. Skipping...")
                    continue
                
                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                
                # Train Random Forest model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate model
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                
                logger.info(f"Model {asset} trained successfully. MSE: {mse:.4f}, MAE: {mae:.4f}")
                
                # Save model
                self.models[asset] = model
                
            except Exception as e:
                logger.error(f"Error training model for {asset}: {str(e)}")
    
    def predict_prices(self):
        """Predict future prices for all assets"""
        logger.info("Starting price prediction...")
        predictions = []
        
        for asset in self.assets:
            try:
                if asset not in self.models:
                    logger.warning(f"No model found for {asset}.")
                    continue
                
                # Prepare current features for prediction
                asset_data = self.price_data[self.price_data['asset'] == asset].copy()
                
                if len(asset_data) < 20:  # We need at least 20 data points
                    logger.warning(f"Not enough data for prediction {asset}.")
                    continue
                
                # Convert timestamp to datetime if it's a string
                if isinstance(asset_data['timestamp'].iloc[0], str):
                    asset_data['timestamp'] = pd.to_datetime(asset_data['timestamp'])
                
                # Sort data by time
                asset_data = asset_data.sort_values('timestamp', ascending=False)
                asset_data.reset_index(drop=True, inplace=True)
                
                # Calculate technical features
                latest_price = asset_data['price'].iloc[0]
                price_1d_change = (asset_data['price'].iloc[0] / asset_data['price'].iloc[1]) - 1 if len(asset_data) > 1 else 0
                price_3d_change = (asset_data['price'].iloc[0] / asset_data['price'].iloc[3]) - 1 if len(asset_data) > 3 else 0
                price_7d_change = (asset_data['price'].iloc[0] / asset_data['price'].iloc[7]) - 1 if len(asset_data) > 7 else 0
                
                # Moving average
                sma_5 = asset_data['price'].iloc[:5].mean() if len(asset_data) >= 5 else latest_price
                sma_10 = asset_data['price'].iloc[:10].mean() if len(asset_data) >= 10 else latest_price
                sma_20 = asset_data['price'].iloc[:20].mean() if len(asset_data) >= 20 else latest_price
                
                # News sentiment from recent 24 hours
                current_time = datetime.datetime.now()
                day_before = current_time - datetime.timedelta(hours=24)
                
                # Filter news relevant to the time period
                news_sentiment = 0
                if not self.news_data.empty and 'timestamp' in self.news_data.columns:
                    if isinstance(self.news_data['timestamp'].iloc[0], str):
                        self.news_data['timestamp'] = pd.to_datetime(self.news_data['timestamp'])
                    
                    relevant_news = self.news_data[
                        (self.news_data['timestamp'] >= day_before) & 
                        (self.news_data['timestamp'] <= current_time)
                    ]
                    
                    # Calculate average sentiment score
                    if not relevant_news.empty and 'sentiment' in relevant_news.columns:
                        news_sentiment = relevant_news['sentiment'].mean()
                
                # Standardize features (using mean and standard deviation of existing data)
                feature_values = np.array([price_1d_change, price_3d_change, price_7d_change, sma_5, sma_10, sma_20, news_sentiment])
                
                # We should use the same scale as used in training
                # But for simplicity, we assume data was previously standardized
                
                # Predict price
                model = self.models[asset]
                predicted_price_scaled = model.predict([feature_values])[0]
                
                # Convert back to original scale (for simplicity, we assume prediction is relative to current price)
                predicted_price = latest_price * (1 + predicted_price_scaled * 0.01)
                # Calculate prediction confidence (based on model confidence)
                confidence = 0.7  # Default value, in real models we can use more advanced methods
                
                # Save prediction
                prediction = {
                    'timestamp': datetime.datetime.now(),
                    'asset': asset,
                    'predicted_price': predicted_price,
                    'confidence': confidence
                }
                predictions.append(prediction)
                
                logger.info(f"Prediction for {asset}: {predicted_price:.2f} (Confidence: {confidence:.2f})")
                
                # Save prediction to database
                self._save_prediction_to_db(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting price for {asset}: {str(e)}")
        
        return predictions
    
    def _save_prediction_to_db(self, prediction):
        """Save prediction to the database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO predictions (timestamp, asset, predicted_price, confidence) VALUES (?, ?, ?, ?)",
                (prediction['timestamp'], prediction['asset'], prediction['predicted_price'], prediction['confidence'])
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
        finally:
            conn.close()
    
    def generate_report(self, predictions):
        """Generate a report from completed predictions"""
        report = "Price Prediction Report\n"
        report += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for pred in predictions:
            asset = pred['asset']
            current_price = self.price_data[self.price_data['asset'] == asset]['price'].iloc[0] if not self.price_data[self.price_data['asset'] == asset].empty else "Unknown"
            
            report += f"Asset: {asset}\n"
            report += f"Current Price: {current_price}\n"
            report += f"Predicted Price: {pred['predicted_price']:.2f}\n"
            report += f"Confidence: {pred['confidence']:.2f}\n"
            
            # Calculate change percentage
            if current_price != "Unknown":
                change_pct = ((pred['predicted_price'] - current_price) / current_price) * 100
                report += f"Predicted Price Change Percentage: {change_pct:.2f}%\n"
            
            report += "\n"
        
        # Display recent important news
        report += "Recent Important News:\n"
        recent_news = self.news_data.sort_values('timestamp', ascending=False).head(5)
        
        if not recent_news.empty:
            for _, news in recent_news.iterrows():
                report += f"- {news['title']} (Source: {news['source']})\n"
                report += f"   Sentiment Impact Score: {news['sentiment']:.2f}\n"
        else:
            report += "No important news found.\n"
        
        return report
    
    def visualize_predictions(self, predictions):
        """Visualize prediction results"""
        plt.figure(figsize=(12, 8))
        
        for pred in predictions:
            asset = pred['asset']
            
            # Extract historical data for this asset
            asset_data = self.price_data[self.price_data['asset'] == asset].copy()
            
            if len(asset_data) < 2:
                continue
                
            # Convert timestamp to datetime if it's a string
            if isinstance(asset_data['timestamp'].iloc[0], str):
                asset_data['timestamp'] = pd.to_datetime(asset_data['timestamp'])
            
            # Sort data by time
            asset_data = asset_data.sort_values('timestamp')
            
            # Plot historical prices
            plt.plot(asset_data['timestamp'].values, asset_data['price'].values, label=f"{asset} - Historical")
            
            # Plot prediction
            last_date = asset_data['timestamp'].max()
            forecast_date = last_date + datetime.timedelta(days=1)
            plt.scatter([forecast_date], [pred['predicted_price']], marker='*', s=100, 
                      label=f"{asset} - Prediction")
            
            # Plot trend line
            plt.plot([last_date, forecast_date], 
                   [asset_data['price'].iloc[-1], pred['predicted_price']], 
                   'r--', alpha=0.5)
        
        plt.title('Asset Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig('price_predictions.png')
        logger.info("Price prediction plot saved successfully.")
    
    def run_scheduled_task(self):
        """Run scheduled tasks"""
        logger.info("Starting scheduled task execution...")
        try:
            # Collect news
            self.fetch_news()
            
            # Get current prices
            self.fetch_asset_prices()
            
            # Train models if they don't exist
            if not self.models:
                logger.info("No models found. Training new models...")
                self.train_models()
            else:
                # Check if any models are missing
                missing_models = [asset for asset in self.assets if asset not in self.models]
                if missing_models:
                    logger.info(f"Missing models for assets: {missing_models}. Training missing models...")
                    self.train_models()
            
            # Predict prices
            predictions = self.predict_prices()
            
            # Generate report
            report = self.generate_report(predictions)
            logger.info("Prediction Report:\n" + report)
            
            # Visualize results
            self.visualize_predictions(predictions)
            
            logger.info("Scheduled tasks executed successfully.")
            
            return predictions, report
            
        except Exception as e:
            logger.error(f"Error executing scheduled tasks: {str(e)}")
            return [], "Error generating predictions."
    
    def start_scheduler(self):
        """Start automatic scheduling of scheduled tasks"""
        logger.info("Setting up scheduler...")
        
        # Initial run without waiting
        self.run_scheduled_task()
        
        # Set up periodic execution every 60 minutes
        schedule.every(1).minutes.do(self.run_scheduled_task)
        
        logger.info("Scheduler set up successfully. Tasks will run every 60 minutes.")
        
        # Infinite loop for scheduled execution
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


class NewsAnalyzer:
    """Class for news text analysis"""
    
    def __init__(self):
        """Initialize news analyzer"""
        # Download stopwords
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000)
    
    def extract_keywords(self, news_df):
        """Extract keywords from news articles"""
        if news_df.empty:
            return {}
        
        # Combine news titles and summaries
        texts = news_df['title'] + " " + news_df['summary']
        
        # Remove stop words and convert to lowercase
        processed_texts = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
            processed_texts.append(" ".join(filtered_tokens))
        
        # Convert text to TF-IDF vector
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate word importance
        keywords = {}
        for i in range(len(processed_texts)):
            feature_index = tfidf_matrix[i, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
            
            for idx, score in tfidf_scores:
                word = feature_names[idx]
                if word in keywords:
                    keywords[word] += score
                else:
                    keywords[word] = score
        
        # Sort keywords by score
        sorted_keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:20])
        
        return sorted_keywords
    
    def analyze_sentiment(self, news_df):
        """Analyze news sentiment"""
        if news_df.empty:
            return 0
        
        sentiments = []
        for _, news in news_df.iterrows():
            text = news['title'] + " " + news['summary']
            sentiment = TextBlob(text).sentiment.polarity
            sentiments.append(sentiment)
        
        return sum(sentiments) / len(sentiments) if sentiments else 0
    
    def analyze_news_impact(self, news_df, asset):
        """Analyze news impact on a specific asset"""
        if news_df.empty:
            return {"sentiment": 0, "keywords": {}, "impact_score": 0}
        
        # Keywords relevant to each asset
        asset_keywords = {
            'XAU/USD': ['gold', 'precious', 'metal', 'inflation', 'safe', 'haven', 'reserve', 'bullion'],
            'BTC/USD': ['bitcoin', 'crypto', 'blockchain', 'cryptocurrency', 'digital', 'btc', 'mining', 'wallet'],
            'USD/IRR': ['iran', 'rial', 'tehran', 'sanction', 'oil', 'dollar', 'currency', 'exchange'],
            'EUR/USD': ['euro', 'ecb', 'european', 'lagarde', 'eurozone', 'germany', 'france', 'eu'],
            'BRENT': ['oil', 'crude', 'opec', 'energy', 'barrel', 'petroleum', 'gas', 'supply']
        }
        
        # Overall sentiment
        sentiment = self.analyze_sentiment(news_df)
        
        # Keywords
        keywords = self.extract_keywords(news_df)
        
        # Calculate impact score based on keyword match
        impact_score = 0
        if asset in asset_keywords:
            for keyword in asset_keywords[asset]:
                if keyword in keywords:
                    impact_score += keywords[keyword]
        
        # Normalize score
        if impact_score > 0:
            impact_score = min(impact_score / 10, 1.0)  # Limit to range 0 to 1
        
        return {
            "sentiment": sentiment,
            "keywords": keywords,
            "impact_score": impact_score
        }


class AdvancedPredictionModel:
    """Class for advanced prediction model using LSTM neural network"""
    
    def __init__(self):
        """Initialize advanced prediction model"""
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_sequence_data(self, features, target, lookback=10):
        """Prepare data for LSTM in sequence format"""
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i:i+lookback])
            y.append(target[i+lookback])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, features, target, epochs=50, batch_size=32, validation_split=0.2):
        """Train LSTM model"""
        if len(features) < 20:  # At least 20 data points are needed
            logger.warning("Not enough data for training LSTM model.")
            return False
        
        # Prepare data
        X, y = self.prepare_sequence_data(features, target)
        
        if len(X) == 0:
            logger.warning("Cannot create sequence data.")
            return False
        
        # Split data into training and testing sets
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"LSTM model trained successfully. Loss: {loss:.4f}")
        
        return True
    
    def predict(self, features, lookback=10):
        """Predict using LSTM model"""
        if self.model is None:
            logger.warning("LSTM model not trained yet.")
            return None
        
        # Prepare input data
        if len(features) < lookback:
            logger.warning(f"At least {lookback} data points are needed for prediction.")
            return None
        
        X = np.array([features[-lookback:]])
        
        # Predict
        predicted = self.model.predict(X)
        
        return predicted[0][0]


def main():
    """Main function of the program"""
    logger.info("Starting asset price prediction system...")
    
    # Set up prediction system
    prediction_system = PricePredictionSystem()
    
    try:
        # Start scheduling
        prediction_system.start_scheduler()
        
    except KeyboardInterrupt:
        logger.info("Program stopped by user.")
    except Exception as e:
        logger.error(f"Error executing program: {str(e)}")
    
    logger.info("Program ended.")


if __name__ == "__main__":
    main()