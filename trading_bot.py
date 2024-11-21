import logging
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import requests
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TOKEN = "7902134422:AAG7XbyGzgo9vk-ic2aprpfk8xtSFi0HNqw"
COINGECKO_API = "https://api.coingecko.com/api/v3"

# Store historical prices for analysis
price_history = {}
HISTORY_LENGTH = 100

class TradingBot:
    def __init__(self):
        self.price_cache = {}
        self.alerts = {}
        self.watchlist = set()
        self.trading_signals = {}
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("📊 Price Check", callback_data='price_menu')],
            [InlineKeyboardButton("🔔 Set Alert", callback_data='alert_menu')],
            [InlineKeyboardButton("📈 Trading Signals", callback_data='signals_menu')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 Welcome to CryptoTrading Bot!\n\n"
            "Available Commands:\n"
            "/price <symbol> - Get price & analysis\n"
            "/alert <symbol> <price> - Set price alert\n"
            "/signals <symbol> - View trading signals\n"
            "/trending - Show trending & moonshot coins\n"
            "/suggest - Get investment suggestions\n"
            "/short_trades - Find short-term opportunities\n"
            "/add <symbol> - Add to watchlist\n"
            "/remove <symbol> - Remove from watchlist\n"
            "/watchlist - Show watchlist",
            reply_markup=reply_markup
        )

    async def price_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Please provide a crypto symbol! Example: /price bitcoin")
            return

        crypto = context.args[0].lower()
        analysis = await self.analyze_crypto(crypto)
        
        if analysis:
            message = (
                f"💰 {crypto.upper()} Analysis:\n\n"
                f"Current Price: ${analysis['price']:,.2f}\n"
                f"24h Change: {analysis['change_24h']:+.2f}%\n"
                f"RSI (14): {analysis['rsi']:.2f}\n"
                f"Signal: {analysis['signal']}\n\n"
                f"Volume 24h: ${analysis['volume']:,.0f}"
            )
            await update.message.reply_text(message)
        else:
            await update.message.reply_text("Error fetching data. Please check the symbol and try again.")

    async def analyze_crypto(self, crypto):
        try:
            # Get current and historical data
            response = requests.get(
                f"{COINGECKO_API}/coins/{crypto}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false"
            )
            data = response.json()
            
            price = data['market_data']['current_price']['usd']
            change_24h = data['market_data']['price_change_percentage_24h']
            change_7d = data['market_data']['price_change_percentage_7d']
            volume = data['market_data']['total_volume']['usd']
            market_cap = data['market_data']['market_cap']['usd']
            
            if crypto not in price_history:
                price_history[crypto] = deque(maxlen=HISTORY_LENGTH)
            price_history[crypto].append(price)
            
            rsi = self.calculate_rsi(list(price_history[crypto])) if len(price_history[crypto]) > 14 else 50
            
            # Calculate volatility
            volatility = abs(change_24h)
            
            # Market sentiment
            sentiment = self.calculate_market_sentiment(change_24h, change_7d, volume, market_cap)
            
            # Generate signals
            signal = self.generate_advanced_signal(rsi, change_24h, volatility, sentiment)
            timeframe = self.recommend_timeframe(volatility, sentiment)
            
            return {
                'price': price,
                'change_24h': change_24h,
                'change_7d': change_7d,
                'volume': volume,
                'market_cap': market_cap,
                'rsi': rsi,
                'volatility': volatility,
                'sentiment': sentiment,
                'signal': signal,
                'timeframe': timeframe
            }
        except Exception as e:
            logger.error(f"Error analyzing crypto: {str(e)}")
            return None

    def calculate_rsi(self, prices, periods=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[:periods])
        avg_loss = np.mean(losses[:periods])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_market_sentiment(self, change_24h, change_7d, volume, market_cap):
        # Simple sentiment score based on multiple factors
        sentiment_score = 0
        
        # Price momentum
        sentiment_score += change_24h * 0.4  # 40% weight to 24h change
        sentiment_score += change_7d * 0.2   # 20% weight to 7d change
        
        # Volume factor (high volume = more conviction)
        volume_factor = 0.2 if volume > market_cap * 0.1 else 0.1
        sentiment_score += volume_factor
        
        # Market cap factor (larger market cap = more stability)
        if market_cap > 100e9:  # >100B
            sentiment_score += 0.2
        elif market_cap > 10e9:  # >10B
            sentiment_score += 0.1
        
        return sentiment_score

    def generate_advanced_signal(self, rsi, change_24h, volatility, sentiment):
        # Combine multiple indicators for a more comprehensive signal
        signal_strength = 0
        
        # RSI factors
        if rsi < 30:
            signal_strength += 2  # Strong buy signal
        elif rsi < 40:
            signal_strength += 1  # Moderate buy signal
        elif rsi > 70:
            signal_strength -= 1  # Moderate sell signal
        elif rsi > 80:
            signal_strength -= 2  # Strong sell signal
        
        # Trend factors
        if change_24h < -5:
            signal_strength += 1  # Potential oversold
        elif change_24h > 5:
            signal_strength -= 1  # Potential overbought
        
        # Sentiment impact
        signal_strength += sentiment
        
        # Generate final signal
        if signal_strength >= 2:
            return "🟢 Strong Buy - High Confidence"
        elif signal_strength >= 1:
            return "🟡 Consider Buy - Moderate Confidence"
        elif signal_strength <= -2:
            return "⛔️ Strong Sell - High Confidence"
        elif signal_strength <= -1:
            return "🔴 Consider Sell - Moderate Confidence"
        return "⚪️ Hold/Neutral - Wait for Better Setup"

    def recommend_timeframe(self, volatility, sentiment):
        # Recommend trading timeframe based on volatility and sentiment
        if volatility > 10:  # High volatility
            if abs(sentiment) > 1:
                return "Scalping (5-15 minutes)"
            return "Short-term (1-4 hours)"
        elif volatility > 5:  # Medium volatility
            if abs(sentiment) > 0.5:
                return "Intraday (1-4 hours)"
            return "Swing Trade (1-3 days)"
        else:  # Low volatility
            if abs(sentiment) > 0.5:
                return "Swing Trade (1-3 days)"
            return "Position Trade (1-2 weeks)"

    async def set_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) != 2:
            await update.message.reply_text("Usage: /alert <symbol> <target_price>")
            return

        crypto = context.args[0].lower()
        try:
            target_price = float(context.args[1])
            self.alerts[crypto] = {
                'price': target_price,
                'user_id': update.effective_user.id
            }
            await update.message.reply_text(f"✅ Alert set for {crypto.upper()} at ${target_price:,.2f}")
        except ValueError:
            await update.message.reply_text("❌ Please enter a valid price!")

    async def check_alerts(self, context: ContextTypes.DEFAULT_TYPE):
        for crypto, alert in list(self.alerts.items()):
            try:
                analysis = await self.analyze_crypto(crypto)
                if analysis:
                    current_price = analysis['price']
                    if current_price >= alert['price']:
                        await context.bot.send_message(
                            alert['user_id'],
                            f"🚨 Alert triggered!\n{crypto.upper()} has reached ${current_price:,.2f}"
                        )
                        del self.alerts[crypto]
            except Exception as e:
                logger.error(f"Error checking alert: {str(e)}")

    async def add_to_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /add <symbol>")
            return
            
        crypto = context.args[0].lower()
        self.watchlist.add(crypto)
        await update.message.reply_text(f"✅ Added {crypto.upper()} to watchlist")

    async def remove_from_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /remove <symbol>")
            return
            
        crypto = context.args[0].lower()
        if crypto in self.watchlist:
            self.watchlist.remove(crypto)
            await update.message.reply_text(f"✅ Removed {crypto.upper()} from watchlist")
        else:
            await update.message.reply_text(f"❌ {crypto.upper()} not found in watchlist")

    async def show_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.watchlist:
            await update.message.reply_text("Watchlist is empty!")
            return
            
        message = "📋 Your Watchlist:\n\n"
        for crypto in self.watchlist:
            analysis = await self.analyze_crypto(crypto)
            if analysis:
                message += (
                    f"{crypto.upper()}:\n"
                    f"Price: ${analysis['price']:,.2f} ({analysis['change_24h']:+.2f}%)\n"
                    f"Signal: {analysis['signal']}\n\n"
                )
        
        await update.message.reply_text(message)

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /signals command with advanced analysis"""
        if not context.args:
            await update.message.reply_text("Please provide a crypto symbol! Example: /signals bitcoin")
            return

        crypto = context.args[0].lower()
        analysis = await self.analyze_crypto(crypto)
        
        if analysis:
            rsi = analysis['rsi']
            change_24h = analysis['change_24h']
            change_7d = analysis['change_7d']
            volatility = analysis['volatility']
            sentiment = analysis['sentiment']
            
            message = (
                f"🤖 AI Trading Analysis for {crypto.upper()}\n\n"
                f"💰 Price: ${analysis['price']:,.2f}\n"
                f"📊 Performance:\n"
                f"• 24h: {change_24h:+.2f}%\n"
                f"• 7d: {change_7d:+.2f}%\n\n"
                f"📈 Technical Indicators:\n"
                f"• RSI (14): {rsi:.2f}\n"
                f"• Volatility: {volatility:.2f}%\n"
                f"• Market Sentiment: {'Bullish 🟢' if sentiment > 0 else 'Bearish 🔴' if sentiment < 0 else 'Neutral ⚪️'}\n\n"
                f"🎯 Trading Signal:\n"
                f"{analysis['signal']}\n\n"
                f"⏱ Recommended Timeframe:\n"
                f"{analysis['timeframe']}\n\n"
                f"💡 Strategy Recommendation:\n"
                f"{self.get_trading_recommendation(rsi, change_24h, volatility, sentiment)}"
            )
            await update.message.reply_text(message)
        else:
            await update.message.reply_text("Error fetching data. Please check the symbol and try again.")

    def get_trading_recommendation(self, rsi, change_24h, volatility, sentiment):
        """Generate detailed AI-like trading recommendation"""
        if rsi < 30 and sentiment > 0:
            entry_type = "Aggressive" if volatility > 5 else "Conservative"
            return (
                f"📗 {entry_type} Buy Opportunity\n"
                f"• Entry: Set limit orders below market price\n"
                f"• Stop Loss: {abs(change_24h/2):.1f}% below entry\n"
                f"• Take Profit: {abs(change_24h*1.5):.1f}% above entry\n"
                f"• Risk Management: Use max 2-3% of portfolio"
            )
        elif rsi > 70 and sentiment < 0:
            exit_type = "Urgent" if volatility > 5 else "Gradual"
            return (
                f"📕 {exit_type} Sell Signal\n"
                f"• Action: Take profits or set trailing stops\n"
                f"• Exit Strategy: {exit_type.lower()} position reduction\n"
                f"• Re-entry: Wait for RSI to cool down\n"
                f"• Risk Management: Lock in profits"
            )
        else:
            action = "Accumulate" if sentiment > 0 else "Hold" if sentiment == 0 else "Reduce"
            return (
                f"📘 {action} Strategy\n"
                f"• Action: {action} positions gradually\n"
                f"• Entry: Use dollar-cost averaging\n"
                f"• Risk: Keep position sizes small\n"
                f"• Strategy: Wait for stronger signals"
            )

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button presses"""
        query = update.callback_query
        await query.answer()  # Answer the callback query

        if query.data == 'price_menu':
            keyboard = [
                [InlineKeyboardButton("Bitcoin", callback_data='price_btc')],
                [InlineKeyboardButton("Ethereum", callback_data='price_eth')],
                [InlineKeyboardButton("Back to Menu", callback_data='main_menu')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                text="Select a cryptocurrency to check price:",
                reply_markup=reply_markup
            )

        elif query.data == 'alert_menu':
            keyboard = [
                [InlineKeyboardButton("Set Bitcoin Alert", callback_data='alert_btc')],
                [InlineKeyboardButton("Set Ethereum Alert", callback_data='alert_eth')],
                [InlineKeyboardButton("Back to Menu", callback_data='main_menu')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                text="Select a cryptocurrency to set alert:",
                reply_markup=reply_markup
            )

        elif query.data == 'signals_menu':
            keyboard = [
                [InlineKeyboardButton("Bitcoin Signals", callback_data='signals_btc')],
                [InlineKeyboardButton("Ethereum Signals", callback_data='signals_eth')],
                [InlineKeyboardButton("Back to Menu", callback_data='main_menu')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                text="Select a cryptocurrency for trading signals:",
                reply_markup=reply_markup
            )

        elif query.data.startswith('price_'):
            crypto = query.data.split('_')[1]
            crypto_name = 'bitcoin' if crypto == 'btc' else 'ethereum'
            analysis = await self.analyze_crypto(crypto_name)
            
            if analysis:
                message = (
                    f"💰 {crypto_name.upper()} Analysis:\n\n"
                    f"Current Price: ${analysis['price']:,.2f}\n"
                    f"24h Change: {analysis['change_24h']:+.2f}%\n"
                    f"RSI (14): {analysis['rsi']:.2f}\n"
                    f"Signal: {analysis['signal']}"
                )
            else:
                message = "Error fetching data. Please try again."

            keyboard = [[InlineKeyboardButton("Back to Menu", callback_data='main_menu')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text=message, reply_markup=reply_markup)

        elif query.data.startswith('signals_'):
            crypto = query.data.split('_')[1]
            crypto_name = 'bitcoin' if crypto == 'btc' else 'ethereum'
            analysis = await self.analyze_crypto(crypto_name)
            
            if analysis:
                message = (
                    f"🤖 AI Trading Analysis for {crypto_name.upper()}\n\n"
                    f"💰 Price: ${analysis['price']:,.2f}\n"
                    f"📊 Performance:\n"
                    f"• 24h: {analysis['change_24h']:+.2f}%\n"
                    f"• 7d: {analysis['change_7d']:+.2f}%\n\n"
                    f"📈 Technical Indicators:\n"
                    f"• RSI (14): {analysis['rsi']:.2f}\n"
                    f"• Volatility: {analysis['volatility']:.2f}%\n"
                    f"• Market Sentiment: {'Bullish 🟢' if analysis['sentiment'] > 0 else 'Bearish 🔴' if analysis['sentiment'] < 0 else 'Neutral ⚪️'}\n\n"
                    f"🎯 Trading Signal:\n"
                    f"{analysis['signal']}\n\n"
                    f"⏱ Recommended Timeframe:\n"
                    f"{analysis['timeframe']}"
                )
            else:
                message = "Error fetching signals. Please try again."

            keyboard = [[InlineKeyboardButton("Back to Menu", callback_data='main_menu')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text=message, reply_markup=reply_markup)

        elif query.data == 'main_menu':
            keyboard = [
                [InlineKeyboardButton("📊 Price Check", callback_data='price_menu')],
                [InlineKeyboardButton("🔔 Set Alert", callback_data='alert_menu')],
                [InlineKeyboardButton("📈 Trading Signals", callback_data='signals_menu')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                text="🤖 Welcome to CryptoTrading Bot!\n\n"
                     "Select an option:",
                reply_markup=reply_markup
            )

        elif query.data == 'refresh_trending':
            # Refresh trending coins
            message = await self.get_trending_analysis()
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data='refresh_trending')],
                [InlineKeyboardButton("📊 Detailed Analysis", callback_data='analyze_trending')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text=message, reply_markup=reply_markup)

        elif query.data == 'analyze_trending':
            # Provide detailed analysis of trending coins
            message = await self.get_detailed_trending_analysis()
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data='refresh_trending')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text=message, reply_markup=reply_markup)

    async def trending_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get trending and potential moonshot cryptocurrencies"""
        try:
            # Get trending coins from CoinGecko
            trending_response = requests.get(f"{COINGECKO_API}/search/trending")
            trending_data = trending_response.json()

            message = "🚀 Trending & Potential Moonshots 🌙\n\n"
            message += " TRENDING COINS:\n"
            
            # Analyze trending coins
            for coin in trending_data['coins'][:5]:  # Top 5 trending
                coin_id = coin['item']['id']
                symbol = coin['item']['symbol'].upper()
                
                analysis = await self.analyze_crypto(coin_id)
                if analysis:
                    price = analysis['price']
                    change_24h = analysis['change_24h']
                    market_cap = analysis['market_cap']
                    
                    potential_rating = self.calculate_moonshot_potential(
                        market_cap, change_24h, analysis['volatility'], analysis['sentiment']
                    )
                    
                    message += (
                        f"\n{symbol} ({coin_id})\n"
                        f"💰 Price: ${price:.8f}\n"
                        f"📊 24h Change: {change_24h:+.2f}%\n"
                        f"🎯 Potential: {potential_rating}\n"
                        f"💡 Market Cap: ${market_cap:,.0f}\n"
                    )

            # Add small cap gems
            message += "\n🔍 POTENTIAL MOONSHOTS (High Risk):\n"
            small_caps = await self.find_small_cap_gems()
            message += small_caps

            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data='refresh_trending')],
                [InlineKeyboardButton("📊 Detailed Analysis", callback_data='analyze_trending')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup)
            
        except Exception as e:
            await update.message.reply_text(f"Error fetching trending coins: {str(e)}")

    def calculate_moonshot_potential(self, market_cap, change_24h, volatility, sentiment):
        """Calculate and rate moonshot potential"""
        score = 0
        
        # Market cap scoring (prefer smaller market caps)
        if market_cap < 1000000:  # < $1M
            score += 5
        elif market_cap < 10000000:  # < $10M
            score += 4
        elif market_cap < 50000000:  # < $50M
            score += 3
        elif market_cap < 100000000:  # < $100M
            score += 2
        
        # Momentum scoring
        if change_24h > 20:
            score += 3
        elif change_24h > 10:
            score += 2
        elif change_24h > 5:
            score += 1
        
        # Volatility scoring (high volatility = higher potential)
        if volatility > 20:
            score += 2
        elif volatility > 10:
            score += 1
        
        # Sentiment impact
        if sentiment > 1:
            score += 2
        elif sentiment > 0:
            score += 1
        
        # Convert score to rating
        if score >= 10:
            return "⭐⭐⭐⭐⭐ Extreme Potential (Very High Risk)"
        elif score >= 8:
            return "⭐⭐⭐⭐ Very High Potential (High Risk)"
        elif score >= 6:
            return "⭐⭐⭐ High Potential (Moderate Risk)"
        elif score >= 4:
            return "⭐⭐ Moderate Potential (Lower Risk)"
        else:
            return "⭐ Limited Potential (Stable)"

    async def find_small_cap_gems(self):
        """Find promising small cap cryptocurrencies"""
        try:
            # Get coins with market cap between $1M and $50M
            response = requests.get(
                f"{COINGECKO_API}/coins/markets",
                params={
                    'vs_currency': 'usd',
                    'order': 'volume_desc',  # Sort by volume
                    'per_page': 250,
                    'sparkline': False,
                    'price_change_percentage': '24h'
                }
            )
            coins = response.json()
            
            gems = ""
            for coin in coins:
                if 1000000 <= coin['market_cap'] <= 50000000:  # $1M to $50M market cap
                    price_change = coin['price_change_percentage_24h']
                    volume = coin['total_volume']
                    market_cap = coin['market_cap']
                    
                    # Calculate volume/market cap ratio (higher is better)
                    volume_mcap_ratio = volume / market_cap
                    
                    if volume_mcap_ratio > 0.3 and price_change is not None:  # High volume relative to market cap
                        gems += (
                            f"\n{coin['symbol'].upper()} ({coin['id']})\n"
                            f"💰 Price: ${coin['current_price']:.8f}\n"
                            f"📊 24h Change: {price_change:+.2f}%\n"
                            f"📈 Volume/MCap: {volume_mcap_ratio:.2f}\n"
                            f"⚠️ Risk Level: Very High\n"
                        )
                        
                        if len(gems.split('\n')) > 15:  # Limit to top 5 gems
                            break
            
            return gems if gems else "No potential gems found at the moment."
        
        except Exception as e:
            return f"Error finding small cap gems: {str(e)}"

    async def suggest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Suggest coins based on different investment strategies"""
        try:
            # Get market data for top 250 coins
            response = requests.get(
                f"{COINGECKO_API}/coins/markets",
                params={
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': 250,
                    'sparkline': False,
                    'price_change_percentage': '24h,7d'
                }
            )
            coins = response.json()

            message = "🤖 Investment Suggestions:\n\n"

            # 1. Safe Bets (Large Cap, Stable)
            message += "🛡️ SAFE BETS (Lower Risk):\n"
            safe_bets = self.find_safe_investments(coins)
            message += safe_bets

            # 2. Growth Potential (Mid Cap, Good Momentum)
            message += "\n📈 GROWTH POTENTIAL (Moderate Risk):\n"
            growth_picks = self.find_growth_investments(coins)
            message += growth_picks

            # 3. Moonshots (Small Cap, High Risk/Reward)
            message += "\n🚀 MOONSHOT OPPORTUNITIES (High Risk):\n"
            moonshots = self.find_moonshot_investments(coins)
            message += moonshots

            # 4. Current Dip Opportunities
            message += "\n📉 DIP BUYING OPPORTUNITIES:\n"
            dips = self.find_dip_opportunities(coins)
            message += dips

            keyboard = [
                [InlineKeyboardButton("🔄 Refresh Suggestions", callback_data='refresh_suggestions')],
                [InlineKeyboardButton("💰 Investment Strategy", callback_data='investment_strategy')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup)

        except Exception as e:
            await update.message.reply_text(f"Error generating suggestions: {str(e)}")

    def find_safe_investments(self, coins):
        """Find stable, large-cap investments"""
        safe_picks = ""
        for coin in coins[:20]:  # Look among top 20 coins
            if coin['market_cap'] > 1e9:  # >$1B market cap
                price_change = coin['price_change_percentage_24h']
                if -5 <= price_change <= 5:  # Relatively stable price
                    safe_picks += (
                        f"\n{coin['symbol'].upper()}:\n"
                        f"💰 Price: ${coin['current_price']:,.2f}\n"
                        f"📊 24h Change: {price_change:+.2f}%\n"
                        f"💎 Strategy: DCA, Long-term Hold\n"
                    )
        return safe_picks if safe_picks else "No safe picks found at the moment.\n"

    def find_growth_investments(self, coins):
        """Find mid-cap coins with good growth potential"""
        growth_picks = ""
        for coin in coins[20:100]:  # Look among top 20-100 coins
            if 100e6 <= coin['market_cap'] <= 1e9:  # $100M to $1B market cap
                price_change = coin['price_change_percentage_24h']
                if price_change > 5:  # Showing momentum
                    growth_picks += (
                        f"\n{coin['symbol'].upper()}:\n"
                        f"💰 Price: ${coin['current_price']:,.2f}\n"
                        f"📊 24h Change: {price_change:+.2f}%\n"
                        f"📈 Strategy: Swing Trade, Set Stop Loss\n"
                    )
        return growth_picks if growth_picks else "No growth picks found at the moment.\n"

    def find_moonshot_investments(self, coins):
        """Find small-cap coins with high potential"""
        moonshots = ""
        for coin in coins[100:]:  # Look beyond top 100
            if coin['market_cap'] < 100e6:  # <$100M market cap
                price_change = coin['price_change_percentage_24h']
                volume = coin['total_volume']
                volume_mcap_ratio = volume / coin['market_cap']
                
                if volume_mcap_ratio > 0.3 and price_change > 10:  # High volume and momentum
                    moonshots += (
                        f"\n{coin['symbol'].upper()}:\n"
                        f"💰 Price: ${coin['current_price']:.8f}\n"
                        f"📊 24h Change: {price_change:+.2f}%\n"
                        f"📈 Volume/MCap: {volume_mcap_ratio:.2f}\n"
                        f"🎯 Strategy: Small Position, High Risk\n"
                    )
        return moonshots if moonshots else "No moonshot opportunities found at the moment.\n"

    def find_dip_opportunities(self, coins):
        """Find coins in potential buying dips"""
        dips = ""
        for coin in coins[:100]:  # Look among top 100 coins
            price_change_24h = coin['price_change_percentage_24h']
            if price_change_24h < -10:  # Significant dip
                dips += (
                    f"\n{coin['symbol'].upper()}:\n"
                    f"💰 Price: ${coin['current_price']:,.2f}\n"
                    f"📉 24h Drop: {price_change_24h:+.2f}%\n"
                    f"💡 Strategy: Staged Buying, Watch RSI\n"
                )
        return dips if dips else "No significant dips found at the moment.\n"

    async def get_investment_strategy(self, coin_symbol):
        """Generate detailed investment strategy for a specific coin"""
        analysis = await self.analyze_crypto(coin_symbol)
        if not analysis:
            return "Could not analyze this coin."

        strategy = (
            f"🎯 Investment Strategy for {coin_symbol.upper()}\n\n"
            f"Entry Strategy:\n"
            f"• {'Aggressive entry' if analysis['volatility'] > 5 else 'Conservative entry'}\n"
            f"• Set limit orders {abs(analysis['volatility']/2):.1f}% below current price\n\n"
            f"Position Size:\n"
            f"• {'1-2%' if analysis['market_cap'] > 1e9 else '0.5-1%'} of portfolio\n"
            f"• Split entry into {3 if analysis['volatility'] > 10 else 2} parts\n\n"
            f"Risk Management:\n"
            f"• Stop Loss: {abs(analysis['volatility']):.1f}% below entry\n"
            f"• Take Profit: {abs(analysis['volatility']*2):.1f}% above entry\n"
            f"• Use trailing stops if trending up\n\n"
            f"Timeframe:\n"
            f"• {analysis['timeframe']}\n"
            f"• {'Scale out in profit' if analysis['sentiment'] > 0 else 'Quick exit if support breaks'}"
        )
        return strategy

    async def short_trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Find the best short-term trading opportunities"""
        try:
            await update.message.reply_text("🔍 Scanning all coins for short-term opportunities... Please wait.")
            
            # Get comprehensive market data
            response = requests.get(
                f"{COINGECKO_API}/coins/markets",
                params={
                    'vs_currency': 'usd',
                    'order': 'volume_desc',
                    'per_page': 250,
                    'sparkline': False,
                    'price_change_percentage': '1h,24h,7d'
                }
            )
            coins = response.json()

            # Analyze and score each coin
            scored_opportunities = []
            for coin in coins:
                try:
                    score = self.calculate_short_term_score(coin)
                    if score > 7:  # Only include high-potential opportunities
                        scored_opportunities.append((score, coin['id'], coin))
                except Exception as e:
                    continue

            # Sort by score (using the first element of the tuple)
            scored_opportunities.sort(key=lambda x: x[0], reverse=True)

            message = "🚀 TOP SHORT-TERM TRADING OPPORTUNITIES:\n\n"
            
            # Categorize by timeframe
            scalping = "⚡ SCALPING (5min-1hr):\n"
            intraday = "📊 INTRADAY (1-4hrs):\n"
            swing = "🌊 SHORT SWING (1-3 days):\n"

            for score, coin_id, coin in scored_opportunities[:15]:  # Top 15 opportunities
                try:
                    analysis = await self.analyze_crypto(coin_id)
                    if analysis:
                        trade_info = self.generate_short_term_trade_info(coin, analysis, score)
                        
                        # Categorize based on volatility and volume
                        if analysis['volatility'] > 15 and coin['total_volume'] > 10000000:
                            scalping += trade_info
                        elif analysis['volatility'] > 8:
                            intraday += trade_info
                        else:
                            swing += trade_info
                except Exception as e:
                    continue

            message += scalping + "\n" + intraday + "\n" + swing

            # Add risk warning
            message += ("\n⚠️ RISK WARNING:\n"
                       "• Use strict stop losses\n"
                       "• Never invest more than you can afford to lose\n"
                       "• High volatility means high risk\n"
                       "• Always do your own research")

            # Split message if too long
            if len(message) > 4096:
                messages = [message[i:i+4096] for i in range(0, len(message), 4096)]
                for msg in messages:
                    await update.message.reply_text(msg)
            else:
                await update.message.reply_text(message)

        except Exception as e:
            await update.message.reply_text(f"Error scanning opportunities: {str(e)}")

    def calculate_short_term_score(self, coin):
        """Calculate score for short-term trading potential"""
        score = 0
        
        try:
            # Volume factors (liquidity is crucial for short-term trading)
            volume = float(coin.get('total_volume', 0))
            market_cap = float(coin.get('market_cap', 0))
            
            # Avoid division by zero
            volume_mcap_ratio = volume / market_cap if market_cap > 0 else 0
            
            if volume > 10000000:  # >$10M daily volume
                score += 3
            elif volume > 5000000:  # >$5M daily volume
                score += 2
            elif volume > 1000000:  # >$1M daily volume
                score += 1

            if volume_mcap_ratio > 0.3:
                score += 2
            elif volume_mcap_ratio > 0.1:
                score += 1

            # Price momentum
            price_change_24h = float(coin.get('price_change_percentage_24h', 0) or 0)
            price_change_1h = float(coin.get('price_change_percentage_1h', 0) or 0)
            
            # Recent momentum (1h)
            if abs(price_change_1h) > 5:
                score += 2
            elif abs(price_change_1h) > 2:
                score += 1

            # Sustained momentum (24h)
            if abs(price_change_24h) > 10:
                score += 2
            elif abs(price_change_24h) > 5:
                score += 1

            # Market cap considerations (prefer medium cap for stability)
            if 100000000 <= market_cap <= 10000000000:  # $100M to $10B
                score += 1

            return score
        except Exception as e:
            return 0

    def generate_short_term_trade_info(self, coin, analysis, score):
        """Generate detailed trading information for short-term opportunities"""
        try:
            price = float(coin.get('current_price', 0))
            change_1h = float(coin.get('price_change_percentage_1h', 0) or 0)
            change_24h = float(coin.get('price_change_percentage_24h', 0) or 0)
            
            # Calculate optimal entry and exit points
            entry_zone = price * 0.99  # 1% below current price
            stop_loss = price * (1 - (analysis['volatility'] / 100))
            take_profit = price * (1 + (analysis['volatility'] / 50))
            
            # Generate trade setup
            return (
                f"\n{coin['symbol'].upper()} (Score: {score}/10)\n"
                f"💰 Price: ${price:.8f}\n"
                f"📊 Changes: 1h {change_1h:+.2f}% | 24h {change_24h:+.2f}%\n"
                f"📈 Volume: ${float(coin.get('total_volume', 0)):,.0f}\n"
                f"🎯 Setup:\n"
                f"  • Entry: ~${entry_zone:.8f}\n"
                f"  • Stop: ${stop_loss:.8f}\n"
                f"  • Target: ${take_profit:.8f}\n"
                f"  • R/R Ratio: {abs((take_profit-entry_zone)/(entry_zone-stop_loss)):.1f}\n"
            )
        except Exception as e:
            return f"\nError processing {coin.get('symbol', 'Unknown')}: {str(e)}\n"

def main():
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()
    bot = TradingBot()

    # Add command handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("price", bot.price_command))
    application.add_handler(CommandHandler("alert", bot.set_alert))
    application.add_handler(CommandHandler("add", bot.add_to_watchlist))
    application.add_handler(CommandHandler("remove", bot.remove_from_watchlist))
    application.add_handler(CommandHandler("watchlist", bot.show_watchlist))
    application.add_handler(CommandHandler("signals", bot.signals_command))
    application.add_handler(CommandHandler("trending", bot.trending_command))
    application.add_handler(CommandHandler("suggest", bot.suggest_command))
    application.add_handler(CommandHandler("short_trades", bot.short_trades_command))
    
    # Add callback query handler for buttons
    application.add_handler(CallbackQueryHandler(bot.button_handler))

    # Start the bot
    logger.info("Starting bot...")
    application.run_polling(poll_interval=1.0)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}")
