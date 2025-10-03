# Import necessary libraries
import backtrader as bt  # The core backtesting framework
import yfinance as yf    # For downloading historical stock data
import pandas as pd      # For data manipulation, especially with DataFrames
import numpy as np       # For numerical operations and statistical calculations
import datetime          # For handling dates and times
import os                # Operating system interfaces (not used in snippet but good practice)
import json              # For handling JSON data (if config is from a file)
import re                # Regular expression operations (not used in snippet but potentially useful)

class DynamicStrategy(bt.Strategy):
    """
    A dynamic backtrader strategy that configures its indicators and trading rules
    based on a dictionary passed during initialization. This allows for flexible
    strategy testing without modifying the core code.

    The strategy incorporates risk management through ATR-based position sizing
    and a maximum number of concurrent positions.
    """
    
    # --- Strategy Parameters ---
    # These can be overridden when adding the strategy to cerebro.
    params = dict(
        risk_free_rate=0.0,            # Annual risk-free rate for Sharpe Ratio calculation
        risk_per_trade_pct=0.02,     # The percentage of portfolio value to risk on a single trade
        atr_period=14,                 # The period for the Average True Range (ATR) indicator
        stop_loss_atr_multiplier=2.0,  # Multiplier for ATR to set the stop-loss distance
        max_positions=50               # The maximum number of concurrent open positions
    )

    def __init__(self, config):
        """
        Initializes the strategy, dynamically creating indicators and crossovers
        based on the provided configuration dictionary.

        Args:
            config (dict): A dictionary defining the indicators and trading rules.
                Expected keys: "indicators" (list of dicts), "rules" (list of dicts).
        """
        # Store the configuration for later use
        self.config = config
        
        # --- Internal State Dictionaries ---
        # Stores instantiated indicator objects for each data feed.
        # Key: (data_feed, indicator_key_string), Value: indicator_object
        self.indicators = {}       
        
        # Stores instantiated crossover objects.
        # Key: (data_feed, "IND1_CROSSOVER_IND2_STR"), Value: CrossOver_indicator_object
        self.crossovers = {}       
        
        # Tracks pending orders for each data feed to avoid placing duplicate orders.
        # Key: data_feed_name (str), Value: pending_order_object
        self.orders = {}           
        
        # --- Performance Tracking Lists ---
        # Records the portfolio value at each step to plot an equity curve.
        self.portfolio_values = [self.broker.getvalue()]
        # Records the corresponding dates for the portfolio values.
        self.dates = []

        # Initialize the dates list with the first available date from the data feeds.
        if self.datas:
            self.dates.append(self.datas[0].datetime.date(0))
        else:
            self.dates.append(datetime.date.today()) # Fallback if no data is loaded

        # --- Dynamic Indicator and Crossover Initialization ---
        # Loop through each data feed (e.g., each stock ticker) added to Cerebro.
        for i, data in enumerate(self.datas):
            # Always add ATR as it's essential for our risk management and sizing logic.
            self.indicators[(data, f"ATR_{self.p.atr_period}")] = bt.indicators.ATR(data, period=self.p.atr_period)

            # Process user-defined indicators from the configuration.
            for ind_cfg in config.get("indicators", []):
                ind_type = ind_cfg["type"]
                
                # Construct a unique string key for the indicator (e.g., "EMA_8", "MACD_12_26_9").
                # This key will be used to access the indicator's value in the 'next' method.
                key_parts = [ind_type]
                if ind_type in ["SMA", "EMA"]:
                    key_parts.append(str(ind_cfg["window"]))
                elif ind_type == "RSI":
                    key_parts.append(str(ind_cfg["period"]))
                elif ind_type == "MACD":
                    key_parts.extend([str(ind_cfg["fast"]), str(ind_cfg["slow"]), str(ind_cfg["signal"])])
                elif ind_type == "BBANDS":
                    dev = ind_cfg["devfactor"]
                    if dev == int(dev): dev = int(dev) # Keep integer if it's a whole number for cleaner keys
                    key_parts.extend([str(ind_cfg["period"]), str(dev).replace('.', '_')]) # Replace '.' for valid eval() context keys
                elif ind_type == "STOCH":
                    key_parts.extend([str(ind_cfg["period"]), str(ind_cfg["percK_period"]), str(ind_cfg["percD_period"])])
                
                ind_key_str = "_".join(key_parts)

                # Instantiate the corresponding backtrader indicator based on its type.
                if ind_type == "SMA":
                    self.indicators[(data, ind_key_str)] = bt.indicators.SimpleMovingAverage(data, period=ind_cfg["window"])
                elif ind_type == "EMA":
                    self.indicators[(data, ind_key_str)] = bt.indicators.ExponentialMovingAverage(data, period=ind_cfg["window"])
                elif ind_type == "RSI":
                    self.indicators[(data, ind_key_str)] = bt.indicators.RSI(data, period=ind_cfg["period"])
                elif ind_type == "MACD":
                    self.indicators[(data, ind_key_str)] = bt.indicators.MACD(
                        data,
                        period_me1=ind_cfg["fast"],
                        period_me2=ind_cfg["slow"],
                        period_signal=ind_cfg["signal"]
                    )
                elif ind_type == "BBANDS":
                    self.indicators[(data, ind_key_str)] = bt.indicators.BollingerBands(
                        data,
                        period=ind_cfg["period"],
                        devfactor=ind_cfg["devfactor"]
                    )
                elif ind_type == "STOCH":
                    self.indicators[(data, ind_key_str)] = bt.indicators.Stochastic(
                        data,
                        period=ind_cfg["period"],
                        period_dfast=ind_cfg["percK_period"],
                        period_dslow=ind_cfg["percD_period"]
                    )
                else:
                    self.log(f"WARNING: Unknown indicator type '{ind_type}' in config for {data._name}. Skipping.")

        # Prepare crossover indicators if any rule conditions contain "_CROSSOVER_".
        for rule in config.get("rules", []):
            cond = rule.get("condition", "")
            # Split the condition string to find tokens that represent a crossover.
            for token in cond.split():
                if "_CROSSOVER_" in token:
                    # Parse the indicator keys from the crossover token.
                    ind1_key_str, ind2_key_str = token.split("_CROSSOVER_")
                    for data in self.datas:
                        # Retrieve the already instantiated indicator objects.
                        ind1_obj = self.indicators.get((data, ind1_key_str))
                        ind2_obj = self.indicators.get((data, ind2_key_str))
                        # If both indicators exist, create the CrossOver indicator.
                        if ind1_obj is not None and ind2_obj is not None:
                            self.crossovers[(data, token)] = bt.indicators.CrossOver(ind1_obj, ind2_obj)
                        else:
                            self.log(f"WARNING: Could not create crossover for '{token}' on {data._name}. Check indicator definitions for '{ind1_key_str}' and '{ind2_key_str}'.")


    def log(self, txt, dt=None):
        """
        Logging function for the strategy.

        Args:
            txt (str): The message to log.
            dt (datetime.date, optional): The date to associate with the log entry. 
                                          Defaults to the current backtest date.
        """
        dt = dt or self.datas[0].datetime.date(0) if self.datas else datetime.date.today()
        print(f"{dt.isoformat()}, {txt}")

    def notify_order(self, order):
        """
        Backtrader's order notification callback. This method is called for any
        change in an order's status.
        
        Args:
            order (bt.Order): The order object with the updated status.
        """
        data_name = order.data._name
        # If the order is submitted or accepted, track it as pending.
        if order.status in [order.Submitted, order.Accepted]:
            self.orders[data_name] = order
            self.log(f"{data_name} ORDER SUBMITTED/ACCEPTED: Ref {order.ref}, Status {order.getstatusname()}.")
            return
        
        # If the order is completed, log the execution details.
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"{data_name} BUY EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size:.0f}, Cost {order.executed.value:.2f}, Comm {order.executed.comm:.2f}. Cash: {self.broker.getcash():.2f}")
            else: # Sell
                self.log(f"{data_name} SELL EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size:.0f}, Cost {order.executed.value:.2f}, Comm {order.executed.comm:.2f}. Cash: {self.broker.getcash():.2f}")
        
        # If the order failed for any reason, log the failure status.
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            reason = order.info.get('reason', 'N/A')
            self.log(f"{data_name} ORDER FAILED/REJECTED: Status {order.getstatusname()}. Reason: {reason}. Cash: {self.broker.getcash():.2f}")
        
        # Once an order is finalized (completed or failed), remove it from the pending orders dictionary.
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
             self.orders[data_name] = None


    def next(self):
        """
        The core logic method of the strategy, called on each bar of data.
        It iterates through all assets, evaluates the dynamic rules, and places
        orders based on signals and risk management.
        """
        # --- Record current portfolio state for equity curve plotting ---
        current_date_in_next = self.datas[0].datetime.date(0) if self.datas else datetime.date.today()
        self.dates.append(current_date_in_next)
        self.portfolio_values.append(self.broker.getvalue())
        self.log(f"--- Current Date: {current_date_in_next.isoformat()} | Cash: {self.broker.getcash():.2f} | Portfolio Value: {self.broker.getvalue():.2f} ---")

        # --- Loop through each data feed (asset) ---
        for data in self.datas:
            data_name = data._name

            # Skip if there's no data for the current bar.
            if not len(data) > 0 or pd.isna(data.close[0]):
                continue

            # Get current price and position information.
            current_close = data.close[0]
            current_position = self.getposition(data)

            # Skip this asset if an order is already pending for it.
            if self.orders.get(data_name) and self.orders[data_name].status in [bt.Order.Submitted, bt.Order.Accepted]:
                continue

            # --- Build the 'context' dictionary for the eval() function ---
            # This context will contain all the necessary variables (prices, indicator values)
            # that the rule conditions can reference.
            context = {f"Close_{data_name}": current_close, "Close": current_close}
            if len(data) > 1:
                context[f"Close(-1)_{data_name}"] = data.close[-1]
                context[f"Close(-1)"] = data.close[-1]

            atr_val = None
            # Populate the context with the current values of all indicators for this data feed.
            for (ind_data, ind_key_str), indicator_obj in self.indicators.items():
                if ind_data == data and not pd.isna(indicator_obj[0]):
                    ind_type = ind_key_str.split('_')[0]

                    if ind_type == "ATR":
                        atr_val = indicator_obj[0]
                        context[ind_key_str] = atr_val
                    # For multi-line indicators, add each line to the context with a descriptive key.
                    elif ind_type == "MACD" and hasattr(indicator_obj, 'macd'):
                        context[f"{ind_key_str}_MACD"] = indicator_obj.macd[0]
                        context[f"{ind_key_str}_SIGNAL"] = indicator_obj.signal[0]
                        context[f"{ind_key_str}_HIST"] = indicator_obj.hist[0]
                    elif ind_type == "BBANDS" and hasattr(indicator_obj, 'top'):
                        context[f"{ind_key_str}_UPPER"] = indicator_obj.top[0]
                        context[f"{ind_key_str}_MIDDLE"] = indicator_obj.mid[0]
                        context[f"{ind_key_str}_LOWER"] = indicator_obj.bot[0]
                    elif ind_type == "STOCH" and hasattr(indicator_obj, 'percK'):
                        context[f"{ind_key_str}_K"] = indicator_obj.percK[0]
                        context[f"{ind_key_str}_D"] = indicator_obj.percD[0]
                    else: # For single-line indicators
                        context[ind_key_str] = indicator_obj[0]

            # Populate the context with crossover signal values (-1 for cross down, 1 for cross up, 0 otherwise).
            for (cross_data, cross_key_str), crossover_obj in self.crossovers.items():
                if cross_data == data and not pd.isna(crossover_obj[0]):
                    context[cross_key_str] = crossover_obj[0]

            # If ATR is not available or zero, we cannot calculate risk, so we skip this asset.
            if atr_val is None or atr_val <= 0:
                continue

            # === DYNAMIC RULE EVALUATION ===
            # Iterate through each rule defined in the configuration.
            for rule in self.config.get("rules", []):
                try:
                    rule_condition_met = False
                    # Evaluate the rule's condition string using the built context.
                    # This is the core of the dynamic strategy logic.
                    try:
                        rule_condition_met = eval(rule["condition"], {}, context)
                    except Exception as e:
                        self.log(f"ERROR evaluating {data_name} rule '{rule['condition']}': {e}")
                        continue

                    # If the condition is met, proceed with signal processing.
                    if rule_condition_met:
                        signal = rule["signal"].upper()
                        self.log(f"{data_name} Rule condition MET: '{rule['condition']}' with signal '{signal}'.")

                        # --- Position Sizing based on Risk Management ---
                        # Calculate the dollar amount to risk on this trade.
                        dollar_risk = self.broker.getvalue() * self.p.risk_per_trade_pct
                        # Calculate the per-share stop-loss distance using ATR.
                        stop_loss_dollar_distance = atr_val * self.p.stop_loss_atr_multiplier
                        
                        target_shares = 0
                        # Calculate the number of shares to trade based on the risk parameters.
                        if stop_loss_dollar_distance > 0 and current_close > 0:
                            calculated_shares = dollar_risk / stop_loss_dollar_distance
                            target_shares = max(1, int(calculated_shares)) # Ensure at least 1 share
                        else:
                            self.log(f"{data_name} Sizing issue: SL distance ({stop_loss_dollar_distance:.4f}) or Close ({current_close:.2f}) invalid.")
                            continue

                        # Get the number of currently active positions.
                        num_active_positions = sum(1 for p in self.positions if self.positions[p].size != 0)

                        # === SIGNAL EXECUTION ===
                        # --- BUY Signal ---
                        if signal == "BUY":
                            # Avoid re-buying if we already hold the target size.
                            if current_position.size == target_shares:
                                continue
                            # Check if we have reached the maximum number of concurrent positions.
                            if num_active_positions >= self.p.max_positions and current_position.size == 0:
                                self.log(f"{data_name} BUY ignored: Max positions reached.")
                                continue
                            # Check for sufficient cash to execute the trade.
                            cash_needed = current_close * target_shares
                            if self.broker.getcash() < cash_needed:
                                self.log(f"{data_name} BUY ignored: Not enough cash.")
                                continue
                            self.log(f"{data_name} BUY -> Target {target_shares} shares.")
                            # Place the order to reach the target size.
                            self.orders[data_name] = self.order_target_size(data=data, target=target_shares)

                        # --- SELL Signal (liquidate position) ---
                        elif signal == "SELL":
                            # Only sell if we currently hold a position.
                            if current_position.size == 0:
                                continue
                            self.log(f"{data_name} SELL -> Liquidate to 0 shares (cash).")
                            # Place the order to close the position.
                            self.orders[data_name] = self.order_target_size(data=data, target=0)

                        # --- SHORT Signal ---
                        elif signal == "SHORT":
                            # Avoid re-shorting if we already hold the target size.
                            if current_position.size == -target_shares:
                                continue
                            # Check max positions limit for opening a new short position.
                            if num_active_positions >= self.p.max_positions and current_position.size == 0:
                                self.log(f"{data_name} SHORT ignored: Max positions reached.")
                                continue
                            self.log(f"{data_name} SHORT -> Target {-target_shares} shares.")
                            # Place the order to short the target number of shares.
                            self.orders[data_name] = self.order_target_size(data=data, target=-target_shares)

                except Exception as e:
                    import traceback
                    self.log(f"ERROR in rule for {data_name}: {e}")
                    traceback.print_exc()



def run_backtest(config):
    """
    Orchestrates the entire backtesting process. It sets up the backtrader
    engine (Cerebro), fetches data, adds the dynamic strategy, runs the backtest,
    and compiles a comprehensive report of performance metrics.

    Args:
        config (dict): The strategy configuration dictionary to be passed to
                       the DynamicStrategy.

    Returns:
        dict: A dictionary containing final portfolio value, PNL, chart data,
              and a detailed breakdown of performance metrics. Returns an
              error dictionary if data loading fails.
    """
    
    # Initialize the backtrader engine.
    cerebro = bt.Cerebro()

    # --- Data and Ticker Configuration ---
    # Define the universe of assets for the backtest.
    stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'PG', 'KO',
                     'XOM', 'CVX', 'UNH', 'LLY', 'JNJ', 'PFE', 'PEP', 'WMT', 'DIS', 'NFLX']
    etf_tickers = ['SPY', 'QQQ', 'DIA', 'IWM', 'GLD']
    all_tickers = stock_tickers + etf_tickers
    
    # Set the date range for the backtest (approximately 3 years).
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=3 * 365)

    # --- Benchmark Data ---
    # Download benchmark data (SPY) for calculating Alpha and Beta.
    benchmark_df = pd.DataFrame() 
    benchmark_ticker = 'SPY'
    try:
        benchmark_df = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        # Data cleaning for consistency.
        if isinstance(benchmark_df.columns, pd.MultiIndex):
            benchmark_df.columns = [col[0] for col in benchmark_df.columns]
        if benchmark_df.index.tz is not None:
            benchmark_df.index = benchmark_df.index.tz_localize(None)
        benchmark_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        if benchmark_df.empty:
            print(f"Warning: No benchmark data downloaded for {benchmark_ticker}.")
    except Exception as e:
        print(f"Warning: Could not download benchmark data for {benchmark_ticker}: {e}. Alpha/Beta might be unavailable.")

    if not all_tickers:
        return {"error": "No tickers provided for backtesting."}

    # --- Asset Data Loading ---
    # Loop through all tickers, download data, and add to Cerebro.
    for ticker in all_tickers:
        try:
            data_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True) 
            if data_df.empty:
                print(f"Warning: No data downloaded for {ticker}. Skipping.")
                continue

            # Clean and format the downloaded DataFrame to be compatible with backtrader.
            if isinstance(data_df.columns, pd.MultiIndex):
                data_df.columns = [col[0] for col in data_df.columns]
            if data_df.index.tz is not None:
                data_df.index = data_df.index.tz_localize(None)
            data_df.columns = [col.lower() for col in data_df.columns]
            
            # Ensure all required columns are present.
            required_cols = ['open', 'high', 'low', 'close', 'volume'] 
            if not all(col in data_df.columns for col in required_cols): 
                 print(f"Warning: Missing expected columns for {ticker}. Skipping.")
                 continue
            
            # Create a backtrader PandasData feed from the DataFrame.
            data = bt.feeds.PandasData(dataname=data_df, name=ticker)
            cerebro.adddata(data)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}. Skipping.")
            continue

    # If no data could be loaded, exit gracefully.
    if not cerebro.datas:
        return {"error": "No data feeds were successfully loaded for backtesting. Check tickers or data availability."}

    # --- Cerebro Configuration ---
    # Add the dynamic strategy to Cerebro, passing the user-defined config.
    cerebro.addstrategy(DynamicStrategy, config=config)

    # Set the initial cash for the backtest.
    start_cash = 10000.0
    cerebro.broker.setcash(start_cash)
    
    # --- CRITICAL CORRECTION: REMOVED THE PERCENTSIZER ---
    # The DynamicStrategy calculates its own position size based on risk (ATR).
    # Using a PercentSizer would conflict with this logic, so it is commented out.
    # cerebro.addsizer(bt.sizers.PercentSizer, percents=100) 

    # --- Add Performance Analyzers ---
    # These analyzers will compute metrics after the backtest runs.
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=config.get('risk_free_rate', 0.0))
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns') 
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # --- Run the Backtest ---
    print("Running backtest...")
    strategies = cerebro.run()
    first_strategy = strategies[0]
    print("Backtest finished.")

    # --- Results Extraction and Processing ---
    final_value = cerebro.broker.getvalue()
    pnl = final_value - start_cash

    # Prepare data for the equity curve chart.
    formatted_dates = []
    portfolio_values_for_chart = []
    if first_strategy.dates and first_strategy.portfolio_values:
        # Ensure dates and values lists are of the same length to avoid errors.
        min_len = min(len(first_strategy.dates), len(first_strategy.portfolio_values))
        formatted_dates = [d.strftime('%Y-%m-%d') for d in first_strategy.dates[:min_len]]
        portfolio_values_for_chart = first_strategy.portfolio_values[:min_len]
    else:
        print("Warning: No dates or portfolio values recorded by the strategy.")

    # Extract results from analyzers with robust handling for None or NaN values.
    sharpe_ratio_result = first_strategy.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_ratio_result.get('sharperatio') if sharpe_ratio_result else None
    sharpe_ratio = round(sharpe_ratio, 2) if sharpe_ratio is not None and not pd.isna(sharpe_ratio) else 0.0
    
    drawdown_result = first_strategy.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown_result.get('max', {}).get('drawdown') if drawdown_result else None
    max_drawdown_len = drawdown_result.get('max', {}).get('len') if drawdown_result else None
    max_drawdown = round(max_drawdown, 2) if max_drawdown is not None and not pd.isna(max_drawdown) else 0.0
    max_drawdown_len = int(max_drawdown_len) if max_drawdown_len is not None and not pd.isna(max_drawdown_len) else 0

    trade_analyzer_result = first_strategy.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analyzer_result.get('total', {}).get('total', 0)
    winning_trades = trade_analyzer_result.get('won', {}).get('total', 0)
    losing_trades = trade_analyzer_result.get('lost', {}).get('total', 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    avg_win = trade_analyzer_result.get('won', {}).get('pnl', {}).get('average')
    avg_win = round(avg_win, 2) if avg_win is not None and not pd.isna(avg_win) else 0.0
    avg_loss = trade_analyzer_result.get('lost', {}).get('pnl', {}).get('average')
    avg_loss = round(avg_loss, 2) if avg_loss is not None and not pd.isna(avg_loss) else 0.0

    # Initialize Alpha and Beta to 0.0
    alpha = 0.0
    beta = 0.0
    
    # Calculate Compound Annual Growth Rate (CAGR).
    num_years = 0.0
    strategy_cagr = 0.0
    if cerebro.datas and len(cerebro.datas) > 0:
        backtest_start_date_data = cerebro.datas[0].datetime.date(0)
        backtest_end_date_data = cerebro.datas[0].datetime.date(-1) 
        num_days = (backtest_end_date_data - backtest_start_date_data).days
        if num_days > 0:
            num_years = num_days / 365.25
            if final_value > 0 and start_cash > 0 and num_years > 0:
                strategy_cagr = ((final_value / start_cash) ** (1/num_years)) - 1
    annual_return_pct = round(strategy_cagr * 100, 2)
    
    # Calculate Alpha and Beta if benchmark data is available.
    if len(first_strategy.portfolio_values) > 1 and not benchmark_df.empty:
        # Create a pandas Series for strategy returns and benchmark returns.
        strategy_pnl_series = pd.Series(first_strategy.portfolio_values, index=pd.to_datetime(first_strategy.dates))
        strategy_daily_returns = strategy_pnl_series.pct_change().dropna()
        benchmark_daily_returns = benchmark_df['close'].pct_change().dropna()
        
        # Align the returns data by date to ensure a fair comparison.
        common_dates = strategy_daily_returns.index.intersection(benchmark_daily_returns.index)
        strategy_returns_aligned = strategy_daily_returns[common_dates]
        benchmark_returns_aligned = benchmark_daily_returns[common_dates]

        # Proceed with calculation only if there's enough aligned data.
        if not strategy_returns_aligned.empty and len(strategy_returns_aligned) > 1 and len(benchmark_returns_aligned) > 1:
            # Calculate covariance matrix to find variance of benchmark and covariance between strategy and benchmark.
            covariance_matrix = np.cov(strategy_returns_aligned, benchmark_returns_aligned)
            cov_strategy_benchmark = covariance_matrix[0, 1]
            var_benchmark = covariance_matrix[1, 1]

            # Calculate benchmark's CAGR.
            benchmark_cagr = 0.0
            if not benchmark_df.empty:
                initial_benchmark_price = benchmark_df['close'].iloc[0]
                final_benchmark_price = benchmark_df['close'].iloc[-1]
                if initial_benchmark_price != 0 and num_years > 0:
                    benchmark_cagr = ((final_benchmark_price / initial_benchmark_price) ** (1/num_years)) - 1
            
            # Calculate Beta and then Alpha using the Capital Asset Pricing Model (CAPM) formula.
            if var_benchmark != 0 and num_years > 0 and not pd.isna(strategy_cagr) and not pd.isna(benchmark_cagr):
                beta = cov_strategy_benchmark / var_benchmark
                if pd.isna(beta): beta = 0.0 # Handle potential NaN result
                
                risk_free_rate = first_strategy.p.risk_free_rate 
                # Alpha = Strategy Return - (Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate))
                alpha_val = strategy_cagr - (risk_free_rate + beta * (benchmark_cagr - risk_free_rate))
                if pd.isna(alpha_val): alpha_val = 0.0 # Handle potential NaN result
                alpha = alpha_val
    
    # Calculate total percentage return.
    total_return_pct = round(((final_value - start_cash) / start_cash) * 100, 2) if start_cash > 0 else 0.0

    # --- Compile Final Metrics Dictionary ---
    metrics = {
        "sharpe_ratio": sharpe_ratio,
        "total_return_pct": total_return_pct, 
        "annual_return_pct": annual_return_pct, 
        "max_drawdown_pct": max_drawdown,
        "max_drawdown_duration_days": max_drawdown_len,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": round(win_rate, 2),
        "average_win": avg_win,
        "average_loss": avg_loss,
        "alpha": round(alpha * 100, 2), # Convert alpha to percentage points
        "beta": round(beta, 2)
    }

    # --- Return the Final Results Package ---
    return {
        "final_portfolio_value": final_value,
        "pnl": pnl,
        "chart_data": {
            "dates": formatted_dates,
            "values": portfolio_values_for_chart,
        },
        "metrics": metrics
    }