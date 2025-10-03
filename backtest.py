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
    
    params = dict(
        risk_free_rate=0.0,
        risk_per_trade_pct=0.02,
        atr_period=14,
        stop_loss_atr_multiplier=2.0,
        max_positions=50
    )

    def __init__(self, config):
        self.config = config
        self.indicators = {}
        self.crossovers = {}
        self.orders = {}
        self.portfolio_values = [self.broker.getvalue()]
        self.dates = [self.datas[0].datetime.date(0) if self.datas else datetime.date.today()]

        for i, data in enumerate(self.datas):
        # ATR for sizing (always add as it's critical for risk management)
            self.indicators[(data, f"ATR_{self.p.atr_period}")] = bt.indicators.ATR(data, period=self.p.atr_period)

            # User-defined indicators from config
            for ind_cfg in config.get("indicators", []):
                ind_type = ind_cfg["type"]
                
                # Construct indicator_key_str (e.g., EMA_8, MACD_12_26_9). This will be the key in context.
                key_parts = [ind_type]
                if ind_type in ["SMA", "EMA"]:
                    key_parts.append(str(ind_cfg["window"]))
                elif ind_type == "RSI": # ATR already handled
                    key_parts.append(str(ind_cfg["period"]))
                elif ind_type == "MACD":
                    key_parts.extend([str(ind_cfg["fast"]), str(ind_cfg["slow"]), str(ind_cfg["signal"])])
                elif ind_type == "BBANDS":
                    dev = ind_cfg["devfactor"]
                    if dev == int(dev): dev = int(dev) # Keep integer if it's a whole number
                    key_parts.extend([str(ind_cfg["period"]), str(dev).replace('.', '_')])
                elif ind_type == "STOCH":
                    key_parts.extend([str(ind_cfg["period"]), str(ind_cfg["percK_period"]), str(ind_cfg["percD_period"])])
                
                ind_key_str = "_".join(key_parts)

                # Instantiate the indicator
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

        for rule in config.get("rules", []):
            cond = rule.get("condition", "")
            for token in cond.split():
                if "_CROSSOVER_" in token:
                    ind1_key_str, ind2_key_str = token.split("_CROSSOVER_")
                    for data in self.datas:
                        ind1_obj = self.indicators.get((data, ind1_key_str))
                        ind2_obj = self.indicators.get((data, ind2_key_str))
                        if ind1_obj is not None and ind2_obj is not None:
                            self.crossovers[(data, token)] = bt.indicators.CrossOver(ind1_obj, ind2_obj)
                        else:
                            self.log(f"WARNING: Could not create crossover for '{token}' on {data._name}.")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0) if self.datas else datetime.date.today()
        print(f"{dt.isoformat()}, {txt}")

    def notify_order(self, order):
        data_name = order.data._name
        if order.status in [order.Submitted, order.Accepted]:
            self.orders[data_name] = order
            self.log(f"{data_name} ORDER SUBMITTED/ACCEPTED: Ref {order.ref}, Status {order.getstatusname()}.")
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"{data_name} BUY EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size:.0f}")
            else:
                self.log(f"{data_name} SELL EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size:.0f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"{data_name} ORDER FAILED/REJECTED: Status {order.getstatusname()}.")
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders[data_name] = None

    def next(self):
        current_date_in_next = self.datas[0].datetime.date(0) if self.datas else datetime.date.today()
        self.dates.append(current_date_in_next)
        self.portfolio_values.append(self.broker.getvalue())
        self.log(f"--- Date: {current_date_in_next.isoformat()} | Cash: {self.broker.getcash():.2f} | Portfolio: {self.broker.getvalue():.2f} ---")

        for data in self.datas:
            data_name = data._name
            if not len(data) > 0 or pd.isna(data.close[0]):
                continue
            current_close = data.close[0]
            current_position = self.getposition(data)

            if self.orders.get(data_name) and self.orders[data_name].status in [bt.Order.Submitted, bt.Order.Accepted]:
                continue

            context = {f"Close_{data_name}": current_close, "Close": current_close}
            if len(data) > 1:
                context[f"Close(-1)_{data_name}"] = data.close[-1]
                context[f"Close(-1)"] = data.close[-1]

            atr_val = None
            for (ind_data, ind_key_str), indicator_obj in self.indicators.items():
                if ind_data == data and not pd.isna(indicator_obj[0]):
                    ind_type = ind_key_str.split('_')[0]

                    if ind_type == "ATR":
                        atr_val = indicator_obj[0]
                        context[ind_key_str] = atr_val

                    elif ind_type == "MACD":
                        # Access macd and signal lines safely
                        macd_line = indicator_obj.macd if hasattr(indicator_obj, 'macd') else indicator_obj.lines.macd
                        signal_line = indicator_obj.signal if hasattr(indicator_obj, 'signal') else indicator_obj.lines.signal
                        
                        # Compute histogram manually
                        hist_val = macd_line[0] - signal_line[0]
                        
                        context[f"{ind_key_str}_MACD"] = macd_line[0]
                        context[f"{ind_key_str}_SIGNAL"] = signal_line[0]
                        context[f"{ind_key_str}_HIST"] = hist_val


                    elif ind_type == "BBANDS":
                        context[f"{ind_key_str}_UPPER"] = indicator_obj.top[0] if hasattr(indicator_obj, 'top') else indicator_obj.lines.top[0]
                        context[f"{ind_key_str}_MIDDLE"] = indicator_obj.mid[0] if hasattr(indicator_obj, 'mid') else indicator_obj.lines.mid[0]
                        context[f"{ind_key_str}_LOWER"] = indicator_obj.bot[0] if hasattr(indicator_obj, 'bot') else indicator_obj.lines.bot[0]

                    elif ind_type == "STOCH":
                        context[f"{ind_key_str}_K"] = indicator_obj.percK[0] if hasattr(indicator_obj, 'percK') else indicator_obj.lines.percK[0]
                        context[f"{ind_key_str}_D"] = indicator_obj.percD[0] if hasattr(indicator_obj, 'percD') else indicator_obj.lines.percD[0]

                    else:
                        context[ind_key_str] = indicator_obj[0]


            for (cross_data, cross_key_str), crossover_obj in self.crossovers.items():
                if cross_data == data and not pd.isna(crossover_obj[0]):
                    context[cross_key_str] = crossover_obj[0]

            if atr_val is None or atr_val <= 0:
                continue

            for rule in self.config.get("rules", []):
                try:
                    rule_condition_met = eval(rule["condition"], {}, context)
                except Exception as e:
                    self.log(f"ERROR evaluating {data_name} rule '{rule['condition']}': {e}")
                    continue

                if rule_condition_met:
                    signal = rule["signal"].upper()
                    self.log(f"{data_name} Rule MET: '{rule['condition']}' -> {signal}")

                    dollar_risk = self.broker.getvalue() * self.p.risk_per_trade_pct
                    stop_loss_dollar_distance = atr_val * self.p.stop_loss_atr_multiplier

                    target_shares = 0
                    if stop_loss_dollar_distance > 0 and current_close > 0:
                        calculated_shares = dollar_risk / stop_loss_dollar_distance
                        target_shares = max(1, int(calculated_shares))
                    else:
                        continue

                    num_active_positions = sum(1 for p in self.positions if self.positions[p].size != 0)

                    if signal == "BUY":
                        if current_position.size == target_shares: continue
                        if num_active_positions >= self.p.max_positions and current_position.size == 0: continue
                        cash_needed = current_close * target_shares
                        if self.broker.getcash() < cash_needed: continue
                        self.orders[data_name] = self.order_target_size(data=data, target=target_shares)

                    elif signal == "SELL":
                        if current_position.size == 0: continue
                        self.orders[data_name] = self.order_target_size(data=data, target=0)

                    elif signal == "SHORT":
                        if current_position.size == -target_shares: continue
                        if num_active_positions >= self.p.max_positions and current_position.size == 0: continue
                        self.orders[data_name] = self.order_target_size(data=data, target=-target_shares)


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
    total_return_pct = round((final_value - start_cash) / start_cash * 100, 2) if start_cash != 0 else 0.0

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

        
    # --- Advanced Metrics Calculation (Annual Return, Alpha, Beta) ---
    alpha = 0.0
    beta = 0.0
    annual_return_pct = 0.0

    if len(first_strategy.portfolio_values) > 1 and not benchmark_df.empty:
        # Convert portfolio values to a pandas Series
        strategy_pnl_series = pd.Series(
            first_strategy.portfolio_values,
            index=pd.to_datetime(first_strategy.dates)
        )

        # Remove duplicate dates
        strategy_pnl_series = strategy_pnl_series[~strategy_pnl_series.index.duplicated(keep="last")]

        # Reindex to match benchmark dates (forward-fill gaps)
        strategy_pnl_series = strategy_pnl_series.reindex(benchmark_df.index, method="ffill")

        # Daily returns
        strategy_daily_returns = strategy_pnl_series.pct_change().dropna()
        benchmark_daily_returns = benchmark_df['close'].pct_change().dropna()

        # Align dates
        common_dates = strategy_daily_returns.index.intersection(benchmark_daily_returns.index)
        if len(common_dates) > 2:
            strategy_returns_aligned = strategy_daily_returns.loc[common_dates]
            benchmark_returns_aligned = benchmark_daily_returns.loc[common_dates]

            # === Annual Return ===
            strategy_annual_return = strategy_returns_aligned.mean() * 252
            benchmark_annual_return = benchmark_returns_aligned.mean() * 252
            annual_return_pct = round(strategy_annual_return * 100, 2)

            # === Beta ===
            covariance_matrix = np.cov(strategy_returns_aligned, benchmark_returns_aligned)
            var_benchmark = covariance_matrix[1, 1]
            if var_benchmark != 0:
                beta = covariance_matrix[0, 1] / var_benchmark

            # === Alpha ===
            risk_free_rate = first_strategy.p.risk_free_rate
            alpha = strategy_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))




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
