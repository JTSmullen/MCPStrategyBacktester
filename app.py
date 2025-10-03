"""
app.py - Flask Web Server for the Algorithmic Trading Backtester

This file sets up a simple Flask web application to serve as the backend and
frontend interface for the dynamic trading strategy backtester.

It provides two main endpoints:
1.  A root endpoint ('/') to serve the main HTML page (`index.html`).
2.  A '/backtest' endpoint to receive natural language strategy descriptions,
    parse them into a machine-readable format, run the backtest, and return
    the performance results.

Dependencies:
- Flask: For creating the web server and handling HTTP requests.
- backtest.py: Contains the `run_backtest` function.
- llm_parser.py: Contains the `parse_strategy` function.
- traceback: For detailed error logging.
"""

# Import necessary modules from Flask and other project files.
from flask import Flask, request, jsonify, render_template
from backtest import run_backtest      # The core backtesting engine function
from llm_parser import parse_strategy  # The function to parse natural language to JSON
import traceback                       # For printing detailed exception information to the console

# Initialize the Flask application instance.
app = Flask(__name__)

# --- Frontend Route ---
@app.route('/')
def index():
    """
    Serves the main frontend page of the application.

    This route handles GET requests to the root URL ('/') and renders the
    'index.html' file from the 'templates' folder.

    Returns:
        Rendered HTML template for the user interface.
    """
    # Flask's render_template function automatically looks for the specified
    # file within a folder named 'templates' in the same directory.
    return render_template('index.html')

# --- Backend API Route ---
@app.route('/backtest', methods=['POST'])
def backtest():
    """
    API endpoint to run a trading strategy backtest.

    This route handles POST requests to '/backtest'. It expects a JSON payload
    containing a natural language description of a trading strategy under the
    key 'strategy'.

    It parses the natural language into a JSON configuration, runs the backtest
    with that configuration, and returns the results.

    Request Body (JSON):
        {
            "strategy": "A natural language description of the trading rules."
        }

    Returns:
        - On success (200 OK): A JSON object with the backtesting results.
        - On parsing error (400 Bad Request): A JSON error message.
        - On server error (500 Internal Server Error): A JSON error message.
    """
    try:
        # --- 1. Receive and Parse the Request ---
        # Get the natural language strategy description from the incoming JSON request.
        strategy_nl = request.json['strategy']
        
        # --- 2. Convert Natural Language to JSON Config ---
        # Call the LLM-based parser to convert the text into a structured JSON
        # configuration that our backtesting engine can understand.
        strategy_json = parse_strategy(strategy_nl)
        
        # If the parser fails to create a valid JSON, return a user-friendly error.
        if not strategy_json:
            error_message = "Could not parse the strategy. Please try a different description or be more specific."
            return jsonify({"error": error_message}), 400
        
        # --- 3. Run the Backtest ---
        # Pass the structured JSON configuration to the backtesting engine.
        results = run_backtest(strategy_json)
        
        # --- 4. Return the Results ---
        # Serialize the results dictionary to JSON and return it to the frontend.
        return jsonify(results)
    
    except Exception as e:
        # --- Robust Error Handling ---
        # If any unexpected error occurs in the backend (e.g., during the backtest),
        # log the full error traceback to the server console for debugging.
        print("--- AN ERROR OCCURRED IN /backtest ENDPOINT ---")
        traceback.print_exc()
        print("---------------------------------------------")
        
        # Return a generic 500 Internal Server Error to the client with the
        # error message, preventing sensitive details from being exposed.
        return jsonify({"error": str(e)}), 500


# --- Application Entry Point ---
if __name__ == '__main__':
    """
    This block runs when the script is executed directly (e.g., `python app.py`).
    It starts the Flask development server.
    """
    # app.run() starts the web server.
    # `debug=True` enables debug mode, which provides helpful error pages
    # and automatically reloads the server when code changes are saved.
    # This should be set to `False` in a production environment.
    app.run(debug=True)