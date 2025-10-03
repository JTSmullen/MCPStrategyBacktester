"""
llm_parser.py - Natural Language to Strategy JSON Parser

This file is responsible for interfacing with the Google Gemini Large Language Model (LLM)
to convert a user's natural language description of a trading strategy into a
structured JSON format that the backtesting engine can understand and execute.

Key functionalities include:
- Configuring the Gemini API.
- Constructing a detailed "few-shot" prompt with examples to guide the LLM.
- Sending the user's query to the model.
- Parsing, cleaning, and validating the model's JSON response.

Dependencies:
- os: To access environment variables for the API key.
- json: To load the JSON string from the LLM into a Python dictionary.
- google.generativeai: The official Google AI Python SDK.
- re: For using regular expressions to reliably extract JSON from the model's output.
"""

import os
import json
import google.generativeai as genai
import re

def extract_json(text: str) -> str:
    """
    Finds and extracts the first JSON object from a given string.

    LLM responses can sometimes include explanatory text or markdown formatting
    (like ```json ... ```) around the JSON object. This function uses a regular
    expression to robustly isolate only the JSON part of the string.

    Args:
        text (str): The string potentially containing a JSON object.

    Returns:
        str: The extracted JSON string.

    Raises:
        ValueError: If no substring resembling a JSON object is found.
    """
    # Use a non-greedy regex to find the content between the first '{' and the last '}'.
    # re.DOTALL allows '.' to match newline characters, which is crucial for multi-line JSON.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        # Return the matched JSON string.
        return match.group(0)
    # If no match is found, raise an error.
    raise ValueError("No JSON object found in model output.")

def normalize_conditions(strategy_json: dict) -> dict:
    """
    Normalizes the 'condition' strings within the strategy rules to be valid
    Python expressions.

    This function replaces common uppercase logical operators (AND, OR) with their
    lowercase Python equivalents (`and`, `or`) that are required by the `eval()`
    function in the backtesting strategy.

    Args:
        strategy_json (dict): The parsed strategy dictionary from the LLM.

    Returns:
        dict: The strategy dictionary with its condition strings normalized.
    """
    if "rules" in strategy_json:
        for rule in strategy_json["rules"]:
            cond = rule.get("condition", "")
            if cond:
                # Replace common logical operators with their Python equivalents.
                cond = cond.replace(" OR ", " or ")
                cond = cond.replace(" AND ", " and ")
                cond = cond.replace(" NOT ", " not ")
                # Update the condition in the dictionary after stripping whitespace.
                rule["condition"] = cond.strip()
    return strategy_json

# --- Configure the Gemini API ---
# This block attempts to configure the generative AI library using an API key.
try:
    # It's best practice to store API keys in environment variables rather than hardcoding them.
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        # Raise an error if the environment variable is not set.
        raise ValueError("GOOGLE_API_KEY environment variable not found.")
    
    # Configure the genai library with the retrieved API key.
    genai.configure(api_key=api_key)
except Exception as e:
    # Handle potential errors during configuration (e.g., invalid key).
    print(f"Error configuring Generative AI: {e}")
    # In a production application, you might want to implement more robust error
    # handling, such as logging the error and exiting gracefully.

def parse_strategy(strategy_nl: str) -> dict:
    """
    Uses the Gemini LLM to parse a natural language strategy into a structured JSON object.

    This function constructs a detailed prompt that tells the model its role,
    provides the exact JSON structure to follow, and gives it several examples
    ("few-shot prompting"). It then sends the user's natural language query,
    processes the response, and returns a clean dictionary.

    Args:
        strategy_nl (str): The user's trading strategy in plain English.

    Returns:
        dict: A dictionary representing the structured strategy, or an empty
              dictionary if parsing fails for any reason.
    """
    
    # Select the specific Gemini model to use. 'gemini-2.5-flash' is fast and capable.
    model = genai.GenerativeModel('gemini-2.5-flash')

    # --- Prompt Engineering ---
    # A detailed, structured prompt is crucial for getting reliable and correctly
    # formatted output from the LLM.
    prompt = f"""
    You are an expert financial analyst who translates natural language trading strategies into a structured JSON format. Analyze the user's request and convert it into the specified JSON structure.

    **JSON Structure Specification:**
    {{
      "name": "StrategyName",
      "indicators": [
        {{"type": "SMA", "window": INTEGER}},
        {{"type": "EMA", "window": INTEGER}},
        {{"type": "RSI", "period": INTEGER}},
        {{"type": "ATR", "period": INTEGER}},
        {{"type": "MACD", "fast": INTEGER, "slow": INTEGER, "signal": INTEGER}},
        {{"type": "BBANDS", "period": INTEGER, "devfactor": FLOAT}},
        {{"type": "STOCH", "period": INTEGER, "percK_period": INTEGER, "percD_period": INTEGER}}
      ],
      "rules": [
        {{"signal": "BUY", "condition": "EXPRESSION"}},
        {{"signal": "SELL", "condition": "EXPRESSION"}}
      ]
    }}

    **Important Condition Expression Rules:**
    - Reference indicators using their type and parameters, like:
        - `SMA_20` (for SMA with window 20)
        - `EMA_50` (for EMA with window 50)
        - `RSI_14` (for RSI with period 14)
        - `ATR_14` (for ATR with period 14)
        - `MACD_12_26_9_MACD` (for the MACD line)
        - `MACD_12_26_9_SIGNAL` (for the MACD Signal line)
        - `MACD_12_26_9_HIST` (for the MACD Histogram)
        - `BBANDS_20_2_UPPER` (for Bollinger Bands Upper band, period 20, devfactor 2.0)
        - `BBANDS_20_2_MIDDLE` (for Bollinger Bands Middle band)
        - `BBANDS_20_2_LOWER` (for Bollinger Bands Lower band)
        - `STOCH_14_3_3_K` (for Stochastic %K line, period 14, %K period 3, %D period 3)
        - `STOCH_14_3_3_D` (for Stochastic %D line)
    - You can also use `Close` to refer to the current closing price and `Close(-1)` for the previous close.
    - Always include one BUY rule and one SELL rule.
    - Ensure all indicators referenced in 'rules' are also present in the 'indicators' list.

    **Example 1: SMA/EMA Crossover**
    User Request: "Buy when the 20-day SMA crosses above the 50-day EMA, sell when it crosses below."
    JSON Output:
    {{
      "name": "SMA_EMA_Crossover",
      "indicators": [
        {{"type": "SMA", "window": 20}},
        {{"type": "EMA", "window": 50}}
      ],
      "rules": [
        {{"signal": "BUY", "condition": "SMA_20 > EMA_50"}},
        {{"signal": "SELL", "condition": "SMA_20 < EMA_50"}}
      ]
    }}

    **Example 2: RSI Overbought/Oversold**
    User Request: "Go long when RSI(14) drops below 30 and close the position when RSI(14) goes above 70."
    JSON Output:
    {{
      "name": "RSI_Overbought_Oversold",
      "indicators": [
        {{"type": "RSI", "period": 14}}
      ],
      "rules": [
        {{"signal": "BUY", "condition": "RSI_14 < 30"}},
        {{"signal": "SELL", "condition": "RSI_14 > 70"}}
      ]
    }}

    **Example 3: MACD Crossover**
    User Request: "Buy when the MACD line crosses above the Signal line (12, 26, 9 setup). Sell when it crosses below."
    JSON Output:
    {{
      "name": "MACD_Crossover",
      "indicators": [
        {{"type": "MACD", "fast": 12, "slow": 26, "signal": 9}}
      ],
      "rules": [
        {{"signal": "BUY", "condition": "MACD_12_26_9_MACD > MACD_12_26_9_SIGNAL"}},
        {{"signal": "SELL", "condition": "MACD_12_26_9_MACD < MACD_12_26_9_SIGNAL"}}
      ]
    }}

    **Example 4: Bollinger Bands with Close Price**
    User Request: "Enter a trade if the Close price falls below the lower Bollinger Band (20, 2.0). Exit if the Close goes above the upper band."
    JSON Output:
    {{
      "name": "BBANDS_Entry_Exit",
      "indicators": [
        {{"type": "BBANDS", "period": 20, "devfactor": 2.0}}
      ],
      "rules": [
        {{"signal": "BUY", "condition": "Close < BBANDS_20_2_LOWER"}},
        {{"signal": "SELL", "condition": "Close > BBANDS_20_2_UPPER"}}
      ]
    }}

    **Example 5: ATR Breakout**
    User Request: "Buy if the current close price is greater than the previous close plus the 14-period ATR. Sell if the close is less than the previous close minus the 14-period ATR."
    JSON Output:
    {{
      "name": "ATR_Breakout",
      "indicators": [
        {{"type": "ATR", "period": 14}}
      ],
      "rules": [
        {{"signal": "BUY", "condition": "Close > (Close(-1) + ATR_14)"}},
        {{"signal": "SELL", "condition": "Close < (Close(-1) - ATR_14)"}}
      ]
    }}

    **Example 6: Stochastic Crossover**
    User Request: "Buy when %K (14,3,3) crosses above %D. Sell when %K crosses below %D."
    JSON Output:
    {{
      "name": "Stochastic_Crossover",
      "indicators": [
        {{"type": "STOCH", "period": 14, "percK_period": 3, "percD_period": 3}}
      ],
      "rules": [
        {{"signal": "BUY", "condition": "STOCH_14_3_3_K > STOCH_14_3_3_D"}},
        {{"signal": "SELL", "condition": "STOCH_14_3_3_K < STOCH_14_3_3_D"}}
      ]
    }}

    **User Request to Process:**
    User Request: "{strategy_nl}"
    JSON Output:
    """

    try:
        # --- Send the request to the Gemini API ---
        response = model.generate_content(prompt)
        
        # --- Process the Response ---
        # 1. Clean the raw text response, removing markdown code fences.
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        # 2. Reliably extract the JSON object from the cleaned text.
        json_text = extract_json(cleaned_response)
        
        # 3. Load the JSON string into a Python dictionary.
        strategy_json = json.loads(json_text)
        
        # 4. Normalize the condition strings to ensure they are valid Python syntax.
        strategy_json = normalize_conditions(strategy_json)
        
        # 5. Return the final, clean dictionary.
        return strategy_json

    except Exception as e:
        # --- Robust Error Handling ---
        # If any part of the process fails (API call, JSON extraction, parsing),
        # log the error and return an empty dictionary to signal failure.
        print(f"--- ERROR PARSING STRATEGY WITH GEMINI ---")
        print(f"Error: {e}")
        # Log the full exception traceback for detailed debugging in the console.
        import traceback
        traceback.print_exc()
        print("------------------------------------------")
        return {}