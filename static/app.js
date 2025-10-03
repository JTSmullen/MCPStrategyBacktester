/**
 * @fileoverview Main JavaScript for the Algorithmic Trading Backtester Frontend.
 *
 * This script handles the user interaction on the index.html page. It does the following:
 * 1. Listens for a click on the "Run Backtest" button.
 * 2. Takes the natural language strategy from the input textarea.
 * 3. Sends this strategy to the backend '/backtest' API endpoint using a POST request.
 * 4. Clears previous results and displays a "loading" message.
 * 5. Handles the JSON response from the backend:
 *    - If an error is returned, it displays the error message.
 *    - On success, it dynamically updates the HTML to display summary results and detailed performance metrics.
 * 6. Renders a portfolio equity curve using the Chart.js library.
 *
 * It maintains a global chart instance variable to properly destroy old charts before rendering new ones.
 */

// A global variable to hold the Chart.js instance. This is necessary so we can
// destroy the previous chart before rendering a new one, preventing memory leaks
// and canvas rendering issues.
let portfolioChart = null;

/**
 * Attaches a click event listener to the main backtest button.
 */
document.getElementById('backtest-btn').addEventListener('click', () => {
    // 1. Get the user's strategy input from the textarea.
    const strategy = document.getElementById('strategy-input').value;
    
    // 2. Provide immediate user feedback by clearing old results and showing a "loading" state.
    document.getElementById('summary-results').innerHTML = '<h2>Backtest Results Summary</h2><p>Running backtest...</p>';
    document.getElementById('performance-metrics').innerHTML = '<h2>Performance Metrics</h2><p>Calculating metrics...</p>';
    
    // If a chart instance already exists, destroy it to clear the canvas.
    if (portfolioChart) {
        portfolioChart.destroy();
    }

    // 3. Use the Fetch API to send the strategy to the backend.
    fetch('/backtest', {
        method: 'POST',
        headers: {
            // Tell the server we are sending JSON data.
            'Content-Type': 'application/json'
        },
        // Convert the JavaScript object into a JSON string for the request body.
        body: JSON.stringify({ strategy: strategy })
    })
    .then(response => response.json()) // Parse the JSON response from the server.
    .then(data => {
        // 4. Handle the response data.
        // Check if the backend returned an error.
        if (data.error) {
            document.getElementById('summary-results').innerHTML = `<h2>Error</h2><p>${data.error}</p>`;
            document.getElementById('performance-metrics').innerHTML = ''; // Clear the metrics section on error.
            return; // Stop further execution.
        }

        // 5. Update the Summary Results section with the returned data.
        // Using toFixed(2) to format the numbers as currency.
        document.getElementById('summary-results').innerHTML = `
            <h2>Backtest Results Summary</h2>
            <p>Final Portfolio Value: $${data.final_portfolio_value.toFixed(2)}</p>
            <p>Profit/Loss (PnL): $${data.pnl.toFixed(2)}</p>
        `;
        
        // 6. Update the Performance Metrics section with detailed stats.
        const metrics = data.metrics;
        document.getElementById('performance-metrics').innerHTML = `
            <h2>Performance Metrics</h2>
            <p>Sharpe Ratio: <span id="sharpe-ratio">${metrics.sharpe_ratio}</span></p>
            <p>Total Return: <span id="total-return">${metrics.total_return_pct}</span>%</p>
            <p>Annual Return (CAGR): <span id="annual-return">${metrics.annual_return_pct}</span>%</p>
            <p>Max Drawdown: <span id="max-drawdown">${metrics.max_drawdown_pct}</span>%</p>
            <p>Max Drawdown Duration: <span id="max-drawdown-duration">${metrics.max_drawdown_duration_days}</span> days</p>
            <p>Total Trades: <span id="total-trades">${metrics.total_trades}</span></p>
            <p>Winning Trades: <span id="winning-trades">${metrics.winning_trades}</span></p>
            <p>Losing Trades: <span id="losing-trades">${metrics.losing_trades}</span></p>
            <p>Win Rate: <span id="win-rate">${metrics.win_rate_pct}</span>%</p>
            <p>Average Win: <span id="avg-win">$${metrics.average_win.toFixed(2)}</span></p>
            <p>Average Loss: <span id="avg-loss">$${metrics.average_loss.toFixed(2)}</span></p>
            <p>Alpha: <span id="alpha">${metrics.alpha.toFixed(2)}</span>%</p>
            <p>Beta: <span id="beta">${metrics.beta.toFixed(2)}</span></p>
        `;
        
        // 7. Render the new portfolio equity curve chart.
        renderPortfolioChart(data.chart_data);
    })
    .catch(error => {
        // Handle network errors or other issues with the fetch request itself.
        console.error('Fetch Error:', error);
        document.getElementById('summary-results').innerHTML = `<h2>Network Error</h2><p>An error occurred while contacting the server: ${error.message}</p>`;
        document.getElementById('performance-metrics').innerHTML = ''; // Clear metrics on error.
    });
});

/**
 * Renders the portfolio value chart using Chart.js.
 * @param {object} chartData - An object containing the data for the chart.
 * @param {string[]} chartData.dates - An array of date strings for the X-axis labels.
 * @param {number[]} chartData.values - An array of portfolio values for the Y-axis data points.
 */
function renderPortfolioChart(chartData) {
    // Get the 2D rendering context of the canvas element.
    const ctx = document.getElementById('portfolioChart').getContext('2d');
    
    // Before creating a new chart, check if an old one exists and destroy it.
    // This is crucial for preventing rendering bugs and memory leaks on subsequent backtests.
    if (portfolioChart) {
        portfolioChart.destroy();
    }

    // Create a new Chart.js instance.
    portfolioChart = new Chart(ctx, {
        type: 'line', // Define the chart type.
        data: {
            labels: chartData.dates, // X-axis labels (dates).
            datasets: [{
                label: 'Portfolio Value Over Time', // Legend label for this dataset.
                data: chartData.values, // Y-axis data points (portfolio values).
                borderColor: 'rgb(0, 123, 255)', // A strong blue for the line color.
                backgroundColor: 'rgba(0, 123, 255, 0.2)', // A light blue fill under the line.
                borderWidth: 2, // Line thickness.
                pointRadius: 0, // Hide the points on the line for a cleaner look.
                fill: true, // Enable the background fill color.
                tension: 0.2 // Apply slight smoothing to the line.
            }]
        },
        options: {
            responsive: true, // Allow the chart to resize with its container.
            maintainAspectRatio: false, // Allow the chart to have a non-standard aspect ratio.
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    },
                    ticks: {
                        autoSkip: true, // Automatically hide some labels to prevent them from overlapping.
                        maxTicksLimit: 15, // Limit the number of visible ticks on the x-axis for clarity.
                        maxRotation: 45, // Rotate labels to prevent collision.
                        minRotation: 45
                    },
                    grid: {
                        display: true, // Show vertical grid lines.
                        color: 'rgba(0, 0, 0, 0.05)' // Use light grid lines.
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Portfolio Value ($)'
                    },
                    // ** CRUCIAL **: Do not force the Y-axis to start at zero.
                    // Financial charts are more readable when the Y-axis is scaled to the data's range.
                    beginAtZero: false, 
                    ticks: {
                        // Callback to format Y-axis labels as currency.
                        callback: function(value, index, values) {
                            return '$' + value.toLocaleString();
                        }
                    },
                    grid: {
                        display: true, // Show horizontal grid lines.
                        color: 'rgba(0, 0, 0, 0.05)' // Use light grid lines.
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top', // Position the legend at the top of the chart.
                },
                tooltip: {
                    // 'index' mode shows a tooltip with data from all datasets at that x-index.
                    mode: 'index',
                    // `false` means the tooltip will appear even if the mouse isn't directly over a point.
                    intersect: false
                }
            },
            hover: {
                // 'nearest' mode finds the nearest item to the mouse pointer.
                mode: 'nearest',
                intersect: true
            }
        }
    });
}