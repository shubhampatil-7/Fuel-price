import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go


# Define product IDs and their corresponding names for the dropdown display
product_ids = [1, 2, 6]
id_to_name = {1: 'Euro-Super 95', 2: 'Automotive gas oil', 6: 'Residual fuel oil'}

# Load and preprocess data
def load_data(file_path):
    """Load CSV data and preprocess."""
    data = pd.read_csv(file_path)
    data['SURVEY_DATE'] = pd.to_datetime(data['SURVEY_DATE'], format='%d-%m-%Y')
    data.set_index('SURVEY_DATE', inplace=True)
    return data

def filter_and_clean_data(data, product_ids):
    """Filter and clean data for specified product IDs with missing week interpolation."""
    filtered_data = data[data['PRODUCT_ID'].isin(product_ids)]
    filled_df = pd.DataFrame()

    for product_id in product_ids:
        product_data = filtered_data[filtered_data['PRODUCT_ID'] == product_id]
        complete_time_range = pd.date_range(start=product_data.index.min(), end=product_data.index.max(), freq='W-MON')
        reindexed_data = product_data.reindex(complete_time_range)
        interpolated_data = reindexed_data.interpolate(method='linear')
        interpolated_data['PRODUCT_ID'] = product_id
        filled_df = pd.concat([filled_df, interpolated_data])

    return filled_df

# Load and preprocess data
file_path = './weekly_fuel_prices_all_data_from_2005_to_20221102-1.csv'
data = load_data(file_path)
cleaned_data = filter_and_clean_data(data, product_ids)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=["./assets/style.css"])

# Define app layout
# app.layout = html.Div([
#     html.H1("Fuel Price Analysis"),
#     dcc.Dropdown(
#         id='product-dropdown',
#         options=[{'label': id_to_name[pid], 'value': pid} for pid in product_ids],
#         value=product_ids[0],  # Default selected product
#         clearable=False  # Prevent clearing selection
#     ),
#     dcc.Graph(id='price-graph'),
#     dcc.Graph(id='seasonal-decomposition-graph'),
#     html.Div([
#         dcc.Input(id='forecast-weeks', type='number', placeholder='Enter Weeks Ahead'),
#         html.Button('Generate SARIMA Forecast', id='sarima-button', n_clicks=0),
#         html.Button('Generate ARIMA Forecast', id='arima-button', n_clicks=0),
#     ]),
#     dcc.Graph(id='sarima-forecast-plot'),
#     dcc.Graph(id='arima-forecast-plot')
# ])

app.layout = html.Div([
    html.H1("Fuel Price Analysis", style={'text-align': 'center', 'margin-top': '20px'}),
    
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': id_to_name[pid], 'value': pid} for pid in product_ids],
        value=product_ids[0],  # Default selected product
        clearable=False,
        style={'width': '50%', 'margin': '20px auto'}
    ),
    
    dcc.Graph(id='price-graph', style={'padding': '20px'}),
    dcc.Graph(id='seasonal-decomposition-graph', style={'padding': '20px'}),
    
    html.Div([
        dcc.Input(
            id='forecast-weeks',
            type='number',
            placeholder='Enter Weeks Ahead',
            style={'width': '200px', 'margin': '10px', 'border-radius':"3px"}
        ),
        # html.Button('Generate SARIMA Forecast', id='sarima-button', n_clicks=0,style={'width': '200px', 'margin': '10px'}),
        # html.Button('Generate ARIMA Forecast', id='arima-button', n_clicks=0, style={'width': '200px', 'margin': '10px'}),
    ], style={'text-align': 'center'}),

    html.Div([
        html.Button('Generate SARIMA Forecast', id='sarima-button', n_clicks=0,style={'width': '200px', 'margin': '10px'}),
        html.Button('Generate ARIMA Forecast', id='arima-button', n_clicks=0, style={'width': '200px', 'margin': '10px'}),
    ], style={'text-align': 'center'}),
    
    
    # Loading component for ARIMA forecast plot
    dcc.Loading(
        id="loading-sarima",
        children=[dcc.Graph(id='sarima-forecast-plot')],
        type="default",
    ),

    dcc.Loading(
        id="loading-arima",
        children=[dcc.Graph(id='arima-forecast-plot')],
        type="default",
    ),

    # dcc.Graph(id='arima-forecast-plot')
], style={'padding': '50px', 'background-color': '#f5f5f5'})

# Define callback to update price graph based on product selection
@app.callback(
    Output('price-graph', 'figure'),
    [Input('product-dropdown', 'value')]
)
def update_price_graph(selected_product_id):
    """Update price graph based on selected product ID."""
    filtered_data = cleaned_data[cleaned_data['PRODUCT_ID'] == selected_product_id]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['PRICE'], mode='lines', name='Price'))
    fig.update_layout(title='Fuel Price Trends', xaxis_title='Date', yaxis_title='Price')
    return fig

# Define callback to update seasonal decomposition graph based on product selection
@app.callback(
    Output('seasonal-decomposition-graph', 'figure'),
    [Input('product-dropdown', 'value')]
)
def update_seasonal_decomposition_graph(selected_product_id):
    """Update seasonal decomposition graph based on selected product ID."""
    product_data = cleaned_data[cleaned_data['PRODUCT_ID'] == selected_product_id]['PRICE']
    decomposition = seasonal_decompose(product_data, model='additive', period=52)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=product_data.index, y=trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=product_data.index, y=seasonal, mode='lines', name='Seasonal'))
    fig.add_trace(go.Scatter(x=product_data.index, y=residual, mode='lines', name='Residual'))
    fig.update_layout(
        title=f"Seasonal Decomposition (Product ID {selected_product_id})",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True,
        height=600,
        width=1000
    )
    return fig

# Define callback to update SARIMA forecast plot based on user inputs
@app.callback(
    Output('sarima-forecast-plot', 'figure'),
    [Input('sarima-button', 'n_clicks')],
    [Input('product-dropdown', 'value'), Input('forecast-weeks', 'value')]
)
def update_sarima_forecast_plot(n_clicks, product_id, n_weeks):
    if n_clicks > 0 and product_id is not None and n_weeks is not None:
        product_data = cleaned_data[cleaned_data['PRODUCT_ID'] == product_id]['PRICE']
        sarima_model = SARIMAX(product_data, order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
        sarima_results = sarima_model.fit()
        forecast = sarima_results.get_forecast(steps=n_weeks)
        forecast_index = pd.date_range(product_data.index[-1] + pd.Timedelta(weeks=1), periods=n_weeks, freq='W-MON')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=product_data.index, y=product_data, mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast.predicted_mean, mode='lines', name='Forecast'))
        fig.update_layout(title='Extended SARIMA Forecast', xaxis_title='Date', yaxis_title='Price', showlegend=True)
        return fig
    else:
        return {}

# Define callback to update ARIMA forecast plot based on user inputs
@app.callback(
    Output('arima-forecast-plot', 'figure'),
    [Input('arima-button', 'n_clicks')],
    [Input('product-dropdown', 'value'), Input('forecast-weeks', 'value')]
)
def update_arima_forecast_plot(n_clicks, product_id, n_weeks):
    if n_clicks > 0 and product_id is not None and n_weeks is not None:
        product_data = cleaned_data[cleaned_data['PRODUCT_ID'] == product_id]['PRICE']
        arima_model = ARIMA(product_data, order=(1, 1, 0))
        arima_results = arima_model.fit()
        forecast = arima_results.forecast(steps=n_weeks)
        forecast_index = pd.date_range(product_data.index[-1] + pd.Timedelta(weeks=1), periods=n_weeks, freq='W-MON')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=product_data.index, y=product_data, mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='ARIMA Forecast'))
        fig.update_layout(title='ARIMA Forecast', xaxis_title='Date', yaxis_title='Price', showlegend=True)
        return fig
    else:
        return {}

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
    