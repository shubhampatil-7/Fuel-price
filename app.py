import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
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
    html.H1("Fuel Price Analysis", style={
        'text-align': 'center',
        'margin-top': '20px',
        'color': '#2176FF',  # Changed to blue to match the theme
        'padding': '10px',
        'font-size': '3em',  # Larger font size for the heading
    }),
    
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': id_to_name[pid], 'value': pid} for pid in product_ids],
        value=product_ids[0],
        clearable=False,
        style={
            'width': '50%',
            'margin': '20px auto',
            'color': '#084298',
            'border-radius': '5px',
            
        }
    ),
    
    dcc.Graph(
        id='price-graph',
        style={'padding': '20px'},
        config={
            'displayModeBar': False  # Hides the mode bar
        }
    ),
    
    html.Div([
        html.Button('Show All', id='button-all', n_clicks=0, className='button-all', style={
            "margin": "3px", 
            "border-radius": "12px", 
            "background-color": "#007BFF", 
            "border": "none",
            "color": "white",
            "padding": "10px 20px",
            "text-align": "center",
            "text-decoration": "none",
            "display": "inline-block",
            "font-size": "16px",
            "margin": "4px 2px",
            "cursor": "pointer"
        }),
        html.Button('Show Trend', id='button-trend', n_clicks=0, className='button-trend', style={
            "margin": "3px", 
            "border-radius": "12px", 
            "background-color": "#17A2B8", 
            "border": "none",
            "color": "white",
            "padding": "10px 20px",
            "text-align": "center",
            "text-decoration": "none",
            "display": "inline-block",
            "font-size": "16px",
            "margin": "4px 2px",
            "cursor": "pointer"
        }),
        html.Button('Show Seasonal', id='button-seasonal', n_clicks=0, className='button-seasonal', style={
            "margin": "3px", 
            "border-radius": "12px", 
            "background-color": "#DC3545", 
            "border": "none",
            "color": "white",
            "padding": "10px 20px",
            "text-align": "center",
            "text-decoration": "none",
            "display": "inline-block",
            "font-size": "16px",
            "margin": "4px 2px",
            "cursor": "pointer"
        }),
        html.Button('Show Residual', id='button-residual', n_clicks=0, className='button-residual', style={
            "margin": "3px", 
            "border-radius": "12px", 
            "background-color": "#28A745",
            "border": "none",
            "color": "white",
            "padding": "10px 20px",
            "text-align": "center",
            "text-decoration": "none",
            "display": "inline-block",
            "font-size": "16px",
            "margin": "4px 2px",
            "cursor": "pointer"
        }),
    ], style={'text-align': 'center'}),

    dcc.Loading(
        id="loading-seasonal",
        children=[
             dcc.Graph(
                id='seasonal-decomposition-graph',
                style={'padding': '20px'},
                config={
                    'displayModeBar': False  # Hides the mode bar
                }
            ),
            # dcc.Graph(id='trend-graph', style={'display': 'inline-block', 'width': '33%'}),
            # dcc.Graph(id='seasonal-graph', style={'display': 'inline-block', 'width': '33%'}),
            # dcc.Graph(id='residual-graph', style={'display': 'inline-block', 'width': '33%'}),
        ],
        type="default"
    ),

    
    
    html.Div([
        dcc.Input(
            id='forecast-weeks',
            type='number',
            placeholder='Enter Weeks Ahead',
            style={
                "height": "60px",
                'width': '200px',
                'margin': '10px auto',
                'border-radius': "5px",
                'border': '1px solid #2176FF',
                'box-shadow': '2px 2px 10px 0 rgba(33, 118, 255, 0.2)',
                'text-align': 'center'
            }
        ),
    ], style={'text-align': 'center'}),

    html.Div([
        html.Button(
            'Generate SARIMA Forecast', id='sarima-button', n_clicks=0,
            style={
                'width': '200px',
                'height': '40px',
                'margin': '10px',
                'background-color': '#28A745',
                'color': 'white',
                'border': 'none',
                'border-radius': '5px',
                'box-shadow': '2px 2px 10px 0 rgba(40, 167, 69, 0.2)',
                'cursor': 'pointer'
            }
        ),
        html.Button(
            'Generate ARIMA Forecast', id='arima-button', n_clicks=0,
            style={
                'width': '200px',
                'height': '40px',
                'margin': '10px',
                'background-color': '#DC3545',
                'color': 'white',
                'border': 'none',
                'border-radius': '5px',
                'box-shadow': '2px 2px 10px 0 rgba(220, 53, 69, 0.2)',
                'cursor': 'pointer'
            }
        ),
    ], style={'text-align': 'center'}),
    
    dcc.Loading(id="loading-sarima", children=[dcc.Graph(id='sarima-forecast-plot')], type="default", style={'margin-top': '20px'} ),
    dcc.Loading(id="loading-arima", children=[dcc.Graph(id='arima-forecast-plot')], type="default", style={'margin-top': '20px'} ),
], style={
    'padding': '50px',
    'background-color': '#e9f4ff',  # A lighter blue theme for the overall background
    'font-family': 'Arial, sans-serif'
})

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
    
    # fig.update_layout(title='Fuel Price Trends', xaxis_title='Date', yaxis_title='Price')
    # Inside your update_price_graph function
    fig.update_layout(
        title='Fuel Price Trends',
        xaxis_title='Date',
        yaxis_title='Price',
        plot_bgcolor='#e9f4ff',  # Graph's plot background color
        paper_bgcolor='#e9f4ff',  # Graph's surrounding paper area background color
    )

    return fig

# Define callback to update seasonal decomposition graph based on product selection
# @app.callback(
#     Output('seasonal-decomposition-graph', 'figure'),
#     [Input('product-dropdown', 'value')]
# )

# @app.callback(
#     Output('seasonal-decomposition-graph', 'figure'),
#     [Input('button-trend', 'n_clicks'),
#      Input('button-seasonal', 'n_clicks'),
#      Input('button-residual', 'n_clicks'),
#      Input('product-dropdown', 'value')],
#     [State('seasonal-decomposition-graph', 'figure')]
# )
# def update_seasonal_decomposition_graph(selected_product_id):
#     """Update seasonal decomposition graph based on selected product ID."""
#     product_data = cleaned_data[cleaned_data['PRODUCT_ID'] == selected_product_id]['PRICE']
#     decomposition = seasonal_decompose(product_data, model='additive', period=52)
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=product_data.index, y=trend, mode='lines', name='Trend'))
#     fig.add_trace(go.Scatter(x=product_data.index, y=seasonal, mode='lines', name='Seasonal'))
#     fig.add_trace(go.Scatter(x=product_data.index, y=residual, mode='lines', name='Residual'))
#     fig.update_layout(
#         title=f"Seasonal Decomposition (Product ID {selected_product_id})",
#         xaxis_title="Date",
#         yaxis_title="Price",
#         showlegend=True,
#         height=600,
#         width=1000,
#         plot_bgcolor='#e9f4ff',
#         paper_bgcolor='#e9f4ff',
#     )
#     return fig

@app.callback(
    Output('seasonal-decomposition-graph', 'figure'),
    [Input('button-all', 'n_clicks'),
     Input('button-trend', 'n_clicks'),
     Input('button-seasonal', 'n_clicks'),
     Input('button-residual', 'n_clicks'),
     Input('product-dropdown', 'value')]
)
def update_graph(btn_all, btn_trend, btn_seasonal, btn_residual, selected_product_id):

    colors = {
        'button-all': '#2176FF', # Blue
        'button-trend': "#17A2B8", # Green
        'button-seasonal': "#DC3545", # Red
        'button-residual': "#28A745", # Cyan
    }
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'button-all'

    product_data = cleaned_data[cleaned_data['PRODUCT_ID'] == selected_product_id]['PRICE']
    decomposition = seasonal_decompose(product_data, model='additive', period=52)
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()

    fig = go.Figure()

    if button_id == 'button-trend':
        fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend', line=dict(color=colors[button_id])))
    elif button_id == 'button-seasonal':
        fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonal', line=dict(color=colors[button_id])))
    elif button_id == 'button-residual':
        fig.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='Residual', line=dict(color=colors[button_id])))
    elif button_id == 'button-all':
        fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend'))
        fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonal'))
        fig.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='Residual'))

    fig.update_layout(
        title=f"Seasonal Decomposition (Product ID {selected_product_id})",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True,
        height=600
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
    