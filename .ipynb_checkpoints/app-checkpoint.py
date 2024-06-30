from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from scipy.stats import multivariate_normal, norm
import math

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Process-centric GPs"),
    html.H2("Bivariate Gaussian Distribution"),
    html.P("This app visualizes the bivariate Gaussian distribution based on the provided mean and covariance matrix."),
    
    html.Div([
        html.Div(dcc.Graph(id='plot'), className='graph-item'),
        html.Div(dcc.Graph(id='contour-plot'), className='graph-item'),
    ], className='graph-container'),

    html.Div([
        html.Div([
            html.Label('Mean X:', className='label-block'),
            dcc.Input(id='mean-x', type='number', value=0, step=0.1, className='input-block'),
            html.Label('Mean Y:', className='label-block'),
            dcc.Input(id='mean-y', type='number', value=0, step=0.1, className='input-block'),
        ], className='input-group'),

        html.Div([
            html.Label('Covariance Matrix:', className='label-block'),
            html.Table([
                html.Tr([html.Td(dcc.Input(id='cov-xx', type='number', value=1, step=0.1, min=0.1, style={'width': '60px'})),
                         html.Td(dcc.Input(id='cov-xy', type='number', value=0.5, step=0.1, min=-1, max=1, style={'width': '60px'}))]),
                html.Tr([html.Td(dcc.Input(id='cov-xy-duplicate', type='number', value=0.5, step=0.1, min=-1, max=1, disabled=True, style={'width': '60px'})),
                         html.Td(dcc.Input(id='cov-yy', type='number', value=1, step=0.1, min=0.1, style={'width': '60px'}))])
            ], style={'margin': '0 auto'})
        ], className='input-group'),
    ], className='input-container')
], style={'maxWidth': '1200px', 'margin': '0 auto'})

@app.callback(
    [Output('plot', 'figure'),
     Output('contour-plot', 'figure'),
     Output('cov-xy-duplicate', 'value')],
    [Input('mean-x', 'value'),
     Input('mean-y', 'value'),
     Input('cov-xx', 'value'),
     Input('cov-xy', 'value'),
     Input('cov-yy', 'value')],
    [State('plot', 'figure'),
     State('contour-plot', 'figure')]
)
def update_plot(mean_x, mean_y, cov_xx, cov_xy, cov_yy, plot_figure, contour_figure):
    # Use default values if inputs are empty
    mean_x = mean_x if mean_x is not None else 0
    mean_y = mean_y if mean_y is not None else 0
    cov_xx = cov_xx if cov_xx is not None else 1
    cov_xy = cov_xy if cov_xy is not None else 0
    cov_yy = cov_yy if cov_yy is not None else 1

    # Ensure no negative values for covariance matrix
    cov_xx = max(cov_xx, 0.1)
    cov_yy = max(cov_yy, 0.1)
    cov_xy = min(max(cov_xy, -1), 1)
        
    # Synchronize duplicate covariance input
    cov_xy_duplicate = cov_xy

    # Parameters for the Gaussian distribution
    mean = [mean_x, mean_y]
    cov = [[cov_xx, cov_xy], [cov_xy, cov_yy]]

    # Generate grid of (x, y) points
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.stack((X, Y), axis=-1)

    # Calculate the Gaussian probability density function
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)

    # Marginal distributions
    marginal_x = norm(loc=mean[0], scale=np.sqrt(cov[0][0]))
    marginal_y = norm(loc=mean[1], scale=np.sqrt(cov[1][1]))

    # Update surface plot
    if not plot_figure:
        surface_fig = go.Figure(data=[go.Surface(
            z=Z, x=X[0], y=Y[:, 0], 
            colorscale='Viridis', 
            cmin=0, cmax=Z.max()
        )])
        surface_fig.update_layout(
            title='Interactive Bivariate Gaussian Distribution',
            autosize=True,
            uirevision='true',
           scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Probability Density'
            ),
            legend=dict(
                x=-0.3,
                y=1,
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                    color='#000'
                ),
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=1
            )
        )

        # Plot marginal distributions along the X axis
        x_kde = np.linspace(-4, 4, 100)
        marginal_x_vals = marginal_x.pdf(x_kde)
        marginal_x_trace = go.Scatter3d(
            x=x_kde, 
            y=np.full_like(x_kde, -4), 
            z=marginal_x_vals, 
            mode='lines',
            line=dict(color='red', width=4),
            name='Marginal X'
        )

        # Plot marginal distributions along the Y axis
        y_kde = np.linspace(-4, 4, 100)
        marginal_y_vals = marginal_y.pdf(y_kde)
        marginal_y_trace = go.Scatter3d(
            x=np.full_like(y_kde, 4), 
            y=y_kde, 
            z=marginal_y_vals, 
            mode='lines',
            line=dict(color='blue', width=4),
            name='Marginal Y'
        )

        surface_fig.add_trace(marginal_x_trace)
        surface_fig.add_trace(marginal_y_trace)

    else:
        plot_figure['data'][0]['z'] = Z
        plot_figure['data'][0]['x'] = X[0]
        plot_figure['data'][0]['y'] = Y[:, 0]
        surface_fig = plot_figure

    contour_fig = go.Figure(data=[go.Contour(
        z=Z, x=X[0], y=Y[:, 0], 
        colorscale='Viridis', 
        showscale=True
    )])
    contour_fig.add_trace(go.Scatter(x=[mean[0]], y=[mean[1]], mode='markers', name='Mean', marker=dict(color='blue', size=12)))
    contour_fig.add_trace(go.Scatter(x=[mean[0]-math.sqrt(cov[0][0]), mean[0]+math.sqrt(cov[0][0])], 
                                     y=[mean[1], mean[1]], mode='lines+markers', name='Variance X', line=dict(color='green')))
    contour_fig.add_trace(go.Scatter(y=[mean[1]-math.sqrt(cov[1][1]), mean[1]+math.sqrt(cov[1][1])], 
                                     x=[mean[0], mean[0]], mode='lines+markers', name='Variance Y', line=dict(color='purple')))
    contour_fig.update_layout(
    title='Contour Plot of Gaussian Distribution',
    autosize=True,
    showlegend=True,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    paper_bgcolor='white',
    plot_bgcolor='white',
    uirevision='true',
    legend=dict(
        x=-0.3,
        y=1,
        traceorder='normal',
        font=dict(
            family='sans-serif',
            size=12,
            color='#000'
        ),
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='rgba(0, 0, 0, 0.5)',
        borderwidth=1
    )
)
    return surface_fig, contour_fig, cov_xy_duplicate   

if __name__ == '__main__':
    app.run_server(debug=True)
