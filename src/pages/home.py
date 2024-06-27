# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:57:57 2024

@author: Gebruiker
"""
import pandas as pd
import numpy as np
import datetime

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
import plotly.graph_objects as go

#from config import colors_config, card_config
import helpers as hl

# =============================================================================
# This is the first page the user will see.
# It contains an overview of the results of the running year where nested dbc Rows and Cols are used to control the placement of the components.
# As a styling choice this dashboard places all elements on a card with a cardheader, the header meant to stand out
# This page has 5 interactive elemants, a linegraph, a kpi card, two barcharts and a table
# The interactivity is provided through sliders (3x) and radiobutton selection 
# 
# =============================================================================

# Required so the dash multipager will take this page on. As it is the first page we have to add "path='/'"
dash.register_page(__name__, path='/')

# downloading data containing all individual stock trades for the running year
#fname = 'dataDT_daash.csv'
fname = 'dataDTmix.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['datetime'], index_col = 'datetime')
df = df[df.index > '01-01-2024']


# some definitions for readability
table1_columns = ['date', 'B/S', 'open', 'close', 'Profit']

background_img = 'linear-gradient(to left, rgba(0,0,0,1), rgba(4,104,125,0.9))'
card_title_img = 'linear-gradient(to left, rgba(1,139,180,0.75), rgba(0,0,0,1))'

# Define a function to create a titled card
def create_titled_card(title, content, color):
    return dbc.Card(
        [
            dbc.CardHeader(title, style={'background-image': color, 'color': 'white'}),
            dbc.CardBody(content, style={})
        ]
    )


# Here we start with the page layout
layout = html.Div(
            style={
                'background-image': background_img,  # Specify the path to your image file
                'background-size': 'cover',  # Cover the entire container
                'background-position': 'center',  # Center the background image
  #              'height': '100vh',  # Set the height to full viewport height
            },
    children = [
    dbc.Card(
        dbc.CardBody([
            # ROW 1
            dbc.Row([
                dbc.Col([
                    create_titled_card('Adjustable Parameter Sliders',
                                           html.Div([
                                                html.H5("Adjust Cutoff to Pause Trading"),
                                                dcc.Slider(
                                                    id='stop-slider',
                                                    min=0,
                                                    max=1,
                                                    step=0.1,
                                                    value=0.5,
                                       #             marks={i: str(i) for i in range(0.15, 0.4)},
                                                    ),
                                                html.Hr(),
                                                html.H5("Adjust Execution Cost (bps)"),
                                                dcc.Slider(
                                                    id='cost-slider',
                                                    min=0,
                                                    max=0.5,
                                                    step=0.1,
                                                    value=0.1,
                                                    ), 
                                                html.Hr(),
                                                html.H5("Adjust Slippage"),
                                                dcc.Slider(
                                                    id='slip-slider',
                                                    min=0,
                                                    max=3,
                                                    step=0.5,
                                                    value=0.5,
                                                    ),
                                                 ]),
                                           card_title_img),
                    html.Div(style = {'height':'5px'}),
                    create_titled_card('Choose Trading Sessions',
                                              dbc.Card(
                                                  dbc.CardBody([
                                                      html.H5("Select TradingHours"),
                                                      dcc.RadioItems(
                                                          id='selection-radio',
                                                          options=[
                                                              {'label': 'Full 23h', 'value': 'opt1'},
                                                              {'label': 'US + EU session', 'value': 'opt2'},
                                                              {'label': 'US session', 'value': 'opt3'}
                                                              ],
                                                          value='opt1',  # default value
                                                          labelStyle={'display': 'inline-block', 'margin-left':'3px', 'margin-right': '20px'}
                                                          ),
                                                      ]
                                                      )                                      
                                                  ), card_title_img),
                    ], width = 3),
                dbc.Col(create_titled_card('YTD Performance', dcc.Graph(id='graph-1', figure = {}), card_title_img), width=6),
                dbc.Col(create_titled_card('KPI',
                                           html.Div([
                                               dbc.Row([
                                                   dbc.Col([
                                                       dbc.Card([
                                                           html.H3(id = 'perf'),
                                                           html.H6('Performance', style={'color':'#1F8094'}),
                                                           html.Br(),                                                           
                                                           html.H3(id = 'winrate'),
                                                           html.H6('% Winning Trades', style={'color':'#1F8094'}),
                                                           html.Br(),
                                                           html.H3(id = 'windays'),
                                                           html.H6('% Winning Days', style={'color':'#1F8094'}),
                                                           html.Br(),
                                                           html.H3(id = 'winmonths'),
                                                           html.H6('% Winning Months', style={'color':'#1F8094'}),
                                                           html.Br(),
                                                           html.H3(id = 'trades'),
                                                           html.H6('Avg Trades per day', style={'color':'#1F8094'}),
                                                           ], style = {'height':'28rem'}),
                                                       ], width = 6),
                                                   dbc.Col([
                                                       dbc.Card([
                                                           html.H3(id ='sharpe'),
                                                           html.H6('Sharpe Ratio', style={'color':'#1F8094'}),
                                                           html.Br(), 
                                                           html.H3(id = 'pr'),
                                                           html.H6('Profit Ratio', style={'color':'#1F8094'}),
                                                           html.Br(),
                                                           html.H3(id = 'dd'),
                                                           html.H6('Max Drawdown', style={'color':'#1F8094'}),
                                                           html.Br(),
                                                           html.H3(id = 'bestday'),
                                                           html.H6('Best Day', style={'color':'#1F8094'}),
                                                           html.Br(),
                                                           html.H3(id = 'worstday'),
                                                           html.H6('Worst Day', style={'color':'#1F8094'}),
                                                           ], style = {'height':'28rem'})
                                                       ], width = 6),
                                                   ]),
                                               ]),                                           
                                           card_title_img), width = 3),
                ], className="equal-height", style={"margin-right": "15px", "margin-left": "15px"}),
            # ROW 2
            html.Br(),
            dbc.Row([
                dbc.Col(create_titled_card('Weekly performance', dcc.Graph(id='graph-2', figure = {}), card_title_img), width=4),
                dbc.Col(create_titled_card('Last 20 days', dcc.Graph(id='graph-3', figure = {}), card_title_img), width=4),
                dbc.Col(create_titled_card('Last Trades', dash_table.DataTable(
                                                                            id='table-1',
                                                                            data=hl.generate_table(df).to_dict('records'),
                                                                            columns=[{'name': col, 'id': col} for col in table1_columns],
                                                                            style_header= {'backgroundColor': '#0E4854', 'color': 'white', 'fontWeight': 'bold'},
                                                                            style_table = {'borderRadius': '10px', 'border':'4px solid #ddd'},
                                                                            style_cell = {
                                                                                'color': '#000000',
                                                                                'font-family':'sans-serif',
                                                                                },
                                                                            page_size = 12,
                                                                            ), card_title_img), width=4),
                ], style={"margin-right": "15px", "margin-left": "15px"}),  # CLOSING ROW 2
            ], style = {'background-image': background_img,}  # Specify the path to your image file
            ) # CLOSING CARDBODY
        ), # CLOSING CARD
    ] #CLOSING children
    ) #CLOSING DIV

# =============================================================================
# # The Outputs contain the interactive elements, this page has 5 interactive elements,
# # 
# # The Inputs are the choices the User can make,
# # either through a slider or a radio-button.
# 
# =============================================================================
@callback(
     [
     Output('graph-1', 'figure'),
     Output('perf', 'children'),
     Output('winrate', 'children'),
     Output('windays', 'children'),
     Output('winmonths', 'children'),
     Output('trades', 'children'),
     Output('sharpe', 'children'),
     Output('pr', 'children'),
     Output('dd', 'children'),
     Output('bestday', 'children'),
     Output('worstday', 'children'),
     Output('graph-2', 'figure'),
     Output('graph-3', 'figure'),
     Output('table-1', 'data'),
     ],
     [
     Input('stop-slider', 'value'),
     Input('cost-slider', 'value'),
     Input('slip-slider', 'value'),
     Input('selection-radio', 'value')
     ],
     )

def update_page1(selected_stop, selected_cost, selected_slip, selected_period):
    
    # Redefining df to exclude days on basis of cutt_off selection
    cut_off = selected_stop / 100
    dfD = df.resample('D').agg({'pnl_ac':'sum'})
    dfD = dfD[dfD.pnl_ac.shift(1) > cut_off]
    
    excluded_dates = dfD.index.normalize()
    dff = df[~df.index.normalize().isin(excluded_dates)]
    
    # Redefine df on basis of cost, slippage and timeperiod
    cost = selected_cost/10000
    slip = selected_slip/(19000)  # divided by value of 1 nasdaq future 

    if selected_period == 'opt1':
        start_hour = 0
        end_hour = 23
    elif selected_period == 'opt2':
        start_hour = 1
        end_hour = 15
    elif selected_period == 'opt3':
        start_hour = 7
        end_hour = 15
    
    dfc = dff[(dff.index.hour >= start_hour) & (dff.index.hour <= end_hour)]
    
    dfc['pnl_ac'][dfc.pnl != 0] = dfc.pnl - cost - slip
    dfc['cr_ac'] = dfc.pnl_ac.cumsum() + 1
    dfc['pnl_plus'] = dfc.pnl_ac * dfc.cr_ac
    dfc['cr_plus'] = dfc.pnl_plus.cumsum() + 1
    
    # Generate interactive graphs and card values.
    figln = hl.generate_line_shaded(dfc)  
  
    performance = hl.Performance(dfc.pnl_plus)
    wr = hl.WinRate(dfc.pnl_plus)
    wd = hl.Windays(dfc.pnl_plus)
    wm = hl.Winmonths(dfc.pnl_plus)
    avgtr = hl.AvgTrades(dfc)
    sharpe = hl.Sharpe(dfc.pnl_plus)
    pr = hl.ProfitRatio(dfc.pnl_plus)
    dd = hl.DrawDown(dfc.pnl_plus)
    bestday = hl.MaxWinDay(dfc.pnl_plus)
    worstday = hl.MaxLossDay(dfc.pnl_plus)
    
    bars = hl.generate_weekly_bars(dfc)
    bars2 = hl.generate_last20days(dfc)
    
    table = hl.generate_table(dfc)
    
    # the order of returns should be the same as the order of Output in the callbacks.
    return [figln, performance, wr, wd, wm, avgtr, sharpe, pr, dd, bestday, worstday, bars, bars2, table.to_dict('records')]