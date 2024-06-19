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

from config import colors_config, card_config

# =============================================================================
# This is the first page the user will see.
# It contains an overview of the results of the running year where nested dbc Rows and Cols are used to control the placement of the components.
# Starting on the left we have a block of dbc Cards which contain interactive information based on the time period the user is selecting.
# In the middle there is a large interactive graph where on the top the user can choose the time period either through a date picker or a button.
# On the top right there is a column with a brief explanation of the different strategies one can choose and below it is a target graph.
# The bottom row than has graphs per month and a multiplot with different statistics.
# 
# =============================================================================

# Required so the dash multipager will take this page on. As it is the first page we have to add "path='/'"
dash.register_page(__name__)

# downloading data containing all individual stock trades for the running year
fname = 'dataDT_daash.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['datetime'], index_col = 'datetime')
df = df[df.index < '01-01-2024']

# some definitions for readability
#table1_columns = [(2022, 'Q1', 'P/L'), (2022, 'Q1', 'Sharpe'), (2022, 'Q2', 'P/L'), (2022, 'Q2', 'Sharpe'), (2022, 'Q3', 'P/L'), (2022, 'Q3', 'Sharpe'), (2022, 'Q4', 'P/L'), (2022, 'Q4', 'Sharpe'), (2023, 'Q1', 'P/L'), (2023, 'Q1', 'Sharpe'), (2023, 'Q2', 'P/L'), (2023, 'Q2', 'Sharpe'), (2023, 'Q3', 'P/L'), (2023, 'Q3', 'Sharpe'), (2023, 'Q4', 'P/L'), (2023, 'Q4', 'Sharpe')]
table1_columns = ['Year','Q1','Q2','Q3','Q4']

background_img = 'linear-gradient(to left, rgba(0,0,0,1), rgba(4,104,125,0.9))'
card_title_img = 'linear-gradient(to left, rgba(1,139,180,0.75), rgba(0,0,0,1))'

# Define a function to create a titled card
def create_titled_card(title, content, color, height=None):
    return dbc.Card(
        [
            dbc.CardHeader(title, style={'background-image': color, 'color': 'white'}),
            dbc.CardBody(content, style={})
        ], style={'height': height}
    )

# Define the rounding function
def round_to_quarter(value):
    return round(value * 4) / 4

def generate_performance_plot(df):
    
    fig = px.line(df, x=df.index, y=df.cr_plus,
                  title = '<b>Growth of Capital</b>')
    
    fig.update_layout(
                        plot_bgcolor= '#000000',
                        paper_bgcolor = '#FFFFFF',
                        font_color = '#025E70',
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':50, 'b':10, 'pad':10},
                        title = {'x':0.5, 'y':0.98, 'font':{'size':16}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'Growth', 'tickformat': '.0%', 'gridcolor':'#808080'},
                        )
    
    fig.update_traces(marker_color='#209BB5')
    
    return fig

def generate_line_shaded(df):
    dfD = df.resample('D').agg({'pnl_plus':'sum', 'cr_plus':'last'})
    dfD = dfD[dfD.pnl_plus != 0]
    x = dfD.index
    y = dfD.cr_plus
    
    offset = 0.025
    y_shadow = y-0.03 * (y-1)
    y_shadow2 = y-0.06 * (y-1)
    y_shadow3 = y-0.1 * (y-1)
    y_shadow4 = y-0.15 * (y-1)
    y_shadow5 = y-0.2 * (y-1)
    y_shadow6 = y-0.25 * (y-1)
    y_shadow7 = y-0.3 * (y-1)
    y_shadow8 = y-0.4 * (y-1)
    y_shadow9 = y-0.5 * (y-1)
    # Create the line trace
    line_trace = go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#C44003', width=2),
        name='Line',
        )

    # Create the shadow trace
    shadow_trace = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y, y_shadow[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.35)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    )
    
    shadow_trace2 = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_shadow, y_shadow2[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.33)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    )
    
    shadow_trace3 = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_shadow2, y_shadow3[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.31)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    )
    shadow_trace4 = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_shadow3, y_shadow4[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.28)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    )
    shadow_trace5 = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_shadow4, y_shadow5[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.25)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    )
    shadow_trace6 = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_shadow5, y_shadow6[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.22)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    ) 
    shadow_trace7 = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_shadow6, y_shadow7[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.18)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    ) 
    shadow_trace8 = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_shadow7, y_shadow8[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.14)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    ) 
    shadow_trace9 = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_shadow8, y_shadow9[::-1]]),
        fill='toself',
        fillcolor='rgba(196, 64, 3, 0.1)',  # Adjust the transparency for fading effect
        line=dict(color='rgba(196, 64, 3, 0)', width=0),
        showlegend=False
    ) 
    # Create the figure
    fig = go.Figure()
    
    # Add the shadow trace first (so it is underneath the line)
    fig.add_trace(shadow_trace9)
    fig.add_trace(shadow_trace8)
    fig.add_trace(shadow_trace7)
    fig.add_trace(shadow_trace6)
    fig.add_trace(shadow_trace5)
    fig.add_trace(shadow_trace4)
    fig.add_trace(shadow_trace3)
    fig.add_trace(shadow_trace2)    
    fig.add_trace(shadow_trace)
    
    # Add the line trace
    fig.add_trace(line_trace)
    
    # Update layout for dark background
    # fig.update_layout(
    #                     plot_bgcolor= '#000000',
    #                     paper_bgcolor = '#FFFFFF',
    #                     font_color = '#025E70',
    #                     font_family = colors_config['colors']['font'],
    #                     margin = {'l':20, 'r':40, 't':50, 'b':10, 'pad':10},
    #                     title = {'text':'<b>Growth of Capital</b>', 'x':0.5, 'y':0.98, 'font':{'size':16}},
    #                     xaxis = {'title':'', 'showgrid':False},
    #                     yaxis = {'title':'Growth', 'tickformat': '.0%', 'showgrid':False},
    #                     yaxis2= {'title':'', 'overlaying':'y', 'side':'right', 'showgrid':False, 'tickvals':fig.layout.yaxis.tickvals},
    #                     showlegend = False,
    #                     )
    
    # Update layout for dark background and double y-axis
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#FFFFFF',
        font_color='#025E70',
        font_family='Arial',  # Replace with your font family if different
        margin={'l':20, 'r':40, 't':50, 'b':10, 'pad':10},
        title={'text':'<b>Growth of Capital</b>', 'x':0.5, 'y':0.98, 'font':{'size':16}},
        xaxis={'title':'', 'showgrid':False},
        yaxis={
            'title':'Growth',
            'tickformat': '.0%',
            'showgrid':False,
        },
        yaxis2={
            'title':'',  # Secondary y-axis title
            'overlaying':'y',  # Overlay on the same plot
            'side':'right',  # Place on the right side
            'showgrid':False,
    #        'tickvals': fig.layout.yaxis.tickvals if fig.layout.yaxis.tickvals else None  # Sync tick values
        },
        showlegend=False,
    )

    
    return fig

def generate_weekly_bars(df):
    dfW = df.resample('W').agg({'pnl_plus':'sum'})
    bars = px.bar(dfW, x=dfW.index, y=['pnl_plus'],
                  title='<b>Weekly P/L</b>')
    bars.update_layout(
                        plot_bgcolor= '#000000',
                        paper_bgcolor = '#FFFFFF',
                        font_color = '#025E70',
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':50, 'b':10, 'pad':10},
                        title = {'x':0.5, 'y':0.98, 'font':{'size':16}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'P/L', 'tickformat': '.1%', 'gridcolor':'#808080'},
                        showlegend = False
                        )  
    bars.update_traces(marker_color='#209BB5')
    
    return bars

def generate_monthly_bars(df):
    dfM = df.resample('M').agg({'pnl_plus':'sum'})
    bars = px.bar(dfM, x=dfM.index, y=['pnl_plus'],
                  title='<b>Monthly P/L</b>')
    bars.update_layout(
                        plot_bgcolor= '#000000',
                        paper_bgcolor = '#FFFFFF',
                        font_color = '#025E70',
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':50, 'b':10, 'pad':10},
                        title = {'x':0.5, 'y':0.98, 'font':{'size':16}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'P/L', 'tickformat': '.1%', 'gridcolor':'#808080'},
                        showlegend = False
                        )  
    bars.update_traces(marker_color='#209BB5')
    
    return bars

def generate_last20days(df):
    dfD = df.resample('D').agg({'pnl_plus':'sum'})
    dfD = dfD[dfD.pnl_plus != 0]
    dfD = dfD[-20:]
    bars = px.bar(dfD, x=['pnl_plus'], y=dfD.index, orientation='h',
                  title='<b>P/L Last 20 trading days</b>')
    bars.update_layout(
                        plot_bgcolor= '#000000',
                        paper_bgcolor = '#FFFFFF',
                        font_color = '#025E70',
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':50, 'b':10, 'pad':10},
                        title = {'x':0.5, 'y':0.98, 'font':{'size':16}},
                        yaxis = {'title':'', 'gridcolor':'#808080'},
                        xaxis = {'title':'P/L', 'tickformat': '.1%', 'gridcolor':'#808080'},
                        showlegend = False,
                        )  
    bars.update_traces(marker_color='#177B90')
    
    return bars

def generate_table(df):
    dfc = df[df.pnl_plus != 0][['buy_open','buy_close','sell_open','sell_close', 'pnl_plus']][-100:]
    dfc['B/S'] = 0
    dfc['B/S'][dfc.buy_open == 0] = 'SELL'
    dfc['B/S'][dfc.sell_open == 0] = 'BUY'
    dfc['open'] = 0
    dfc['open'][dfc.buy_open == 0] = dfc.sell_open
    dfc['open'][dfc.sell_open == 0] = dfc.buy_open
    dfc['close'] = 0
    dfc['close'][dfc.buy_open == 0] = dfc.sell_close
    dfc['close'][dfc.sell_open == 0] = dfc.buy_close
    dfc['date'] = dfc.index.date
    
    dftable = dfc[['date','B/S', 'open', 'close', 'pnl_plus']]
    dftable['close'] = dftable['close'].apply(round_to_quarter)
    dftable['pnl_plus'] = dftable['pnl_plus'].map(lambda x: f"{x:.2%}")
#    dftable.rename({'pnl_plus':'P/L'})
    dftable = dftable.sort_index(ascending=False)
    
    return dftable

def calculate_sharp(df):
    sharpe = Sharpe(df)
    return pd.Series({'Sharpe': sharpe})

def calculate_drawdown(df):
    dd = DrawDown(df)
    return pd.Series({'DrawDown': dd})

def datetime_to_quarter_str(date):
    quarter = (date.month - 1) // 3 + 1
    return f'Q{quarter}'

def generate_QPtable(df):
    #Calculates table for Quarterly performance per Year
    dfQ = df.resample('Q').agg({'pnl_plus':'sum'})
    
    # Extract years and quarters
    dfQ['Year'] = dfQ.index.year
    dfQ.reset_index(inplace=True)
    dfQ['Quarter'] = dfQ['datetime'].apply(datetime_to_quarter_str)

    dfQ['pnl_plus'] = dfQ['pnl_plus'].map(lambda x: f"{x:.2%}")
    
    dftable = dfQ.pivot(index='Year', columns='Quarter', values='pnl_plus')
    dftable.reset_index(inplace=True)

    return dftable

def generate_QStable(df):
    #Calculates table for Quarterly Sharpe per Year
    dfQ = df.resample('Q').agg({'pnl_plus':'sum'})
    dfQ['Sharpe'] = df['pnl_plus'].resample('Q').apply(calculate_sharp)
    # Extract years and quarters
    dfQ['Year'] = dfQ.index.year
    dfQ.reset_index(inplace=True)
    dfQ['Quarter'] = dfQ['datetime'].apply(datetime_to_quarter_str)
    
    dftable = dfQ.pivot(index='Year', columns='Quarter', values='Sharpe')
    dftable.reset_index(inplace=True)

    return dftable

def generate_QDDtable(df):
    #Calculate table for Quarterly Drawdown per year
    dfQ = df.resample('Q').agg({'pnl_ac':'sum'})
    dfQ['DrawDown'] = df['pnl_ac'].resample('Q').apply(calculate_drawdown)
    # Extract years and quarters
    dfQ['Year'] = dfQ.index.year
    dfQ.reset_index(inplace=True)
    dfQ['Quarter'] = dfQ['datetime'].apply(datetime_to_quarter_str)
    print(dfQ)
    dftable = dfQ.pivot(index='Year', columns='Quarter', values='DrawDown')
    dftable.reset_index(inplace=True)
    print(dftable)
    return dftable    
    


def generate_violin(df):
    df = df[df.pnl_plus != 0]
    violin = px.violin(df, y='pnl_plus', box=True, points='all', title='Spread of Trading Profits')
    violin.update_layout(
        yaxis_title='Trading Profits',
        xaxis_title='',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
        )
    return violin 

def generate_histo(df):
    df = df[df.pnl_plus != 0]
    fig = px.histogram(df, x='pnl_plus', nbins=80, title='<b>Distribution of Trading Profits</b>')
    fig.update_layout(
                        plot_bgcolor= '#000000',
                        paper_bgcolor = '#FFFFFF',
                        font_color = '#025E70',
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':50, 'b':10, 'pad':10},
                        title = {'x':0.5, 'y':0.98, 'font':{'size':16}},
                        yaxis = {'title':'', 'gridcolor':'#808080'},
                        xaxis = {'title':'P/L', 'tickformat': '.1%', 'gridcolor':'#808080'},
                        showlegend = False,
                        bargap = 0.1,
                        )  
    fig.update_traces(marker_color='#177B90')
    return fig

#helper functions for metrics
def Performance(pnl):
    """ Calculate the total return """
    return "{:.2%}".format(pnl.sum())
def Sharpe(pnl):
    """ Calculate annualised Sharpe Ratio """
    # Resample the pnl series to daily frequency and sum the values for each day
    pnlD = pnl.resample('D').sum()
    
    # Filter out days with zero profit/loss
    pnlD = pnlD[pnlD != 0]
    
    # Calculate the daily return mean and standard deviation
    daily_return_mean = pnlD.mean()
    daily_return_std = pnlD.std()
    
    # Calculate the annualized Sharpe Ratio
    sharpe = (daily_return_mean * 252) / (daily_return_std * (252 ** 0.5))
    return round(sharpe, 2)

def WinRate(pnl):
    """ Calculate the winners vs the losers """
    winrate = len(pnl[pnl > 0]) / len(pnl[pnl != 0])
    return "{:.2%}".format(winrate)
def ProfitRatio(pnl):
    """ Calculate the average profitable trades vs the average losing trade """
    profitratio = -pnl[pnl > 0].mean() / pnl[pnl < 0].mean()
    return round(profitratio, 2)
def Windays(pnl):
    pnlD = pnl.resample('D').agg({'pnl_plus':'sum'})
    winrate = len(pnlD[pnlD.pnl_plus > 0]) / len(pnlD[pnlD.pnl_plus != 0])
    return "{:.2%}".format(winrate)
def Winmonths(pnl):
    pnlM = pnl.resample('M').agg({'pnl_plus':'sum'})
    winrate = len(pnlM[pnlM.pnl_plus > 0]) / len(pnlM)
    return "{:.2%}".format(winrate)

def DrawDown(pnl):
    """Calculate drawdown, or the max losing streak, given a return series."""
    wealth_index = 1000 * (1 + pnl).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdownser = (wealth_index - previous_peaks) / previous_peaks
    drawdown = drawdownser.min()
    return "{:.2%}".format(drawdown)

def MaxWinDay(pnl):
    """Calculate best day"""
    pnlD = pnl.resample('D').agg({'pnl_plus':'sum'})
    return "{:.2%}".format(pnlD.pnl_plus.max())

def MaxLossDay(pnl):
    """Calculate worst day"""
    pnlD = pnl.resample('D').agg({'pnl_plus':'sum'})
    return "{:.2%}".format(pnlD.pnl_plus.min())

def AvgTrades(df):
    trades = len(df[df.pnl != 0])
    dfD = df.resample('D').agg({'pnl_plus':'sum'})
    dfD = dfD[dfD.pnl_plus != 0]
    days = len(dfD)
    return round(trades/days, 2)


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
                    create_titled_card('Choose TimePeriod',
                                           html.Div([
                                               dbc.Button('2022', id='y2022', n_clicks=0, style={'margin-left': '0px'}),
                                               dbc.Button('2023', id='y2023', n_clicks=0, style={'margin-left': '12px'}),
                                               dbc.Button('TOTAL', id='total', n_clicks=0, style={'margin-left': '12px'}),
                                                 ]),
                                           card_title_img),
                    html.Br(),
                    create_titled_card('KPI',
                                               html.Div([
                                                   dbc.Row([
                                                       dbc.Col([
                                                           dbc.Card([
                                                               html.H3(id = 'perf-2'),
                                                               html.H6('Performance', style={'color':'#1F8094'}),
                                                               html.Br(),                                                           
                                                               html.H3(id = 'winrate-2'),
                                                               html.H6('% Winning Trades', style={'color':'#1F8094'}),
                                                               html.Br(),
                                                               html.H3(id = 'windays-2'),
                                                               html.H6('% Winning Days', style={'color':'#1F8094'}),
                                                               html.Br(),
                                                               html.H3(id = 'winmonths-2'),
                                                               html.H6('% Winning Months', style={'color':'#1F8094'}),
                                                               html.Br(),
                                                               html.H3(id = 'trades-2'),
                                                               html.H6('Avg Trades per day', style={'color':'#1F8094'}),
                                                               ], style = {'height':'28rem'}),
                                                           ], width = 6),
                                                       dbc.Col([
                                                           dbc.Card([
                                                               html.H3(id ='sharpe-2'),
                                                               html.H6('Sharpe Ratio', style={'color':'#1F8094'}),
                                                               html.Br(), 
                                                               html.H3(id = 'pr-2'),
                                                               html.H6('Profit Ratio', style={'color':'#1F8094'}),
                                                               html.Br(),
                                                               html.H3(id = 'dd-2'),
                                                               html.H6('Max Drawdown', style={'color':'#1F8094'}),
                                                               html.Br(),
                                                               html.H3(id = 'bestday-2'),
                                                               html.H6('Best Day', style={'color':'#1F8094'}),
                                                               html.Br(),
                                                               html.H3(id = 'worstday-2'),
                                                               html.H6('Worst Day', style={'color':'#1F8094'}),
                                                               ], style = {'height':'28rem'})
                                                           ], width = 6),
                                                       ]),
                                                   ]),                                           
                                               card_title_img),
                    html.Br(),
                    create_titled_card('Distribution Profits of Single Trades',
                                       dcc.Graph(id = 'stripper',
                                                 figure = generate_histo(df),
                                                 style = {'height':'35vh'}
                                                 ),
                                       card_title_img,
                                       height = '42vh'
                                       )
                    ], width = 3),
                dbc.Col([
                    create_titled_card('Performance', dcc.Graph(id='graph-1-2', figure = {}), card_title_img),
                    html.Br(),
                    create_titled_card('Quarterly Overview Growth of Capital', dash_table.DataTable(
                                                            id='table-1-2',
                                                            data=generate_QPtable(df).to_dict('records'),
                                                            columns=[{'name': col, 'id': col} for col in table1_columns],
                                                            style_header= {'backgroundColor': '#0E4854', 'color': 'white', 'fontWeight': 'bold'},
                                                            style_table = {'borderRadius': '10px', 'border':'4px solid #ddd'},
                                                            style_cell = {
                                                                'color': '#000000',
                                                                'font-family':'sans-serif',
                                                                },
                                                            merge_duplicate_headers=True,
          #                                                  page_size = 12,
                                                            ), card_title_img), 
                    html.Div(style={'height':'2px'}),
                    create_titled_card('Quarterly Overview Sharpe Ratio', dash_table.DataTable(
                                                            id='table-2-2',
                                                            data=generate_QStable(df).to_dict('records'),
                                                            columns=[{'name': col, 'id': col} for col in table1_columns],
                                                            style_header= {'backgroundColor': '#0E4854', 'color': 'white', 'fontWeight': 'bold'},
                                                            style_table = {'borderRadius': '10px', 'border':'4px solid #ddd'},
                                                            style_cell = {
                                                                'color': '#000000',
                                                                'font-family':'sans-serif',
                                                                },
                                                            merge_duplicate_headers=True,
          #                                                  page_size = 12,
                                                            ), card_title_img), 
                    html.Div(style={'height':'2px'}),
                    create_titled_card('Quarterly Overview Max Drawdown', dash_table.DataTable(
                                                            id='table-2-3',
                                                            data=generate_QDDtable(df).to_dict('records'),
                                                            columns=[{'name': col, 'id': col} for col in table1_columns],
                                                            style_header= {'backgroundColor': '#0E4854', 'color': 'white', 'fontWeight': 'bold'},
                                                            style_table = {'borderRadius': '10px', 'border':'4px solid #ddd'},
                                                            style_cell = {
                                                                'color': '#000000',
                                                                'font-family':'sans-serif',
                                                                },
                                                            merge_duplicate_headers=True,
          #                                                  page_size = 12,
                                                            ), card_title_img), 
                    
                    ], width=5),

                dbc.Col([
                    create_titled_card('Monthly performance', dcc.Graph(id='graph-2-2', figure = {}), card_title_img),
                    html.Br(),
                    create_titled_card('Weekly performance', dcc.Graph(id='graph-3-2', figure = {}), card_title_img),
                    ],width=4),
                ], className="equal-height", style={"margin-right": "15px", "margin-left": "15px"}),
            # ROW 2
            ], style = {'background-image': background_img,}  # Specify the path to your image file
            ) # CLOSING CARDBODY
        ), # CLOSING CARD
    ] #CLOSING children
    ) #CLOSING DIV

# =============================================================================
# # The Outputs contain the interactive elements, this page has 8 interactive elements,
# # with the multiplot containing another 8 elements
# # The Inputs are the choices the User can make, which on this page is only time period related,
# # either through a drop picker or a button.
# 
# =============================================================================
@callback(
     [
     Output('graph-1-2', 'figure'),
     Output('perf-2', 'children'),
     Output('winrate-2', 'children'),
     Output('windays-2', 'children'),
     Output('winmonths-2', 'children'),
     Output('trades-2', 'children'),
     Output('sharpe-2', 'children'),
     Output('pr-2', 'children'),
     Output('dd-2', 'children'),
     Output('bestday-2', 'children'),
     Output('worstday-2', 'children'),
     Output('graph-2-2', 'figure'),
     Output('graph-3-2', 'figure'),
     Output('table-1-2', 'data'),
     ],
     [
     Input('y2022', 'n_clicks'),
     Input('y2023', 'n_clicks'),
     Input('total', 'n_clicks'),
     ],
     )

def update_page1(y2022, y2023, total):
    
#     #update the dates according to the preferred button click
#     #also updating the titles to show the user the graphs are updating to their selection in the titles
#     # target updates for the target graph is chosen to be (a bit random) 70% of the average performance over the last 6 years
    stop = 0.25
    cost = 0.1/10000
    slip = 0.5/(19000)  # divided by value of 1 nasdaq future 

    ctx = dash.callback_context
    button_id = None
    if ctx.triggered:
   #     return dash.no_update
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Determine date range based on button click

    if button_id == 'y2022':
        start_date = '01-01-2022'
        end_date = '31-12-2022'
        plot_title = '2022'
    elif button_id == 'y2023':
        start_date = '01-01-2023'
        end_date = '31-12-2023'
        plot_title = '2023'
    elif button_id == 'total':
        start_date = '01-01-2018'
        end_date = '31-12-2023'
        plot_title = 'Total'
    else:
        start_date = '01-01-2018'
        end_date = '31-12-2023'
        plot_title = 'Total' 

    
  #  df = df
    dfc = df[(df.index >= start_date) & (df.index <= end_date)]
    

    dfc['pnl_ac'][dfc.pnl != 0] = dfc.pnl - cost - slip
    dfc['cr_ac'] = dfc.pnl_ac.cumsum() + 1
    dfc['pnl_plus'] = dfc.pnl_ac * dfc.cr_ac
    dfc['cr_plus'] = dfc.pnl_plus.cumsum() + 1
    
 #   figln = generate_performance_plot(df)
    figln = generate_line_shaded(dfc)  
  
    performance = Performance(dfc.pnl_plus)
    wr = WinRate(dfc.pnl_plus)
    wd = Windays(dfc.pnl_plus)
    wm = Winmonths(dfc.pnl_plus)
    avgtr = AvgTrades(dfc)
    sharpe = Sharpe(dfc.pnl_ac)
    pr = ProfitRatio(dfc.pnl_plus)
    dd = DrawDown(dfc.pnl_ac)
    bestday = MaxWinDay(dfc.pnl_ac)
    worstday = MaxLossDay(dfc.pnl_ac)
    
    bars = generate_monthly_bars(dfc)
    bars2 = generate_weekly_bars(dfc)

    
    table = generate_QPtable(dfc)
#     # Graphs are all created on a helper page plots_generator. All other metrics are calculated on a helper page metrics_generator.
    
#     # the order of returns should be the same as the order of Output in the callbacks.
    return [figln, performance, wr, wd, wm, avgtr, sharpe, pr, dd, bestday, worstday, bars, bars2, table.to_dict('records')]