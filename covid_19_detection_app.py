import datetime
import json
import pandas as pd
import plotly
import io
import numpy as np
from base64 import decodestring
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import base64
import cv2
import keras
from keras import backend as K
from model import get_extractor
import joblib

import tensorflow as tf
from tensorflow.keras import models

import plotly.express as px


app = dash.Dash()

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

model = keras.models.load_model("AdemNetV4")

def predict_covid(model):
    img = cv2.imread('uploaded_image.png', cv2.IMREAD_GRAYSCALE)
    gry = cv2.resize(img, (128,128))
    gry = gry.reshape((1, 128, 128, 1))/255.

    extract_layer = model.get_layer("extracted")
    grad_model = models.Model([model.inputs], [extract_layer.output, model.output])

    with tf.GradientTape() as gtape:
        extract_output, predictions = grad_model(gry)
        loss = predictions[:, np.argmax(predictions[0])]
        averaged_grads = K.mean(gtape.gradient(loss, extract_output), axis=(0, 1, 2))

    gradcam = tf.reduce_mean(tf.multiply(averaged_grads, extract_output), axis=-1)
    gradcam = np.maximum(gradcam, 0)
    max_grad = np.max(gradcam)

    epsilon = 1e-8
    gradcam /= (max_grad + epsilon)
    
    #Create GradCam figure
    def scale(x):
        """Scale heatmap and convert to RGB."""
        heat = (((x - x.min()) / x.max()) * 255.)
        heat = heat.astype(np.uint8)
        return heat
    if np.array(predictions).argmax() == 0:
        fig = px.imshow(gry.reshape(128,128), color_continuous_scale='gray')
        fig.update_layout(coloraxis_showscale=False, margin={"l": 0, "r": 0, "t": 0, "b": 0}, width=350, height=350)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(hovertemplate=None, hoverinfo='skip')
        
    else:
        alpha1 = 0.5
        alpha2 = 0.8
        gamma = 0
        reinforce = 3.0
        heatmap = scale(cv2.resize(gradcam.reshape(11,11)**3 , (128, 128)))

        heatmap = np.clip(heatmap, 0, 150)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_MAGMA)

        heatmap[100:, :] = 0

        imm = np.repeat(scale(gry.reshape(128, 128, 1)), 3, axis=2)

        layered = cv2.addWeighted(imm,
                                    alpha1,
                                    heatmap,
                                    alpha2,
                                    gamma
                                    )

        fig = px.imshow(layered)

        fig.update_layout(coloraxis_showscale=False, margin={"l": 0, "r": 0, "t": 0, "b": 0}, width=300, height=300)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(hovertemplate=None, hoverinfo='skip')

    preds = np.array(predictions[0])

    return preds, fig 


def parse_contents(contents):
    return html.Div([
                                                                                                                                                      
        html.Img(src=contents, style={'height':438, 'width':500}),
    ])

@app.callback(
            [
            Output('output-image-upload', 'children'),
            Output('uploaded-image', 'children'),
            Output('show-result-button', 'hidden')
            ],
            [Input('upload-image', 'contents')],
            State('upload-image', 'filename'),)
def update_output(images, file_name):
    if not images:
        return

    for i, image_str in enumerate(images):
        image = image_str.split(',')[1]
        data = decodestring(image.encode('ascii'))
        with open(f"uploaded_image.png", "wb") as f:
            f.write(data)

    children = [parse_contents(i) for i in images]

    return children, "Uploaded Image", False


button_information = html.Div(
                                [
                                    dbc.Button("Show me Results", id="example-button", className="mr-2", style={"background-color": "#8A2BE2", 'font-size':'20px'}),
                                    html.Span(id="example-output", style={"vertical-align": "middle"}),
                                ], style={'padding-left':'20px'}
                            )



@app.callback(
    Output("example-output", "children"), 
    [Input("example-button", "n_clicks")]
)
def on_button_click(n):
    if n is None:
        return " "
    else:
        predictions, figure = predict_covid(model)

        pred_table = pd.DataFrame({'Normal Probability':f"{round(predictions[0]*100, 2)}%",
                                    'COVID-19 Probability' : f"{round(predictions[1]*100, 2)}%", 
                                    'Pneumonia Probability' : f"{round(predictions[2]*100, 2)}%"}, index=[0])

        return html.Div(
                        [
                        html.Br(),
                        dbc.Row([
                                dbc.Table.from_dataframe(pred_table, striped=True, bordered=True, hover=True, size='md', dark=False, style={'width':"100%",
                                                                                                                                            'textAlign': 'center',} )
                                ], justify = 'center'),
                        dbc.Row([
                                html.Div(
                                            dcc.Graph(id='grad-figure', figure= figure)
                                        )
                                ], justify = 'center')
                        ]
                        )


app.layout = html.Div([
    html.P("COVID-19 Detection Application", style={'background-color':'RoyalBlue', 'color':'white', 'padding':'5px', 'font-size':'30px'}), 
    html.P("Please upload X-ray image using following section.", style={'textAlign': 'center', 'font-size':'20px'}),
    dbc.Col(
            [
                dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'textAlign': 'center',
                            },
                            # Allow multiple files to be uploaded                                                                                                                                                  
                            multiple=True
                        ),
                ],width={'size':8, 'offset':2}, style={'background-color':'#E6E6FA', 'padding':'0px'}
            ),
    html.Br(),
    html.Hr(),
    html.Div(
            [
                dbc.Row([
                    dbc.Col([  
                                html.P(id="uploaded-image",                     
                                        style={
                                                'textAlign': 'center',
                                                'font-size':'20px',
                                            },
                                    ),
                                html.Div(id='output-image-upload'),
                                
                            ], width=4),
                    dbc.Col(
                            dbc.Row(html.Div(id='show-result-button', hidden=True, children=button_information), justify='center')
                            )
                    ]),
                ],
            style={'padding-left':'20px'}
        ),
    html.Hr(),
], style={'padding':'5px 10px 10px 10px', 'background-color':'#f2f2f2'})


if __name__ == '__main__':
    app.run_server(port='8051', debug=False)