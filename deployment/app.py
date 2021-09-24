from os import link
import flask
from flask.globals import request
from flask import Flask, render_template
# library used for prediction
import numpy as np
import pandas as pd
import pickle
# library used for insights
import json
import plotly
import plotly.express as px

app = Flask(__name__, template_folder = 'templates')

link_active = None
# render home template
@app.route('/')
def main():
    return(render_template('home.html', title = 'Home'))

# load pickle file
model = pickle.load(open('model/optimized_rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/form')
def form():
    show_prediction = False
    link_active = 'Form'
    return(render_template('form.html', title = 'Form', show_prediction = show_prediction, link_active = link_active))

@app.route('/insights')
def insights():
    link_active = 'Insights'

    df = pd.read_csv('Churn_Modelling.csv')
    df['Exited'] = np.where(df['Exited']==1, 'Yes', 'No')
    color_map = {'Yes' : '#FF033E', 'No' : '#ACE1AF'}
    
    df_sorted = df.sort_values('Exited', ascending = 'Yes')

    fig1 = px.box(df, x = 'Exited', y='Age', color='Exited', color_discrete_map=color_map)
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    dist_prod = df.groupby(['NumOfProducts', "Exited"]).count()[["CustomerId"]]
    cat_group = df.groupby(['NumOfProducts']).count()[["CustomerId"]]
    dist_prod["percentage"] = dist_prod.div(cat_group, level = 'NumOfProducts') * 100
    dist_prod.reset_index(inplace = True)
    dist_prod.columns = ['NumOfProducts', "Exited", "count", "percentage"]
    dist_prod = dist_prod.sort_values(['NumOfProducts', 'Exited'], ascending=True)

    fig2 = px.bar(
        dist_prod, x = 'NumOfProducts', y='percentage', title='Customer churn by #of Products', range_y = [0, 100],
        color='Exited', color_discrete_map=color_map, barmode="group",
        labels = {
            "NumOfProducts": "Number of Products"
        }
    )
    fig2.update_layout(legend_traceorder='reversed')
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    df['IsActiveMember'] = np.where(df['IsActiveMember']==1, 'Yes', 'No')
    dist_active = df.groupby(['IsActiveMember', "Exited"]).count()[["CustomerId"]]
    cat_group = df.groupby(['IsActiveMember']).count()[["CustomerId"]]
    dist_active["percentage"] = dist_active.div(cat_group, level = 'IsActiveMember') * 100
    dist_active.reset_index(inplace = True)
    dist_active.columns = ['IsActiveMember', "Exited", "count", "percentage"]
    dist_active = dist_active.sort_values(['IsActiveMember', 'Exited'], ascending=True)

    fig3 = px.bar(
        dist_active, x = 'IsActiveMember', y='percentage', title='Customer churn by Activity', range_y = [0, 100],
        color='Exited', color_discrete_map=color_map, barmode="group",
        labels = {
            "IsActiveMember": "Is Active Member"
        }
    )
    fig3.update_layout(legend_traceorder='reversed')
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    dist_geo = df.groupby(['Geography', "Exited"]).count()[["CustomerId"]]
    cat_group = df.groupby(['Geography']).count()[["CustomerId"]]
    dist_geo["percentage"] = dist_geo.div(cat_group, level = 'Geography') * 100
    dist_geo.reset_index(inplace = True)
    dist_geo.columns = ['Geography', "Exited", "count", "percentage"]
    dist_geo = dist_geo.sort_values(['Geography', 'Exited'], ascending=True)

    fig4 = px.bar(
        dist_geo, x = 'Geography', y='percentage', title='Customer churn by Geography', range_y = [0, 100],
        color='Exited', color_discrete_map=color_map, barmode="group"
    )
    fig4.update_layout(legend_traceorder='reversed')
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)   

    return(render_template('insights.html', title = 'Insights', link_active = link_active, graph1JSON = graph1JSON, graph2JSON = graph2JSON, graph3JSON = graph3JSON, graph4JSON = graph4JSON))

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering prediction result.
    '''
    link_active = 'Result'
    show_prediction = True

    # retrieve data
    Age = int(request.form.get('Age'))
    Gender_Male = int(request.form.get('Gender'))
    Geography = request.form.get('Geography')
    EstimatedSalary = float(request.form.get('EstimatedSalary'))
    Tenure = int(request.form.get('Tenure'))
    IsActiveMember = int(request.form.get('IsActiveMember'))
    Balance = float(request.form.get('Balance'))
    NumOfProducts = int(request.form.get('NumOfProducts'))
    HasCrCard = int(request.form.get('HasCrCard'))
    CreditScore = int(request.form.get('CreditScore'))

    # set previously known values for one-hot encoding
    known_Geography = ['France', 'Germany', 'Spain']

    # encode the categorical value
    Geography_type = pd.Series([Geography])
    Geography_type = pd.Categorical(Geography_type, categories = known_Geography)
    Geography_input = pd.get_dummies(Geography_type, prefix = 'Geography', drop_first=True)

    # concat new data
    onehot_result = list(pd.concat([Geography_input], axis = 1).iloc[0])
    new_data = [[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary] + onehot_result + [Gender_Male]]

    scaled_input = scaler.transform(new_data)
    prediction = model.predict(scaled_input)

    
    if prediction == 1:
        prediction_churn = True
    else:
        prediction_churn = False

    output = {0: 'less likely to churn', 1: 'more likely to churn'}

    return render_template('form.html', title = 'Prediction', show_prediction = show_prediction, prediction_text = 'The Customer will {}.'.format(output[prediction[0]]), link_active = link_active, prediction_churn = prediction_churn)

if __name__ == '__main__':
    app.run(debug = True)
