from pyexpat import model
from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)
df = pd.read_csv('clean_df.csv')
model = pickle.load(open('lr_model.pkl','rb'))
@app.route('/')
def index():
    companies = sorted(df['company'].unique())
    models    = sorted(df['name'].unique())
    fule_types    = sorted(df['fuel_type'].unique())
    years = sorted(df['year'].unique(),reverse=True)
    return render_template('index.html',companies = companies,models = models , years = years,fule_types=fule_types)

@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    fule_type = request.form.get('fule_type')
    km_travel = int(request.form.get('km_travel'))
    year = int(request.form.get('year'))
    prediction = model.predict(pd.DataFrame([[car_model,company,year,km_travel,fule_type]],columns=['name','company','year','kms_driven','fuel_type']))
 
    return str(round(np.exp(prediction[0]),2))
if __name__=='__main__':
    app.run(debug=True)