from flask import Flask
from flask import render_template, request

# modeling packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# create the flask object
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------------ #
# -------- DATA SCIENCE TIME --------- #
# ------------------------------------ #

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/stat')
def stat():
    return render_template('stat.html')

@app.route('/result', methods=['GET','POST'])
def result():
    data = {}   # data object to be passed back to the web page
    if request.form:
        # get the input data
        form_data = request.form
        data['form'] = form_data
        predict_sex = form_data['predict_sex']
        predict_age = float(form_data['predict_age'])
        predict_fever = float(form_data['predict_fever'])
        predict_cough = float(form_data['predict_cough'])
        predict_sputum = float(form_data['predict_sputum'])
        predict_headache = float(form_data['predict_headache'])
        predict_wuhan = float(form_data['predict_wuhan'])
        predict_china = float(form_data['predict_china'])
        predict_breath = float(form_data['predict_breath'])
        
        # convert the sex from text to binary
        if predict_sex == 'M':
            predict_sex = 0
        else:
            predict_sex = 1
        input_data = np.array([predict_sex, predict_age, predict_fever, predict_cough, predict_sputum, predict_headache, predict_wuhan, predict_china, predict_breath])
        input_data = input_data.astype(np.float64)
        print(input_data.reshape(1, -1))
#         # get prediction
        prediction = L1_logistic.predict_proba(input_data.reshape(1, -1))
        prediction = prediction[0][1] # probability of carrying coronavirus
        data['prediction'] = '{:.1f}% Chance of Carrying Coronavirus'.format(prediction * 100 - 10)
        print(data)
    return render_template('result.html', data=data)

if __name__ == '__main__':
    # build a basic model for coronavirus infection rate
    virus_df = pd.read_csv('data/final.tsv',encoding="utf-8",sep='\t')
    virus_df['sex'].replace({"male": 0, "female": 1}, inplace=True)

    train_df, test_df = train_test_split(virus_df)

    virus_df_copy = virus_df.copy()

     
    # virus_df['sex_binary'] = virus_df['predict_sex'].map({'female': 1, 'male': 0})

    # choose our features and create test and train sets
    features = [u'sex', u'age', u'fever', u'cough', u'sputum', u'headache', u'wuhan', u'china', u'breath', 'positive']
    train_df, test_df = train_test_split(virus_df)
    train_df = train_df[features].dropna()
    test_df = test_df[features].dropna()

    features.remove('positive')
    X_train = train_df[features]
    y_train = train_df['positive']
    X_test = test_df[features]
    y_test = test_df['positive']

    # fit the model
    L1_logistic = LogisticRegression(C=1.0, penalty='l1')
    y_train = np.random.randint(2,size=len(y_train))    
    
    L1_logistic.fit(X_train, y_train)

    # check the performance
    target_names = ['Negative', 'Positive']
    y_pred = L1_logistic.predict(X_test)
    # import ipdb; ipdb.set_trace()
    print(classification_report(y_test, y_pred, target_names=target_names))
    

    # start the app
    app.run(debug=True)
