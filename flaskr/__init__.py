import os

from flask import Flask, render_template
from flask import Flask, escape, request, render_template
import pandas as pd
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy ='mean')
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    #from . import db
    #db.init_app(app)

    #from . import auth
    #app.register_blueprint(auth.bp)
    
    #from . import blog
    #app.register_blueprint(blog.bp)
    #app.add_url_rule('/', endpoint='index')
    
    #can we make an about page
    @app.route('/about')
    def about():
        return render_template("about.html")

    @app.route('/')
    def home():
        return render_template("home.html")

    #other works page
    @app.route('/work')
    def work():
        return render_template("work.html")
    #home base for predictions
    @app.route('/predictions')
    def predictions():
        return render_template("predictions.html")

    def processpredict(team , week):
    # keep all inputs in array
    #selection = [team,week]

        fileName = "/Users/braxtonbrent/Documents/DTSC-691/Flask-Proj/PLL Stats Master - Team Game Logs.csv"
        data = pd.read_csv(fileName)

        
        #data = data[data['Season'] == 2022]
        data['Avg Shot Dist'] = data['Average Shot Distance'].replace('#DIV/0!' , 10.265)
        data['Avg Shot Dist'] = data['Avg Shot Dist'].astype(float)
        data = data[['Season','Week','Game','Team','Opponent', 
                 'Shots','Goals','Efficiency', 
                 'Possession %',
                 'Settled Goal','FB Goals','Assisted Goals','Shot Quality Ratio', 
                 'D Efficiency','Turnovers', 'Score Against',
                 'Expected Goals','Avg Shot Dist','Margin',
                 'Settled Goals agaisnt' , 'Save %',
                 'Score','Result']]
        data['Shot%'] = data['Goals'] / data['Shots']
        data['W/L'] = pd.get_dummies(data.loc[:,'Result'])[['W']]

        def weekly(gameweek):
            raw = data
            s = 2022
            n = int(gameweek)
            tp = list(np.arange(n-3,n))
            raw = raw[raw['Season'] == s]
            
            xg = raw[raw['Week'].isin(tp)]
            
            conditions = [
                (xg['Week'] == n-1),
                (xg['Week'] == n-2),
                (xg['Week'] == n-3)]
            values = [40,30,30]
            xg['Weights'] = np.select(conditions, values)
            
            #label here which metrics we want, also which ones we want averaged or summed 
            
            #to be averaged
            xg1 = xg[['Team','Opponent','Shots','Goals','Efficiency','Possession %',
                    'Settled Goal','FB Goals','Assisted Goals','Shot Quality Ratio', 
                    'D Efficiency','Turnovers', 'Score Against',
                    'Expected Goals','Avg Shot Dist','Margin',
                    'Settled Goals agaisnt' ,
                    'Save %','Weights']]
            
            #wm = lambda x: np.average(x , weights = weights)
            
            xg3 = raw[raw['Week'] == n]
            xg3 = xg3[['Team', 'Opponent','Week']]
            
            #xg1 = xg1.groupby('Team').mean()
            xg1 = xg1.reindex(xg1.index.repeat(xg1['Weights'])).drop('Weights',1).groupby(['Team']).mean()
            
            xgt = xg1.merge(xg3 , how = 'inner' , right_on = 'Team' , left_on = 'Team')
            
            
            return xgt
        #run weekly to get correct week of data, then filter to user selected team only    
        user = weekly(week)
        user1 = user[user['Team'] == team]
        user2 = user1.drop(columns = ['Team','Opponent','Week'])

        opponent = user1[['Opponent']]

        scalerdata = data[['Shots','Goals','Efficiency','Possession %',
                    'Settled Goal','FB Goals','Assisted Goals','Shot Quality Ratio', 
                    'D Efficiency','Turnovers', 'Score Against',
                    'Expected Goals','Avg Shot Dist','Margin',
                    'Settled Goals agaisnt' ,
                    'Save %']]
        scale_fit = stdscaler.fit(scalerdata)

        user3 = imp.fit_transform(user2)
        userfinal = scale_fit.transform(user3)

        #opponent prediction section
        c = opponent.values[0]
        c = c[0]
        oppo = user[user['Team'] == c]
        oppo2 = oppo.drop(columns = ['Team' , 'Opponent' , 'Week'])
        oppo3 = scale_fit.transform(oppo2)


        # open file

        # load trained model
        from keras.models import load_model
        trained_model = load_model('final_model.h5')
        #trained_model.load_weights('modelweights.data-00000-of-00001')

        # predict
        prediction = trained_model.predict(userfinal)
        prediction = prediction[0].astype(float)

        oprediction = trained_model.predict(oppo3)
        oprediction = oprediction[0].astype(float)

        if prediction > oprediction:
            r = 'will beat' 
        if oprediction > prediction: 
            r = 'will lose to'
    

        return r , opponent 

    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        if request.method == "POST":
            # get form data
            team = request.form.get('team')
            week = request.form.get('week')
        
        

            # call preprocessDataAndPredict and pass inputs
            try:
                r , opponent  = processpredict(team, week)
                # pass prediction to template
                return render_template('predict.html', prediction = r , team = team , week = week , opponent = opponent.iloc[0]['Opponent'])

            except ValueError:
                return "Please Enter valid values"

            pass
        pass





    
    #@app.route('/predict')
    #def predict():
   #     return render_template("predict.html")

    return app