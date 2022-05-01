from flask import Flask ,render_template, request, session, redirect, url_for, jsonify, flash
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
from flask_session import Session
from flask import send_file
#app = Flask(__name__)
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
 
""" @app.route("/") 
def hello_world(): 
    return render_template("index.html")
 """

@app.route("/")
def index():
  # check if the users exist or not
    if not session.get("name"):
        # if not there in the session then redirect to the login page
        return redirect("/login")
    return render_template('index.html')

@app.route("/login", methods=["POST", "GET"])
def login():
  # if form is submited
    if request.method == "POST":
        # record the user name
        session["name"] = request.form.get("name")
        # redirect to the main page
        return redirect("/")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session["name"] = None
    return redirect("/")

@app.route('/data_model', methods = ['POST'])
def data_model():
    if request.method == 'POST':
        problemTypeDic = {"1":"Classification","2":"Regression"}
        modelNameDic = {"1":"Linear Regression",
                        "2":"Logistic Regression",
                        "3":"Random Forest",
                        "4":"Decision Tree",
                        "5":"kNN",
                        "6":"SVM"}
        #title = request.form["title"]
        #firstName = request.form['nameFirst']
        #lastName = request.form['nameLast']
        email = request.form['email']
        contact = request.form['number']
        modelName = request.form['modelName']
        problemType = request.form['problemType']
        dataSet = request.files['dataSet']
        ytrainName = request.form["ytrain"]
        problem = problemTypeDic[problemType]
        model = modelNameDic[modelName]    
        #userData = [title,firstName,lastName,email,contact,problem,model]
        
        dataframe = pd.read_csv(dataSet)
        X = dataframe.loc[:, dataframe.columns != ytrainName] # Features
        y = dataframe[ytrainName]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        userId = email[:email.index("@")]
       # print(userId)
        if problem == "Classification":
            
            
            session["modelName"] = userId+"_"+"finalized_model.sav"
            if model == "Linear Regression":
                values,modeld = linearRegression(X_train, X_test, y_train, y_test)
                pickle.dump(modeld, open(session["modelName"], 'wb'))
                pass
            elif model =="Logistic Regression":
                values,modeld = logisticRegression(X_train, X_test, y_train, y_test)
                pickle.dump(modeld, open(session["modelName"], 'wb'))
                pass
            elif model == "Random Forest":
                values,modeld = randomForestClassifier(X_train, X_test, y_train, y_test)
                pickle.dump(modeld, open(session["modelName"], 'wb'))
                pass
            elif model == "Decision Tree":
                values,modeld = decisionTreeClassifier(X_train, X_test, y_train, y_test)
                pickle.dump(modeld, open(session["modelName"], 'wb'))
                pass
            elif model == "kNN":
                values,modeld = kNN(X_train, X_test, y_train, y_test)
                pickle.dump(modeld, open(session["modelName"], 'wb'))
                pass
            elif model =="SVM":
                values,modeld = sVM(X_train, X_test, y_train, y_test)
                pickle.dump(modeld, open(session["modelName"], 'wb'))
                pass
        #    fileSender(path)
            session["path"] = str(session["name"]+'finalized_model.sav')
            return render_template('output.html',values=values) 
        
        if problem == "Regression": 
            if model == "Linear Regression":
                pass
            pass

@app.route("/return-file")
def fileSender():
    q= session["modelName"]
    return send_file(q,
                     as_attachment=True)

def decisionTreeClassifier(X_train, X_test, y_train, y_test):
    model = sklearn.tree.DecisionTreeClassifier()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
  
    cm = confusion_matrix(y_test, y_pred)

    values = {"cm":cm,"acc":acc}
    return(values,model)

def randomForestClassifier(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
   
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values,model)

def linearRegression(X_train, X_test, y_train, y_test):
    model = sklearn.linear_model.LinearRegression()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values,model)

def logisticRegression(X_train, X_test, y_train, y_test):
    from sklearn import linear_model
    model = linear_model.LogisticRegression()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values,model)

def sVM(X_train, X_test, y_train, y_test):
    from sklearn import svm
    model = svm.SVC()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
  
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values,model)

def kNN(X_train, X_test, y_train, y_test):
    model = sklearn.neighbors.KNeighborsClassifier()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
   
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values,model)





if __name__ == '__main__':
    app.run(debug=True)