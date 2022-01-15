from flask import Flask ,render_template, request, session, redirect, url_for, jsonify, flash
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
app = Flask(__name__)
 
@app.route("/") 
def hello_world(): 
    return render_template("index.html")

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
        title = request.form["title"]
        firstName = request.form['nameFirst']
        lastName = request.form['nameLast']
        email = request.form['email']
        contact = request.form['number']
        modelName = request.form['modelName']
        problemType = request.form['problemType']
        dataSet = request.files['dataSet']
        ytrainName = request.form["ytrain"]
        problem = problemTypeDic[problemType]
        model = modelNameDic[modelName]    
        userData = [title,firstName,lastName,email,contact,problem,model]
        
        dataframe = pd.read_csv(dataSet)
        X = dataframe.loc[:, dataframe.columns != ytrainName] # Features
        y = dataframe[ytrainName]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        

        if problem == "Classification":
            if model == "Linear Regression":
                values = linearRegression(X_train, X_test, y_train, y_test)
                pass
            elif model =="Logistic Regression":
                values = logisticRegression(X_train, X_test, y_train, y_test)
                pass
            elif model == "Random Forest":
                values = randomForestClassifier(X_train, X_test, y_train, y_test)
                pass
            elif model == "Decision Tree":
                values = decisionTreeClassifier(X_train, X_test, y_train, y_test)
                pass
            elif model == "kNN":
                values = kNN(X_train, X_test, y_train, y_test)
                pass
            elif model =="SVM":
                values = sVM(X_train, X_test, y_train, y_test)
                pass
            return render_template('output.html',values=values)
    

def decisionTreeClassifier(X_train, X_test, y_train, y_test):
    model = sklearn.tree.DecisionTreeClassifier()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
  
    cm = confusion_matrix(y_test, y_pred)

    values = {"cm":cm,"acc":acc}
    return(values)

def randomForestClassifier(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
   
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values)

def linearRegression(X_train, X_test, y_train, y_test):
    model = sklearn.linear_model.LinearRegression()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values)

def logisticRegression(X_train, X_test, y_train, y_test):
    model = sklearn.linear_model.LogisticRegression()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
  
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values)

def sVM(X_train, X_test, y_train, y_test):
    from sklearn import svm
    model = svm.SVC()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
  
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values)

def kNN(X_train, X_test, y_train, y_test):
    model = sklearn.neighbors.KNeighborsClassifier()
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
   
    cm = confusion_matrix(y_test, y_pred)
    values = {"cm":cm,"acc":acc}
    return(values)



if __name__ == '__main__':
    app.run(debug=True,port = int(os.environ.get('PORT', 5000)))