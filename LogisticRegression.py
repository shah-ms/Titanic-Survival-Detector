import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import countplot
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def MarvellousLogisticRegression():
	
	#Load Data
	
	titanic_data = pd.read_csv("Titanic.csv")		
	
	print(titanic_data.head())

	print("Number of passengers = ", len(titanic_data))

	#Visualize Data
	
	target = "Survived"
	
	figure()
	countplot(data=titanic_data, x=target).set_title("Survived & Non-Survived Passengers")
	show()	
		
	figure()
	countplot(data=titanic_data, x=target, hue="Sex").set_title("Survived & Non-Survived Passengers based on Gender")
	show()
	
	figure()
	countplot(data=titanic_data, x=target, hue="Pclass").set_title("Survived & Non-Survived Passengers based on Class")
	show()
	
	figure()
	titanic_data["Age"].plot.hist().set_title("Passengers based on Age in the Titanic")
	show()	
		
	#Data Cleaning
	
	Sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
	
	Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
	
	titanic_data = pd.concat([titanic_data, Pclass, Sex], axis=1)
	
	titanic_data.drop(["Sex", "Pclass", "SibSp", "Parch", "Embarked", "Cabin", "Ticket", "Name"], axis=1, inplace=True)
	
	titanic_data = titanic_data.dropna()
	
	print(titanic_data.head())
	
	print("Number of passengers = ", len(titanic_data))
	print(titanic_data.shape)
	
	x = titanic_data.drop("Survived", axis=1)
	
	y = titanic_data["Survived"]
	
	#Training
	
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25)
	
	logreg = LogisticRegression()

	logreg.fit(xtrain, ytrain)	
	
	#Testing
	
	predictions = logreg.predict(xtest)
	
	#Calculating Accuracy
	
	print(classification_report(ytest, predictions))
	
	print(confusion_matrix(ytest, predictions))
	
	print(accuracy_score(ytest, predictions))
	
def main():

	MarvellousLogisticRegression()

if __name__ == "__main__":
	main()
