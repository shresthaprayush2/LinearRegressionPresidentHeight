import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style('whitegrid')
sns.set_palette('bright')

traininData = pd.read_csv('training.csv')
print(traininData.head(10).to_string())

#Checking if the data is clean
#Check for null vlaues
print(traininData.isnull().sum())
#Data contains no null value so we dont need to do anything

#Seperate data into feature and target set
x = traininData['height(cm)']
y = traininData['weight(kg)']

#Rough Visualize
sns.scatterplot(x=x,y=y)
plt.savefig('Original_DataSet.png')

x = traininData['height(cm)']
y = traininData['weight(kg)']
#for scikit learn out data should be two dimensional
X2D = x[:,np.newaxis]
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
#Training the model
model.fit(X2D,y)

#Model Analysis and interpretation
print(f'The slope is {model.coef_}')
print(f'The intercept is {model.intercept_}')
print(f"Starting from {model.intercept_} each 1 cm increase in height results in increase in weight by {model.coef_[0]}")

dataToPredict = pd.read_csv('toPredict.csv')
Xpred = dataToPredict['height(cm)']
Xpred2d = Xpred[:,np.newaxis]

yPred = model.predict(Xpred2d)
print(yPred)

#Creating a dataframe from predicted
dataToPredict['Predicted Weight'] = yPred
print(dataToPredict.to_string())

#Rough Visualize
plt.scatter(x,y)
plt.plot(Xpred,yPred)
plt.savefig('Visualized_DataSet.png')
