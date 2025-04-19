# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ADARSH CHOWARY R
RegisterNumber:  212223040166
*/
```
```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:

### DATA HEAD:

![image](https://github.com/user-attachments/assets/36f71567-44c3-47ca-ade2-2b145c3cf85a)

### DATA INFO:

![image](https://github.com/user-attachments/assets/5f085362-1948-44d5-aa7c-94147fe302dd)


### ISNULL() AND SUM():

![image](https://github.com/user-attachments/assets/ae764e08-d6fe-4787-ad9a-49fa64adace5)

### DATA HEAD FOR SALARY:

![image](https://github.com/user-attachments/assets/76b11f14-89f9-4721-b973-e0a7a7ee4f3c)

### MEAN SQUARED ERROR:

![image](https://github.com/user-attachments/assets/dbc9783d-7e03-43a0-be72-7407cca4faa4)

### R2 VALUE:

![image](https://github.com/user-attachments/assets/f6305b78-6dfc-4d2b-894f-c5bee62715b1)

### DATA PREDICTION:

![image](https://github.com/user-attachments/assets/7be75e53-bd8f-46a3-94ae-5c23fac6a81c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
