# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The Neural network model contains input layer, two hidden layers and output layer. Input layer contains a single neuron. Output layer also contains single neuron. First hidden layer contains six neurons and second hidden layer contains four neurons. A neuron in input layer is connected with every neurons in a first hidden layer. Similarly, each neurons in first hidden layer is connected with all neurons in second hidden layer. All neurons in second hidden layer is connected with output layered neuron. Relu activation function is used here and the model is a linear neural network model. Data is the key for the working of neural network and we need to process it before feeding to the neural network. In the first step, we will visualize data which will help us to gain insight into the data. We need a neural network model. This means we need to specify the number of hidden layers in the neural network and their size, the input and output size. Now we need to define the loss function according to our task. We also need to specify the optimizer to use with learning rate. Fitting is the training step of the neural network. Here we need to define the number of epochs for which we need to train the neural network. After fitting model, we can test it on test data to check whether the case of overfitting.



## Neural Network Model





![image](https://user-images.githubusercontent.com/103016346/191755297-6f63b47f-2a01-48d9-8be1-021cc0bc8160.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
import matplotlib.pyplot as plt

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DL Data').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float','Output':'float'})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df.head()

x=df[['Input']].values
x

y=df[['Output']].values
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=11)

Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_test1=Scaler.transform(x_test)
x_train1

ai_brain = Sequential([
    Dense(6,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])

ai_brain.compile(
    optimizer='rmsprop',
    loss='mse'
)
ai_brain.fit(x_train1,y_train,epochs=4000)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
plt.title('Training Loss Vs Iteration Plot')

ai_brain.evaluate(x_test1,y_test)

x_n1=[[66]]
x_n1_1=Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```
## Dataset Information
![22t](https://user-images.githubusercontent.com/103016346/191753814-bfdac076-a349-416a-95f5-5c00b9a342df.png)




## OUTPUT



I![187120262-1531b3a8-d780-483d-9163-8ef0a0c27a37](https://user-images.githubusercontent.com/103016346/191754169-f66ea2c0-1c40-41dc-8cb3-07ccb0c793cf.png)


### Test Data Root Mean Squared Error
![187120317-8cba881a-8b24-499e-8eed-f6d36ca7b522](https://user-images.githubusercontent.com/103016346/191754503-fcb3f58e-ddae-4fd3-827a-89b7816d5c97.png)








### New Sample Data Prediction
![187120386-36370204-9cac-4de4-a554-bf6be2034ffc](https://user-images.githubusercontent.com/103016346/191754727-303d2489-c014-4e5f-b847-0347bf98f023.png)





## RESULT:
Succesfully created and trained a neural network regression model for the given dataset.
