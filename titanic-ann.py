# Artificial Neural Network to predict the survival of passengers on the Titanic
# Goal is to find out whether Jack would have survived or not

# Step1 - Preparing the dataset

import pandas as pd
import numpy as np

# Reading the training data
train_data = pd.read_csv('train.csv')

# This variable contains whether the passenger survived or not
y = train_data['Survived'].values

# The features to train on
features = ['Pclass', 'Sex', 'Age']
# Taking only the features we need
df_X = train_data[features]

# To encode the feature Sex 
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
df_X['Sex'] = le_sex.fit_transform(df_X['Sex'])
# male = 1, female = 0

# Converting Pclass to dummy variables
df_X = pd.concat([df_X, pd.get_dummies(df_X['Pclass'])], axis=1)
df_X = df_X.drop('Pclass', axis=1) # Deleting the original Pclass column

# Replacing nan with the mean of age
df_X['Age'] = df_X['Age'].fillna(df_X['Age'].mean())
# To normalize Age and convert them to a value between 0 & 1
from sklearn.preprocessing import MinMaxScaler
mmsc = MinMaxScaler()
df_X['Age'] = mmsc.fit_transform(df_X['Age'].reshape(len(df_X), 1))

# Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_X.values, y, test_size=0.2)


# Step2 - Building the model

# Importing the required modules
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the model
model = Sequential()

# Adding the input layer and first hidden layer
model.add(Dense(input_dim=5, units=32, activation='relu'))
model.add(Dropout(rate=0.3)) # Try reducing the dropout rate

# Adding the second hidden layer
model.add(Dense(units=16, activation='relu'))

# Adding the output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, batch_size=10, epochs=500)
# Try reducing the batch size and changing the number of epochs


# Step3 - Making predictions 

predictions = model.predict(X_test)
predictions[predictions >= 0.6] = 1 # survived
predictions[predictions < 0.6] = 0 # did not survive

correct_count=0
for i in range(len(X_test)):
    if predictions[i] == y_test[i]:
        correct_count+=1

accuracy = correct_count / len(X_test)
print("Actual Accuracy = ", accuracy)

jack = [1, mmsc.transform(20), 0, 0, 1]
jack_survived = model.predict(np.array(jack).reshape(1, 5))
print("Chances of Jack surviving = ", jack_survived[0][0]*100)

rose = [0, mmsc.transform(17), 1, 0, 0]
rose_survived = model.predict(np.array(rose).reshape(1, 5))
print("Chances of Rose surviving = ", rose_survived[0][0]*100)











