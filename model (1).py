import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("Latest Covid-19 India Status.csv")

print(df.head())

# Select independent and dependent variable
x = df[["Total Cases", "Active", "Discharged"]]
y = df["Deaths"]

# Split the dataset into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)
len(train_x),len(test_x)

# Feature scaling
#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test= sc.transform(x_test)

# Instantiate the model
model=LinearRegression()

# Fit the model
model.fit(train_x,train_y)

# Make pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))
