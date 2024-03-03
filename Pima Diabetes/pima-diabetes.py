# Imports
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error,mean_absolute_error
import mlflow

import torch
import torch.nn as nn
import torch.nn.functional as F
get_ipython().run_line_magic('matplotlib', 'inline')



df = pd.read_csv('/pima-diabetes/diabetes.csv')


df.head()


df.isnull().sum()


# Change to categorical variable for visualization
df['Outcome'] = np.where(df['Outcome'] == 1, "Diabetic","Non-Diabetic")


df.head()


# Seaborn takes category features. That is inform of strings
sns.pairplot(df,hue="Outcome");


df = pd.read_csv('/pima-diabetes/diabetes.csv')


X = df.drop('Outcome',axis=1).values ## Independent variables
y = df['Outcome'].values ## Dependent variable


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)



## Creating Tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)



## Creating Model with PyTorch

class Diabetic_Model(nn.Module):
    def __init__(self,input_layer=8, h_layer1=20, h_layer2=20,output_layer=2):
        super().__init__()
        self.dense_layer1 = nn.Linear(input_layer,h_layer1)
        self.dense_layer2 = nn.Linear(h_layer1,h_layer2)
        self.output_layer = nn.Linear(h_layer2,output_layer)
        
    def forward(self,x):
        x = F.relu(self.dense_layer1(x))
        x = F.relu(self.dense_layer2(x))
        x = self.output_layer(x)
        
        return x


## Instantiate model
torch.manual_seed(0) # For application of the initial weights

model = Diabetic_Model()


model.parameters


# Backward propagation 
## a) Define loss function
## b) Define the Optimizer
lr = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


epochs = 1000 # Epochs

final_loses = []

for epc in range(epochs):
    epc + 1
    y_pred = model.forward(X_train)
    loss = loss_func(y_pred, y_train)
    final_loses.append(loss)

    if epc % 100 ==1:
        print(f"Epoch nunber: {epc} =========> Loss for epoch number {epc}: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# # plot the loss function
final_loses_np = [fl.detach().numpy() for fl in final_loses]
plt.plot(range(epochs),final_loses_np)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Function")


## Prediction on test data
predictions = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_preds = model(data)
        predictions.append(y_preds.argmax().item())
        # print(y_preds.argmax().item())


cm = confusion_matrix(y_test, predictions)
cm


plt.figure(figsize=(10,5));
sns.heatmap(cm,annot=True);
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');


# Accuracy Score
score = accuracy_score(y_test, predictions)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Pima Diabetes")


with mlflow.start_run():
    
    # log the hyperparameters
    mlflow.log_params(epochs)
    mlflow.log_params(lr)
    
    # Log meetrics
    mlflow.log_metric("accurcay", score)
    mlflow.log_metric("mae",mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse",rmse)
    
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Deep learning model for Pima diabetes.")



# # Save model
# torch.save(model,'/home/jothammasila/Projects/Models/diabetes.pt')


## Load model

# t_model = torch.load('/home/jothammasila/Projects/Models/diabetes.pt')



#Check the model parameters

t_model.eval()

## New data

class Data:
    def __init__(self, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age
        
        self.data = pd.DataFrame({
            'Pregnancies': [self.Pregnancies],
            'Glucose': [self.Glucose],
            'BloodPressure': [self.BloodPressure],
            'SkinThickness': [self.SkinThickness],
            'Insulin': [self.Insulin],
            'BMI': [self.BMI],
            'DiabetesPedigreeFunction': [self.DiabetesPedigreeFunction],
            'Age': [self.Age]
        })
        
        self.data = torch.FloatTensor(self.data.values)
        

data = torch.tensor([4, 111, 67,32,7, 30.8,0.887,78])


# function for prediction.

def prediction(data):
    with torch.no_grad():
        model = torch.load('/Models/diabetes.pt')
        outcome = [model(data),model(data).argmax().item()]
    return outcome


print(prediction(data))

