# Heart-Disease-Prediction-using-DL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
import warnings
warnings.filterwarnings('ignore')
import random
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
# Data Loading
data = pd.read_csv(r"heart_disease_uci.csv")
type(data)
data.shape
# Data set Description
data.head()
data.tail()
data.describe()
data.info()
1. age: The person's age in years

2. sex: The person's sex (1 = male, 0 = female)

3. cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)

4. trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)

5. chol: The person's cholesterol measurement in mg/dl

6. fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

7. restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)

8. thalach: The person's maximum heart rate achieved

9. exang: Exercise induced angina (1 = yes; 0 = no)

10. oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)

11. slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

12. ca: The number of major vessels (0-3)

13. thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

14. target: Heart disease (0 = no, 1 = yes)

Heart disease risk factors to the following: high cholesterol, high blood pressure, diabetes, weight, family history and smoking .

According to another source , the major factors that can't be changed are: increasing age, male gender and heredity.

Note that thalassemia, one of the variables in this dataset, is heredity.

Major factors that can be modified are: Smoking, high cholesterol, high blood pressure, physical inactivity, and being overweight and having diabetes.

Other factors include stress, alcohol and poor diet/nutrition.
#checking null values in dataset
data.isnull().sum()
###So, we have  missing values
data['fbs'].value_counts()
data['exang'].value_counts()
data['restecg'].value_counts()
data['slope'].value_counts()
data.shape
# Identify numerical columns
numerical_columns = data.select_dtypes(include=['number']).columns

# Fill missing values with the median for numerical columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

data.isnull().sum()
# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Fill missing values with the mode for categorical columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

data.isnull().sum()
data.shape
data.head(2)
# LabelEncoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
objList = ['sex','cp','fbs','restecg','exang','slope','thal']
for feat in objList:
    data[feat] = le.fit_transform(data[feat].astype(str))
print (data.info())
data
#dropping unnecesary columns
data.drop(['id'],axis = 1, inplace = True)
data.drop(['dataset'],axis = 1, inplace = True)
# Exploratory Data Analysis (EDA)
y = data["num"]
#sns.countplot(y)
target_temp = data.num.value_counts()
print(target_temp)
y
# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
sns.countplot(x=data.num)
plt.title('Countplot of num')
plt.xlabel('stages')
plt.ylabel('Count')
plt.show()
# Percentage of patient with or without heart problems in the given dataset
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/920,2)))
print("Percentage of patience with heart problem 1 st stage: "+str(round(target_temp[1]*100/920,2)))
print("Percentage of patience with heart problem 2nd stage: "+str(round(target_temp[2]*100/920,2)))
print("Percentage of patience with heart problem 3rd stage: "+str(round(target_temp[3]*100/920,2)))
print("Percentage of patience with heart problem 4th stage : "+str(round(target_temp[4]*100/920,2)))

data["sex"].unique()
unique_values = data["sex"].unique()
# Plotting
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
sns.countplot(x="sex", data=data, palette="Set2")  # Assuming "sex" is a categorical variable
plt.title('Count of unique values in "sex" column')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(ticks=range(len(unique_values)), labels=unique_values)  # Setting x-axis ticks to be the unique values
plt.show()
### Here 0 is female and 1 is male patients
countFemale = len(data[data.sex == 0])
countMale = len(data[data.sex == 1])
print("Percentage of Female Patients:{:.2f}%".format((countFemale)/(len(data.sex))*100))
print("Percentage of Male Patients:{:.2f}%".format((countMale)/(len(data.sex))*100))
# Heart Disease Frequency for ages
pd.crosstab(data.age,data.num).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()
# Heart Disease frequency for sex
pd.crosstab(data.sex,data.num).plot(kind="bar",figsize=(20,10),color=['blue','#AA1111','green','yellow','black' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No Heart disease", "Stage A","Stage B","Stage C","Stage D"])
plt.ylabel('Frequency')
plt.show()
# Analysing the chest pain (4 types of chest pain)

#[Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic]
data["cp"].unique()
# Get unique values and their counts
value_counts = data["cp"].value_counts()
unique_values = value_counts.index.tolist()
# Plotting
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
sns.countplot(x="cp", data=data, order=unique_values, palette="Set2")  # Assuming "cp" is a categorical variable
plt.title('Count of unique values in "cp" column')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.xticks(ticks=range(len(unique_values)), labels=unique_values)  # Setting x-axis ticks to be the unique values
plt.show()
# Analysing The person's resting blood pressure (mm Hg on admission to the hospital)
data["trestbps"].unique()
# Get unique values
unique_values = data["trestbps"].unique()

# Plotting
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
sns.countplot(x="trestbps", data=data, palette="Set1")  # Assuming "trestbps" is a categorical variable
plt.title('Count of unique values in "trestbps"')
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()
## people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2'
# Analysing the slope of the peak exercise ST segment
data["slope"].unique()
# Get unique values
unique_values = data["slope"].unique()

# Plotting
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
sns.countplot(x="slope", data=data, palette="Set2", order=unique_values)  # Assuming "slope" is a categorical variable
plt.title('Count of unique values in "slope" column')
plt.xlabel('Slope')
plt.ylabel('Count')
plt.show()
Slope '0' causes heart pain much more than Slope  '1'and'2'
#Heart disease according to Fasting Blood sugar

pd.crosstab(data.fbs,data.num).plot(kind="bar",figsize=(20,10),color=['#4286f4','#f49242','green','black','red'])
plt.title("Heart disease according to FBS")
plt.xlabel('FBS- (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation=90)
plt.legend(["No Heart disease", "Stage A","Stage B","Stage C","Stage D"])
plt.ylabel('Disease or not')
plt.show()
# Correlation plot
Correlation analysis is a method of statistical evaluation used to study the strength of a relationship between two, numerically measured, continuous variables (e.g. height and weight)
data.columns
plt.figure(figsize=(14,10))
sns.heatmap(data.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
plt.show()
# Splitting the dataset to Train and Test
data.shape
predictors = data.drop("num",axis=1)
target = data["num"]

target.value_counts()
# !pip install imbalanced-learn==0.6.0
# !pip install scikit-learn==0.22.1
from sklearn.utils import resample
# Separate majority and minority classes
df_majority = data[data['num']== 0]
df_minority1 = data[data['num']== 1]
df_minority2 = data[data['num']== 2]
df_minority3 = data[data['num']== 3]
df_minority4 = data[data['num']== 4]

# Downsample majority class and upsample the minority class
df_minority_upsampled = resample(df_minority4, replace=True,n_samples=8000,random_state=100)
df_majority_1 = resample(df_minority3, replace=True,n_samples=7000,random_state=100)
df_majority_2 = resample(df_minority2, replace=True,n_samples=8000,random_state=100)
df_majority_3 = resample(df_minority1, replace=True,n_samples=9000,random_state=100)
df_majority_4= resample(df_majority, replace=True,n_samples=8000,random_state=100)

# Combine minority class with downsampled majority class
df_balanced = pd.concat([df_minority_upsampled, df_majority_1,df_majority_2,df_majority_3,df_majority_4])

# Display new class counts
df_balanced['num'].value_counts()
df_balanced
predictors = df_balanced.drop("num",axis=1)
target = df_balanced["num"]
predictors.shape,target.shape
# from imblearn.over_sampling import SMOTE
# print("Before OverSampling, counts of label '0': {}".format(sum(target == 0)))
# print("Before OverSampling, counts of label '1': {}".format(sum(target == 1)))
# print("Before OverSampling, counts of label '2': {}".format(sum(target == 2)))
# print("Before OverSampling, counts of label '3': {}".format(sum(target == 3)))
# print("Before OverSampling, counts of label '4': {}\n".format(sum(target == 4)))

# # import SMOTE module from imblearn library
# # pip install imblearn (if you don't have imblearn in your system)
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(sampling_strategy={0: 7000, 1: 7000, 2: 7000, 3: 7000, 4: 7000}, random_state=42)
# predictors_res, target_res = sm.fit_resample(predictors,target.ravel())

# print('After OverSampling, the shape of train_X: {}'.format(predictors_res.shape))
# print('After OverSampling, the shape of train_y: {} \n'.format(target_res.shape))

# print("After OverSampling, counts of label '0': {}".format(sum(target_res == 0)))
# print("After OverSampling, counts of label '1': {}".format(sum(target_res == 1)))
# print("After OverSampling, counts of label '2': {}".format(sum(target_res == 2)))
# print("After OverSampling, counts of label '3': {}".format(sum(target_res == 3)))
# print("After OverSampling, counts of label '4': {}".format(sum(target_res == 4)))


from keras.utils import to_categorical
target = to_categorical(target)
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(predictors,target,stratify=target,random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#from keras.utils.np_utils import to_categorical#convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPool1D,GlobalAvgPool1D,GlobalMaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed

# LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(13, 1)))
# model.add(LSTM(100, input_shape=(41)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              # optimizer='sgd', # almost same
              optimizer="Adam",
              metrics=['accuracy'])
history1 = model.fit(X_train,y_train, batch_size= 128,
                    epochs = 100, validation_data = (X_test,y_test))
plt.plot(history1.history['accuracy'], 'r')
plt.plot(history1.history['val_accuracy'], 'b')
plt.legend({'Train Accuracy': 'r', 'Test Accuracy':'b'})
plt.show()
# Now you can evaluate the model on the test data
score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
#plot confusion matrix
from sklearn.metrics import confusion_matrix
class_names = ['No Heart disease','1st stage','2nd stage','3rd stage','final stage']
df_heatmap = pd.DataFrame(confusion_matrix(np.argmax((model.predict(X_test)),axis = 1),np.argmax(y_test,axis=1)),columns = class_names, index = class_names)
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Class names
class_names = ['No Heart Disease', '1st Stage', '2nd Stage', '3rd Stage', 'Final Stage']

# Predict classes
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)


# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# Extract Precision and Recall
precision = {class_name: report[class_name]['precision'] for class_name in class_names}
recall = {class_name: report[class_name]['recall'] for class_name in class_names}

# Convert to DataFrame for better visualization
precision_recall_df = pd.DataFrame({'Precision': precision, 'Recall': recall})
print("Precision and Recall for Each Class:\n")
print(precision_recall_df)

# Optionally plot precision and recall
precision_recall_df.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange'])
plt.title('Precision and Recall per Class')
plt.ylabel('Score')
plt.xlabel('Classes')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()

#classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n")
print(report)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Number of classes
n_classes = len(class_names)

# Binarize the labels for multi-class ROC
y_test_binarized = label_binarize(np.argmax(y_test, axis=1), classes=range(n_classes))
y_pred_proba = model.predict(X_test)  # Predicted probabilities

# Plot AUC-ROC curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {class_names[i]} (AUC = {roc_auc:.2f})")

# Plotting the diagonal
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Add labels and legend
plt.title("AUC-ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
