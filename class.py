###############################################
#About
###############################################
#Source: https://github.com/georgetown-analytics/machine-learning/blob/master/examples/bbengfort/census.ipynb
#Instructor Ben Bengfort


###############################################
#Import
###############################################
import os 
import requests 
import pandas as pd 
import seaborn as sns
import json 
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
import pickle 

###############################################
#Data
###############################################
CENSUS_DATASET = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names", 
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", 
)

def download_data(path='data', urls=CENSUS_DATASET):
    if not os.path.exists(path):
        os.mkdir(path) 
    
    for url in urls:
        response = requests.get(url)
        name = os.path.basename(url) 
        with open(os.path.join(path, name), 'w') as f: 
            f.write(response.content)

download_data()

###############################################
#Data Exploration and Cleaning
###############################################

names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

data = pd.read_csv('data/adult.data', names=names)
data.head()

chart1 = sns.countplot(y='occupation', hue='income', data=data,)
chart2 = sns.countplot(y='education', hue='income', data=data,)
print chart1,chart2

meta = {
    'target_names': list(data.income.unique()),
    'feature_names': list(data.columns),
    'categorical_features': {
        column: list(data[column].unique())
        for column in data.columns
        if data[column].dtype == 'object'
    },
}

with open('data/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

def load_data(root='data'):
    # Load the meta data from the file 
    with open(os.path.join(root, 'meta.json'), 'r') as f:
        meta = json.load(f) 
    
    names = meta['feature_names']
    
    # Load the readme information 
    with open(os.path.join(root, '/Users/lesleychapman/Desktop/README.md'), 'r') as f:
        readme = f.read() 
    
    # Load the training and test data, skipping the bad row in the test data 
    train = pd.read_csv(os.path.join(root, 'adult.data'), names=names)
    test  = pd.read_csv(os.path.join(root, 'adult.test'), names=names, skiprows=1)
    
    # Remove the target from the categorical features 
    meta['categorical_features'].pop('income')
    
    # Return the bunch with the appropriate data chunked apart
    return Bunch(
        data = train[names[:-1]], 
        target = train[names[-1]], 
        data_test = test[names[:-1]], 
        target_test = test[names[-1]], 
        target_names = meta['target_names'],
        feature_names = meta['feature_names'], 
        categorical_features = meta['categorical_features'], 
        DESCR = readme,
    )

dataset = load_data()
gender = LabelEncoder() 
gender.fit(dataset.data.sex)
print(gender.classes_)
print(gender.transform([
    ' Female', ' Female', ' Male', ' Female', ' Male'
]))

class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None. 
    """
    
    def __init__(self, columns=None):
        self.columns  = columns 
        self.encoders = None
    
    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode. 
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns 
        
        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns 
        }
        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame. 
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])
        
        return output

encoder = EncodeCategorical(dataset.categorical_features.keys())
data = encoder.fit_transform(dataset.data)

class ImputeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None. 
    """
    
    def __init__(self, columns=None):
        self.columns = columns 
        self.imputer = None
    
    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to impute. 
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns 
        
        # Fit an imputer for each column in the data frame
        self.imputer = Imputer(missing_values=0, strategy='most_frequent')
        self.imputer.fit(data[self.columns])

        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame. 
        """
        output = data.copy()
        output[self.columns] = self.imputer.transform(output[self.columns])
        
        return output

    
imputer = ImputeCategorical(['workclass', 'native-country', 'occupation'])
data = imputer.fit_transform(data)
# we need to encode our target data as well. 
yencode = LabelEncoder().fit(dataset.target)

# construct the pipeline 
census = Pipeline([
        ('encoder',  EncodeCategorical(dataset.categorical_features.keys())),
        ('imputer', ImputeCategorical(['workclass', 'native-country', 'occupation'])), 
        ('classifier', LogisticRegression())
    ])

# fit the pipeline 
census.fit(dataset.data, yencode.transform(dataset.target))
# encode test targets, and strip traililng '.' 
y_true = yencode.transform([y.rstrip(".") for y in dataset.target_test])

# use the model to get the predicted value
y_pred = census.predict(dataset.data_test)

# execute classification report 
print classification_report(y_true, y_pred, target_names=dataset.target_names)
def dump_model(model, path='data', name='classifier.pickle'):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(model, f)
        
dump_model(census)
def load_model(path='data/classifier.pickle'):
    with open(path, 'rb') as f:
        return pickle.load(f) 


def predict(model, meta=meta):
    data = {} # Store the input from the user
    
    for column in meta['feature_names'][:-1]:
        # Get the valid responses
        valid = meta['categorical_features'].get(column)
    
        # Prompt the user for an answer until good 
        while True:
            val = " " + raw_input("enter {} >".format(column))
            if valid and val not in valid:
                print "Not valid, choose one of {}".format(valid)
            else:
                data[column] = val
                break
    
    # Create prediction and label 
    yhat = model.predict(pd.DataFrame([data]))
    return yencode.inverse_transform(yhat)
            
    
# Execute the interface 
model = load_model()
predict(model)

