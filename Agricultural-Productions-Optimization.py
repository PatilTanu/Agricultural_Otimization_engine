#!/usr/bin/env python
# coding: utf-8

# In[1]:


# for manipulations
import numpy as np
import pandas as pd

# for data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# for interactivity
from ipywidgets import interact


# In[2]:


# lets read the dataset.

data = pd.read_csv("Agricultural data.csv")


# In[3]:


# lets check the shape of the dataset.

print("Shape of the Dataset :", data.shape)


# In[4]:


# lets check the head of the dataset

data.head()


# In[5]:


# lets check of there is any missing value present in the dataset.

data.isnull().sum()


# In[6]:


# lets check the Crops present in this dataset
data['label'].value_counts()


# In[7]:


# lets check the summary for all the crops

print("Average Ratio of Nitrogen in the Soil :{0:.2f}".format(data['N'].mean()))
print("Average Ratio of Phosphorous in th Soil :{0:.2f}".format(data['P'].mean()))
print("Average Ratio of Potassium in the Soil : {0:.2f}".format(data['K'].mean()))
print("Average Tempature in the Celsius : {0:.2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in % : {0:.2f}".format(data['humidity'].mean()))
print("Average PH Value of the Soil : {0:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))


# In[8]:


# lets check the Summary Statistics for each of the Crops

@interact
def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label']==crops]
    print('---------------------------------------------------------------')
    print("Statistics for Nitrogen")
    print("Minimum Nitrogen Required ", x['N'].min())
    print("Average Nitrogen Required ", x['N'].mean())
    print("Maximum Nitrogen Required ", x['N'].max())
    print('---------------------------------------------------------------')
    print("Statistics for Phosphorous")
    print("Minimum Phosphorous Required ", x['P'].min())
    print("Average Phosphorous Required ", x['P'].mean())
    print("Maximum Phosphorous Required ", x['P'].max())
    print('---------------------------------------------------------------')
    print("Statistics for Potassium")
    print("Minimum Potassium Required :", x['K'].min())
    print("Average Potassium Required :", x['K'].mean())
    print("Maximum Potassium Required :", x['K'].max())
    print('---------------------------------------------------------------')
    print("Statistics for Tempature")
    print("Minimum Tempature Required : {0:.2f}".format(x['temperature'].min()))
    print("Average Tempature Required : {0:.2f}".format(x['temperature'].mean()))
    print("Maximum Tempature Required : {0:.2f}".format(x['temperature'].max()))
    print('---------------------------------------------------------------')
    print("Statistics for Humidity")
    print("Minimum Humidity Required : {0:.2f}".format(x['humidity'].min()))
    print("Average Humidity Required : {0:.2f}".format(x['humidity'].mean()))
    print("Maximum Humidity Required : {0:.2f}".format(x['humidity'].max()))
    print('---------------------------------------------------------------')
    print("Statistics for PH")
    print("Minimum PH Required : {0:.2f}".format(x['ph'].min()))
    print("Average PH Required : {0:.2f}".format(x['ph'].mean()))
    print("Maximum PH Required : {0:.2f}".format(x['ph'].max()))
    print('---------------------------------------------------------------')
    print("Statistics for Rainfall")
    print("Minimum Rainfall Required : {0:.2f}".format(x['rainfall'].min()))
    print("Average Rainfall Required : {0:.2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall Required : {0:.2f}".format(x['rainfall'].max()))
    print('---------------------------------------------------------------')
    


# In[9]:


# lets compare the Average Requirement for each Crops with average conditions

@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
   print("Average Value for",conditions,"is {0:.2f}".format(data[conditions].mean()))
   print("Rice        : {0:.2f}".format(data[(data['label']== 'rice')][conditions].mean()))  
   print("Maize       : {0:.2f}".format(data[(data['label']== 'maize')][conditions].mean()))  
   print("Jute        : {0:.2f}".format(data[(data['label']== 'jute')][conditions].mean()))  
   print("Cotton      : {0:.2f}".format(data[(data['label']== 'cotton')][conditions].mean()))  
   print("Coconut     : {0:.2f}".format(data[(data['label']== 'coconut')][conditions].mean()))  
   print("Papaya      : {0:.2f}".format(data[(data['label']== 'papaya')][conditions].mean()))  
   print("Orange      : {0:.2f}".format(data[(data['label']== 'orange')][conditions].mean()))  
   print("Apple       : {0:.2f}".format(data[(data['label']== 'apple')][conditions].mean()))  
   print("Muskmelon   : {0:.2f}".format(data[(data['label']== 'muskmelon')][conditions].mean()))  
   print("Watermelon  : {0:.2f}".format(data[(data['label']== 'watermelon')][conditions].mean()))  
   print("Grapes      : {0:.2f}".format(data[(data['label']== 'grapes' )][conditions].mean()))  
   print("Mango       : {0:.2f}".format(data[(data['label']== 'mango')][conditions].mean()))  
   print("Banana      : {0:.2f}".format(data[(data['label']== 'banana')][conditions].mean()))  
   print("Pomegranate : {0:.2f}".format(data[(data['label']== 'pomegranate')][conditions].mean()))  
   print("Lentil      : {0:.2f}".format(data[(data['label']== 'lentil')][conditions].mean()))  
   print("Blackgram   : {0:.2f}".format(data[(data['label']== 'blackgram')][conditions].mean()))  
   print("Mungbean    : {0:.2f}".format(data[(data['label']== 'mungbean')][conditions].mean()))  
   print("Mothbeans   : {0:.2f}".format(data[(data['label']== 'mothbeans')][conditions].mean()))  
   print("Pigeonpeas  : {0:.2f}".format(data[(data['label']== 'pigeonpeas')][conditions].mean()))  
   print("Kidneybeans : {0:.2f}".format(data[(data['label']== 'kidneybeans')][conditions].mean()))  
   print("Chickpea    : {0:.2f}".format(data[(data['label']== 'chickpea')][conditions].mean()))  
   print("Coffee      : {0:.2f}".format(data[(data['label']== 'coffee')][conditions].mean()))  


# In[10]:


#Checking the below and above Average Conditions

@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Crops which require greater than average", conditions,'\n')
    print(data[data[conditions]>data[conditions].mean()]['label'].unique())
    print("--------------------------------------------------------------------------")
    print("Crops which require less than average", conditions,'\n')
    print(data[data[conditions]<=data[conditions].mean()]['label'].unique())


# In[11]:


# Define the columns you want to plot
cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
col_title =['Nitrogen', 'Phosphorous', 'Potassium', 'temperature', 'humidity', 'ph', 'rainfall']

# Create dictionary for put different color for each columns
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta']


# Set up the FacetGrid with 2 rows and 4 columns
g = sns.FacetGrid(data[cols].melt(var_name='Columns'), col='Columns', col_wrap=4, sharex=False, sharey=False, height=4,despine=True)
# g = sns.FacetGrid(data, col='columns', palette=my_palette)

# Create a histogram for each column
g.map(sns.histplot,'value',stat = 'density',color=(0.5, 0.6, 0.6), alpha=1.0)
g.map(sns.kdeplot, 'value', color=(1, 0.2, 0.4), alpha=1.0,linewidth=2.5, marker='s', markersize=5)



# Set the title of each subplot
for ax, title in zip(g.axes.flat, col_title):
    ax.set_title(title)

# Remove unused plots
for i in range(len(cols), len(g.axes.flat)):
    g.axes.flat[i].remove()

# Add a main title to the entire figure
g.fig.suptitle('Distribution of Agricultural Data')

# Adjust layout to avoid overlapping labels
g.fig.tight_layout()

# Display the plot
plt.show()


# In[12]:


#Checking that crops those have unusual requirements

print("Some Interesting Patterns")
print("...........................................")
print("Crops that require very High Ratio of Nitrogen Content in Soil:", data[data['N'] > 120]['label'].unique())
print("Crops that require very High Ratio of Phosphorous Content in Soil:", data[data['P'] > 100]['label'].unique())
print("Crops that require very High Ratio of Potassium Content in Soil:", data[data['K'] > 200]['label'].unique())
print("Crops that require very High Rainfall:", data[data['rainfall'] > 200]['label'].unique())
print("Crops that require very Low Temperature:", data[data['temperature'] < 10]['label'].unique())
print("Crops that require very High Temperature:", data[data['temperature'] > 40]['label'].unique())
print("Crops that require very Low Humidity:", data[data['humidity'] < 20]['label'].unique())
print("Crops that require very Low pH:", data[data['ph'] < 4]['label'].unique())
print("Crops that require very High pH:", data[data['ph'] > 9]['label'].unique())


# In[13]:


#Checking which crop to be grown according to the season

print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("...........................................")
print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("...........................................")
print("Monsoon Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())


# In[15]:


from sklearn.cluster import KMeans

#removing the labels column
x = data.drop(['label'], axis=1)

#selecting all the values of data
x = x.values

#checking the shape
print(x.shape)


# In[16]:


#Determining the optimum number of clusters within the Dataset

plt.rcParams['figure.figsize'] = (10,4)

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 2000, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
#Plotting the results

plt.plot(range(1,11), wcss)
plt.title('Elbow Method', fontsize = 20)
plt.xlabel('No of Clusters')
plt.ylabel('wcss')
plt.show


# In[17]:


#Implementation of K Means algorithm to perform Clustering analysis

km = KMeans(n_clusters = 4, init = 'k-means++',  max_iter = 2000, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

#Finding the results
a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})

#Checking the clusters for each crop
print("Lets Check the results after applying K Means Clustering Analysis \n")
print("Crops in First Cluster:", z[z['cluster'] == 0]['label'].unique())
print("...........................................")
print("Crops in Second Cluster:", z[z['cluster'] == 1]['label'].unique())
print("...........................................")
print("Crops in Third Cluster:", z[z['cluster'] == 2]['label'].unique())
print("...........................................")
print("Crops in Fourth Cluster:", z[z['cluster'] == 3]['label'].unique())


# In[18]:


#Splitting the Dataset for predictive modelling

y = data['label']
x = data.drop(['label'], axis=1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[19]:


#Creating training and testing sets for results validation
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("The Shape Of x train:", x_train.shape)
print("The Shape Of x test:", x_test.shape)
print("The Shape Of y train:", y_train.shape)
print("The Shape Of y test:", y_test.shape)


# In[20]:


#Creating a Predictive Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[21]:


#Evaluating the model performance
from sklearn.metrics import confusion_matrix

#Printing the Confusing Matrix
plt.rcParams['figure.figsize'] = (10,10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'Wistia')
plt.title('Confusion Matrix For Logistic Regression', fontsize = 15)
plt.show()


# In[22]:


#Defining the classification Report
from sklearn.metrics import classification_report

#Printing the Classification Report
cr = classification_report(y_test, y_pred)
print(cr)


# In[23]:


#head of dataset
data.head()


# In[29]:


prediction = model.predict((np.array([[90, 40, 40, 20, 80, 7, 200]])))
print("The Suggested Crop for given climatic condition is :",prediction)

