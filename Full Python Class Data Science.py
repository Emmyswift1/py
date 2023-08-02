#!/usr/bin/env python
# coding: utf-8

# In[5]:


print ("Hello World")


# In[4]:


type ("Hello World")


# In[5]:


type (20)


# In[6]:


type (2.0)


# In[13]:


type (None)


# In[9]:


x = 200


# In[10]:


x


# In[28]:


type yes


# In[8]:


y = "Jesus loves you"
y


# In[9]:


y


# In[11]:


x = 10
y = 50
z = "Jesus loves"
x, y ,z


# In[11]:


x = 10
y = 50
z = "Jesus loves" #we use quotation because of string 
print (x)
print (y) 
print (z)


# ## Data Operations  

# y

# In[322]:


t = (20, 10, 11)
t


# In[323]:


s = (5, 14, 13)
s


# In[324]:


print (t + s)


# In[20]:


print (t * s)


# In[21]:


print (t / s)


# In[22]:


print (t + s)


# In[23]:


t / s


# In[24]:


print (t * s)
print (t / s)
print (t + s)


# ## The python programming FUNCTION ##

# In[18]:


def add (z, t):
    return (z + t)


# In[19]:


add (5, 6)


# In[20]:


def me (d, f):
    return (d - f)


# In[21]:


me (90, 50)


# In[22]:


def x (t , s):
    a=t+s 
    b=t*s
    c=t-s
    d=t/s
    return (a,b,c,d)


# In[23]:


x (500, 20)


# In[49]:


x (80, 5)


# IF STATEMENT

# In[24]:


g = 100
if (g > 50):
    print ("g is greater than 50")
else:
    print ("g is less than 50")


# Data Collection in Python
# 
# Tuple

# In[25]:


c = (30, 2.0, 50, 32, 56, 81, 100)
c


# In[26]:


for go in c:
    print (go)


# Slicing in Tuple

# In[27]:


print (c[0])
print (c[1])
print (c[2])
print (c[-5])
print (c[-1])


# LIST

# In[29]:


k = ["a", 2.5, 70, 100]
k


# In[32]:


k.append (50)


# In[89]:


k


# In[92]:


print (k[0])
print (k[1])
print (k[2])
print (k[-3])
print (k[-1])


# For Loop

# In[95]:


for item in k:
    print (item)


# Operation on List

# In[98]:


[1, 5, 6] + [2, 8, 9]


# In[34]:


[1]*10


# In[103]:


h = ("Jesus loves you all expecially me")
h


# In[106]:


print (h[-13])


# Concatenating two strings

# In[35]:


Firstname = "Temilayo"
Lastname = "Onayemi"
Country = "United State"


# In[36]:


Firstname + " " + Lastname + " " + Country


# In[4]:


Z = str(2)
Z


# In[7]:


type (Z)


# In[11]:


"Dolapo" + " " + str(2.4)


# .split function

# In[17]:


name = "Yusuf Ayodele Emmanuel Nigeria"
name


# In[18]:


name.split(" ")


# In[26]:


Firstname = name.split(" ")[0]
Middlename = name.split(" ")[1]
Lastname = name.split(" ")[2]
Country = name.split(" ")[3]

print (Firstname)
print (Middlename)
print (Lastname)
print (Country)


# In[27]:


Food = "Rice Beans Eba Yam"
Food


# In[28]:


Food.split (" ")


# In[31]:


Breakfast = Food.split(" ")[0]
Brunch = Food.split(" ")[1]
Lunch = Food.split(" ")[2]
Dinner = Food.split(" ")[3]

print (Breakfast)
print (Brunch)
print (Lunch)
print (Dinner)


# ## Dictionary

# In[45]:


userDetails = {"name" : "Yusuf", "password" : "Swiftss", "Country" : "China"}
userDetails


# In[46]:


userDetails.keys ( )


# In[47]:


userDetails.values ( )


# In[51]:


userDetails.items ( )


# In[37]:


userDetails ["name"]


# In[38]:


userDetails ["password"]


# Iterating over all of the values

# In[56]:


for i in userDetails.values():
    print(i)


# In[57]:


for me in userDetails.keys():
    print(me)


# Iterate over both keys and values

# In[75]:


for n, p in userDetails.items():
    print (n)
    print (p)
    
    


# In[78]:


my_list = [ ]
for number in range (0, 10000):
    if number %50 == 0:
        my_list.append(number)
my_list


# In[79]:


my_list = [number for number in range (0, 10000) if number %5 == 0]
my_list


# In[316]:


import numpy as np


# Creating an array

# In[81]:


h = np.array ([4, 5, 6])
h


# In[82]:


z_list = [6, 7, 9]
z = np.array (z_list)
z


# Multi-dimentional Array

# In[85]:


j = np.array ([[3, 5, 7, 9], [4, 7, 10, 12], [5, 8, 12, 20]])   #multi_dimentional array
j


# Array Dimension

# In[88]:


np.shape (j)


# In[99]:


n = np.arange  (0, 30, 2)
n


# .reshape

# In[93]:


d = np.array ([[2, 3, 4, 5, 7], [3, 2, 60, 70, 40], [11, 23, 44, 10, 55], [2, 45, 21, 45, 20], [21, 46, 71, 25, 100]])
d


# In[94]:


np.shape (d)


# In[101]:


n.reshape (5, 3)


# In[102]:


d.reshape (25)


# .ones

# In[104]:


np.ones ([5, 3])


# .zeros

# In[105]:


np.zeros ([4, 2])


# .eye

# In[319]:


np.eye (5)


# #### Operations in Numpy

# In[318]:


f = np.array([2, 4, 6])
f


# In[327]:


g = np.array([5, 10, 15])
g


# In[328]:


print (f + g)


# In[329]:


print (f - g)


# In[331]:


print (f * g)


# In[332]:


print (f / g)


# In[334]:


n = np.arange  (0, 20, 2)
n


# In[339]:


j= n.reshape(5, 2)
j


# In[340]:


j.T


# In[341]:


t = np.array ([5, 10, 15, 6, 9, 13, 14, 16])
t


# In[342]:


t.sum()


# In[343]:


t.max()


# In[344]:


t.min()


# In[345]:


t.mean()


# In[346]:


t.std()


# In[347]:


t.argmax()


# In[348]:


t.argmin()


# In[349]:


s=np.arange(0, 13,1)**2
s


# In[350]:


s[0],s[5], s[7]


# In[352]:


s[2:7]


# In[353]:


z = np.arange(36)
z.resize((6, 6))
z


# In[354]:


z[3,4]


# In[355]:


z[5,3]


# In[356]:


z[3, 3:7]


# In[358]:


h= z[z>25]
h


# In[360]:


h.reshape(5,2)**4


# In[333]:


print (f**3)


# In[313]:


#### Continution using your note


# In[362]:


import pandas as pd


# In[364]:


attd =['Fola', 'Omolara', 'Christine']
pd.Series (attd)


# In[365]:


attd= pd.Series (['Fola', 'Omolara', 'Christine'])
attd


# #### Creating pandas using dictionary

# In[366]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Takwuando': 'South Korea'}
s= pd.Series(sports)
s


# In[367]:


s.index


# In[368]:


l = pd.Series(['Tiger', 'Bear', 'Moose'],
    index= ['India', 'America', 'Canada'])
l


# In[369]:


l.iloc[2]


# In[373]:


l.loc['India']


# In[3]:


import pandas as pd


# In[34]:


df = pd.read_csv('olympics.csv')


# In[374]:


l.loc['Nigeria'] = 'Cat'
l


# In[ ]:





# In[7]:


df.head() #Print only the first 5 rows of our data set


# In[7]:


len(df)


# In[36]:


df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)


# In[40]:


df.head(10)


# In[42]:


df.columns


# In[45]:


for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+ col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+ col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+ col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+ col[1:]}, inplace=True)
df.head()


# ## Querying a dataframe
# 
# Team that won at least one Gold in the summer

# In[46]:


df['Gold'] > 0


# In[47]:


only_gold = df[df['Gold'] > 0]
only_gold.head()


# ## How many teams won at least one gold medal in the summer Olympics

# In[48]:


only_gold['Gold'].count()


# In[49]:


df['Gold'].count()


# Teams that win atleast one gold in summer and winter

# In[50]:


onegoldsw= df[(df['Gold']> 0) & (df['Gold.1'] > 0)]
onegoldsw.head()


# In[23]:


len(df[(df['Gold']> 0) & (df['Gold.1'] > 0)]) # and operator & and or |


# ## Missing values
# 
#     1. looking for the main dataset source online
#     2. Dropping the missing values by row or column
#     3. filling the missing values with the mean, median and mode

# In[24]:


### gold = df.where(df['Gold'] > 0)


# In[27]:


### gold.head()


# Check if a dataframe conatains nan

# In[26]:


gold.isnull().values.any()


# In[28]:


only_gold.isnull().values.any()


# In[27]:


gold.isnull().sum()


# Dropping missing values

# In[29]:


gold.dropna(axis=0) #Drop by row


# In[30]:


len(gold.dropna(axis=0))


# In[31]:


gold.dropna(axis=1) #Drop by column


# Filling missing value

# In[32]:


df.mean()


# In[34]:


# Filling with mean
gold_fillna_mean= gold.fillna(df.mean())


# In[35]:


gold.head()


# In[36]:


gold_fillna_mean.head()


# In[37]:


df.median()


# In[38]:


# Filling with median
gold_fillna_median= gold.fillna(df.median())


# In[40]:


gold_fillna_median.head()


# In[41]:


df.mode()


# In[42]:


# Filling with mode
gold_fillna_mode= gold.fillna(df.mode())


# In[45]:


gold_fillna_mode.head()


# ## Python programming: Object Oriented Programming (OOP)
#     
# How to define a class

# In[46]:


class Employee:
    pass


# In[51]:


class Employee:
    def __init__(self, name, age):
        self.name = name
        self.age= age


# In[52]:


employee1 = Employee('John Doe', 27)


# In[53]:


employee1.name


# In[54]:


employee1.age


# In[108]:


class Employee:
    def __init__(self, name, age, department):
        self.name = name
        self.age= age
        self.department = department
        
    #instance method
    def office(self):
        return f" {self.name} works in the {self.department} department "
    def description(self):
         return f" {self.name} is {self.age} years old"
    def detail(self):
         return f" {self.name} is {self.age} years old and in {self.department} department "


# In[109]:


employee1 = Employee("John Doe", 34, "Engineering")
employee2 = Employee("Joseph Matthew", 40, "Marketing")
employee3 = Employee("Luke Jesus", 53, "Finance")


# In[110]:


employee1.office()


# In[111]:


employee1.age


# In[112]:


employee2.description()


# In[113]:


employee3.description()


# In[114]:


employee3.detail()


# ## Parent class and child class: Inheritance

# In[71]:


class Salary(Employee):   # can inherit properties and methods
    pass


# In[74]:


employeeOneSalary= Salary("Elon Musk", 55, "Engineering")


# In[75]:


employeeOneSalary.office()


# In[76]:


employeeOneSalary.name


# In[121]:


class Employee:
    def __init__(self, name, age, department, language):
        self.name = name
        self.age= age
        self.department = department
        self.language = language
        
    #instance method
    def office(self):
        return f" {self.name} works in the {self.department} department "
    def description(self):
         return f" {self.name} is {self.age} years old"
    


# In[122]:


class Salary(Employee): 
    def speak(self):
        return f" {self.name} speaks {self.language} language "


# In[123]:


employeeOneSalary= Salary("John Doe", 55, "Engineering", "English")


# In[124]:


employeeOneSalary.speak()


# In[88]:


employeeTwoSalary= Employee("John Doe", 55, "Engineering", "English")


# In[ ]:


employeeTwoSalary() #Can't work 


# ## DATA ANALYSIS

# ### Machine Learning

# In[141]:


from sklearn.datasets import make_classification, make_blobs
import matplotlib.pyplot as plt


# In[142]:


# synthetic dataset for simple regression
from sklearn.datasets import make_regression


# In[148]:


#make_regression(n_samples= 100, n_features=1)


# In[156]:


plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples = 100, n_features= 1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()


# ### Linear regression

# In[158]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[161]:


X_train, X_test, y_train, y_test = train_test_split(X_R1,y_R1,
                                                   random_state = 0)
#X_train


# In[160]:


linreg = LinearRegression().fit(X_train, y_train)


# In[168]:


#linreg = LinearRegression?


# In[ ]:


linreg = LinearRegression


# In[163]:


print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_train, y_train)))


# In[289]:


import pandas as pd


# In[287]:


dope = pd.read_csv('Veg.csv')


# In[288]:


import matplotlib.pyplot as plt


# #### BOX PLOT

# In[290]:


import seaborn as sns

sns.boxplot(data=data, x= 'Price_tk', y='Season', width=0.5)
plt.title('Boxplot for Price of Seasonal Vegetable', fontsize=16)
plt.show()


# #### Distribution Plot

# In[291]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(data, x='Price_tk', kind='kde')
plt.title('Distribution of Price of Vegetables', fontsize=16)
plt.show()


# #### Exploded Pie Chart

# In[292]:


import matplotlib.pyplot as plt
import seaborn as sns

#define data
x = data['Season'].value_counts()
labels = data['Season'].value_counts().index
myexplode = [0, 0, 0.2]

#define Seaborn color plette to use
colors = sns.color_palette('bright')[0:3]

#create pie chart
plt.pie(x, labels = labels, explode=myexplode, colors = colors, autopct= '%.0f%%')
plt.title('Exploded Pie Chart for Seasonal Vegetable', fontsize=16)
plt.show()


# #### Pie Chart

# In[293]:


import matplotlib.pyplot as plt
import seaborn as sns

#define data
x = data['Season'].value_counts()
labels = data['Season'].value_counts().index

#define Seaborn color plette to use
colors = sns.color_palette('bright')[0:3]

#create pie chart
plt.pie(x, labels = labels, colors = colors, autopct= '%.0f%%')
plt.title('Pie Chart for Seasonal Vegetable', fontsize=16)
plt.show()


# #### Histogram

# In[294]:


import matplotlib.pyplot as plt
plt.figure(figsize =(8,5))
plt.hist(data['Price_tk'], color='#066471')
plt.xlabel('Price Interval')
plt.ylabel('Frequency')
plt.title('Histogram for Price')
plt.show()


# #### Bar Chart

# In[295]:


import matplotlib.pyplot as plt
x=data ['Season']. value_counts().index
y=data ['Season']. value_counts()
plt.figure(figsize =(8,5))
plt.bar(x,y, color='#066471', width=0.5)
plt.xlabel('Seasons')
plt.ylabel('Frequency')
plt.title('Bar Plot for Season')
plt.show()


# #### Line Plot

# In[296]:


import matplotlib.pyplot as plt
plt.figure(figsize =(7,5))
plt.plot(data['Price_tk'])
plt.xlabel('No. of Data Points')
plt.ylabel('Weight (gm)')
plt.title('Line Plot for Price')
plt.show()


# #### Scatter Plot

# In[297]:


import matplotlib.pyplot as plt
plt.figure(figsize =(7,5))
plt.scatter(data['Price_tk'], data['Weight_gm'])
plt.xlabel('Price (tk)')
plt.ylabel('Weight (gm)')
plt.title('Scatter Plot Price vs Weight')
plt.show()


# ##### Continuation

# In[172]:


plt.figure(figsize=(5,4))
plt.scatter(X_R1, y_R1, marker= 'o', s=50, alpha=0.8)
plt.plot(X_R1, linreg.coef_* X_R1 + linreg.intercept_, 'r-')
plt.title('Least-squares linear regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
plt.show()


# ### Logistic Regression

# In[173]:


from sklearn.datasets import load_breast_cancer


# In[174]:


# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)


# In[175]:


cancer


# In[176]:


y_cancer


# In[177]:


X_cancer


# In[178]:


X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)


# In[179]:


from sklearn.linear_model import LogisticRegression


# In[180]:


clf = LogisticRegression().fit(X_train, y_train)


# In[181]:


print('Breast cancer dataset')
print('Accuracy of Logistic reggresion classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_train, y_train)))


# ### Multi class classification using Support vector machine (SVM)

# In[182]:


import pandas as pd


# In[183]:


fruits = pd.read_table('fruit_data_with_colors.txt')


# In[184]:


fruits.head()


# In[185]:


fruits.describe()


# In[186]:


fruits.isnull().sum()


# In[189]:


feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']


# In[200]:


from sklearn.svm import SVC


# In[201]:


X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state = 0)


# In[202]:


clf = SVC(C=5, random_state = 67).fit(X_train, y_train)


# In[203]:


clf.score(X_test, y_test)


# In[204]:


clf.score(X_train, y_train)


# In[206]:


clf.predict(X_test)


# In[28]:


import pandas as pd


# In[382]:


df = pd.DataFrame ()


# In[383]:


print (df)


# In[384]:


f = [10, 20, 30, 40, 50, 60]


# In[385]:


# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(f, columns=['Numbers'])


# In[386]:


df


# #### Creating Pandas DataFrame from lists of lists

# In[214]:


# initilize list of lists
data = [['tom', 10], ['nick', 15], ['juli', 14]]


# In[33]:


data = [['tom', 10, 40], ['nick', 15, 60], ['juli', 14, 90], ['temi', 12, 60]]
dp = pd.DataFrame (data, columns = ['Name', 'Age', 'Grade'],
                index = ['A', 'B', 'C', 'D'])
dp


# In[392]:


game = pd.read_csv('Cars.csv')
game


# In[215]:


# Creat the pandas DataFrame
df = pd.DataFrame (data, columns = ['Name', 'Age'])
df


# #### Creating DataFrame from dict of narray/list

# In[217]:


#Initialize data of lists.
kong = {'Nmae' : ["Tom", 'nick', 'krish', 'jack'],
       'Age': [20, 21, 19, 18]}

#Create DataFrame
df = pd.DataFrame(kong)

df


# #### Creating DataFrame by proving index label explicitly

# In[219]:


#Initialize data of lists.
kong = {'Name' : ["Tom", 'Jack', 'nick', 'juli'],
       'Marks': [99, 98, 95, 90]}

#Create pandas DataFrame
df = pd.DataFrame(kong, index=['rank1',
                               'rank2',
                               'rank3',
                               'rank4'])

df


# #### Creating DataFrame from list of dicts

# In[221]:


#Initialize data of lists.
kong = [{'a' : 1, 'b' : 2, 'c' : 3},
        {'a' : 10, 'b' : 20, 'c' : 30}]

#Create pandas DataFrame
df = pd.DataFrame(kong)

df


# #### Creating DataFrame from list of dicts and row indexes

# In[229]:


#Initialize data of lists.
kong = [{ 'b' : 2, 'c' : 3}, {'a' : 10, 'b' : 20, 'c' : 30}]

#Create pandas DataFrame by passing
#lists of dictionaries and row index.
df = pd.DataFrame(kong, index=['first', 'second'])

df


# #### Creating DataFrame from list of dicts and row indexes as well as column index

# In[237]:


#Initialize lists data
mate = [{'a' : 1, 'b' : 2},
        {'a' : 5, 'b' : 10},
        {'a' : 5, 'b' : 10}]


# With two column indices, values same 
# as dictionary keys
df1 = pd.DataFrame(mate, index= ['frist', 'secoud', 'third'], columns=['a', 'b', 'c'])

# With two column indices with 
# one index with other name
df2 = pd.DataFrame(mate, index= ['frist', 'secoud', 'third' ], columns=['a', 'b1', 'c'])

print (df1, "\n")

print (df2)


# #### Creating DataFrame using zip () functions.
# 
# Two lists can be merged by using list (zip()) fuction. Now, create the pandas
# DataFrame by calling pd.DataFrame()function

# In[238]:


# Initialize data of lists.
# List1
Name = ["Tom", 'nick', 'krish', 'jack']

# List2
Age = [25, 30, 26, 22]

# get the list of tuples from two lists.
# and merge them by using zip().
list_of_tuples = list(zip(Name, Age))

# Assign data to tuples.
list_of_tuples

# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(list_of_tuples,
                  columns=['Name', 'Age'])

df


# #### Creating dataframe from series
# 
# To create a dataframe from series, we must pass series as argument to DataFrame() function.

# In[239]:


# Initialize data to series.
d = pd.Series([10, 20, 30, 40])

# creates Dataframe.
df = pd.DataFrame(d)

# print the data.
df


# #### Creating DataFrame from Dictionary of series
# 
# To create DataFrame from Dict of series, dictionary can be passed to form a DataFrame. 
# The resultant index is the union of all the series of passed indexed

# In[240]:


# Initialize data to Dicts of series.
d = {'one' : pd.Series ([10, 20, 30, 40],
                        index=['a', 'b', 'c', 'd']),
     'two' : pd.Series ([10, 20, 30, 40],
                        index=['a', 'b', 'c', 'd'])}

# Creates Dataframe.
df = pd.DataFrame(d)

df


# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('Vegetables.csv')


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


data.head() #Print only the first 5 rows of our data set


# #### Frequency Table
# Frequency is the count of the number of occurrences of a value. The frequency table is the representation of frequency in a tabular form. 

# In[257]:


frequency=data.groupby(['Vegetable', 'Season']).size().reset_index(name='Frequency')
frequency


# #### Scatter Plot

# In[264]:


import matplotlib.pyplot as plt
plt.figure(figsize =(7,4))
plt.scatter(data['Price (tk)'], data['Weight (gm)'])
plt.xlabel('Price (tk)')
plt.ylabel('Weight (gm)')
plt.title('Scatter Plot Price vs Weight')
plt.show()


# In[9]:


sns.scatterplot(x='Price (tk)',y='Weight (gm)', data=data)
plt.show()


# In[17]:


sns.scatterplot(x='Price (tk)', y='Weight (gm)', data=data, , hue='Season')
plt.show()


# #### Line Plot
# By nature, the line plot is similar to the scatter plot, but the points are connected with continuous lines sequentially. Rather than data distribution, the plot is preferable for finding data flow in a two-dimensional space

# In[266]:


import matplotlib.pyplot as plt
plt.figure(figsize =(7,5))
plt.plot(data['Price (tk)'])
plt.xlabel('No. of Data Points')
plt.ylabel('Weight (gm)')
plt.title('Line Plot for Price')
plt.show()


# #### Bar Chart
# The bar chart is mainly used to represent the frequency of categorical variables with bars. Different heights of the bar indicat the frequency

# In[275]:


import matplotlib.pyplot as plt
x=data ['Season']. value_counts().index
y=data ['Season']. value_counts()
plt.figure(figsize =(8,5))
plt.bar(x,y, color='#066471', width=0.5)
plt.xlabel('Seasons')
plt.ylabel('Frequency')
plt.title('Bar Plot for Season')
plt.show()


# In[8]:


sns.countplot(x='Season', data=data)
plt.show()


# #### Histogram
# The concept of the histogram is the same as the bar chart. In a bar chart, the frequency is shown in discrete bars of categorical variables; however, a histogram shows the frequency of a continuous interval. Basically, it is used to find the frequency of continuous variables within intervals.

# In[276]:


import matplotlib.pyplot as plt
plt.figure(figsize =(8,5))
plt.hist(data['Price (tk)'], color='#066471')
plt.xlabel('Price Interval')
plt.ylabel('Frequency')
plt.title('Histogram for Price')
plt.show()


# #### Pie chart
# The pie chart shows the frequency in terms of percentage in a circular manner. Each element holds the area of the circle according to its frequency percentage.

# In[280]:


import matplotlib.pyplot as plt
import seaborn as sns   #For plotting the pie chart, i have used the seaborn libary

#define data
x = data['Season'].value_counts()
labels = data['Season'].value_counts().index

#define Seaborn color plette to use
colors = sns.color_palette('bright')[0:3]

#create pie chart
plt.pie(x, labels = labels, colors = colors, autopct= '%.0f%%')
plt.title('Pie Chart for Seasonal Vegetable', fontsize=16)
plt.show()


# #### Exploded Pie Chart
# The pie chart and exploded pie chart are the same. In the exploded pie chart, you can segregate a portion of a pie chart to highlight elements.

# In[281]:


import matplotlib.pyplot as plt
import seaborn as sns

#define data
x = data['Season'].value_counts()
labels = data['Season'].value_counts().index
myexplode = [0, 0, 0.2]

#define Seaborn color plette to use
colors = sns.color_palette('bright')[0:3]

#create pie chart
plt.pie(x, labels = labels, explode=myexplode, colors = colors, autopct= '%.0f%%')
plt.title('Exploded Pie Chart for Seasonal Vegetable', fontsize=16)
plt.show()


# #### Distribution Plot
# We can understand how a continuous variable's values are distributed with the distrution plot

# In[282]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(data, x='Price (tk)', kind='kde')
plt.title('Distribution of Price of Vegetables', fontsize=16)
plt.show()


# #### Box Plot
# The following graph is a box plot where the box indicates the range between the 1st(Q1) and 3rd(Q3) quartiles. The vertical line of the left and right sides indicates the outlier fences, values beyoud the lines are considered outliers. The yellow vertical line inside the box indicates the 2nd quartile (Q2)

# In[5]:


import seaborn as sns
# This plot is used to find the central tendency, and dispersion can also be estimated.
# A codified example of a box plot is given below

sns.boxplot(data=data, x= 'Price (tk)', y='Season', width=0.5)
plt.title('Boxplot for Price of Seasonal Vegetable', fontsize=16)
plt.show()


# In[6]:


import seaborn as sns


# In[7]:


data.nunique()


# In[ ]:




