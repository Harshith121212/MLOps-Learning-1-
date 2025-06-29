medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'

from urllib.request import urlretrieve
urlretrieve(medical_charges_url, 'medical.csv')

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

medical_df = pd.read_csv('medical.csv')

mean_age = medical_df['age'].mean()
print(mean_age)

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

fig = px.histogram(medical_df, x='age', marginal='box', nbins=47, title='Age Distribution')
fig.update_layout(bargap=0.1)
fig.show()

fig1 = px.histogram(medical_df, x='bmi', marginal='box', title='BMI Distribution')
fig1.update_layout(bargap=0.1)
fig1.show()

fig2 = px.histogram(medical_df, x='charges', marginal='box', color='region', color_discrete_sequence=['green', 'grey', 'yellow', 'blue'], title='medical charges')
fig2.update_layout(bargap=0.1)
fig2.show()

px.histogram(medical_df, x='smoker', color='sex', title='smoker')

fig6 = px.scatter(medical_df, x='age', y='charges', color='smoker', opacity=0.8, hover_data=['sex'], title='Age vs Charges')
fig6.update_traces(marker_size=5)
fig6.show()

fig9 = px.histogram(medical_df, x='region', marginal='box', color='region', color_discrete_sequence=['green', 'pink'])
fig9.show()

smoker_value = {'no': 0, 'yes': 1}
smoker_num = medical_df.smoker.map(smoker_value)

corr_of_smokers = medical_df.charges.corr(smoker_num)
print(corr_of_smokers)

non_smoker_df = medical_df[medical_df.smoker == 'no']
fig32 = px.scatter(non_smoker_df, x='age', y='charges', color='smoker', opacity=0.8, hover_data=['sex'], title='Age vs Charges')
fig32.update_traces(marker_size=5)
fig32.show()

def estimate_charges(age, w, b):
    return w * age + b

w = 50
b = 100

ages = non_smoker_df.age
estimated_charges = estimate_charges(ages, w, b)
print(estimated_charges)

plt.scatter(ages, estimated_charges)
plt.xlabel('Age')
plt.ylabel('Estimated Charges')

target = non_smoker_df.charges
plt.plot(ages, estimated_charges, 'r', alpha=0.9)
plt.scatter(ages, target, s=8, alpha=0.8)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend(['Estimated Charges', 'Actual Charges'])

def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    estimated_charges = estimate_charges(ages, w, b)
    plt.plot(ages, estimated_charges, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.8)
    plt.xlabel('Ages')
    plt.ylabel('Charges')
    plt.legend(['Estimated Charges', 'Actual Charges'])

try_parameters(60, 200)
try_parameters(200, -200)

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

w = 50
b = 100
targets = non_smoker_df['charges']
predicted = estimate_charges(non_smoker_df.age, w, b)

t = 300
u = -3250
try_parameters(t, u)
print(rmse(targets, estimate_charges(non_smoker_df.age, t, u)))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges

print('input shape:', inputs.shape)
print('target shape:', targets.shape)

model.fit(inputs, targets)
print(model.predict(np.array([[30], [45], [61]])))

predictions = model.predict(inputs)
print(rmse(targets, predictions))

plt.plot(ages, predictions, 'r', alpha=0.9)
plt.scatter(ages, target, s=8, alpha=0.8)
plt.xlabel('Ages')
plt.ylabel('Charges')
plt.legend(['Estimated Charges', 'Actual Charges'])

print(model.coef_)
print(model.intercept_)

inputs, targets = non_smoker_df[['age', 'bmi', 'children']], non_smoker_df.charges
model = LinearRegression().fit(inputs, targets)
predictions = model.predict(inputs)

inputs, targets = medical_df[['age', 'bmi', 'children']], medical_df.charges
model = LinearRegression().fit(inputs, targets)
predictions = model.predict(inputs)
print(model.predict([[22, 23, 0]]))

from sklearn.preprocessing import StandardScaler

numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])

print(scaler.mean_)
print(scaler.var_)
