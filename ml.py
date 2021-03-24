import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataset

## Generate Train Population and Plot Problem Definition Graph
fig = plt.figure()
ax = fig.gca()
fig.suptitle("Sample of Maintenance Sensor Readings. 1 = Failure.\nTraining Population.")
ax.set_xlabel('Time')
ax.set_ylabel('Sensor Reading')
pop = dataset.create_population(1000)
sensor_records = []
for i,r in pop.iterrows():
    fr = dataset.create_sensor_record(r)
    sensor_records.append(fr)
    if i < 10: plt.plot(fr['t'], fr['sensor_reading'])
all_sensor_records = pd.concat(sensor_records, axis=0, ignore_index=True)
fig.gca().legend(['Robot ' + str(i) for i in range(10)])
ax.axhline(y=1, linestyle='dashed', color='red')

## Perform Multivariate Polynomial Regression on the Data
poly_transformer = PolynomialFeatures(degree=3)
regr = linear_model.LinearRegression()
all_augmented_features = []
for i, robot in pop.iterrows():
    augmented_features = pd.DataFrame(np.repeat(pd.DataFrame(robot).T.values, 1000, axis=0), columns=pop.columns)
    augmented_features['t'] = np.linspace(0, 1000, 1000)
    all_augmented_features.append(augmented_features)
all_augmented_features = pd.concat(all_augmented_features, axis=0, ignore_index=True)
all_linearized_features = poly_transformer.fit_transform(all_augmented_features)
regr.fit(all_linearized_features, all_sensor_records['sensor_reading'])

## Store Feature / Coefficient Mapping for Later
corrected_names = poly_transformer.get_feature_names()
for i in range(len(corrected_names)):
    for j in range(len(dataset.features)+1):
        corrected_names[i] = corrected_names[i].replace('x'+str(j), (dataset.features+['t'])[j])
coeff_df = pd.DataFrame({
    'name': corrected_names,
    'val': regr.coef_
}).sort_values('val', ascending=False, ignore_index=True)[:10]

## Get New (Test) Population
pop = dataset.create_population(12)
sensor_records = []
for i,r in pop.iterrows():
    fr = dataset.create_sensor_record(r)
    sensor_records.append(fr)
    plt.plot(fr['t'], fr['sensor_reading'])
all_sensor_records = pd.concat(sensor_records, axis=0, ignore_index=True)

## Plot Fit Results
fig2, ax2 = plt.subplots(3,4)
ax2 = ax2.reshape(-1)
fig2.suptitle("Maintenance Sensor Readings & ML Fit Curve. 1 = Failure.\nTesting Population.")
fig2.text(0.5, 0.04, 'Time', ha='center')
fig2.text(0.04, 0.5, 'Sensor Reading', va='center', rotation='vertical')
for i in range(12):
    ax2[i].plot(sensor_records[i]['t'], sensor_records[i]['sensor_reading'])
    y_pred = regr.predict(all_linearized_features[1000*i:1000*(i+1),:])
    ax2[i].plot(sensor_records[i]['t'], y_pred)
    ax2[i].axhline(y=1, linestyle='dashed', color='red')
handles, labels = ax2[-1].get_legend_handles_labels()
plt.subplots_adjust(bottom=0.2)
fig2.legend(handles, labels=['Actual Sensor Readings', 'Predicted Sensor Readings'], loc='lower left')

## Plot Coefficients for Inference
C = min(coeff_df['val'])
coeff_df['val'] = coeff_df['val'].div(np.abs(C))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(coeff_df)
    print("C = " + str(C))
plt.show()