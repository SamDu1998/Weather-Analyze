import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
from sklearn.svm import SVR

df_ferrara = pd.read_csv('WeatherData/ferrara_270615.csv')
df_milano = pd.read_csv('WeatherData/milano_270615.csv')
df_mantova = pd.read_csv('WeatherData/mantova_270615.csv')
df_ravenna = pd.read_csv('WeatherData/ravenna_270615.csv')
df_torino = pd.read_csv('WeatherData/torino_270615.csv')
df_asti = pd.read_csv('WeatherData/asti_270615.csv')
df_bologna = pd.read_csv('WeatherData/bologna_270615.csv')
df_piacenza = pd.read_csv('WeatherData/piacenza_270615.csv')
df_cesena = pd.read_csv('WeatherData/cesena_270615.csv')
df_faenza = pd.read_csv('WeatherData/faenza_270615.csv')

# y1 = df_ravenna['temp']
# x1 = df_ravenna['day']
# y2 = df_faenza['temp']
# x2 = df_faenza['day']
# y3 = df_cesena['temp']
# x3 = df_cesena['day']
# y4 = df_milano['temp']
# x4 = df_milano['day']
# y5 = df_asti['temp']
# x5 = df_asti['day']
# y6 = df_torino['temp']
# x6 = df_torino['day']

dist = [df_ravenna['dist'][0],
    df_cesena['dist'][0],
    df_faenza['dist'][0],
    df_ferrara['dist'][0],
    df_bologna['dist'][0],
    df_mantova['dist'][0],
    df_piacenza['dist'][0],
    df_milano['dist'][0],
    df_asti['dist'][0],
    df_torino['dist'][0]
]

temp_max = [df_ravenna['temp'].max(),
    df_cesena['temp'].max(),
    df_faenza['temp'].max(),
    df_ferrara['temp'].max(),
    df_bologna['temp'].max(),
    df_mantova['temp'].max(),
    df_piacenza['temp'].max(),
    df_milano['temp'].max(),
    df_asti['temp'].max(),
    df_torino['temp'].max()
]

temp_min = [df_ravenna['temp'].min(),
    df_cesena['temp'].min(),
    df_faenza['temp'].min(),
    df_ferrara['temp'].min(),
    df_bologna['temp'].min(),
    df_mantova['temp'].min(),
    df_piacenza['temp'].min(),
    df_milano['temp'].min(),
    df_asti['temp'].min(),
    df_torino['temp'].min()
]

# day_ravenna = [parser.parse(x) for x in x1]
# day_faenza = [parser.parse(x) for x in x2]
# day_cesena = [parser.parse(x) for x in x3]
# day_milano = [parser.parse(x) for x in x4]
# day_asti = [parser.parse(x) for x in x5]
# day_torino = [parser.parse(x) for x in x6]

# fig, ax = plt.subplots()
# plt.xticks(rotation=70)

# hours = mdates.DateFormatter('%H:%M')
# ax.xaxis.set_major_formatter(hours)

# ax.plot(day_ravenna, y1, 'r', day_faenza, y2, 'r', day_cesena, y3, 'r')
# ax.plot(day_milano, y4, 'g', day_asti, y5, 'g', day_torino, y6, 'g')

# fig, ax = plt.subplots()
# ax.plot(dist, temp_max, 'ro')

# dist1 = dist[0:5]
# dist2 = dist[5:10]

# dist1 = [[x] for x in dist1]
# dist2 = [[x] for x in dist2]

# temp_max1 = temp_max[0:5]
# temp_max2 = temp_max[5:10]

# svr_lin1 = SVR(kernel='linear', C=1e3)
# svr_lin2 = SVR(kernel='linear', C=1e3)

# svr_lin1.fit(dist1, temp_max1)
# svr_lin2.fit(dist2, temp_max2)

# xp1 = np.arange(10, 100, 10).reshape((9, 1))
# xp2 = np.arange(50, 400, 50).reshape((7, 1))
# yp1 = svr_lin1.predict(xp1)
# yp2 = svr_lin2.predict(xp2)

# ax.set_xlim(0, 400)

# ax.plot(xp1, yp1, c='b', label='Strong sea effect')
# ax.plot(xp2, yp2, c='g', label='Light sea effect')

# plt.axis((0,400,15,25))
# plt.plot(dist, temp_min, 'bo')

y1 = df_ravenna['humidity']
x1 = df_ravenna['day']
y2 = df_faenza['humidity']
x2 = df_faenza['day']
y3 = df_cesena['humidity']
x3 = df_cesena['day']
y4 = df_milano['humidity']
x4 = df_milano['day']
y5 = df_asti['humidity']
x5 = df_asti['day']
y6 = df_torino['humidity']
x6 = df_torino['day']

fig, ax = plt.subplots()
plt.xticks(rotation=70)

day_ravenna = [parser.parse(x) for x in x1]
day_faenza = [parser.parse(x) for x in x2]
day_cesena = [parser.parse(x) for x in x3]
day_milano = [parser.parse(x) for x in x4]
day_asti = [parser.parse(x) for x in x5]
day_torino = [parser.parse(x) for x in x6]

hours = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(hours)

ax.plot(day_ravenna, y1, 'r', day_faenza, y2, 'r', day_cesena, y3, 'r')
ax.plot(day_milano, y4, 'g', day_asti, y5, 'g', day_torino, y6, 'g')

plt.show()

hum_max = [df_ravenna['humidity'].max(),
df_cesena['humidity'].max(),
df_faenza['humidity'].max(),
df_ferrara['humidity'].max(),
df_bologna['humidity'].max(),
df_mantova['humidity'].max(),
df_piacenza['humidity'].max(),
df_milano['humidity'].max(),
df_asti['humidity'].max(),
df_torino['humidity'].max()
]

plt.plot(dist, hum_max, 'bo')
plt.show()

hum_min = [df_ravenna['humidity'].min(),
df_cesena['humidity'].min(),
df_faenza['humidity'].min(),
df_ferrara['humidity'].min(),
df_bologna['humidity'].min(),
df_mantova['humidity'].min(),
df_piacenza['humidity'].min(),
df_milano['humidity'].min(),
df_asti['humidity'].min(),
df_torino['humidity'].min()
]

plt.plot(dist, hum_min, 'bo')
plt.show()

plt.plot(df_ravenna['wind_deg'], df_ravenna['wind_speed'], 'ro')
plt.show()