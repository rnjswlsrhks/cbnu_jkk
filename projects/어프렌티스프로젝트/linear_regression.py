from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("elec_power.csv")
df.head()

# AT    시간별 평균 주변 온도
# V     시간별 평균 배출 진공도
# AP    시간별 평균 주변 압력
# RH    시간별 평균 주변 습도
# EP    공장의 시간당 전기에너지 출력

x = df[['AT','V','AP','RH']]
y = df[['EP']]
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

#my_test = [[23.64,58.47,1011.4,74.2]]
#my_predict = mlr.predict(my_test)
#print(my_predict)

y_predict = mlr.predict(x_test)

print("회귀계수 : ")
print(mlr.coef_)

print("결정계수 : ")
print(mlr.score(x_train, y_train))

plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual EP")
plt.ylabel("Predicted EP")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

plt.scatter(df[['AT']], df[['EP']], alpha=0.4)
plt.xlabel("AT")
plt.show()

plt.scatter(df[['V']], df[['EP']], alpha=0.4)
plt.xlabel("V")
plt.show()

plt.scatter(df[['AP']], df[['EP']], alpha=0.4)
plt.xlabel("AP")
plt.show()

plt.scatter(df[['RH']], df[['EP']], alpha=0.4)
plt.xlabel("RH")
plt.show()

