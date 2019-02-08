import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# print data description
print(open("OnlineNewsPopularity.names", encoding="utf8").read())

# load csv
csv = pd.read_csv("OnlineNewsPopularity.csv")

# remove outliers
csv = csv[np.abs(csv[' shares']-csv[' shares'].mean()) <= (3*csv[' shares'].std())]

# data analysis
print("\n"+ str(csv.describe()))

# get all but first (url, non-predictive), second (timedelta, predictive) and last (shares, target) column
data = csv.iloc[:,2:-1]
# get the target column (shares)
target = csv.iloc[:,-1]

# normalise data (only numerical values)
numerical = [' n_tokens_title', ' n_tokens_content', ' num_hrefs', ' num_self_hrefs', ' num_imgs',' num_videos', ' average_token_length',' num_keywords',' self_reference_min_shares',' self_reference_max_shares', ' self_reference_avg_sharess']
data[numerical] = MinMaxScaler().fit_transform(data[numerical])

# convert dataframe to value array
data = data.values.astype(np.float)
target = target.values.astype(np.float)

# show input data shape
print("\nShape of data array: " + str(data.shape))
print("Shape of target array: " + str(target.shape))

# split data
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# train linear regression
linear_regression = LinearRegression()
linear_regression.fit(data_train, target_train)

# show example article share value prediction
id = 1337
linear_regression_prediction = linear_regression.predict(data_test[id,:].reshape(1,-1))
print("Model predicted for article {0} value {1}".format(id, linear_regression_prediction))
print("Real value for article \"{0}\" is {1}".format(id, target_test[id]))

# evaluate the model
print("Mean squared error of a learned model: %.2f" % mean_squared_error(target_test, linear_regression.predict(data_test)))
print('Variance score: %.2f' % r2_score(target_test, linear_regression.predict(data_test)))
scores = cross_val_score(LinearRegression(), data, target, cv=4)
print(scores)

# plot 500 example scores
some_X_data = data_train[:500]
some_y_data = target_train[:500]
df_someXdata = pd.DataFrame(linear_regression.predict(some_X_data),list(some_y_data) )
df_someXdata.reset_index(level=0, inplace=True)
df_someXdata_LR = df_someXdata.rename(index=str, columns={"index": "Actual shares", 0: "Predicted shares"})
f, ax = plt.subplots(figsize=(17, 3))
sns.regplot(x=df_someXdata_LR["Actual shares"], y=df_someXdata_LR["Predicted shares"])
plt.show()