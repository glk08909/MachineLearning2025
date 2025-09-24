import pandas as pd
import urllib.request
import numpy as np

# Download dataset
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
urllib.request.urlretrieve(url, "car_fuel_efficiency.csv")

# Load dataset
df = pd.read_csv("car_fuel_efficiency.csv")

# Q1: Pandas version
print("Pandas version:", pd.__version__)

# Q2: Number of records
print("Number of records:", len(df))

# Q3: Origin counts
print(df['origin'].value_counts())

# Q4: Columns with missing values
num_missing_cols = df.isnull().sum()[df.isnull().sum() > 0].shape[0]
print("Number of columns with missing values:", num_missing_cols)

# Q5: Max fuel efficiency
max_mpg = df['fuel_efficiency_mpg'].max()
print("Maximum fuel efficiency (mpg):", max_mpg)

# Q6: Median horsepower and fill missing values
median_before = df['horsepower'].median()
print("Median before filling missing values:", median_before)

mode_hp = df['horsepower'].mode()[0]
df['horsepower'].fillna(mode_hp, inplace=True)
median_after = df['horsepower'].median()
print("Median after filling missing values:", median_after)

if median_after > median_before:
    print("Answer: Yes, it increased")
elif median_after < median_before:
    print("Answer: Yes, it decreased")
else:
    print("Answer: No")

# Q7: Linear regression
df_asia = df[df['origin'] == 'asia']
X = df_asia[['vehicle_weight', 'model_year']].iloc[:7].to_numpy()


XTX_inv = np.linalg.pinv(X.T @ X)  # pseudo-inverse to avoid singular error
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print("Sum of all elements in w:", w.sum())
