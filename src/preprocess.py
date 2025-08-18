import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
df = pd.read_csv('../datasets/Employee-Attrition.csv')
df


# lets check for missing values
df.isnull().sum() # result: no missing value

# lets check datatypes

df.tail() # check how many rows (0-1469) total of 1470

# next will be converting into categorical columns
    # converting object types to category dtype  for efficiency  and clarity

df.shape

cat_cols = df.select_dtypes(include='object').columns
# now lets use for loop since it has 34 columns
for col in cat_cols:
    df[col] = df[col].astype('category')

print("Converted dtypes:\n", df.info()) # 35 columns (0-34)
# check if there is duplication
df.duplicated().sum() # result: no duplication (0)

# df = df.drop_duplicates() # this doesn't matter since there is no duplication

# lets check outliers

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Display outliers per column using IQR method
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"Outliers for {col}: {len(outliers)}")
# since the outliers has no errors we should keep them

print("================================")
# lets check if there is typos
for col in cat_cols:
    print(f"{col}: {df[col].unique()}")
    
# so far there's no typos
print("==================================")

# Check for columns with only one unique value
# Remove constant columns directly
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
# this are the results:
# Constant column: EmployeeCount = [1]
# Constant column: Over18 = ['Y']
# Categories (1, object): ['Y']
# Constant column: StandardHours = [80]

# since they are redundant lets remove them
df = df.drop(columns=constant_cols)
print(f"Removed columns: {constant_cols}")
print(f"New shape: {df.shape}")

# lets check for impossible combinations
print("===========================")
# YearsAtCompany should not exceed TotalWorkingYears
print(df[df['YearsAtCompany'] > df['TotalWorkingYears']]) # results: all good

# Age vs experience consistency
print(df[df['Age'] < df['TotalWorkingYears'] + 18])   # results: all good

print("==============================================")
print("Final Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

df.to_csv('Employee-Attrition-Cleaned.csv', index=False) # save the cleaned dataset