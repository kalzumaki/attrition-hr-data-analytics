import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

df = pd.read_csv('../datasets/Employee-Attrition.csv')
df.head()

# viewing dataset info like columns with its total of rows.
df.info()

# displaying numerical data (statistical methods)
df.describe()

pd.isna(df)
pd.isna(df).sum()

# total number of elements in the dataframe
df.size # --->> (51450) so to get the total of rows we need to divide the total of size 51450 / 35 (columns) and the answer should be the total of rows = 1470

# since we don't have any missing value we will proceed on identifying outliers (column by column)

# find all columns that are int and float data types
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns


outlier_counts = [] # --->> a variable for empty list to store results

for col in numeric_cols: # --->> creating a for loop to check all columns that has int and float data types
    
    q1 = df[col].quantile(0.25) # --->> 25th percentile
    q3 = df[col].quantile(0.75) # --->> 75th percentile
    
    iqr = q3 - q1 # --->> formula on getting IQR
    
    lower = q1 - 1.5 * iqr # --->> lower bound for outliers
    upper = q3 + 1.5 * iqr # --->> higher bound for outliers
    
    outliers = df[(df[col] < lower) | (df[col] > upper)] # --->> formula for finding outliers on each columns
    outlier_counts.append({'column': col, 'outlier_count': len(outliers)}) # --->> added dictionary with its name and the number of outliers found to `outlier_count` column
    
    
outliers = pd.DataFrame(outlier_counts) # --->> the list of dictionaries were converted into DataFrame

outliers # -->> results


n_cols = len(numeric_cols) # --->> counts total of columns (26)
n_per_row = 5  # --->> Number of plots per row
n_rows = (n_cols + n_per_row - 1) // n_per_row  # --->> Ceiling division for rows. formula (26 + 5 - 1) // 5 = 6. 6 rows 5 columns will be shown on the boxplot graphs

fig, axes = plt.subplots(n_rows, n_per_row, figsize=(5 * n_per_row, 5 * n_rows)) # --->> creates the figure and subplots grid

axes = axes.flatten() # --->> Flattens the axes array into a 1D list so you can easily iterate and assign each plot.

for i, col in enumerate(numeric_cols): # --->> loops through each numeric columns. enumerate(numeric_cols) gives you both the index (i) and column name (col).
    sns.boxplot(y=df[col], ax=axes[i]) # --->> sns.boxplot(y=df[col], ax=axes[i]): plots a boxplot for the column’s data on its assigned subplot.
    axes[i].set_title(f'Boxplot of {col}', fontsize=13) # --->> sets a title for each subplot, showing the column name.
    axes[i].set_xlabel('') # --->> removes the x-axis label (since it’s not needed for these boxplots).
    axes[i].set_ylabel(col, fontsize=11) # --->> adds the column name as the y-axis label for clarity.

# Remove unused subplot axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Outliers for all Numeric Columns", fontsize=22, fontweight='bold', y=1.03) # --->> adds a main title to the entire figure, describing what the plot shows.
plt.tight_layout() # --->> adjusts subplot spacing so titles and labels don’t overlap.
plt.show()