import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# setup plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.figure(figsize=(10, 6))


df = pd.read_csv('../datasets/Employee-Attrition-Cleaned.csv')
print("Dataset loaded for EDA")
print(f"Shape: {df.shape}") #(rows, columns)
print(f"Columns: {df.columns.tolist()}") # display all columns

# ============== 1. SUMMARY STATISTICS ==============
print("\n" + "="*60)
print("1. SUMMARY STATISTICS")
print("="*60)

# basic info
print("\nDataset Info:")
print(df.info())

# summary statistics for all columns
print("\nSummary Statistics:")
print(df.describe(include='all'))

# target variable distribution

print(df['Attrition'].value_counts())
print("\nAttrition Percentage:")
print(df['Attrition'].value_counts(normalize=True) * 100)

# ============== 2. VISUALIZATIONS ==============

print("\n" + "="*60)
print("2. VISUALIZATIONS")
print("="*60)

# target variable distribution
# shows how many employees left vs stayed (visual confirmation of the problem scale)
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Attrition')
plt.title('Employee Attrition Distribution')
plt.ylabel('Count')
# annotate each bar
for p in ax.patches:
    ax.annotate(
        str(int(p.get_height())),
        (p.get_x() + p.get_width() / 2, p.get_height() + 10),
        ha='center', va='bottom'
    )

plt.show()

# attrition by department
# shows which departments lose more employees
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition by Department')
plt.xticks(rotation=45)
plt.show()


# attrition by job role
# shows which specific jobs have retention problems Business insight: "Sales Representatives are leaving more than managers"
plt.figure(figsize=(14, 8))
sns.countplot(data=df, y='JobRole', hue='Attrition', order=df['JobRole'].value_counts().index)
plt.title('Attrition by Job Role')
plt.show()

# age distribution
# shows do younger/older employees leave more?
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='Age', hue='Attrition', bins=20, alpha=0.7)
plt.title('Age Distribution by Attrition')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Attrition', y='Age')
plt.title('Age vs Attrition')
plt.tight_layout()
plt.show()

# monthly income distribution
# shows do low-paid employees leave more? 
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='MonthlyIncome', hue='Attrition', bins=30, alpha=0.7)
plt.title('Monthly Income Distribution by Attrition')

# plt.subplot(1, 2, 2)
# sns.boxplot(data=df, x='Attrition', y='MonthlyIncome')
# plt.title('Monthly Income vs Attrition')
# plt.tight_layout()
# plt.show()

# years at company
#shows when do employees typically leave?
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='YearsAtCompany', hue='Attrition', bins=20, alpha=0.7)
plt.title('Years at Company Distribution by Attrition')

# plt.subplot(1, 2, 2)
# sns.boxplot(data=df, x='Attrition', y='YearsAtCompany')
# plt.title('Years at Company vs Attrition')
# plt.tight_layout()
# plt.show()

# work-life balance and job satisfaction
# shows how satisfaction levels relate to leaving
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.countplot(data=df, x='WorkLifeBalance', hue='Attrition', ax=axes[0,0])
axes[0,0].set_title('Work-Life Balance vs Attrition')

sns.countplot(data=df, x='JobSatisfaction', hue='Attrition', ax=axes[0,1])
axes[0,1].set_title('Job Satisfaction vs Attrition')

sns.countplot(data=df, x='EnvironmentSatisfaction', hue='Attrition', ax=axes[1,0])
axes[1,0].set_title('Environment Satisfaction vs Attrition')

sns.countplot(data=df, x='JobInvolvement', hue='Attrition', ax=axes[1,1])
axes[1,1].set_title('Job Involvement vs Attrition')

plt.tight_layout()
plt.show()

# ============== 3. CORRELATION ANALYSIS ==============
print("\n" + "="*60)
print("3. CORRELATION ANALYSIS")
print("="*60)


# get numeric columns

# Which variables move together (positive correlation)
# Which variables move opposite (negative correlation)
# Business insight: "Age and YearsAtCompany are highly correlated (0.68) - makes sense!"
num_cols = df.select_dtypes(include=['int64']).columns
print(f"Numeric columns for correlation: {num_cols.tolist()}")

# correlation matrix
correlation_matrix = df[num_cols].corr()

# Plot correlation heatmap
# find relationships between numeric variables
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()


# find highly correlated pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((correlation_matrix.columns[i], 
                                  correlation_matrix.columns[j], 
                                  correlation_matrix.iloc[i, j]))

print("\nHighly Correlated Pairs (|correlation| > 0.7):")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
    
# ============== ANSWERING YOUR BUSINESS QUESTIONS ==============
print("\n" + "="*60)
print("ANSWERING YOUR SPECIFIC BUSINESS QUESTIONS")
print("="*60)

# 1. What are the main reasons employees leave the company?
print('\n1. What are the main reasons employees leave the company?')
print("="*50)

# analyze satisfaction levels for employees who left
left_employees = df[df['Attrition'] == 'Yes']

print(" Satisfaction Levels of Employees Who Left:")
print(f"Job Satisfaction (1-4 scale):")
job_sat_left = left_employees['JobSatisfaction'].value_counts().sort_index()
for level, count in job_sat_left.items():
    percentage = (count / len(left_employees)) * 100
    print(f"   Level {level}: {count} employees ({percentage:.1f}%)")

print(f"\nWork-Life Balance (1-4 scale):")
wlb_left = left_employees['WorkLifeBalance'].value_counts().sort_index()
for level, count in wlb_left.items():
    percentage = (count / len(left_employees)) * 100
    print(f"   Level {level}: {count} employees ({percentage:.1f}%)")

print(f"\nEnvironment Satisfaction (1-4 scale):")
env_sat_left = left_employees['EnvironmentSatisfaction'].value_counts().sort_index()
for level, count in env_sat_left.items():
    percentage = (count / len(left_employees)) * 100
    print(f"   Level {level}: {count} employees ({percentage:.1f}%)")

# overtime analysis
overtime_left = left_employees['OverTime'].value_counts()
print(f"\nOvertime Status of Employees Who Left:")
for status, count in overtime_left.items():
    percentage = (count / len(left_employees)) * 100
    print(f"   {status}: {count} employees ({percentage:.1f}%)")

# 2. Which departments or teams have the highest turnover rates?
print('\n\n2. Which departments or teams have the highest turnover rates?')
print("="*50)

dept_attrition = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
dept_counts = df.groupby('Department').size()
dept_left_counts = df[df['Attrition'] == 'Yes'].groupby('Department').size()

print(" Department Turnover Analysis:")
for dept in dept_attrition.index:
    total = dept_counts[dept]
    left = dept_left_counts.get(dept, 0)
    rate = dept_attrition[dept]
    print(f"   {dept}:")
    print(f"      Total employees: {total}")
    print(f"      Employees left: {left}")
    print(f"      Turnover rate: {rate:.1f}%")
    print()

# job roles with highest turnover
print("💼 Job Roles with Highest Turnover:")
role_attrition = df.groupby('JobRole')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
for role, rate in role_attrition.head(5).items():
    role_total = df[df['JobRole'] == role].shape[0]
    role_left = df[(df['JobRole'] == role) & (df['Attrition'] == 'Yes')].shape[0]
    print(f"   {role}: {rate:.1f}% ({role_left}/{role_total} employees)")

# 3. Are there any patterns in employee attrition based on age, experience, or job level?
print('\n\n3. Are there any patterns in employee attrition based on age, experience, or job level?')
print("="*50)

# age patterns
print("👥 Age Patterns:")
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
age_attrition = df.groupby('AgeGroup', observed=True)['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
for age_group, rate in age_attrition.items():
    age_total = df[df['AgeGroup'] == age_group].shape[0]
    age_left = df[(df['AgeGroup'] == age_group) & (df['Attrition'] == 'Yes')].shape[0]
    print(f"   {age_group} years: {rate:.1f}% turnover ({age_left}/{age_total} employees)")

# experience patterns (Total Working Years)
print("\n💼 Experience Patterns:")
df['ExperienceGroup'] = pd.cut(df['TotalWorkingYears'], bins=[0, 5, 10, 20, 50], labels=['0-5 years', '6-10 years', '11-20 years', '20+ years'])
exp_attrition = df.groupby('ExperienceGroup', observed=True)['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
for exp_group, rate in exp_attrition.items():
    exp_total = df[df['ExperienceGroup'] == exp_group].shape[0]
    exp_left = df[(df['ExperienceGroup'] == exp_group) & (df['Attrition'] == 'Yes')].shape[0]
    print(f"   {exp_group}: {rate:.1f}% turnover ({exp_left}/{exp_total} employees)")

# job Level patterns
print("\n Job Level Patterns:")
level_attrition = df.groupby('JobLevel')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
for level, rate in level_attrition.items():
    level_total = df[df['JobLevel'] == level].shape[0]
    level_left = df[(df['JobLevel'] == level) & (df['Attrition'] == 'Yes')].shape[0]
    print(f"   Level {level}: {rate:.1f}% turnover ({level_left}/{level_total} employees)")

# 4. How does employee attrition affect overall company productivity and profitability?
print('\n\n4. How does employee attrition affect overall company productivity and profitability?')
print("="*50)

# calculate financial impact estimates
total_employees = len(df)
employees_left = len(left_employees)
avg_income_left = left_employees['MonthlyIncome'].mean()
avg_income_stayed = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()

print(" Financial Impact Analysis:")
print(f"   Total employees: {total_employees:,}")
print(f"   Employees who left: {employees_left:,}")
print(f"   Average monthly income of employees who left: ${avg_income_left:,.0f}")
print(f"   Average monthly income of employees who stayed: ${avg_income_stayed:,.0f}")

# Estimate replacement costs (typically 20-200% of annual salary)
annual_income_left = avg_income_left * 12
replacement_cost_low = annual_income_left * 0.2  # Conservative estimate
replacement_cost_high = annual_income_left * 1.0  # Moderate estimate

total_replacement_cost_low = replacement_cost_low * employees_left
total_replacement_cost_high = replacement_cost_high * employees_left

print(f"\n Estimated Replacement Costs:")
print(f"   Per employee (conservative): ${replacement_cost_low:,.0f}")
print(f"   Per employee (moderate): ${replacement_cost_high:,.0f}")
print(f"   Total cost (conservative): ${total_replacement_cost_low:,.0f}")
print(f"   Total cost (moderate): ${total_replacement_cost_high:,.0f}")

# Experience loss
avg_experience_left = left_employees['TotalWorkingYears'].mean()
avg_tenure_left = left_employees['YearsAtCompany'].mean()

print(f"\n Knowledge/Experience Loss:")
print(f"   Average total experience lost per employee: {avg_experience_left:.1f} years")
print(f"   Average company tenure lost per employee: {avg_tenure_left:.1f} years")
print(f"   Total years of company experience lost: {left_employees['YearsAtCompany'].sum():.0f} years")

# 5. Can we predict which employees are most likely to leave soon?
print('\n\n5. Can we predict which employees are most likely to leave soon?')
print("="*50)

print(" Prediction Readiness Assessment:")
print("    We have labeled data (Attrition: Yes/No)")
print("    We have multiple features (demographic, job, satisfaction)")
print("    We have sufficient data points (1,470 employees)")
print("    We identified key risk factors")

print("\n High-Risk Employee Profile (Based on Analysis):")
print("   • Department: Sales")
print("   • Age: Under 30 years")
print("   • Experience: 0-5 years")
print("   • Job Level: Level 1-2")
print("   • Overtime: Yes")
print("   • Low Job Satisfaction (Level 1-2)")
print("   • Poor Work-Life Balance (Level 1-2)")

print("\n Next Steps for Prediction Model:")
print("   1. Feature engineering and encoding")
print("   2. Train machine learning models")
print("   3. Evaluate model performance")
print("   4. Create risk scoring system")
print("   5. Deploy for real-time predictions")

