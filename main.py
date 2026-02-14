import pandas as pd

# Load dataset
df = pd.read_csv("titanic.csv")

print("Missing values BEFORE cleaning:")
print(df.isnull().sum())

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column
df = df.drop(columns=['Cabin'])

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

# Remove duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after removal:", df.duplicated().sum())

# ---- Outlier Detection using IQR ----

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nLower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

before_rows = df.shape[0]

df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

after_rows = df.shape[0]

print("Rows before outlier removal:", before_rows)
print("Rows after outlier removal:", after_rows)

# Save cleaned data
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved as cleaned_titanic.csv")
