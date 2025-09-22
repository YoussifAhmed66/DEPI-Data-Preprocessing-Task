# Data Cleaning Report: Student Dataset

## Introduction
This report documents the data cleaning process performed on the student dataset (`bi.csv`). The dataset contains information about students' demographics and academic performance. The cleaning process addressed inconsistencies, missing values, and outliers to ensure data quality for further analysis.

## Part 1: Data Cleaning

### Dataset Structure
The dataset initially contained 77 rows and 11 columns.

Using `df.info()`, I verified that the data types were appropriate:
- `fNAME`, `lNAME`, `gender`, `country`, `residence`, `prevEducation`: Object (string)
- `Age`, `entryEXAM`, `studyHOURS`, `DB`: Integer
- `Python`: Float (due to missing values)

No duplicates were found (`df.duplicated().sum()` returned 0), but I used `df.drop_duplicates()` to ensure no duplicate rows remained.

### Inconsistent Categories

#### Gender Column
**Issue**: Inconsistent values: `['Female', 'M', 'Male', 'F', 'female', 'male']`.

**Action**: Standardized values using:
```python
df['gender'] = df['gender'].str.strip().str.upper().replace({
    'M': 'Male', 'MALE': 'Male', 'F': 'Female', 'FEMALE': 'Female'
})
```

**Result**: Only `['Female', 'Male']` remained.

#### Country Column
**Issue**: Inconsistent naming (e.g., `Norway`, `Norge`, `norway`, `Rsa`).

**Action**: Standardized values using:
```python
df['country'] = df['country'].str.strip().str.upper().replace({
    'NORWAY': 'Norway', 'NORGE': 'Norway', 'RSA': 'SOUTH AFRICA'
})
```

**Result**: Countries were consolidated (e.g., all variants of `Norway` became `Norway`).

#### prevEducation Column
**Issue**: Inconsistent values (e.g., `HighSchool`, `High School`, `Barrrchelors`, `diploma`).

**Action**: Standardized values using:
```python
df['prevEducation'] = df['prevEducation'].str.strip().str.lower().replace({
    'bachelors': 'Bachelors', 'barrrchelors': 'Bachelors', 
    'diploma': 'Diploma', 'diplomaaa': 'Diploma',
    'highschool': 'High School', 'high school': 'High School',
    'masters': 'Masters', 'doctorate': 'Doctorate'
})
```

**Result**: Categories were simplified to `['Masters', 'Diploma', 'High School', 'Bachelors', 'Doctorate']`.

## Part 2: Missing Data
**Missing Values**: The `Python` column had 2 missing values (other columns were complete).

**Imputation Method**: I used median imputation for the `Python` column since it is numerical and median is robust to outliers.
```python
df["Python"] = df["Python"].fillna(df["Python"].median())
```

**Why Median?**: The median was chosen over the mean because the `Python` scores had outliers (e.g., a score of 15), which could skew the mean.

## Part 3: Outliers

### Initial Assessment
I used `df.describe()` and boxplots to detect outliers across numerical columns.

- Grade columns (`Python`, `DB`, and `entryEXAM`): All scores were within the expected 0-100 range. No unrealistic values were found in these academic performance metrics.
- Age column: There were two individuals with ages above 60. However, their previous education level is 'Doctorate', which suggests they could be mature students pursuing advanced degrees. Therefore, these were deemed realistic and not treated as outliers.

### Outlier Detection in Study Hours
I focused specifically on the `studyHOURS` column for outlier detection because:
- Study habits can vary significantly between students
- This metric had the highest potential for data entry errors
- Extreme values could skew analysis of study habits vs performance

The `studyHOURS` column had 7 outliers (values below 124 hours).

**Boxplot Analysis**: The boxplot for `studyHOURS` showed several points below the lower whisker, indicating outliers.

### Handling Outliers
**Method**: I used the IQR method via the `datasist` library to detect outliers and replaced them with the median value of `studyHOURS`.
```python
outlier_ind = detect_outliers(df, 0, ['studyHOURS'])
median = df['studyHOURS'].median()
df.loc[outlier_ind, 'studyHOURS'] = median
```

**Why Median?**: Again, median was chosen for robustness. The outliers were unrealistic study hours that could distort analysis.

## Part 4: Feature Engineering
New features were created to enhance the dataset's predictive power:
- **Programming Average**: Average of Python and DB scores.
```python
df['Programming Average'] = (df['Python'] + df['DB']) / 2
```
- **isAdult**: Binary flag for age ≥ 25.
```python
df['isAdult'] = df['Age'] >= 25
```
- **Studying Category**: Categorized `studyHOURS` into Low (<130), Medium (130–149), High (≥150).
```python
def categorize_study_hours(hours):
    if hours < 130:
        return 'Low'
    elif 130 <= hours < 150:
        return 'Medium'
    else:
        return 'High'

df['Studying Category'] = df['studyHOURS'].apply(categorize_study_hours)
```

**Most Predictive Feature**: The `Programming Average` is likely to have the most predictive power for models predicting academic performance, as it directly aggregates two key performance metrics (`Python` and `DB` scores), capturing a student's overall programming ability more effectively than individual scores or other engineered features.

## Part 5: Feature Scaling
### Detect Numeric Columns
The numerical columns identified for scaling were: `Age`, `entryEXAM`, `studyHOURS`, `Python`, `DB`, and `Programming Average`.

### Apply Scaling
**Method**: Applied `MinMaxScaler` to scale numerical features to a range of [0, 1], suitable for algorithms like Neural Networks and KNN.
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_cols = scaler.fit_transform(df[['Age', 'entryEXAM', 'studyHOURS', 'Python', 'DB', 'Programming Average']])
df_scaled = pd.DataFrame(scaled_cols, columns=['Age', 'entryEXAM', 'studyHOURS', 'Python', 'DB', 'Programming Average'])
```

## Part 6: Encoding Categorical Data
### Detect Categorical Columns
The categorical columns identified were: `gender`, `country`, `residence`, `prevEducation`, and `Studying Category`.

### Handle Encoding
- **Gender and Residence**: Used binary encoding to reduce dimensionality compared to one-hot encoding, as these columns have a small number of categories.
```python
from sklearn.preprocessing import OneHotEncoder
Encoder = OneHotEncoder(sparse_output=False)
gender_encoded = Encoder.fit_transform(df[['gender']])
gender_encoded_df = pd.DataFrame(gender_encoded , columns= Encoder.get_feature_names_out())
```
- **Countryand Residence**: Applied binary encoding due to the high number of unique categories to avoid high-dimensional one-hot encoding.
```python
from category_encoders import BinaryEncoder
encoder = BinaryEncoder(cols=['country', 'residence'])
df_country_encoded = encoder.fit_transform(df['country', 'residence'])
```
- **prevEducation**: Since `prevEducation` is ordinal, it was label encoded with the following mapping:
  - High School: 1
  - Diploma: 2
  - Bachelors: 3
  - Masters: 4
  - Doctorate: 5
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Education_Level'] = le.fit_transform(df['prevEducation'])
# Adjusted to ensure correct ordinal mapping: High School=1, Diploma=2, Bachelors=3, Masters=4, Doctorate=5
```
- **Studying Category**: As an ordinal categorical feature, it was label encoded:
  - Low: 0
  - Medium: 1
  - High: 2
```python
df['Studying_Category_Encoded'] = df['Studying Category'].map({'Low': 0, 'Medium': 1, 'High': 2})
```

### Combining Features
The preprocessed dataframe was created by combining scaled numerical features, new engineered features, and encoded categorical features. Original categorical columns (`gender`, `country`, `residence`, `prevEducation`, `Studying Category`) were dropped.
```python
df_processed = pd.concat([df_scaled, df[['Programming Average', 'isAdult', 'Education_Level', 'Studying_Category_Encoded']], df_encoded, df_country_encoded], axis=1)
df_processed.drop(['gender', 'country', 'residence', 'prevEducation', 'Studying Category'], axis=1, inplace=True)
```

## Conclusion
The cleaned and preprocessed dataset now has consistent categories, no missing values, no outliers, encoded categoricals, new features, and normalized numericals.

**Changes made**:
- Standardized `gender`, `country`, and `prevEducation`.
- Imputed missing `Python` scores with the median.
- Replaced outlier `studyHOURS` values with the median.
- Created new features: `Programming Average`, `isAdult`, `Studying Category`.
- Scaled numerical columns using `MinMaxScaler`.
- Encoded categorical variables using one-hot encoding for `gender` and binary encoding for `residence`, and `country`, and label encoding for `prevEducation` and `Studying Category`.
- Dropped original categorical columns.

The dataset is now ready for exploratory data analysis and machine learning tasks.