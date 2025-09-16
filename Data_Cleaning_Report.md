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
Before addressing outliers, I first examined the grade columns (Python, DB, and entryEXAM) using descriptive statistics `df.describe()`:
- All scores were within the expected 0-100 range
- No unrealistic values were found in these academic performance metrics

### Outlier Detection in Study Hours
I focused specifically on the studyHOURS column for outlier detection because:
- Study habits can vary significantly between students
- This metric had the highest potential for data entry errors
- Extreme values could skew analysis of study habits vs performance

I focused specifically on the studyHOURS column for outlier detection using the IQR method then replaced them with the median value of `studyHOURS`:
```
outlier_ind = detect_outliers(df, 0, ['studyHOURS'])
median = df['studyHOURS'].median()              
df.loc[outlier_ind, 'studyHOURS'] = median
```

**Why Median?**: Again, median was chosen for robustness. The outliers were unrealistic study hours that could distort analysis.

## Conclusion
The cleaned dataset (`cleaned_students.csv`) now has consistent categories, no missing values, and no outliers.

**Changes made**:
- Standardized `gender`, `country`, and `prevEducation`.
- Imputed missing `Python` scores with the median.
- Replaced outlier `studyHOURS` values with the median.

The dataset is now ready for exploratory data analysis and machine learning tasks.
