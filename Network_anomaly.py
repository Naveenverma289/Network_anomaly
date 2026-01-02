#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data_df=pd.read_csv('C:\\Users\\DimpleMourya\\Downloads\\Network_anomaly_data.csv')


# In[4]:


data_df


# In[7]:


data_df.shape


# In[11]:


data_df.describe()


# In[12]:


data_df.info()


# In[4]:


data_df


# # Block 2: EDA and Hypothesis testing
# 

# # Distribution of Each Feature:
# 
# #Checking Data Types
# 
# numerical_features = data_df.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = data_df.select_dtypes(include=['object', 'category']).columns
# 

# In[7]:


numerical_features


# In[8]:


categorical_features


# In[9]:


for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data_df, x=col, order=data_df[col].value_counts().index)
    plt.title(f'Count of Categories in {col}')
    plt.xticks(rotation=45)
    plt.show()


# In[10]:


for col in numerical_features:
    plt.figure(figsize=(10, 4))
    sns.histplot(data_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[11]:


sns.countplot(data=data_df, x='label')
plt.title('Anomaly vs Normal Traffic')
plt.xlabel('Label (0 = Normal, 1 = Anomaly)')
plt.ylabel('Count')
plt.show()


# In[16]:


corr_matrix = data_df.corr(numeric_only=True)
corr_matrix


# In[15]:


plt.figure(figsize=(24, 15))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


Conclusions from the Correlation Heatmap
Presence of Strong Multicollinearity:

Several features exhibit very high correlations (|correlation| > 0.9), especially among:

serrorrate & srvserrorrate

rerrorrate & srvrerrorrate

dsthostserrorrate & dsthostsrvserrorrate

dsthostrerrorrate & dsthostsrvrerrorrate

Features Highly Correlated with Each Other:

The count and srvcount pair shows a moderate to high correlation.

Features related to dsthost* are often strongly correlated with each other
suggesting they capture similar traffic behaviors from different angles.


# In[21]:


# Outlier detection


# Select only numeric columns
numeric_cols = data_df.select_dtypes(include=['int64', 'float64']).columns

# Plot boxplots for each numeric column
plt.figure(figsize=(20, len(numeric_cols) * 4))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols), 1, i)
    sns.boxplot(x=data_df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()

plt.show()


# In[ ]:


# Key Observations from Outlier Detection:
Many features have extreme values (visible as points far above or below the box).

Features such as srcbytes, dstbytes, duration, wrong_fragment, and num_failed_logins display significant outliers.

Outliers could be important in anomaly detection (e.g., unusually high srcbytes or duration might indicate suspicious activity).


# In[23]:


#Feature Engineering:

# Log Transformation (for skewed features)


data_df['log_srcbytes'] = np.log1p(data_df['srcbytes'])
data_df['log_dstbytes'] = np.log1p(data_df['dstbytes'])
data_df['log_duration'] = np.log1p(data_df['duration'])



# In[25]:


numeric_cols = data_df.select_dtypes(include=[np.number]).columns

# Check skewness
skewed_feats = data_df[numeric_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
print("Skewed Features:\n", skewed_feats[skewed_feats > 1]) 


# In[26]:


# Apply log1p (log(1+x)) to handle zero values
skewed_cols = skewed_feats[skewed_feats > 1].index

df_log_transformed = data_df.copy()
df_log_transformed[skewed_cols] = np.log1p(df_log_transformed[skewed_cols])


# In[31]:


# Example for one column
col = skewed_cols[4]  # pick one skewed column

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data_df[col], bins=50, kde=True)
plt.title(f"Original Distribution: {col}")

plt.subplot(1, 2, 2)
sns.histplot(df_log_transformed[col], bins=50, kde=True)
plt.title(f"Log-Transformed Distribution: {col}")

plt.tight_layout()
plt.show()


# In[30]:


before = data_df[skewed_cols].apply(lambda x: x.skew())
after = df_log_transformed[skewed_cols].apply(lambda x: x.skew())

skew_comparison = pd.DataFrame({'Before': before, 'After': after})
print(skew_comparison)


# In[35]:


# Ratio Features
# Creating ratio features to capture the relationship between source and destination bytes or packets.

data_df['srcbytes'] = data_df['srcbytes'].replace(0, 1)
data_df['dstbytes'] = data_df['dstbytes'].replace(0, 1)

data_df['bytes_ratio'] = data_df['srcbytes'] / data_df['dstbytes'].replace(0, 1)


# In[39]:


# Calculate ratios
data_df['bytes_ratio']   = data_df['srcbytes'] / data_df['dstbytes']
data_df['packets_ratio'] = data_df['srcpkts'] / data_df['dstpkts']


# In[8]:


# Missing values count and percentage
missing_count = data_df.isnull().sum()
missing_percent = (missing_count / len(data_df)) * 100

# Create a summary DataFrame
missing_df = pd.DataFrame({
    'Missing Values': missing_count,
    'Percentage': missing_percent
}).sort_values(by='Missing Values', ascending=False)

print("Missing Values Analysis:\n")
print(missing_df[missing_df['Missing Values'] > 0])

# Optional visualization
import matplotlib.pyplot as plt

missing_df[missing_df['Missing Values'] > 0]['Percentage'].plot(
    kind='bar',
    figsize=(10, 4),
    color='salmon'
)
plt.ylabel("Percentage of Missing Values")
plt.title("Missing Values Percentage by Feature")
plt.show()


# In[9]:


# Missing values count and percentage
missing_count = data_df.isnull().sum()
missing_percent = (missing_count / len(data_df)) * 100

missing_df = pd.DataFrame({
    'Missing Values': missing_count,
    'Percentage': missing_percent
}).sort_values(by='Missing Values', ascending=False)

print("Missing Values Analysis:\n")

if (missing_df['Missing Values'] > 0).any():
    print(missing_df[missing_df['Missing Values'] > 0])

    # Optional visualization
    import matplotlib.pyplot as plt
    missing_df[missing_df['Missing Values'] > 0]['Percentage'].plot(
        kind='bar',
        figsize=(10, 4),
        color='salmon'
    )
    plt.ylabel("Percentage of Missing Values")
    plt.title("Missing Values Percentage by Feature")
    plt.show()
else:
    print("✅ No missing values found in the dataset.")


# In[ ]:


# Possible Hypotheses to Test


# In[12]:


# Network Traffic Volume and Anomalies:
#Hypothesis: Network connections with unusually high or low traffic volume (bytes transferred) are more likely to be anomalous.
#Tests: Use t-tests or ANOVA to compare the means of Src_bytes and Dst_bytes in normal versus anomalous connections.


import pandas as pd
import numpy as np
from scipy import stats

# Hypothesis:
# H0: Mean traffic volume (bytes) is the same for normal and anomalous connections
# H1: Mean traffic volume differs between normal and anomalous connections


normal_df = data_df[data_df['attack'] == 'normal']
anomaly_df = data_df[data_df['attack'] != 'normal']  # everything else considered anomaly

# --- T-Test for srcbytes ---
t_stat_src, p_val_src = stats.ttest_ind(normal_df['srcbytes'], anomaly_df['srcbytes'], equal_var=False)

# --- T-Test for dstbytes ---
t_stat_dst, p_val_dst = stats.ttest_ind(normal_df['dstbytes'], anomaly_df['dstbytes'], equal_var=False)

print("=== Network Traffic Volume Hypothesis Test ===")
print(f"Src_bytes: t-stat={t_stat_src:.4f}, p-value={p_val_src:.4f}")
print(f"Dst_bytes: t-stat={t_stat_dst:.4f}, p-value={p_val_dst:.4f}")

# Interpretation
alpha = 0.05
if p_val_src < alpha:
    print(" Src_bytes: Significant difference between normal and anomaly traffic.")
else:
    print(" Src_bytes: No significant difference found.")

if p_val_dst < alpha:
    print(" Dst_bytes: Significant difference between normal and anomaly traffic.")
else:
    print(" Dst_bytes: No significant difference found.")


# In[15]:


# Impact of Protocol Type on Anomaly Detection:
# Hypothesis: Certain protocols are more frequently associated with network anomalies.

import pandas as pd
from scipy.stats import chi2_contingency

# Create a contingency table of protocol_type vs attack
contingency_table = pd.crosstab(data_df['protocoltype'], data_df['attack'])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("P-value:", p)

if p < 0.05:
    print("Significant difference: Protocol type distribution differs between normal and anomalous connections.")
else:
    print("No significant difference: Protocol type distribution is similar.")


# In[18]:


# Tests: Chi-square test to determine if the distribution of Protocol_type differs significantly in normal and anomalous connections.

import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
#data_df = pd.read_csv("Network_anomaly_data.csv")

# Adjust these names if needed based on your dataset
protocol_col = "protocoltype"  
label_col = "attack"             

# Create a contingency table
contingency_table = pd.crosstab(data_df[protocol_col], data_df[label_col])

print("\nContingency Table (Protocol vs Anomaly Status):")
print(contingency_table)

# Perform Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-square Test Results:")
print(f"Chi2 Statistic: {chi2}")
print(f"p-value: {p}")
print(f"Degrees of Freedom: {dof}")

if p < 0.05:
    print(" Significant association between protocol type and anomalies.")
else:
    print(" No significant association found.")

# Show anomaly proportion per protocol
print("\nAnomaly Proportion per Protocol:")
anomaly_ratio = contingency_table.apply(lambda row: row.max() / row.sum(), axis=1)
print(anomaly_ratio.sort_values(ascending=False))


# # Role of Service in Network Security:
# # Hypothesis: Specific services are targets of network anomalies more often than others.
# # Tests: Chi-square test to compare the frequency of services in normal versus anomaly-flagged connections.
# 
# 
# service_col = "service"  # service name column
# label_col = "attack"      # anomaly label column
# 
# # contingency table
# contingency_table = pd.crosstab(data_df[service_col], data_df[label_col])
# 
# print("\nContingency Table (Service vs Anomaly Status):")
# print(contingency_table)
# 
# # Chi-square test
# chi2, p, dof, expected = chi2_contingency(contingency_table)
# 
# print("\nChi-square Test Results:")
# print(f"Chi2 Statistic: {chi2}")
# print(f"p-value: {p}")
# print(f"Degrees of Freedom: {dof}")
# 
# if p < 0.05:
#     print("✅ Significant association between service type and anomalies.")
# else:
#     print("❌ No significant association found.")
# 
# # Calculation of anomaly proportion per service
# print("\nAnomaly Proportion per Service:")
# anomaly_ratio = contingency_table.apply(lambda row: row.max() / row.sum(), axis=1)
# print(anomaly_ratio.sort_values(ascending=False))
# 

# In[22]:


import statsmodels.api as sm

flag_col = "flag"   # Connection status column


data_df[label_col] = data_df[label_col].apply(lambda x: 1 if str(x).lower() == "anomaly" else 0)

# One-hot encode the 'flag' feature
flag_dummies = pd.get_dummies(data_df[flag_col], drop_first=True)

# Prepare features (X) and target (y)
X = sm.add_constant(flag_dummies)  # Adds intercept
y = data_df[label_col]

# Fit logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Show results
print(result.summary())

# Interpretation based on p-values
print("\nInterpretation:")
for var, pval in result.pvalues.items():
    if pval < 0.05:
        print(f"✅ {var} is a significant predictor of anomalies (p={pval:.4f})")
    else:
        print(f"❌ {var} is NOT a significant predictor of anomalies (p={pval:.4f})")


# In[23]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report


# Column names
flag_col = "flag"
label_col = "attack"

# Binary label
data_df[label_col] = data_df[label_col].apply(lambda x: 1 if str(x).lower() == "anomaly" else 0)

# One-hot encode flag
encoder = OneHotEncoder(drop='first', sparse=False)
X = encoder.fit_transform(data_df[[flag_col]])
y = data_df[label_col]

# Fit logistic regression with regularization
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Coefficients with feature names
feature_names = encoder.get_feature_names_out([flag_col])
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"{name}: {coef:.4f}")

# Classification report
y_pred = model.predict(X)
print("\nClassification Report:\n", classification_report(y, y_pred))


# In[29]:


# Data Cleaning: Handle missing values, remove duplicates, and correct errors in the dataset.

import pandas as pd

# ------------------------------
# 1. Check initial info
# ------------------------------
print("Initial shape:", data_df.shape)
print(data_df.info())

# ------------------------------
# 2. Handling missing values
# ------------------------------

# Counting missing values
print("\nMissing values before handling:")
print(data_df.isnull().sum())

# Filling numeric NaNs with median
num_cols = data_df.select_dtypes(include=['float64', 'int64']).columns
data_df[num_cols] = data_df[num_cols].fillna(data_df[num_cols].median())

#  Filling categorical NaNs with mode
cat_cols = data_df.select_dtypes(include=['object']).columns
data_df[cat_cols] = data_df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

# ------------------------------
# 3. Removing duplicates
# ------------------------------
duplicates_count = data_df.duplicated().sum()
print(f"\nNumber of duplicates found:", duplicates_count)
data_df = data_df.drop_duplicates()

# ------------------------------
# 4. Correcting basic errors
# ------------------------------

#  Ensuring text columns are lowercase & stripped of spaces
for col in cat_cols:
    data_df[col] = data_df[col].str.strip().str.lower()

# Removing impossible negative values in byte/packet counts
for col in ['srcbytes', 'dstbytes', 'srcpkts', 'dstpkts']:
    if col in data_df.columns:
        data_df[col] = data_df[col].apply(lambda x: max(x, 0))

# ------------------------------
# 5. Final check
# ------------------------------
print("\nShape after cleaning:", data_df.shape)
print("Missing values after cleaning:")
print(data_df.isnull().sum())

# Save cleaned data
data_df.to_csv("Network_anomaly_data_cleaned.csv", index=False)
print("\n✅ Data cleaning complete. Saved as Network_anomaly_data_cleaned.csv")


# In[36]:


# Feature Engineering: Develop new features that could enhance model performance, such as aggregating data over specific time windows or creating interaction terms.

# ======================
# 1. Ratio Features
# ======================
# Avoid division by zero
data_df['srcbytes'] = data_df['srcbytes'].replace(0, 1)
data_df['dstbytes'] = data_df['dstbytes'].replace(0, 1)
data_df['src_pkts'] = data_df['src_pkts'].replace(0, 1)
data_df['dstpkts'] = data_df['dstpkts'].replace(0, 1)

data_df['bytes_ratio'] = data_df['srcbytes'] / data_df['dstbytes']
data_df['packets_ratio'] = data_df['src_pkts'] / data_df['dstpkts']

# ======================
# 2. Interaction Terms
# ======================
data_df['bytes_times_packets'] = (data_df['srcbytes'] + data_df['dstbytes']) * (data_df['src_pkts'] + data_df['dstpkts'])
data_df['avg_packet_size_src'] = data_df['srcbytes'] / data_df['src_pkts']
data_df['avg_packet_size_dst'] = data_df['dstbytes'] / data_df['dstpkts']

# ======================
# 3. Aggregation over Time (if time column exists)
# ======================
if 'timestamp' in data_df.columns:
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df = data_df.sort_values('timestamp')

    # Rolling sum and mean over 5-minute windows
    data_df['rolling_bytes_sum'] = data_df['srcbytes'].rolling(window=5, min_periods=1).sum()
    data_df['rolling_packets_mean'] = data_df['src_pkts'].rolling(window=5, min_periods=1).mean()

# ======================
# 4. Binary Features
# ======================
data_df['is_heavy_traffic'] = (data_df['srcbytes'] + data_df['dstbytes']) > data_df['srcbytes'].median()
data_df['is_packet_burst'] = (data_df['src_pkts'] + data_df['dstpkts']) > data_df['src_pkts'].quantile(0.90)

# ======================
# 5. Log Transform for skewed features
# ======================
for col in ['srcbytes', 'dstbytes', 'bytes_ratio', 'packets_ratio']:
    data_df[f'log_{col}'] = np.log1p(data_df[col])  # log1p avoids log(0) issues

print("✅ Feature engineering complete. New columns added:")
print(data_df.columns)


# In[7]:


# Feature Engineering: Create new ratio and interaction features

# Avoid division by zero
data_df['srcbytes'] = data_df['srcbytes'].replace(0, 1)
data_df['dstbytes'] = data_df['dstbytes'].replace(0, 1)

# Ratio feature
data_df['bytes_ratio'] = data_df['srcbytes'] / data_df['dstbytes']

# Aggregated feature: total traffic
data_df['total_traffic'] = data_df['srcbytes'] + data_df['dstbytes']

print(" Feature engineering completed. New features created:")
print(data_df[['bytes_ratio', 'total_traffic']].head())


# In[9]:


#Data Transformation: Scale and normalize data, especially for algorithms that are sensitive to the scale of input features, like neural networks and distance-based algorithms.
    
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Select only numeric columns for scaling
numeric_cols = data_df.select_dtypes(include=['float64', 'int64']).columns

# Standardization (mean=0, std=1)
scaler = StandardScaler()
data_df[[col + "_std" for col in numeric_cols]] = scaler.fit_transform(data_df[numeric_cols])

# Min-Max Normalization (values between 0 and 1)
minmax_scaler = MinMaxScaler()
data_df[[col + "_norm" for col in numeric_cols]] = minmax_scaler.fit_transform(data_df[numeric_cols])

print("✅ Data Transformation completed. Added standardized (_std) and normalized (_norm) features to data_df.")

# Show first 5 rows of transformed columns
print(data_df.head())

    


# In[10]:


#Train-Test Split: Divide the data into training and testing sets to ensure the model can be evaluated on unseen data.

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Select only numeric columns for scaling
numeric_cols = data_df.select_dtypes(include=['float64', 'int64']).columns

# Standardization (mean=0, std=1)
scaler = StandardScaler()
data_df[[col + "_std" for col in numeric_cols]] = scaler.fit_transform(data_df[numeric_cols])

# Min-Max Normalization (values between 0 and 1)
minmax_scaler = MinMaxScaler()
data_df[[col + "_norm" for col in numeric_cols]] = minmax_scaler.fit_transform(data_df[numeric_cols])

print("✅ Data Transformation completed. Added standardized (_std) and normalized (_norm) features to data_df.")

# Show first 5 rows of transformed columns
print(data_df.head())


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ✅ Features and Target
X = data_df.drop(columns=['attack'])
y = data_df['attack']

# ✅ Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

# ✅ Preprocessing: OneHotEncode categorical, passthrough numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

# ✅ Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ✅ Train & Evaluate each model in a pipeline
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n🔹 {name}:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ✅ Encode categorical features
categorical_cols = data_df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    data_df[col] = le.fit_transform(data_df[col].astype(str))

# ✅ Features & target
X = data_df.drop('attack', axis=1)
y = data_df['attack']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Scale features for faster convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Optimized Logistic Regression
log_reg = LogisticRegression(
    solver='saga',     # fast & supports large datasets
    max_iter=200,      # increase if convergence warning
    n_jobs=-1,         # use all CPU cores
    verbose=1          # show progress while training
)

print("🚀 Training Logistic Regression...")
log_reg.fit(X_train_scaled, y_train)

# ✅ Predictions & Evaluation
y_pred = log_reg.predict(X_test_scaled)
print("\n📊 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ================== Data Preprocessing ==================
# Encode categorical columns
categorical_cols = data_df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    data_df[col] = le.fit_transform(data_df[col].astype(str))

# Features and Target
X = data_df.drop('attack', axis=1)
y = data_df['attack']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================== Ensemble Techniques ==================

# 1. Bagging (with Decision Tree as base)
bagging = BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)
bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

# 2. Boosting (AdaBoost)
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
y_pred_ada = adaboost.predict(X_test)

# 3. Gradient Boosting
gboost = GradientBoostingClassifier(n_estimators=100, random_state=42)
gboost.fit(X_train, y_train)
y_pred_gb = gboost.predict(X_test)

# 4. Stacking (combine Logistic Regression + SVM + Random Forest)
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ('svc', SVC(probability=True, kernel='linear'))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=200, solver='saga'),
    n_jobs=-1
)
stacking.fit(X_train, y_train)
y_pred_stack = stacking.predict(X_test)

# ================== Results ==================
print("\n📊 Ensemble Models Performance:")

print("\nBagging Accuracy:", accuracy_score(y_test, y_pred_bag))
print(classification_report(y_test, y_pred_bag))

print("\nAdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))

print("\nGradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

print("\nStacking Accuracy:", accuracy_score(y_test, y_pred_stack))
print(classification_report(y_test, y_pred_stack))


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ================== Data Preprocessing ==================
# Encode categorical columns
categorical_cols = data_df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    data_df[col] = le.fit_transform(data_df[col].astype(str))

# Features and Target
X = data_df.drop('attack', axis=1)
y = data_df['attack']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================== Ensemble Techniques ==================

# 1. Bagging (with Decision Tree as base)
bagging = BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)
bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

# 2. Boosting (AdaBoost)
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
y_pred_ada = adaboost.predict(X_test)

# 3. Gradient Boosting
gboost = GradientBoostingClassifier(n_estimators=100, random_state=42)
gboost.fit(X_train, y_train)
y_pred_gb = gboost.predict(X_test)

# 4. Stacking (combine Logistic Regression + SVM + Random Forest)
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ('svc', SVC(probability=True, kernel='linear'))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=200, solver='saga'),
    n_jobs=-1
)
stacking.fit(X_train, y_train)
y_pred_stack = stacking.predict(X_test)

# ================== Results ==================
print("\n📊 Ensemble Models Performance:")

print("\nBagging Accuracy:", accuracy_score(y_test, y_pred_bag))
print(classification_report(y_test, y_pred_bag))

print("\nAdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))

print("\nGradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

print("\nStacking Accuracy:", accuracy_score(y_test, y_pred_stack))
print(classification_report(y_test, y_pred_stack))


# In[ ]:


# Unsupervised Learning Models: When labels are not available,
# Clustering Models: K-means, DBSCAN, or hierarchical clustering to identify unusual patterns or groups.


# In[26]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================== Data Preprocessing ==================
# Drop label column if exists
if "attack" in data_df.columns:
    X_unsup = data_df.drop("attack", axis=1)
else:
    X_unsup = data_df.copy()

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
categorical_cols = X_unsup.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    X_unsup[col] = le.fit_transform(X_unsup[col].astype(str))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsup)

# ================== K-Means Clustering ==================
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # you can change n_clusters
kmeans.fit(X_scaled)

# Cluster assignments
data_df['Cluster'] = kmeans.labels_

print("✅ K-Means clustering completed.")
print(data_df[['Cluster']].head())

# ================== Elbow Method (to find best k) ==================
inertia = []
K = range(1, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[27]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ================== Data Preprocessing ==================
X_unsup = data_df.copy()

# Drop label column if present
if "attack" in X_unsup.columns:
    X_unsup = X_unsup.drop("attack", axis=1)

# Encode categorical columns
categorical_cols = X_unsup.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    X_unsup[col] = le.fit_transform(X_unsup[col].astype(str))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsup)

# ================== DBSCAN ==================
dbscan = DBSCAN(eps=1.5, min_samples=10)  # adjust eps & min_samples
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add cluster labels to dataset
data_df['DBSCAN_Cluster'] = dbscan_labels

print("✅ DBSCAN clustering done. Clusters assigned ( -1 = anomalies/outliers ).")
print(data_df[['DBSCAN_Cluster']].value_counts())


# In[ ]:


# Dimensionality Reduction: PCA or t-SNE for anomaly detection in a reduced dimensional space


# In[28]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ================== PCA ==================
pca = PCA(n_components=2)  # reduce to 2D for visualization
X_pca = pca.fit_transform(X_scaled)

# Add PCA results
data_df['PCA1'] = X_pca[:,0]
data_df['PCA2'] = X_pca[:,1]

print("✅ PCA completed. Variance explained:", pca.explained_variance_ratio_)

# ================== Visualization ==================
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c='blue', alpha=0.5, s=10)
plt.title("PCA - 2D Projection of Network Data")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()


# In[29]:


from sklearn.manifold import TSNE

# ================== t-SNE ==================
sample_data = X_scaled[:2000]   # use subset (e.g. 2000 rows) to speed up
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(sample_data)

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c='blue', alpha=0.5, s=10)
plt.title("t-SNE - 2D Projection of Network Data")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.show()


# In[ ]:


# Cross-Validation: Use techniques like k-fold cross-validation to assess model performance across different subsets of the dataset.


# In[5]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

# ================== Cross-Validation Setup ==================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=5000)
log_reg_scores = cross_val_score(log_reg, X_scaled, y, cv=cv, scoring='accuracy')
print("Logistic Regression CV Accuracy:", log_reg_scores)
print("Mean Accuracy:", np.mean(log_reg_scores))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
print("\nRandom Forest CV Accuracy:", rf_scores)
print("Mean Accuracy:", np.mean(rf_scores))

# Support Vector Machine
svm = SVC(kernel='rbf', random_state=42)
svm_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
print("\nSVM CV Accuracy:", svm_scores)
print("Mean Accuracy:", np.mean(svm_scores))


# In[ ]:


# Performance Metrics:
# For supervised models: Accuracy, Precision, Recall, F1-score, and ROC-AUC.


# In[32]:


# Logistic Regression 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Train model
log_reg = LogisticRegression(max_iter=2000, solver='lbfgs')
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:,1]

# Metrics
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# In[6]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

# -------------------------------
# Encode target column
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(data_df['attack'])   # 'attack' column is target
X = data_df.drop(columns=['attack'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Scale features (important for SVM & Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Define Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)   # probability=True for ROC-AUC
}

# -------------------------------
# Train & Evaluate Each Model
# -------------------------------
for name, model in models.items():
    print(f"\n🔹 {name} Results:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # If model supports probability, get predict_proba
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
    print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))

    # ROC-AUC (multiclass)
    if y_prob is not None:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        print("ROC-AUC (OvR, macro):", roc_auc_score(y_test_bin, y_prob, average='macro', multi_class="ovr"))

# -------------------------------
# Neural Network (using scikit-learn MLPClassifier)
# -------------------------------
from sklearn.neural_network import MLPClassifier

print("\n🔹 Neural Network Results:")
nn = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
y_prob_nn = nn.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Precision (macro):", precision_score(y_test, y_pred_nn, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred_nn, average='macro'))
print("F1-score (macro):", f1_score(y_test, y_pred_nn, average='macro'))
print("ROC-AUC (OvR, macro):", roc_auc_score(label_binarize(y_test, classes=np.unique(y_test)), 
                                             y_prob_nn, average='macro', multi_class="ovr"))


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# -------------------------------
# Encode target column
# -------------------------------
le = LabelEncoder()
y = le.fit_transform(data_df['attack'])   # Target variable
X = data_df.drop(columns=['attack'])

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

# Preprocessing: OneHotEncode categorical, Scale numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# -------------------------------
# Define Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
}

# -------------------------------
# Train & Evaluate Each Model
# -------------------------------
for name, model in models.items():
    print(f"\n🔹 {name} Results:")
    
    # Build pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # If model supports probability, get predict_proba
    y_prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
    print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))

    # ROC-AUC (multiclass, OvR)
    if y_prob is not None:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        print("ROC-AUC (OvR, macro):", roc_auc_score(y_test_bin, y_prob, average='macro', multi_class="ovr"))


# In[9]:


from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------
# Prepare Data (exclude label/attack column for unsupervised learning)
# -------------------------------
X_unsup = data_df.drop(columns=['attack'])

# Identify categorical and numeric columns
categorical_cols = X_unsup.select_dtypes(include=['object']).columns
numeric_cols = X_unsup.select_dtypes(exclude=['object']).columns

# Preprocessing: OneHotEncode categorical, Scale numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# -------------------------------
# Define Unsupervised Models
# -------------------------------
unsup_models = {
    "KMeans (k=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative Clustering (k=3)": AgglomerativeClustering(n_clusters=3)
}

# -------------------------------
# Train & Evaluate
# -------------------------------
for name, model in unsup_models.items():
    print(f"\n🔹 {name} Results:")
    
    # Preprocess data
    X_proc = preprocessor.fit_transform(X_unsup)
    
    # Fit model
    labels = model.fit_predict(X_proc)
    
    # Handle edge case: silhouette needs >1 cluster
    if len(set(labels)) > 1 and -1 not in set(labels):  
        score = silhouette_score(X_proc, labels)
        print("Silhouette Score:", score)
    else:
        print("Silhouette Score: Not applicable (only 1 cluster or noise only)")


# In[10]:


from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Drop target column
X_unsup = data_df.drop(columns=['attack'])

# Identify categorical and numeric columns
categorical_cols = X_unsup.select_dtypes(include=['object']).columns
numeric_cols = X_unsup.select_dtypes(exclude=['object']).columns

# Preprocessing with sparse output
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=True), categorical_cols)
    ]
)

# Transform (sparse output preserved)
X_proc = preprocessor.fit_transform(X_unsup)

print(f"Shape after preprocessing: {X_proc.shape}, type: {type(X_proc)}")

# Use only a sample if dataset is too big
if X_proc.shape[0] > 50000:
    idx = np.random.choice(X_proc.shape[0], 50000, replace=False)
    X_proc = X_proc[idx]
    print("⚠️ Using a 50,000-row sample for clustering due to memory limits.")

# Models
unsup_models = {
    "KMeans (k=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative Clustering (k=3)": AgglomerativeClustering(n_clusters=3)
}

# Train & evaluate
for name, model in unsup_models.items():
    print(f"\n🔹 {name} Results:")
    labels = model.fit_predict(X_proc)

    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(X_proc, labels, sample_size=5000)  # sample to save memory
        print("Silhouette Score:", score)
    else:
        print("Silhouette Score: Not applicable (1 cluster or all noise)")


# In[11]:


# Confusion Matrix: Analyze the true positives, true negatives, false positives, and false negatives to understand the model’s performance in different scenarios.


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example: Logistic Regression (replace with your model & predictions)
y_pred = log_reg.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print raw values
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Plot heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[12]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Fit the model first
log_reg.fit(X_train, y_train)

# ✅ Predict on test set
y_pred = log_reg.predict(X_test)

# ✅ Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print values
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# ✅ Plot heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=['Normal','Attack'],
            yticklabels=['Normal','Attack'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# Step 1: Encode categorical features
# ==============================
df = data_df.copy()

# Encode all object (string) columns
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("✅ All categorical columns encoded successfully!")

# ==============================
# Step 2: Train-Test Split
# ==============================
X = df.drop("attack", axis=1)   # Features
y = df["attack"]                # Target (assuming 'attack' is your label column)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==============================
# Step 3: Train Multiple Models
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(kernel="linear")
}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"✅ {name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


# In[ ]:




