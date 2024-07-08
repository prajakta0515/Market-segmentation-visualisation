#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


# In[2]:


vacmotdesc = pd.read_csv("vacation_complete_dataset.csv")


# In[3]:


vacmotdesc.head()


# In[4]:


vacmotdesc.tail()


# In[5]:


vacmotdesc.info()


# In[6]:


vacmotdesc.describe()


# In[7]:


vacmotdesc.columns


# In[8]:


vacmotdesc.isna().sum()*100


# In[9]:


percent_missing=(vacmotdesc.isna().sum()/vacmotdesc.shape[0])*100


# In[10]:


percent_missing


# In[11]:


from sklearn.impute import SimpleImputer
# Identify categorical and numerical columns
categorical_columns = vacmotdesc.select_dtypes(include=['object']).columns
numerical_columns = vacmotdesc.select_dtypes(include=['number']).columns

print("Before imputation:")
print(vacmotdesc.isnull().mean() * 100)

# Impute missing values in numerical columns with the mean
num_imputer = SimpleImputer(strategy='mean')
vacmotdesc[numerical_columns] = num_imputer.fit_transform(vacmotdesc[numerical_columns])

# Impute missing values in categorical columns with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
vacmotdesc[categorical_columns] = cat_imputer.fit_transform(vacmotdesc[categorical_columns])

print("After imputation:")
print(vacmotdesc.isnull().mean() * 100)

# Apply one-hot encoding to categorical columns
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cats = encoder.fit_transform(vacmotdesc[categorical_columns])

# Create a DataFrame from the encoded categories
encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate the new encoded columns
vacmotdesc_numeric = vacmotdesc.drop(columns=categorical_columns).reset_index(drop=True)
vacmotdesc_numeric = pd.concat([vacmotdesc_numeric, encoded_cats_df], axis=1)


# In[12]:


# Identify categorical columns
categorical_columns = vacmotdesc.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cats = encoder.fit_transform(vacmotdesc[categorical_columns])

# Create a DataFrame from the encoded categories
encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate the new encoded columns
vacmotdesc_numeric = vacmotdesc.drop(columns=categorical_columns).reset_index(drop=True)
vacmotdesc_numeric = pd.concat([vacmotdesc_numeric, encoded_cats_df], axis=1)

# Perform clustering
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(vacmotdesc_numeric)

# Store segment membership
C6 = kmeans.labels_

vacmotdesc['segment'] = C6


# In[13]:


cluster_counts = pd.Series(C6).value_counts().sort_index()
print("\nCluster membership counts:")
print(cluster_counts)

# Output the results as a table similar to R
print("\nC6")
for cluster, count in cluster_counts.items():
    print(f"{cluster+1}\t{count}")


# In[14]:


vacmotdesc['C6'] = pd.Categorical(C6)


# In[15]:


C6_Gender = pd.crosstab(vacmotdesc['C6'], vacmotdesc['Gender'])
print(C6_Gender)


# In[16]:


vacmotdesc['C6'] = pd.Categorical(C6)

# Create a Segment column with the segment labels
vacmotdesc['Segment'] = 'Segment ' + vacmotdesc['C6'].astype(str)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data=vacmotdesc, x='Age', hue='Segment', multiple='dodge', ax=ax)
ax.set_title('Histogram of Age by Segment')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
plt.show()


# In[17]:


vacmotdesc['C6'] = pd.Categorical(C6)

# Create a Segment column with the segment labels
vacmotdesc['Segment'] = 'Segment ' + vacmotdesc['C6'].astype(str)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
vacmotdesc.groupby(['Segment', 'Obligation']).size().unstack(fill_value=0).plot(kind='bar', ax=ax)
ax.set_xlabel('Segment')
ax.set_ylabel('Obligation')
ax.set_title('Obligation by Segment')
plt.show()


# In[18]:


vacmotdesc['C6'] = pd.Categorical(C6)

# Create the boxplot
boxplot = vacmotdesc.boxplot(column='Age', by='C6', grid=False)

# Set the labels and title
boxplot.set_xlabel('Segment number')
boxplot.set_ylabel('Age')
boxplot.set_title('Boxplot of Age by Segment')

# Remove the automatic title to avoid duplication
plt.suptitle('')

# Display the plot
plt.show()


# In[19]:


vacmotdesc['C6'] = pd.Categorical(C6)

# Create the boxplot
boxplot = vacmotdesc.boxplot(column='Obligation', by='C6', 
                             notch=True)

# Set the labels and title
boxplot.set_xlabel('Segment number')
boxplot.set_ylabel('Moral obligation')
boxplot.set_title('Boxplot of Moral Obligation by Segment')

# Remove the automatic title to avoid duplication
plt.suptitle('')

# Display the plot
plt.show()


# In[20]:


import scipy.stats as stats
# Example observed frequencies (replace with your actual data)
observed = np.array([[235, 189, 174, 139, 94, 169]])  # Replace with your observed frequencies for each segment

# Perform chi-squared test
chi2, p, df, expected = stats.chi2_contingency(observed)

# Print results
print("Pearson's Chi-squared test:")
print(f"Chi-square statistic: {chi2:.2f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of freedom: {df}")
print(f"Expected frequencies:\n{expected}")


# In[26]:


C6_moblig = vacmotdesc.groupby('C6')['Obligation'].mean()
print(C6_moblig)



# In[30]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

data = pd.read_csv('vacation_complete_dataset.csv')
model = ols('Obligation ~ C6', data=data).fit()
anova_table = anova_lm(model)
print(anova_table)


# In[70]:


import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Assuming you have loaded your dataset 'vacmotdesc' into a pandas DataFrame
# Example dataset creation (replace with your actual dataset loading)
data = {
    'C6': np.random.choice([1, 2, 3, 4, 5, 6], size=1000),  # Example outcome variable C6 (categorical)
    'Obligation': np.random.randint(1, 10, size=1000)  # Example numeric variable
}

vacmotdesc = pd.DataFrame(data)

# Perform pairwise t-tests
pairwise_results = pairwise_tukeyhsd(endog=vacmotdesc['Obligation'], groups=vacmotdesc['C6'], alpha=0.05)

print(pairwise_results)


# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Example dataset creation (replace with your actual dataset loading)
data = {
    'C6': np.random.choice([1, 2, 3, 4, 5, 6], size=1000),  # Example outcome variable C6 (categorical)
    'Obligation': np.random.randint(1, 10, size=1000)  # Example numeric variable
}

vacmotdesc = pd.DataFrame(data)

# Perform Tukey HSD test
tukey_results = pairwise_tukeyhsd(endog=vacmotdesc['Obligation'], groups=vacmotdesc['C6'], alpha=0.05)

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 8))
tukey_results.plot_simultaneous(ax=ax)
ax.set_xlabel('Mean difference')
ax.set_ylabel('Segment pairs')
ax.set_title('Tukey HSD test for Obligation across C6 segments')
plt.grid(True)

# Adding custom text to the plot
plt.text(-2, -0.5, 'Pairs of segments', rotation=90, va='center', ha='center')

plt.show()


# In[72]:


import pandas as pd
import statsmodels.api as sm
import numpy as np

# Example dataset creation (replace with your actual dataset loading)
data = {
    'C6': np.random.choice([1, 2, 3, 4, 5, 6], size=1000),  # Example categorical variable C6
    'Age': np.random.randint(18, 65, size=1000)  # Example numeric variable Age
}

vacmotdesc = pd.DataFrame(data)

# Create dummy variables for C6 without an intercept
C6_dummies = pd.get_dummies(vacmotdesc['C6'], drop_first=False)
C6_dummies.columns = [f'C6_{col}' for col in C6_dummies.columns]

# Perform the linear regression
model = sm.OLS(vacmotdesc['Age'], C6_dummies).fit()

# Print the summary of the regression
print(model.summary())



# In[77]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

# Example dataset creation (replace with your actual dataset loading)
data = {
    'C6': np.random.choice([1, 2, 3, 4, 5, 6], size=1000),  # Example categorical variable C6
    'Age': np.random.randint(18, 65, size=1000)  # Example numeric variable Age
}

vacmotdesc = pd.DataFrame(data)

# Fit linear regression model without intercept
model_without_intercept = sm.OLS.from_formula('Age ~ C6 - 1', data=vacmotdesc)
result_without_intercept = model_without_intercept.fit()

# Print the summary of the model without intercept
print(result_without_intercept.summary())

# Fit linear regression model with intercept
model_with_intercept = sm.OLS.from_formula('Age ~ C6', data=vacmotdesc)
result_with_intercept = model_with_intercept.fit()

# Print the summary of the model with intercept
print(result_with_intercept.summary())



# In[78]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

# Example dataset creation (replace with your actual dataset loading)
data = {
    'C6': np.random.choice(['1', '2', '3', '4', '5', '6'], size=1000),  # Example categorical variable C6 as strings
    'Age': np.random.randint(18, 65, size=1000).astype(float)  # Example numeric variable Age as floats
}

vacmotdesc = pd.DataFrame(data)

# Convert C6 to numeric if it's intended as categorical
vacmotdesc['C6'] = vacmotdesc['C6'].astype(int)

# Fit linear regression model without intercept
model_without_intercept = sm.OLS.from_formula('Age ~ C6 - 1', data=vacmotdesc)
result_without_intercept = model_without_intercept.fit()

# Print the summary of the model without intercept
print(result_without_intercept.summary())

# Fit linear regression model with intercept
model_with_intercept = sm.OLS.from_formula('Age ~ C6', data=vacmotdesc)
result_with_intercept = model_with_intercept.fit()

# Print the summary of the model with intercept
print(result_with_intercept.summary())



# In[80]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

# Example dataset creation (replace with your actual dataset loading)
data = {
    'C6': np.random.choice(['1', '2', '3', '4', '5', '6'], size=1000),  # Example categorical variable C6 as strings
    'Age': np.random.randint(18, 65, size=1000).astype(float),  # Example numeric variable Age as floats
    'Obligation2': np.random.randn(1000)  # Example numeric variable Obligation2
}

vacmotdesc = pd.DataFrame(data)

# Convert C6 to numeric if it's intended as categorical
vacmotdesc['C6'] = vacmotdesc['C6'].astype(int)

# Fit multinomial logistic regression model
model_mnlogit = sm.MNLogit.from_formula('C6 ~ Age + Obligation2', data=vacmotdesc)
result_mnlogit = model_mnlogit.fit()

# Print the summary of the model
print(result_mnlogit.summary())


# In[86]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Example dataset creation (replace with your actual dataset loading)
data = {
    'C6': np.random.choice(['1', '2', '3', '4', '5', '6'], size=1000),  # Example categorical variable C6 as strings
    'Age': np.random.randint(18, 65, size=1000).astype(float)  # Example numeric variable Age as floats
}

vacmotdesc = pd.DataFrame(data)

# Separate X (independent variables) and y (dependent variable)
X = vacmotdesc.drop(columns=['C6'])
y = vacmotdesc['C6']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize and fit the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Predict probabilities for each class
pred_prob = model.predict_proba(X_test)

# Convert predictions to a DataFrame for plotting
predicted = pd.DataFrame({
    'prob': pred_prob[:, 5],  # Adjust index based on your class order (0-indexed)
    'observed': y_test,
    'predicted': model.classes_[5]  # Adjust index based on your class order (0-indexed)
})

# Plotting boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(predicted[predicted['observed'] == '6']['prob'])
plt.xlabel('Segment')
plt.ylabel('Probability')
plt.title('Boxplot of Probabilities for Segment 6')
plt.xticks([1], ['6'])  # Adjust ticks based on your class order
plt.grid(True)
plt.show()


# In[91]:





# In[81]:


import pandas as pd
import numpy as np

# Example dataset creation (replace with your actual dataset loading)
data = {
    'C6': np.random.choice(['1', '2', '3', '4', '5', '6'], size=1000),  # Example categorical variable C6 as strings
    'Age': np.random.randint(18, 65, size=1000).astype(float)  # Example numeric variable Age as floats
}

vacmotdesc = pd.DataFrame(data)

# Convert C6 to numeric if it's intended as categorical
vacmotdesc['C6'] = vacmotdesc['C6'].astype(int)


# In[85]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Separate X (independent variables) and y (dependent variable)
X = vacmotdesc.drop(columns=['C6'])
y = vacmotdesc['C6']  # Assuming C6 is a categorical variable with six levels

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize decision tree classifier
clf = DecisionTreeClassifier(random_state=1234)

# Fit the classifier
clf.fit(X_train, y_train)

# Visualize the decision tree (optional for interpretation)
# Visualize the decision tree (optional for interpretation)
plt.figure(figsize=(12, 6))
plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=[str(i) for i in np.unique(y)])
plt.show()


# In[ ]:




