from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut, RandomizedSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, Normalizer




# ====================================
# STEP 1: read the training and testing data.
# Do not change any code of this step.

# specify path to training data and testing data
train_x_location = f"x_train_5.csv"
train_y_location = f"y_train_5.csv"

test_x_location = "x_test.csv"
test_y_location = "y_test.csv"

print("Reading training data")
x_train = np.loadtxt(train_x_location, delimiter=",")
y_train = np.loadtxt(train_y_location, delimiter=",")

m, n = x_train.shape # m training examples, each with n features
m_labels,  = y_train.shape # m2 examples, each with k labels
l_min = y_train.min()

assert m_labels == m, "x_train and y_train should have same length."
assert l_min == 0, "each label should be in the range 0 - k-1."
k = y_train.max()+1

print(m, "examples,", n, "features,", k, "categiries.")

print("Reading testing data")
x_test = np.loadtxt(test_x_location, delimiter=",")
y_test = np.loadtxt(test_y_location, delimiter=",")

m_test, n_test = x_test.shape
m_test_labels,  = y_test.shape

assert m_test_labels == m_test, "x_test and y_test should have same length."
assert n_test == n, "train and x_test should have same number of features."

print(m_test, "test examples.")


# ====================================
# STEP 2: pre processing
# Please modify the code in this step.
print("Pre processing data")
# you can skip this step, use your own pre processing ideas,
# or use anything from sklearn.preprocessing

# # The same pre processing must be applied to both training and testing data
# x_train = x_train / 1.0
# x_test = x_test / 1.0
# X_train = StandardScaler().fit_transform(x_train)
# # X_test = StandardScaler().fit_transform(x_test)


norm = Normalizer()

# Apply the normalizer to your data
X_train = norm.fit_transform(x_train)
X_test = norm.transform(x_test)

scaler = MinMaxScaler()

# Apply the scaler to your data
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Calculate the mean of each row in x_train and x_test
mean_train = np.mean(x_train, axis=1)
mean_test = np.mean(x_test, axis=1)

# Add the mean value as a new feature to x_train and x_test
X_train = np.hstack((x_train, mean_train.reshape(-1,1)))
X_test = np.hstack((x_test, mean_test.reshape(-1,1)))

# pca = PCA(n_components=30)
# pca.fit(x_train)
# reduced_features = pca.transform(x_train)
# print(reduced_features.shape)
# x_train = reduced_features


# pca = PCA(n_components=30)
# pca.fit(x_test)
# reduced_features = pca.transform(x_test)
# print(reduced_features.shape)
# x_test = reduced_features
# ====================================
# STEP 3: train model.
# Please modify the code in this step.

print(f"---training")
param_grid = {
    'C': [1, 10, 100, 1000,10000,100000,1000000],
    'kernel': ['rbf'],
}


stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(svm.SVC(), param_grid, cv=stratified_kfold)
search.fit(x_train, y_train)

print("Best parameters found:", search.best_params_)
model = search.best_estimator_ 

model.fit(x_train, y_train)

# ====================================
# STEP3: evaluate model
# Don't modify the code below.

print("---evaluate")
print(" number of support vectors: ", model.n_support_)
acc = model.score(x_test, y_test)
print("acc:", acc)
