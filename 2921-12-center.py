import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Load data
df = pd.read_csv('/Users/anderson/Desktop/df2921(unbalance).csv')

# Define categorical columns
cat_cols = ["gender", "HBP", "DM", "PancreatitisHis", "HepatitisHis", "gallbladder", "Gallstones", 
            "IntrahepaticStones", "NumberofStones", "Diverticulum", "DifficultIntubation", "EST", "EPBD"]

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=cat_cols)

# Define predictors and target variable
X = df.drop(['PEP', 'center'], axis=1)
y = df['PEP']

# Define centres
centers = ["main", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Split the data into training and testing sets
X_train, X_test1, y_train, y_test1 = train_test_split(X[df['center'] == 'main'], y[df['center'] == 'main'], test_size=0.2, random_state=42)

# Create a XGBoost Classifier
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model using the training sets
clf.fit(X_train, y_train)

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Center', 'AUC', 'AUC_CI_Lower', 'AUC_CI_Upper'])

# Loop over each center
for center in centers:
    # Subset the data based on the 'center' column
    X_test = X[df['center'] == center]
    y_test = y[df['center'] == center]

    # Check if there are at least two classes present in the test set
    if len(np.unique(y_test)) < 2:
        print(f"Center {center} skipped due to having only one class in the test set.")
        continue

    # Predict the probabilities for test dataset
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    # AUC Score
    auc = roc_auc_score(y_test, y_pred_prob)

    # Bootstrap 95% CI for AUC
    n_iterations = 1000
    auc_scores = []
    for i in range(n_iterations):
        # Prepare a pseudo test set
        pseudo_test_y, pseudo_test_pred_prob = resample(y_test, y_pred_prob)

        # Check if there are at least two classes present in the pseudo test set
        if len(np.unique(pseudo_test_y)) < 2:
            continue

        # Compute the AUC score
        pseudo_auc = roc_auc_score(pseudo_test_y, pseudo_test_pred_prob)

        # Store the score
        auc_scores.append(pseudo_auc)

    # Calculate the 95% CI for AUC
    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()

    # Calculate the lower and upper 95% interval
    confidence_lower = sorted_scores[int(0.025 * len(auc_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(auc_scores))]

    # Append the results to the DataFrame
    results = results.append({'Center': center, 'AUC': auc, 'AUC_CI_Lower': confidence_lower, 'AUC_CI_Upper': confidence_upper}, ignore_index=True)

# Save the results to a CSV file
results.to_csv('/Users/anderson/Desktop/results-xgboost.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Load data
df = pd.read_csv('/Users/anderson/Desktop/df2921(unbalance).csv')

# Define categorical columns
cat_cols = ["gender", "HBP", "DM", "PancreatitisHis", "HepatitisHis", "gallbladder", "Gallstones", 
            "IntrahepaticStones", "NumberofStones", "Diverticulum", "DifficultIntubation", "EST", "EPBD"]

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=cat_cols)

# Define predictors and target variable
X = df.drop(['PEP', 'center'], axis=1)
y = df['PEP']

# Define centres
centers = ["main", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Split the data into training and testing sets
X_train, X_test1, y_train, y_test1 = train_test_split(X[df['center'] == 'main'], y[df['center'] == 'main'], test_size=0.2, random_state=42)

# Create a SVM Classifier
clf = SVC(probability=True)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Center', 'AUC', 'AUC_CI_Lower', 'AUC_CI_Upper'])

# Loop over each center
for center in centers:
    # Subset the data based on the 'center' column
    X_test = X[df['center'] == center]
    y_test = y[df['center'] == center]

    # Check if there are at least two classes present in the test set
    if len(np.unique(y_test)) < 2:
        print(f"Center {center} skipped due to having only one class in the test set.")
        continue

    # Predict the probabilities for test dataset
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    # AUC Score
    auc = roc_auc_score(y_test, y_pred_prob)

    # Bootstrap 95% CI for AUC
    n_iterations = 1000
    auc_scores = []
    for i in range(n_iterations):
        # Prepare a pseudo test set
        pseudo_test_y, pseudo_test_pred_prob = resample(y_test, y_pred_prob)

        # Check if there are at least two classes present in the pseudo test set
        if len(np.unique(pseudo_test_y)) < 2:
            continue

        # Compute the AUC score
        pseudo_auc = roc_auc_score(pseudo_test_y, pseudo_test_pred_prob)

        # Store the score
        auc_scores.append(pseudo_auc)

    # Calculate the 95% CI for AUC
    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()

    # Calculate the lower and upper 95% interval
    confidence_lower = sorted_scores[int(0.025 * len(auc_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(auc_scores))]

    # Append the results to the DataFrame
    results = results.append({'Center': center, 'AUC': auc, 'AUC_CI_Lower': confidence_lower, 'AUC_CI_Upper': confidence_upper}, ignore_index=True)

# Save the results to a CSV file
results.to_csv('/Users/anderson/Desktop/results-svm.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Load data
df = pd.read_csv('/Users/anderson/Desktop/df2921(unbalance).csv')

# Define categorical columns
cat_cols = ["gender", "HBP", "DM", "PancreatitisHis", "HepatitisHis", "gallbladder", "Gallstones", 
            "IntrahepaticStones", "NumberofStones", "Diverticulum", "DifficultIntubation", "EST", "EPBD"]

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=cat_cols)

# Define predictors and target variable
X = df.drop(['PEP', 'center'], axis=1)
y = df['PEP']

# Define centres
centers = ["main", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Split the data into training and testing sets
X_train, X_test1, y_train, y_test1 = train_test_split(X[df['center'] == 'main'], y[df['center'] == 'main'], test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Center', 'AUC', 'AUC_CI_Lower', 'AUC_CI_Upper'])

# Loop over each center
# Loop over each center
for center in centers:
    # Subset the data based on the 'center' column
    X_test = X[df['center'] == center]
    y_test = y[df['center'] == center]

    # Check if there are at least two classes present in the test set
    if len(np.unique(y_test)) < 2:
        print(f"Center {center} skipped due to having only one class in the test set.")
        continue

    # Predict the probabilities for test dataset
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    # AUC Score
    auc = roc_auc_score(y_test, y_pred_prob)

    # Bootstrap 95% CI for AUC
    # Bootstrap 95% CI for AUC
    n_iterations = 1000
    auc_scores = []
    for i in range(n_iterations):
        # Prepare a pseudo test set
        pseudo_test_y, pseudo_test_pred_prob = resample(y_test, y_pred_prob)

        # Check if there are at least two classes present in the pseudo test set
        if len(np.unique(pseudo_test_y)) < 2:
            continue

        # Compute the AUC score
        pseudo_auc = roc_auc_score(pseudo_test_y, pseudo_test_pred_prob)

        # Store the score
        auc_scores.append(pseudo_auc)

    # Calculate the 95% CI for AUC
    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()

    # Calculate the lower and upper 95% interval
    confidence_lower = sorted_scores[int(0.025 * len(auc_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(auc_scores))]

    # Append the results to the DataFrame
    results = results.append({'Center': center, 'AUC': auc, 'AUC_CI_Lower': confidence_lower, 'AUC_CI_Upper': confidence_upper}, ignore_index=True)

# Print the results
print(results)
results.to_csv('/Users/anderson/Desktop/results-rf.csv', index=False)




import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Load data
df = pd.read_csv('/Users/anderson/Desktop/df2921(unbalance).csv')

# Define categorical columns
cat_cols = ["gender", "HBP", "DM", "PancreatitisHis", "HepatitisHis", "gallbladder", "Gallstones", 
            "IntrahepaticStones", "NumberofStones", "Diverticulum", "DifficultIntubation", "EST", "EPBD"]

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=cat_cols)

# Define predictors and target variable
X = df.drop(['PEP', 'center'], axis=1)
y = df['PEP']

# Define centres
centers = ["main", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Split the data into training and testing sets
X_train, X_test1, y_train, y_test1 = train_test_split(X[df['center'] == 'main'], y[df['center'] == 'main'], test_size=0.2, random_state=42)

# Create a GLM Classifier
clf = LogisticRegression(max_iter=1000)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Center', 'AUC', 'AUC_CI_Lower', 'AUC_CI_Upper'])

# Loop over each center
for center in centers:
    # Subset the data based on the 'center' column
    X_test = X[df['center'] == center]
    y_test = y[df['center'] == center]

    # Check if there are at least two classes present in the test set
    if len(np.unique(y_test)) < 2:
        print(f"Center {center} skipped due to having only one class in the test set.")
        continue

    # Predict the probabilities for test dataset
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    # AUC Score
    auc = roc_auc_score(y_test, y_pred_prob)

    # Bootstrap 95% CI for AUC
    n_iterations = 1000
    auc_scores = []
    for i in range(n_iterations):
        # Prepare a pseudo test set
        pseudo_test_y, pseudo_test_pred_prob = resample(y_test, y_pred_prob)

        # Check if there are at least two classes present in the pseudo test set
        if len(np.unique(pseudo_test_y)) < 2:
            continue

        # Compute the AUC score
        pseudo_auc = roc_auc_score(pseudo_test_y, pseudo_test_pred_prob)

        # Store the score
        auc_scores.append(pseudo_auc)

    # Calculate the 95% CI for AUC
    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()

    # Calculate the lower and upper 95% interval
    confidence_lower = sorted_scores[int(0.025 * len(auc_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(auc_scores))]

    # Append the results to the DataFrame
    results = results.append({'Center': center, 'AUC': auc, 'AUC_CI_Lower': confidence_lower, 'AUC_CI_Upper': confidence_upper}, ignore_index=True)

# Save the results to a CSV file
results.to_csv('/Users/anderson/Desktop/results-lm.csv', index=False)




import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Load data
df = pd.read_csv('/Users/anderson/Desktop/df2921(unbalance).csv')

# Define categorical columns
cat_cols = ["gender", "HBP", "DM", "PancreatitisHis", "HepatitisHis", "gallbladder", "Gallstones", 
            "IntrahepaticStones", "NumberofStones", "Diverticulum", "DifficultIntubation", "EST", "EPBD"]

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=cat_cols)

# Define predictors and target variable
X = df.drop(['PEP', 'center'], axis=1)
y = df['PEP']

# Define centres
centers = ["main", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Split the data into training and testing sets
X_train, X_test1, y_train, y_test1 = train_test_split(X[df['center'] == 'main'], y[df['center'] == 'main'], test_size=0.2, random_state=42)

# Create a Naive Bayes Classifier
clf = GaussianNB()

# Train the model using the training sets
clf.fit(X_train, y_train)

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Center', 'AUC', 'AUC_CI_Lower', 'AUC_CI_Upper'])

# Loop over each center
for center in centers:
    # Subset the data based on the 'center' column
    X_test = X[df['center'] == center]
    y_test = y[df['center'] == center]

    # Check if there are at least two classes present in the test set
    if len(np.unique(y_test)) < 2:
        print(f"Center {center} skipped due to having only one class in the test set.")
        continue

    # Predict the probabilities for test dataset
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    # AUC Score
    auc = roc_auc_score(y_test, y_pred_prob)

    # Bootstrap 95% CI for AUC
    n_iterations = 1000
    auc_scores = []
    for i in range(n_iterations):
        # Prepare a pseudo test set
        pseudo_test_y, pseudo_test_pred_prob = resample(y_test, y_pred_prob)

        # Check if there are at least two classes present in the pseudo test set
        if len(np.unique(pseudo_test_y)) < 2:
            continue

        # Compute the AUC score
        pseudo_auc = roc_auc_score(pseudo_test_y, pseudo_test_pred_prob)

        # Store the score
        auc_scores.append(pseudo_auc)

    # Calculate the 95% CI for AUC
    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()

    # Calculate the lower and upper 95% interval
    confidence_lower = sorted_scores[int(0.025 * len(auc_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(auc_scores))]

    # Append the results to the DataFrame
    results = results.append({'Center': center, 'AUC': auc, 'AUC_CI_Lower': confidence_lower, 'AUC_CI_Upper': confidence_upper}, ignore_index=True)

# Save the results to a CSV file
results.to_csv('/Users/anderson/Desktop/results-nb.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Load data
df = pd.read_csv('/Users/anderson/Desktop/df2921(unbalance).csv')

# Define categorical columns
cat_cols = ["gender", "HBP", "DM", "PancreatitisHis", "HepatitisHis", "gallbladder", "Gallstones", 
            "IntrahepaticStones", "NumberofStones", "Diverticulum", "DifficultIntubation", "EST", "EPBD"]

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=cat_cols)

# Define predictors and target variable
X = df.drop(['PEP', 'center'], axis=1)
y = df['PEP']

# Define centres
centers = ["main", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Split the data into training and testing sets
X_train, X_test1, y_train, y_test1 = train_test_split(X[df['center'] == 'main'], y[df['center'] == 'main'], test_size=0.2, random_state=42)

# Create a Naive Bayes Classifier
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the model using the training sets
clf.fit(X_train, y_train)

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Center', 'AUC', 'AUC_CI_Lower', 'AUC_CI_Upper'])

# Loop over each center
for center in centers:
    # Subset the data based on the 'center' column
    X_test = X[df['center'] == center]
    y_test = y[df['center'] == center]

    # Check if there are at least two classes present in the test set
    if len(np.unique(y_test)) < 2:
        print(f"Center {center} skipped due to having only one class in the test set.")
        continue

    # Predict the probabilities for test dataset
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    # AUC Score
    auc = roc_auc_score(y_test, y_pred_prob)

    # Bootstrap 95% CI for AUC
    n_iterations = 1000
    auc_scores = []
    for i in range(n_iterations):
        # Prepare a pseudo test set
        pseudo_test_y, pseudo_test_pred_prob = resample(y_test, y_pred_prob)

        # Check if there are at least two classes present in the pseudo test set
        if len(np.unique(pseudo_test_y)) < 2:
            continue

        # Compute the AUC score
        pseudo_auc = roc_auc_score(pseudo_test_y, pseudo_test_pred_prob)

        # Store the score
        auc_scores.append(pseudo_auc)

    # Calculate the 95% CI for AUC
    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()

    # Calculate the lower and upper 95% interval
    confidence_lower = sorted_scores[int(0.025 * len(auc_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(auc_scores))]

    # Append the results to the DataFrame
    results = results.append({'Center': center, 'AUC': auc, 'AUC_CI_Lower': confidence_lower, 'AUC_CI_Upper': confidence_upper}, ignore_index=True)

# Save the results to a CSV file
results.to_csv('/Users/anderson/Desktop/results-dt.csv', index=False)