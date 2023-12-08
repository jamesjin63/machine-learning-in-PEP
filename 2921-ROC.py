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

# Split the data into training and testing sets
X_train, X_test1, y_train, y_test1 = train_test_split(X[df['center'] == 'main'], y[df['center'] == 'main'], test_size=0.2, random_state=42)

# Create a GLM Classifier
clf = LogisticRegression(max_iter=1000)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the probabilities for test dataset
y_pred_prob1 = clf.predict_proba(X_test1)[:, 1]

# AUC Score
auc1 = roc_auc_score(y_test1, y_pred_prob1)

# Bootstrap 95% CI for AUC
n_iterations = 1000
auc_scores1 = []
for i in range(n_iterations):
    # Prepare a pseudo test set
    pseudo_test_y1, pseudo_test_pred_prob1 = resample(y_test1, y_pred_prob1)

    # Check if there are at least two classes present in the pseudo test set
    if len(np.unique(pseudo_test_y1)) < 2:
        continue

    # Compute the AUC score
    pseudo_auc1 = roc_auc_score(pseudo_test_y1, pseudo_test_pred_prob1)

    # Store the score
    auc_scores1.append(pseudo_auc1)

# Calculate the 95% CI for AUC
sorted_scores1 = np.array(auc_scores1)
sorted_scores1.sort()

# Calculate the lower and upper 95% interval
confidence_lower1 = sorted_scores1[int(0.025 * len(auc_scores1))]
confidence_upper1 = sorted_scores1[int(0.975 * len(auc_scores1))]

# Print the AUC and its 95% CI
print(f"AUC for y_test1: {auc1}")
print(f"95% CI for AUC of y_test1: ({confidence_lower1}, {confidence_upper1})")


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import matplotlib.pyplot as plt
# AUC Score
auc1 = roc_auc_score(y_test1, y_pred_prob1)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test1, y_pred_prob1)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Save ROC curve data to a CSV file
roc_data = pd.DataFrame({
    'FPR': fpr,
    'TPR': tpr,
    'Thresholds': thresholds
})
roc_data.to_csv('/Users/anderson/Desktop/roc_data.csv', index=False)



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb

def calculate_and_plot_auc(clf, X_train, y_train, X_test, y_test, model_name):
    # Train the model and predict probabilities
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC for {model_name}: {auc}")

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

    # Print the 95% CI for AUC
    print(f"95% CI for AUC of {model_name}: ({confidence_lower}, {confidence_upper})")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    # Save ROC data to a CSV file
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Thresholds': thresholds
    })
    roc_data.to_csv(f'/Users/anderson/Desktop/roc_data_{model_name}.csv', index=False)

    # Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(probability=True),
    'GLM': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

# Use each model to predict AUC and plot ROC curve
plt.figure(figsize=(10, 10))
for model_name, model in models.items():
    calculate_and_plot_auc(model, X_train, y_train, X_test1, y_test1, model_name)
plt.show()

