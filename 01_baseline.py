#IoT Botnet Detection - Baseline

## 1. Importing Libraries
## 2. Loading Dataset
## 3. Inspecting Dataset
## 4. Prepare Features and Labels
## 5. Train-Test Split
## 6. Train Baseline Model
## 7. Evaluate Baseline Model
## 8. Notes

#========== IMPORTING AND LOADING DATASET==========

import pandas as pd

benign = pd.read_csv(r"C:\Users\Mulkum\OneDrive\Рабочий стол\Uni Related\research\Dataset\Danmini_Doorbell\benign_traffic.csv")
malicious = pd.read_csv(r"C:\Users\Mulkum\OneDrive\Рабочий стол\Uni Related\research\Dataset\Danmini_Doorbell\mirai_attacks\udp.csv")

#========== INSPECTING BOTH: MALICIOUS AND BENIGN ==========

#Inspecting Benign Data
print("=== BENIGN DATA ===")
print("Head:")
print(benign.head()) #this one prints the first 5 rows of the dataset for quick visual check

print("\n== Shape: ==") #this one tells the number of rows and columns in dataset
print(benign.shape)

print("\n== Info: ==")
benign.info() #.info() prints to console, no need for print()


#Inspecting Malicious Data
print("\n\n=== MALICIOUS DATA ===")
print("Head:")
print(malicious.head()) #this one prints the first 5 rows of the dataset for quick visual check

print("\n== Shape: ==") #this one tells the number of rows and columns in dataset
print(malicious.shape)

print("\n== Info: ==")
malicious.info() #.info() prints to console, no need for print()

# === Check whether columns of both datasets (malicious & benign) are the same and in the same order ===
print("\n\n === COLUMN NAMES IN BENIGN DATA ===")
print(benign.columns.tolist())

print("\n\n === COLUMN NAMES IN MALICIOUS DATA ===")
print(malicious.columns.tolist())

print("\n\n === ALL COLUMNS ARE SAME OR NOT ? T/F ===")
print(benign.columns.equals(malicious.columns))

# === Lets Label the datasets: 0 for benign and 1 for malicious (adding one last column)===
benign["label"] = 0 # adds label column to benign dataset and assigns 0 to all rows
malicious["label"] = 1 # adds label column to malicious dataset and assigns 1 to all rows

# === Take 50k rows from each dataset to create a balanced sample for training ===
min_rows = min(len(benign),len(malicious)) #this one checks which dataset has fewer rows and takes that number as the sample size to ensure balance

benign_sample = benign.sample(n=min_rows, random_state=42) #random_state is for reproducibility
malicious_sample = malicious.sample(n=min_rows, random_state=42)

# === Merge these two sample datasets ===
# df is final dataset that we will use for training and testing our model. It contains 100k rows (50k benign + 50k malicious) and all the original columns plus the new "label" column.
df = pd.concat([benign_sample, malicious_sample], ignore_index=True) #ignore_index resets the row numbers after concatenation

#Inspect the results
print("=== MERGED DATASET ===")
print("=== Head: ===")
print(df.head())

print("\n== Shape: ==") #this one tells the number of rows and columns in dataset
print(df.shape) #should be (min_rows*2, 116) 

print("\nClass Balance:")
print(df["label"].value_counts()) #this one counts how many 0s and 1s we have in the label column to check if our dataset is balanced or not. Should be 50000 each.

# === Separate the features and labels: X and y ===
X = df.drop("label", axis=1) #this one drops the "label" column from df and axis=1 means we are dropping a column (not a row). The resulting X will contain all the original features but not the label.
y = df["label"] #this one selects only the "label" column from df and assigns it to y. So y will be a Series containing only the labels (0s and 1s) that we want to predict.


#Inspect X and y
print("\n=== FEATURES (X) ===")
print(X.head())
print(X.shape) #should be (min_rows*2, 115) because we dropped the label column

print("\n=== LABELS (y) ===")
print(y.head())
print(y.shape) #should be (min_rows*2,) because it's just a single column of labels

# === Train-Test Split ===
from sklearn.model_selection import train_test_split

#this one splits the data into training and testing sets. test_size=0.2 means 20% of the data will be used for testing and 80% for training. random_state=42 is for reproducibility. Stratify = y means we want to maintain the same class distribution in both training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 


# === TRAINING A BASELINE MODEL USING DECISION TREE CLASSIFIER ===
from sklearn.tree import DecisionTreeClassifier

print("\nInitializing the Decision Tree Model...")
dt_model = DecisionTreeClassifier(random_state=42) #random_state for reproducibility

print("Training the model on the 80% practice data. Please wait...")
#fit function does the actual training
dt_model.fit(X_train, y_train)

print("Model training is done! The model has learned the patterns. ")

# === Testing the trained model ===
print("\nTesting the model on the unseen 20% data...")

#The model takes the exam (20% of the unseen data) l y_pred_dt is the model's predictions for the test set. It will be an array of 0s and 1s, where 0 means the model thinks it's benign and 1 means it thinks it's malicious.
y_pred_dt = dt_model.predict(X_test) 

print("Predictions complete!")

#Let's peek at the first 10 predictions vs the true answers
print("\n---Let's grade the first 10 rows ---")
print(f"Model's Guesses: {y_pred_dt[:10]}")
print(f"True Answers: {y_test[:10].values}")

# === Evaluating the model's performance ===
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n=== GRADING THE MODEL ===")
print("Accuracy Score:", accuracy_score(y_test, y_pred_dt)) #this one calculates the overall accuracy of the model by comparing the predicted labels (y_pred_dt) with the true labels (y_test). It returns a value between 0 and 1, where 1 means perfect accuracy.

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt)) #this one creates a confusion matrix which is a table that shows the counts of true positives, true negatives, false positives, and false negatives. It helps us understand how well the model is performing in terms of correctly identifying benign and malicious samples.

print("\nClassification Report:")
print(classification_report(y_test,y_pred_dt)) #this one generates a detailed classification report that includes precision, recall, f1-score, and support for each class (benign and malicious). It gives us a more comprehensive view of the model's performance beyond just accuracy.



# Now we are trying to know which features were used most by the Decision Tree to make its decisions. This can help us understand which aspects of the network traffic are most important for distinguishing between benign and malicious behavior.
print("\n=== TOP 5 MOST IMPORTANT FEATURES: ===")

#Extract the importnace scores from the trained model
importances = dt_model.feature_importances_

#Create  a clean table matching feature names to their scores
feature_names = X.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

#Sort the features by importance in descending order
importance_df = importance_df.sort_values(by = "Importance", ascending=False)
print(importance_df.head(5)) 