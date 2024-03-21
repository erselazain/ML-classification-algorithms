# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv("iphone_purchase_records.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Step 2 - Convert Gender to number
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

# Step 3 - Split Data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Step 4 - Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 5 - RF
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

# Step 6 - Predict
y_pred = classifier.predict(X_test)

# Step 7 - Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred) 
print(cm)
print(classification_report(y_test, y_pred))

# Step 8 - Make New Predictions
x1 = sc.transform([[1,21,45000]])
x2 = sc.transform([[0,51,80000]])

print("Laki-laki umur 21 dengan pendapatan $45k :", classifier.predict(x1))
print("Perempuan umur 51 dengan pendapatan $80k :", classifier.predict(x2))