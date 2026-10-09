import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df=pd.read_excel("Dry_Bean_Dataset.xlsx")

print(df.head())

print("Dataset loaded successfully.\n")
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("number of unique classes:", df['Class'].nunique())
print("Unique classes:", df['Class'].unique())
print("Sample per class: ",df['Class'].value_counts())
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("Original shape:", df.shape)

df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)
print("Duplicates remaining:", df.duplicated().sum())

X=df.drop('Class', axis=1)
y=df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
X_train_final,X_val,y_train_final,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42,stratify=y_train)

print("Training set shape:", X_train_final.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)
scaler=StandardScaler()
X_train_final_scaled=scaler.fit_transform(X_train_final)
X_test_scaled=scaler.transform(X_test)
X_val_scaled=scaler.transform(X_val)

print("Training:", X_train_final_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)

k_values=[1,3,5,7,9,1,13]
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_final_scaled,y_train_final)
    y_val_pred=knn.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_val_pred)

print(f"k = {k:2d}  →  Validation Accuracy = {accuracy:.4f}")
print(f"k = 7 accuracy: {0.9359:.6f}")
print(f"k = 9 accuracy: {0.9359:.6f}")
for k in [7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_final_scaled, y_train_final)

    y_val_pred = knn.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_val_pred)

    print(f"k = {k} → Validation Accuracy = {accuracy:.8f}")
X_train_full = pd.concat([X_train_final, X_val])
y_train_full = pd.concat([y_train_final, y_val])

print("Final training set:", X_train_full.shape)
print("Final target:", y_train_full.shape)

final_scaler = StandardScaler()

X_train_full_scaled = final_scaler.fit_transform(X_train_full)
X_test_final_scaled = final_scaler.transform(X_test)
final_knn = KNeighborsClassifier(n_neighbors=7)

final_knn.fit(X_train_full_scaled, y_train_full)
y_test_pred = final_knn.predict(X_test_final_scaled)

test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
cm = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
joblib.dump(final_knn, "beansense_knn.pkl")
joblib.dump(final_scaler, "beansense_scaler.pkl")

