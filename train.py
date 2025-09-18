# ===========================
# Smartphone Addiction ML Project
# ===========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping

# Metaheuristics
from mealpy.swarm_based import GWO


# ===========================
# 1. Load Dataset
# ===========================
df = pd.read_csv("teen_phone_addiction_dataset.csv")

# Binary Target
df["Addiction_Class"] = df["Addiction_Level"].apply(lambda x: 1 if x >= 8.0 else 0)

# Features & Target
X = df.drop(["ID", "Name", "Location", "Addiction_Level", "Addiction_Class"], axis=1)
y = df["Addiction_Class"].values

# Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# 2. Train-Test Split (ML models)
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ===========================
# 3. Decision Tree
# ===========================
# (Removed)

# ===========================
# 4. Random Forest (GridSearch)
# ===========================
# (Removed)

# ===========================
# 5. XGBoost
# ===========================
# (Removed)

# ===========================
# 6. Bi-LSTM optimized with GWO
# ===========================
X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y, test_size=0.3, random_state=42, stratify=y
)

# Global tracker for best accuracy
best_accuracy_seen = 0.0

# Fitness function for GWO → Bi-LSTM
def bilstm_fitness(params):
    global best_accuracy_seen

    units1 = int(params[0])        # first LSTM layer units
    units2 = int(params[1])        # second LSTM layer units
    dropout1 = float(params[2])    # dropout rate 1
    dropout2 = float(params[3])    # dropout rate 2
    lr = float(params[4])          # learning rate
    
    model = Sequential()
    model.add(Input(shape=(X_train_lstm.shape[1], 1)))
    model.add(Bidirectional(LSTM(units1, return_sequences=True)))
    model.add(Dropout(dropout1))
    model.add(Bidirectional(LSTM(units2)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train briefly for evaluation
    history = model.fit(
        X_train_lstm, y_train_lstm,
        validation_data=(X_test_lstm, y_test_lstm),
        epochs=5, batch_size=128, verbose=0
    )
    
    acc = history.history['val_accuracy'][-1] * 100  # percentage
    
    # Print each evaluation
    print(f"[BiLSTM Evaluation] Params={params}, Accuracy={acc:.2f}%")
    
    # Track the best accuracy seen so far
    if acc > best_accuracy_seen:
        best_accuracy_seen = acc
        print(f"New Best Accuracy Found: {best_accuracy_seen:.2f}%")
    
    return 1 - (acc / 100)   # minimize error

# Search space: [units1, units2, dropout1, dropout2, lr]
problem = {
    "fit_func": bilstm_fitness,
    "lb": [16, 8, 0.05, 0.05, 0.00001],
    "ub": [512, 256, 0.6, 0.6, 0.05],
    "minmax": "min",
}

# Run GWO
gwo_model = GWO.OriginalGWO(epoch=5, pop_size=5)
best_params, best_fitness = gwo_model.solve(problem)

print("\n[Optimized Bi-LSTM Params with GWO]")
print("Best Params:", best_params)
print(f"Best Accuracy Achieved During GWO: {best_accuracy_seen:.2f}%")

# ===========================
# Final Bi-LSTM training with best params + EarlyStopping
# ===========================
units1 = int(best_params[0])
units2 = int(best_params[1])
dropout1 = float(best_params[2])
dropout2 = float(best_params[3])
lr = float(best_params[4])

final_model = Sequential()
final_model.add(Input(shape=(X_train_lstm.shape[1], 1)))
final_model.add(Bidirectional(LSTM(units1, return_sequences=True)))
final_model.add(Dropout(dropout1))
final_model.add(Bidirectional(LSTM(units2)))
final_model.add(Dense(32, activation='relu'))
final_model.add(Dropout(dropout2))
final_model.add(Dense(1, activation='sigmoid'))

final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = final_model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_test_lstm, y_test_lstm),
    epochs=200,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

loss, acc = final_model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
print(f"Final GWO-Optimized Bi-LSTM Accuracy: {acc*100:.2f}%")

# ===========================
# 7. Evaluation
# ===========================
print("\n--- Final Model Accuracy ---")
print(f"Bi-LSTM (GWO-Optimized) Accuracy: {acc*100:.2f}%")

# ===========================
# 8. New Prediction Example
# ===========================
new_data = pd.DataFrame([{
    "Age": 16,
    "Gender": "Male",
    "School_Grade": "11th",
    "Daily_Usage_Hours": 9,
    "Sleep_Hours": 3.5,
    "Academic_Performance": 45,
    "Social_Interactions": 2,
    "Exercise_Hours": 0.5,
    "Anxiety_Level": 8,
    "Depression_Level": 7,
    "Self_Esteem": 2,
    "Parental_Control": 1,
    "Screen_Time_Before_Bed": 3,
    "Phone_Checks_Per_Day": 120,
    "Apps_Used_Daily": 15,
    "Time_on_Social_Media": 5,
    "Time_on_Gaming": 4,
    "Time_on_Education": 0.5,
    "Phone_Usage_Purpose": "Gaming",
    "Family_Communication": 2,
    "Weekend_Usage_Hours": 12
}])

# Encode + scale
for col in new_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    new_data[col] = le.fit_transform(new_data[col].astype(str))

new_data_scaled = scaler.transform(new_data)
new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
prediction_lstm = final_model.predict(new_data_lstm)

print("\n--- New User Prediction ---")
print("Bi-LSTM (GWO-Optimized):", int(prediction_lstm[0] > 0.5))
