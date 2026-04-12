import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump


df = pd.read_csv('weather_outfits.csv')
df.columns = df.columns.str.strip()
le = LabelEncoder()
df["outfit_encoded"]=le.fit_transform(df["general_outfit"])
x=df[["temperature", "humidity", "wind_speed","precipitation"]]
y=df["outfit_encoded"]
print("Class distribution:")
print(df["general_outfit"].value_counts())

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("\nModel evaluation:")
print(classification_report(y_test, y_pred, labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0))
dump(model, "outfit_model.joblib")
dump(le, "label_encoder.joblib")
label_map = pd.DataFrame({"outfit": le.classes_, "outfit_encoded": le.transform(le.classes_)})
label_map.to_csv("label_map.csv", index=False)
print("Model and label encoder saved successfully.")