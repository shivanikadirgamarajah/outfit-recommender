        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from joblib import dump

        df = pd.read_csv('weather_outfits.csv')
        le = LabelEncoder()
        df["outfit_encoded"]=le.fit_transform(df["outfit"])
        x=df[["temperature", "humidity", "wind_speed","precipitation"]]
        y=df["outfit_encoded"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x, y)

        dump(model, "outfit_model.joblib")
        dump(le, "label_encoder.joblib")

        print("Model and label encoder saved successfully.")