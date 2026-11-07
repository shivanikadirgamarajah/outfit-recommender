
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
]

TARGET = "general_outfit"
OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)


# 1. Load and validate data
df = pd.read_csv("weather_outfits.csv")
df.columns = df.columns.str.strip()

required = FEATURES + [TARGET]
df = df.dropna(subset=required).copy()

for column in FEATURES:
    df[column] = pd.to_numeric(df[column], errors="raise")

df[TARGET] = df[TARGET].str.strip()

print("Original rows:", len(df))
print("Classes:", df[TARGET].nunique())


# 2. Check data quality
exact_duplicates = df.duplicated().sum()
print("Exact duplicate rows:", exact_duplicates)

df = df.drop_duplicates().reset_index(drop=True)

conflicting = (
    df.groupby(FEATURES)[TARGET]
    .nunique()
)
conflicting_count = (conflicting > 1).sum()

print("Weather inputs with conflicting labels:", conflicting_count)
print("\nClass distribution:")
print(df[TARGET].value_counts().to_string())

df[TARGET].value_counts().to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    header=["count"],
)

# 3. Encode outfit labels
encoder = LabelEncoder()
y = pd.Series(
    encoder.fit_transform(df[TARGET]),
    index=df.index,
)
X = df[FEATURES]


# 4. Split data
# If identical weather inputs occur more than once,
# keep those inputs together to avoid train/test leakage.
repeated_inputs = df.duplicated(subset=FEATURES).any()

if repeated_inputs:
    print("\nUsing group-aware split to prevent leakage.")

    groups = (
        df[FEATURES]
        .astype(str)
        .agg("|".join, axis=1)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

else:
    # Classes appearing once cannot be stratified.
    counts = y.value_counts()
    rare_labels = counts[counts < 2].index

    rare_mask = y.isin(rare_labels)

    X_rare = X.loc[rare_mask]
    y_rare = y.loc[rare_mask]

    X_common = X.loc[~rare_mask]
    y_common = y.loc[~rare_mask]

    class_count = y_common.nunique()
    test_count = max(
        int(len(X_common) * 0.2 + 0.999),
        class_count,
    )

    if (
        class_count == 0
        or len(X_common) - test_count < class_count
    ):
        raise ValueError(
            "Not enough repeated labels for "
            "a stratified evaluation. "
            "Collect more examples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_common,
        y_common,
        test_size=test_count,
        random_state=42,
        stratify=y_common,
    )

    # Singleton labels are training-only.
    X_train = pd.concat([X_train, X_rare])
    y_train = pd.concat([y_train, y_rare])

    train_idx = X_train.index.to_numpy()
    test_idx = X_test.index.to_numpy()

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print(
    "Note: Rare classes may not be represented "
    "in the test set."
)


# 5. Define the baseline
class TemperatureBaseline:
    def fit(self, X, y):
        self.default_label = y.mode().iloc[0]
        self.band_labels = {}

        bands = (X["temperature"] // 5).astype(int)

        for band in bands.unique():
            labels = y.loc[bands == band]
            self.band_labels[band] = labels.mode().iloc[0]

        return self

    def predict(self, X):
        bands = (X["temperature"] // 5).astype(int)
        return [
            self.band_labels.get(b, self.default_label)
            for b in bands
        ]


# 6. Compare three models
models = {
    "Temperature baseline": TemperatureBaseline(),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            random_state=42,
        ),
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions,
        average="macro", zero_division=0,
    )
    macro_recall = recall_score(
        y_test, predictions,
        average="macro", zero_division=0,
    )

    print(f"\n========== {name} ==========")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Macro recall: {macro_recall:.3f}")

    print(
        classification_report(
            y_test,
            predictions,
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=range(len(encoder.classes_)),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=encoder.classes_,
        columns=encoder.classes_,
    )

    filename = name.lower().replace(" ", "_")
    matrix_df.to_csv(
        OUTPUT_DIR / f"{filename}_confusion_matrix.csv"
    )

    results.append({
        "model": name,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_recall": round(macro_recall, 4),
    })


# 7. Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# 8. Train production Random Forest on all data
# Keep production model separate from test evaluation.
production_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
)

production_model.fit(X, y)

dump(production_model, "outfit_model.joblib")
dump(encoder, "label_encoder.joblib")

label_map = pd.DataFrame({
    "outfit": encoder.classes_,
    "outfit_encoded": encoder.transform(
        encoder.classes_
    ),
})
label_map.to_csv("label_map.csv", index=False)

print("\nEvaluation files saved in evaluation/")
print("Production Random Forest saved successfully.")
