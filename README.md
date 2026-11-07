# WEATHER OUTFIT RECOMMENDER

A simple web application that recommends an outfit based on real-time weather data using a machine learning model. Built using **Flask**, **TailwindCSS**, and **scikit-learn**.


---
## Live Demo

[Try the Weather Outfit Recommender](https://outfit-recommender-phi.vercel.app/)

---
## Preview

<img alt="Screenshot 1" src="screenshot.png" width="100%" />


<img alt="Screenshot 2" src="screenshottwo.png" width="100%" />



---


## Features

- Real-time weather data via [Open-Meteo API](https://open-meteo.com/)
- Machine learning-based outfit prediction
- Responsive UI styled with TailwindCSS

---

## Machine Learning Evaluation

### Models Compared
Three prediction methods were evaluated:

- **Temperature baseline:** Predicts the most common outfit category within each 5°C temperature band.
- **Logistic Regression:** A linear classification model with standardized weather features.
- **Random Forest:** An ensemble classifier using 200 trees with a maximum depth of 10.

### Evaluation Method
The dataset is checked for duplicate records and conflicting labels before training. A held-out test set is used to evaluate model performance, with stratification where feasible and group-aware splitting when repeated weather inputs are present.

Metrics include accuracy, macro F1, macro recall, per-class recall, and confusion matrices.

### Results

| Model | Accuracy | Macro F1 | Macro Recall |
|---|---:|---:|---:|
| Temperature baseline | 0.3182 | 0.1568 | 0.3000 |
| Logistic Regression | 0.3636 | 0.1815 | 0.2778 |
| Random Forest | 0.4545 | 0.2983 | 0.3725 |

### Dataset Provenance
The dataset `weather_outfits.csv` contains weather features and corresponding outfit recommendations.

**Data source:** The repository includes a small manually curated dataset of weather-to-outfit examples stored in `weather_outfits.csv`; these examples were created for the project rather than collected from a large external benchmark.

**Labeling method:** Outfit recommendations were assigned by hand based on weather conditions and general outfit suitability, rather than by automatic annotation from user behavior or a large-scale crowdsourced labeling process.

### Limitations

- The dataset contains a small number of examples relative to the number of outfit categories.
- Some outfit categories are rare and may not appear in the test set.
- Recommendations are subjective and depend on individual comfort preferences.
- Performance on held-out examples does not guarantee generalization to real-world users.
- Additional independently collected weather/outfit examples are needed for more reliable evaluation.

---

##  How It Works

1. The user clicks the **"Predict Outfit"** button.
2. The browser fetches weather data from Open-Meteo, using geolocation data that it fetches via HTML5 Geolocation API
3. A POST request is sent to the Flask backend with:
   - Temperature
   - Humidity
   - Wind speed
   - Precipitation
4. The backend loads a trained `RandomForestClassifier` and label encoder.
5. The model predicts the best outfit based on input.
6. The predicted outfit is displayed on the frontend.

---

###  Prerequisites

Make sure you have:

- Python 3.7 or above
- pip (Python package manager)
- Git (to clone the project, or download ZIP)

---

Clone or Download the Repository

If using Git:

```bash
git clone https://github.com/shivanikadirgamarajah/outfit-recommender.git
cd outfit-recommender
```

---
### Create a virtual enviornment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---
### Install dependencies

```bash
python -m pip install -r requirements.txt
```

---
### Run the application

```bash
python app.py
```

---
