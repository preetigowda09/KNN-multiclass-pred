# 🌱 BeanSense — Dry Bean Classification

BeanSense is a machine learning mini-project that predicts dry bean varieties using numerical shape features and the **K-Nearest Neighbors (KNN)** algorithm.

This project is being developed as part of my machine learning learning journey, from understanding classification algorithms to building a functional web application.

## 🚧 Project Status

* **Version 1 — Current:** Numerical-feature-based classification using KNN with a Flask web interface.
* **Version 2 — Planned:** Bean image upload, automatic shape-feature extraction using computer vision, and integration with the existing classifier.

*Image upload and automatic feature extraction are not implemented in the current version.*

## ✨ Features

* Interactive web application built with Flask.
* Multiclass classification using KNN.
* StandardScaler preprocessing.
* Seven dry bean varieties.
* Sample measurements from the dataset that users can inspect and edit.
* Prediction results with a neighbor-vote breakdown.
* Model evaluation using a separate test set.

## 🧠 Machine Learning Workflow

1. Load and explore the Dry Bean dataset.
2. Remove duplicate records and prepare the data.
3. Separate features and target labels.
4. Split the data into training, validation, and test sets.
5. Scale numerical features using StandardScaler.
6. Select the KNN hyperparameter using validation accuracy.
7. Evaluate the selected model on unseen test data.
8. Save the trained model and scaler.
9. Integrate the model into a Flask web application.

## 📊 Dataset

This project uses the UCI Dry Bean dataset, which contains 16 numerical features describing bean shape and geometry.

The seven classes are:

* BARBUNYA
* BOMBAY
* CALI
* DERMASON
* HOROZ
* SEKER
* SIRA

Examples of features include area, perimeter, major and minor axis lengths, eccentricity, compactness, and shape factors.

## 📈 Model Performance

The selected KNN model achieved approximately **91% accuracy on the test set** in the current evaluation.

Some visually similar bean varieties may be confused by the model. Actual performance depends on the dataset, preprocessing, and evaluation setup.

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Flask
* **Frontend:** HTML, CSS
* **Model Persistence:** Joblib
* **Dataset Format:** Excel

## 📁 Project Structure

```text
BeanSense/
├── app.py
├── beansense_knn.pkl
├── beansense_scaler.pkl
├── Dry_Bean_Dataset.xlsx
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html
│   ├── demo.html
│   └── result.html
└── static/
    └── style.css
```

## 🚀 How to Run Locally

### Prerequisites

* Python installed
* Git installed
* The project files, including the trained model, scaler, and dataset

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/BeanSense.git
cd BeanSense
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the Required Files

Ensure `app.py`, `beansense_knn.pkl`, `beansense_scaler.pkl`, and `Dry_Bean_Dataset.xlsx` are in the project root, with the HTML files inside `templates/` and the CSS file inside `static/`.

### 5. Start the Application

```bash
python app.py
```

### 6. Open in Your Browser

Visit:

http://127.0.0.1:5000

Explore the homepage, open the demonstration page, inspect or edit the sample measurements, and view the predicted bean variety.

## ⚠️ Current Limitations

* The current version requires 16 numerical input features.
* Users cannot upload a bean image for automatic classification yet.
* Neighbor-vote percentages represent the proportions of the selected neighbors voting for each class; they are not calibrated probabilities.
* Predictions are for educational demonstration and are not agricultural certification.

## 🔮 Future Improvements — Version 2

* Add bean image upload.
* Use computer vision to segment a bean from its background.
* Extract and validate the required geometric features.
* Integrate the extracted measurements with the trained KNN classifier.
* Improve image-processing reliability and user experience.

## 👩‍💻 Author

**Preethi A**

CSE (AI & ML) student exploring machine learning through practical projects.

---

*Built with curiosity, data, and a little love for machine learning. 🌱*
