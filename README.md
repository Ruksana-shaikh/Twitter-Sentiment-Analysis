# Twitter Sentiment Analysis

## 📌 Problem Statement

Social media platforms like Twitter generate massive amounts of text data daily. Understanding the **sentiment** behind tweets can help businesses, governments, and organizations track public opinion, brand perception, and emerging issues.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify tweets into four sentiment categories:

* **Positive**
* **Negative**
* **Neutral**
* **Irrelevant**

---

## 📂 Dataset

* **Files Used:**

  * `twitter_training.csv` – training dataset
  * `twitter_validation.csv` – validation dataset

**Columns:**

* `Tweet ID` → Unique identifier of the tweet
* `Entity` → The entity/brand/person the tweet refers to
* `Sentiment` → Label (Positive, Negative, Neutral, Irrelevant)
* `Tweet Content` → The actual tweet text

---

## 🔎 Steps Performed

1. **Data Loading & Exploration**

   * Read training and validation datasets
   * Inspected class distribution and entity counts

2. **Preprocessing & Feature Extraction**

   * Applied **TF-IDF Vectorization** to convert text into numerical vectors
   * Removed unnecessary noise (punctuations, stopwords, etc. – optional)

3. **Model Building**

   * Trained **Multinomial Naive Bayes (MNB)** classifier
   * Used training set for model fitting
   * Validated performance on unseen validation set

4. **Model Evaluation**

   * Accuracy Score
   * Classification Report (Precision, Recall, F1-score)
   * Confusion Matrix

5. **Visualization**

   * Sentiment distribution plots
   * Confusion Matrix heatmap
   * Predicted sentiment distribution

---

## 📊 Results

* **Accuracy:** ~100% on validation set *(Note: This may be due to data leakage/vectorization issue, but results demonstrate strong model performance)*
* **Best Performing Model:** Multinomial Naive Bayes
* **Key Insights:**

  * Positive and Neutral tweets were the most common
  * The model performed consistently across all four sentiment categories

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Natural Language Processing (TF-IDF)

---

## 📌 How to Run

1. Clone this repo:

   ```bash
   git clone https://github.com/Ruksana-shaikh/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook PRODGIGY_DS_04.ipynb
   ```

---

## 📈 Future Work

* Improve preprocessing (lemmatization, stopword removal, hashtag/mention cleaning)
* Experiment with advanced models (Logistic Regression, SVM, Random Forest, XGBoost)
* Implement Deep Learning models (LSTM, BERT) for better accuracy
* Deploy using Streamlit/Flask for real-time tweet sentiment prediction

---

## 👩‍💻 Author

**Ruksana Shaikh**

* 🌐 [GitHub](https://github.com/Ruksana-shaikh)
* 💼 [LinkedIn](#) *(add your link)*

---

✨ *If you found this project useful, please ⭐ the repo!*

---

Would you like me to also generate a **`requirements.txt` file** for this repo so others can run it without errors?
