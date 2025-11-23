# SMS-Spam-Detection-Systems
"Built a Natural Language Processing (NLP) prototype to identify spam messages. I utilized the CountVectorizer to transform text data into numerical vectors and trained a Multinomial Naive Bayes classifier to distinguish between legitimate and spam content."
# 📧 SMS Spam Classifier (NLP Project)

## 🎯 Project Goal
This project implements a foundational Machine Learning model to accurately classify incoming text messages (or emails) as either **Spam** (unwanted) or **Ham** (legitimate). Developed as part of my focus on applying core Computer Science concepts to Artificial Intelligence challenges.

## ✨ Key Concepts Demonstrated
The model successfully showcases end-to-end execution of a text classification task:

1.  **Natural Language Processing (NLP):** Handling raw text data.
2.  **Feature Extraction:** Converting unstructured text into a numerical matrix.
3.  **Supervised Learning:** Training a model on labeled data.

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Core Library:** Scikit-learn (sklearn)
* **Data Handling:** Pandas
* **Algorithm:** Multinomial Naive Bayes (chosen for its efficiency and strong baseline performance in text classification)

## 🧠 Model Pipeline (How it Works)

1.  **Vectorization:** Raw messages are processed using the **`CountVectorizer`** (Bag of Words model) to transform text into a numerical feature matrix.
2.  **Training:** The matrix is fed into the **`MultinomialNB`** classifier, which learns the probability of specific words appearing in Spam vs. Ham messages.
3.  **Prediction:** The model calculates the probability scores for new, unseen messages and assigns the highest probability label (Spam or Ham).

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed. Then, install the required dependencies:
```bash
pip install pandas scikit-learn
