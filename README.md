# Final capstone
# Credit Card Fraud prediction
Author - Pallab Bhandari

Course - Data Science Bootcamp by UpGrad

Published - March 15th 2025
# Introduction credit card
Credit cards are widely used for online shopping and payments, but they are also at risk of fraud. Fraud happens when someone steals and uses another person's credit card details without permission. This causes financial losses for cardholders and creates challenges for credit card companies in detecting and stopping these illegal activities.

As more financial services move online, fraudsters have become more advanced, using new methods to get around security systems. Reports show that credit card fraud costs billions of dollars each year globally. To fight this, fraud detection systems need to keep improving. Machine learning and artificial intelligence (AI) are now key tools in this fight. They can quickly analyze millions of transactions and spot unusual patterns that older systems might miss. This helps protect consumers and reduce fraud.

# Credit Card Basics  

# What is a Credit Card?  
A credit card is a small plastic or metal card given by a bank or financial company. It lets you borrow money to buy things or pay for services at places that accept credit cards.  
Paying Back the Money  
When you use a credit card, you must pay back the money you borrowed. You can either pay the full amount by the due date or pay it back slowly over time. If you don’t pay it all at once, you’ll be charged extra fees called interest.  
Cash Advances  
Some credit cards let you borrow cash, not just pay for things. You can get cash from:  
- Bank tellers  
- ATMs  
- Special checks linked to your card  
But cash advances usually have higher fees and no grace period (meaning interest starts right away).  
Spending Limits  
Every credit card has a limit on how much you can spend. This limit is based on your credit score and financial history.  

# Credit Cards vs. Debit Cards  
Both credit and debit cards are used to pay for things, but they work differently:  
Debit Cards: Take money directly from your bank account. Most checking accounts are free unless you spend more money than you have.  
Credit Cards: Let you borrow money to pay for things. If you don’t pay it back by the due date, you’ll be charged interest.  
Credit Cards in Europe  
-	Visa and Mastercard are the most popular credit cards in Europe.  
-	Many European countries use contactless payments (just tap your card to pay).  
-	Since 2015, the EU has limited fees on credit card transactions to 0.3%.  
-	European credit cards often use chip-and-PIN technology for security, instead of signatures.  
-	European countries have strong consumer protection rules for credit cards, but these rules can vary.  
-	In some countries, like Germany, people use debit cards more often than credit cards.

# Dataset overview
The dataset contains credit card transactions made by European cardholders over two days in September 2013. It includes 284,807 transactions, with only 492 being fraudulent. This means frauds make up just 0.172% of the data, making the dataset highly unbalanced. Detecting fraud in such cases is challenging because the number of fraudulent transactions is very small compared to legitimate ones.
The dataset has 28 features (V1 to V28) that were transformed using Principal Component Analysis (PCA), a technique used to simplify data while keeping its structure

# Goal
The primary goal of this project is to develop a highly accurate and reliable fraud detection model for credit card transactions. Fraudulent transactions can cause significant financial losses, and this model aims to minimize such risks by detecting fraudulent transactions with high precision and recall.

 # Step 1: Importing Required Libraries
I begin by importing the necessary Python libraries for data handling, visualization, feature selection, model training, and evaluation. Libraries like Pandas and NumPy help with data manipulation, Matplotlib and Seaborn assist in data visualization, while Scikit-learn provides tools for model selection, training, and validation.

# Step 2: Loading the Dataset
I load the `creditcard.csv` dataset using Pandas. This dataset contains transaction details such as time, amount, and anonymized features, along with labels indicating whether a transaction is fraudulent. Proper loading and initial exploration of data are crucial to understanding its structure and preparing it for analysis.

# Step 3: Exploratory Data Analysis (EDA)
Exploratory Data Analysis helps in understanding the data distribution and identifying patterns that may impact fraud detection. Key EDA steps include:
Checking Data Information: I inspect the dataset for missing values and outliers, ensuring data quality.
Class Distribution: The dataset is highly imbalanced, so I visualize the proportion of fraudulent and non-fraudulent transactions using a pie chart to highlight the imbalance.
Feature Distribution: I use boxplots to analyze how different features are distributed and identify potential outliers.
Transaction Amount Analysis: A comparison of transaction amounts between fraudulent and non-fraudulent transactions helps understand transaction patterns.
Histogram Visualization: Histograms of numerical features provide insights into data distribution, skewness, and potential normalization needs.
Correlation Matrix: A heatmap is used to examine relationships between different features, which aids in feature selection and elimination of redundant variables.

# Step 4: Handling Missing Values
Missing values can negatively impact machine learning models. I fill missing values with the median of their respective columns, as median imputation is robust against outliers and preserves the overall data structure.

# Step 5: Feature Engineering & Data Transformation
Feature engineering involves modifying and creating new features to improve model performance:
Standardization: I normalize `Amount` and `Time` using `StandardScaler` to ensure uniformity across features.
Handling Imbalance with SMOTE: Since fraudulent transactions are rare, I apply Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic fraud samples, balancing the dataset and improving model learning.

# Step 6: Feature Selection
Selecting the most relevant features is essential for model efficiency. I use `SelectKBest` with the `f_classif` function to select the 20 best features based on their correlation with fraud occurrences. This step enhances model performance by reducing noise and irrelevant data.

# Step 7: Splitting Data for Training and Testing
The dataset is split into 80% training and 20% testing using `train_test_split`. This ensures that the model generalizes well to unseen data and prevents overfitting.

# Step 8: Model Selection
-	I evaluate multiple machine learning models, including:
-	Random Forest Classifier (selected due to its efficiency, interpretability, and robustness against overfitting)
-	Gradient Boosting Classifier (performs well on complex patterns but is computationally expensive)
-	Logistic Regression (simple and interpretable but less effective for non-linear relationships)
-	Decision Tree Classifier (prone to overfitting but useful for feature importance analysis)

-	I select Random Forest as the final model due to its superior accuracy and recall in detecting fraud cases.

# Step 9: Model Training & Hyperparameter Tuning
Hyperparameter Tuning: I optimize model parameters using `RandomizedSearchCV`, which efficiently searches for the best hyperparameters to improve performance.
Training the Model: I train the Random Forest model with the optimal parameters, ensuring it learns patterns from both fraudulent and non-fraudulent transactions effectively.

# Step 10: Model Validation
Model validation ensures that the trained model performs well on unseen data. I use the following evaluation metrics:
Accuracy Score (Overall correctness of predictions)
Precision, Recall, and F1-score (Important for handling imbalanced data)
Confusion Matrix (Visual representation of correct vs. incorrect classifications)
ROC-AUC Score (Measures model reliability in distinguishing fraud vs. non-fraud transactions)

# Step 11: Saving the Model
The trained model is saved as a `.pkl` file using `joblib`, allowing for easy reuse and deployment in real-time fraud detection systems.

# Step 12: Downloading, Using My Fraud Detection Model
1. Saving My Model
After training my model (rf_model), I save it using joblib.dump(), so I don’t have to retrain it every time I want to use it.
2. Downloading and Storing the Model
To keep things organized, I save my model directly in the Downloads folder at:/Users/pallabbhandari/Downloads/credit_card_model.pkl
I use os.path.expanduser("~/Downloads/") to ensure compatibility across different systems.
3. Loading My Model and Making Predictions
Once saved, I can reload my model using joblib.load(). I generate a random transaction sample with the same feature structure as my training data (X_train) and predict if it's fraud or not. The script prints out either "Fraud Transaction" or "Normal Transaction", making it easy to test my model.

# Step 13: Testing the Model
The saved model is loaded and tested with a randomly generated sample to verify its effectiveness in detecting fraudulent transactions.
Key Findings
- The dataset is highly imbalanced, making fraud detection challenging.
- Feature selection significantly improves model efficiency.
- SMOTE enhances fraud detection by providing a balanced dataset.
- Random Forest performs best among tested models, offering high recall and precision.
- Fraud detection requires balancing precision and recall to minimize both false negatives and false positives.
Result and Discussion
The trained Random Forest model achieves:
High accuracy, ensuring effective classification of fraudulent and non-fraudulent transactions.
Strong recall, reducing missed fraudulent transactions.
Good precision, minimizing false fraud alerts.
Balanced F1-score, providing a well-rounded fraud detection system.

Fraud detection models should prioritize recall, as missing fraud cases can result in financial losses. The use of SMOTE significantly improved the model’s ability to identify fraudulent transactions. Future improvements can focus on integrating additional domain-specific features, such as transaction location and customer behavior.

# Conclusion
This fraud detection model effectively identifies fraudulent transactions with high precision and recall. Using SMOTE and feature selection enhances performance, and the model can be further optimized with deep learning techniques for real-time applications.

# Future Work
Real-Time Deployment: Implement fraud detection in live banking systems.
Self-Learning Models: Develop models that adapt over time for improved fraud detection.
Unsupervised Learning: Use anomaly detection techniques to identify novel fraud patterns.
Scalability: Expand to large financial datasets with distributed computing.

# Business Benefits
•	Reduced Financial Loss Early detection minimizes fraud-related losses.

•	Enhanced Security: Protects customers from fraudulent activities.

•	Improved Customer Trust: Reliable fraud prevention strengthens brand reputation.

•	Regulatory Compliance: Helps banks meet fraud prevention standards.

# Summary
This project presents a complete credit card fraud detection pipeline, covering data preprocessing, exploratory data analysis, feature engineering, model selection, training, validation, and deployment. The use of SMOTE, feature selection, and hyperparameter tuning ensures a highly effective fraud detection system. The model can be further improved by incorporating deep learning techniques, real-time detection capabilities, and self-learning mechanisms to adapt to evolving fraud patterns.

# Why my Code is Better

•	Balanced Data Handling: Uses SMOTE to address class imbalance.

•	Feature Selection: Improves efficiency by selecting the most relevant features.

•	Hyperparameter Tuning: Optimizes performance.

•	Comprehensive EDA: Provides detailed insights into fraud patterns.

•	Scalability: Can be extended with deep learning and real-time processing.


