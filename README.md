TECHNICAL PROJECT REPORT
Sentiment Analysis and Customer Classification
of All Flipkart Reviews
Decoding Customer Sentiments: A Comprehensive Analysis of E-commerce Reviews

Domain	Natural Language Processing / Machine Learning
Platform	Flipkart E-commerce
Task Type	Multi-class Sentiment Classification
Tools Used	Python, Scikit-learn, NLTK, TensorFlow, Seaborn
Models Evaluated	SVC, Random Forest, Logistic Regression, Gradient Boosting
Best Accuracy	~89.93% (Random Forest with TF-IDF)
Dataset	Dataset-SA.csv (Flipkart Reviews)
 
1. Executive Summary
This report presents a comprehensive technical analysis of a sentiment classification project applied to customer reviews collected from Flipkart, one of India's largest e-commerce platforms. The project, titled "Decoding Customer Sentiments: A Comprehensive Analysis of E-commerce Reviews," leveraged natural language processing (NLP) techniques and multiple supervised machine learning classifiers to categorize reviews into three sentiment classes — Positive, Negative, and Neutral.
Four classification algorithms were benchmarked: Support Vector Classifier (SVC), Random Forest, Logistic Regression, and Gradient Boosting. All models achieved accuracy scores in the range of 89.38%–89.93%, with Random Forest (TF-IDF vectorization) attaining the highest accuracy of 89.93%. A critical finding across all models was the consistent failure to correctly classify the Neutral sentiment class, yielding F1-scores of 0.00 for that class, which points to a severe class imbalance problem in the dataset.
The analysis also revealed that approximately 60% of Flipkart reviewers awarded a 5-star rating, and the overall sentiment of the review corpus is predominantly Neutral, followed by Positive, with a very small proportion of Negative reviews.
2. Project Overview
E-commerce platforms generate millions of customer reviews every day. Understanding the sentiment embedded in these reviews is critical for businesses to gauge product quality, customer satisfaction, and areas requiring improvement. This project performs end-to-end sentiment analysis on Flipkart product reviews, covering data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and comparison.

Project Title	Sentiment Analysis and Customer Classification of All Flipkart Reviews
Subtitle	Decoding Customer Sentiments: A Comprehensive Analysis of E-commerce Reviews
Programming Language	Python 3.x
Key Libraries	Pandas, NumPy, Scikit-learn, NLTK, TensorFlow/Keras, Matplotlib, Seaborn, Plotly, WordCloud
Vectorization Methods	Bag-of-Words (CountVectorizer), TF-IDF (TfidfVectorizer)
Sentiment Labeling	VADER (Valence Aware Dictionary and Sentiment Reasoner) — NLTK
Train-Test Split	80% Training / 20% Testing (random_state=42)
Target Classes	Positive, Negative, Neutral

3. Problem Statement
In the highly competitive e-commerce space, customer reviews serve as a direct channel of feedback. However, manually reading and categorizing thousands of reviews is time-consuming and error-prone. There exists a need for an automated, scalable, and accurate system that can:
•	Classify customer reviews into meaningful sentiment categories (Positive, Negative, Neutral).
•	Provide businesses with actionable intelligence about product satisfaction.
•	Identify patterns in customer language that correlate with different sentiment polarities.
•	Compare the effectiveness of multiple machine learning algorithms for the sentiment classification task.

The challenge is further complicated by the natural ambiguity of language — sarcasm, informal writing styles, abbreviations, and mixed sentiments within a single review make accurate classification difficult. A specific bottleneck identified in this project is the classification of Neutral sentiments, which are linguistically less distinct than Positive or Negative reviews.
4. Objectives
•	Perform comprehensive exploratory data analysis to understand the distribution and characteristics of Flipkart reviews.
•	Preprocess text data through cleaning, tokenization, stop-word removal, and stemming.
•	Apply VADER sentiment analysis to generate polarity scores (Positive, Negative, Neutral) for each review.
•	Engineer text features using Bag-of-Words (CountVectorizer) and TF-IDF representations.
•	Train and evaluate four machine learning classifiers: SVC, Random Forest, Logistic Regression, and Gradient Boosting.
•	Analyze and compare model performance using accuracy, precision, recall, F1-score, and confusion matrices.
•	Derive actionable insights from the analysis to support business decision-making.
5. Dataset Description
The dataset used for this project is "Dataset-SA.csv," a collection of customer product reviews scraped from Flipkart. The dataset contains both textual review content and associated metadata, including product price and customer star ratings.

File Name	Dataset-SA.csv
Key Columns	Review (text), Rate (star rating 1–5), Sentiment (label), product_price
Target Variable	Sentiment — derived from VADER polarity scores
Preprocessing Applied	Null removal, deduplication, lowercasing, punctuation removal, stemming
Rating Distribution	~60% of reviews carry a 5-star rating
Dominant Sentiment	Neutral (by VADER aggregate polarity scoring)

After initial loading, the dataset underwent null value detection (df.isnull().sum()), followed by dropping all rows with missing values (df.dropna(inplace=True)), and removal of duplicate records (df.drop_duplicates(inplace=True)). The cleaned dataframe was stored as df2 to preserve the original Sentiment labels before VADER-based polarity decomposition.
6. Methodology & Project Process
The project follows a structured end-to-end NLP pipeline encompassing data collection, preprocessing, feature engineering, model training, evaluation, and interpretation.

6.1 Data Loading & Initial Inspection
•	Dataset loaded using Pandas: df = pd.read_csv('Dataset-SA.csv')
•	Shape inspection with df.shape; data types and null counts verified via df.info() and df.describe().
•	Null rows dropped; duplicate records removed to ensure data integrity.

6.2 Text Preprocessing Pipeline
A multi-step text normalization pipeline was applied to the Review column:
•	Lowercasing: All review text converted to lowercase to normalize casing inconsistencies.
•	Punctuation Removal: Python's string.punctuation module used to strip all punctuation.
•	URL & HTML Tag Removal: Regex-based cleaning removed hyperlinks and HTML artifacts.
•	Digit/Number Removal: Numeric tokens eliminated using regex pattern \w*\d\w*.
•	Stop-word Removal: NLTK English stop-words list applied to filter uninformative words.
•	Stemming: NLTK SnowballStemmer (English) reduced words to their morphological roots, reducing vocabulary dimensionality.

6.3 Exploratory Data Analysis
•	Distribution of product_price visualized using a histogram (sns.histplot).
•	Rating distribution visualized using a donut/pie chart via Plotly Express.
•	Sentiment distribution visualized using sns.countplot.
•	Word frequency histogram plotted for feature corpus (bins=50, range 0–5000).
•	Word Cloud generated from the cleaned review text to surface dominant vocabulary.

6.4 VADER Sentiment Scoring
The NLTK VADER SentimentIntensityAnalyzer was applied to every review to compute continuous polarity scores:
•	Positive Score: df['Positive'] — proportion of positive sentiment.
•	Negative Score: df['Negative'] — proportion of negative sentiment.
•	Neutral Score: df['Neutral'] — proportion of neutral/objective content.
An aggregate scoring function (sentiment_score) determined the overall corpus sentiment by comparing the sum of all Positive, Negative, and Neutral scores — determining that Neutral dominates the corpus.

6.5 Feature Engineering
•	Bag-of-Words (CountVectorizer): Transforms text into token frequency vectors. max_features=10,000 applied to cap vocabulary size.
•	TF-IDF (TfidfVectorizer): Applied with a custom stemmed_words analyzer using Porter Stemmer, max_features=10,000. Produces term-frequency-inverse-document-frequency weighted features that down-weight common terms.
•	Sparsity Analysis: Feature matrix density computed as (NNZ / (rows × cols)) × 100.

6.6 Model Training & Evaluation
All four models used the same data split strategy:
•	80/20 train-test split: train_test_split(..., test_size=0.2, random_state=42)
•	Feature representation: CountVectorizer (primary) and TF-IDF (secondary).
•	Metrics computed: Accuracy score, Precision, Recall, F1-Score (per class), Confusion Matrix.

Models evaluated:
•	Support Vector Classifier (SVC) — kernel: default RBF
•	Random Forest Classifier — default n_estimators=100
•	Logistic Regression — default solver
•	Gradient Boosting Classifier — default parameters
7. Exploratory Data Analysis (EDA)
7.1 Star Rating Distribution
The distribution of star ratings was analyzed using a Plotly donut pie chart. The results were highly revealing:
Star Rating	Approximate Share	Interpretation
5 Stars	~60%	Majority — highly satisfied customers
4 Stars	~15–18%	Satisfied customers with minor issues
3 Stars	~8–10%	Mixed reviews — average experience
2 Stars	~5–7%	Dissatisfied customers
1 Star	~5–8%	Highly dissatisfied customers

The dominant 5-star rating suggests that the majority of Flipkart customers are highly satisfied, but it also introduces a class imbalance problem — since most reviews are positive-skewed, neutral and negative classes are underrepresented.

7.2 VADER Aggregate Sentiment Scores
The sum of VADER polarity scores across all reviews revealed the following overall sentiment distribution:
Sentiment Dimension	Aggregate Score	Relative Magnitude
Positive (x)	High	Second largest — many positive-toned reviews
Negative (y)	Low	Smallest — relatively few negative expressions
Neutral (z)	Highest	Dominant — most review language is objective/neutral
Polarity (x – y)	Positive	Net positive — reviews lean favorable overall

The VADER aggregate result confirms that customer reviews are primarily neutral in language — this is expected, as many reviews describe product features objectively (e.g., "the color is red, size is medium") rather than expressing strong emotions.

7.3 Vocabulary & Feature Analysis
Metric	Value
Total vocabulary (CountVectorizer, all features)	Variable (unbounded)
Vocabulary (capped, max_features=10,000)	10,000 unique tokens
Single-occurrence features	Large count (Hapax Legomena — high sparsity signal)
Feature matrix shape	(num_reviews, 10,000)
Matrix density	Low — sparse representation (<<10%)
Top token distribution	Highly skewed — few words dominate, most are rare

The word frequency histogram (bins=50, range 0–5000) revealed a classic power-law (Zipfian) distribution — a small number of high-frequency tokens account for the majority of word occurrences, while the vast majority of tokens appear only once or twice.
8. Model Development & Numerical Results
8.1 Model Performance Summary Table
Model	Vectorizer	Accuracy	Neg. Precision	Neg. Recall	Neg. F1	Neu. F1	Pos. Precision	Pos. Recall	Pos. F1
SVC	BoW	89.91%	0.85	0.75	0.80	0.00	0.91	0.99	0.95
Random Forest	BoW	89.92%	0.85	0.76	0.80	0.00	0.91	0.99	0.95
Logistic Reg.	BoW	89.92%	0.85	0.76	0.80	0.00	0.91	0.99	0.95
Gradient Boost	BoW	89.38%	0.85	0.72	0.78	0.00	0.90	0.99	0.94
Random Forest	TF-IDF	89.93%	0.85	0.76	0.80	0.00	0.91	0.99	0.95

8.2 Detailed Per-Model Results
8.2.1 Support Vector Classifier (SVC)
Class	Precision	Recall	F1-Score	Support
Negative	0.85	0.75	0.80	—
Neutral	0.00	0.00	0.00	~Very Few
Positive	0.91	0.99	0.95	—
Overall Accuracy	—	—	89.91%	—

Key Observations: SVC achieved 89.91% overall accuracy. The high positive recall (0.99) indicates the model almost always correctly identifies positive reviews. The Neutral F1-score of 0.00 reveals a complete failure to detect neutral samples — the model never predicted the Neutral class, suggesting severe class imbalance. The negative F1-score of 0.80 shows reasonable performance for minority negative class detection.

8.2.2 Random Forest Classifier
Class	Precision	Recall	F1-Score
Negative	0.85	0.76	0.80
Neutral	0.29	0.00	0.00
Positive	0.91	0.99	0.95
Overall Accuracy	—	—	89.92%

Key Observations: Random Forest marginally outperformed SVC with 89.92% accuracy. The Neutral precision of 0.29 suggests the model did predict some Neutral instances (29% of its Neutral predictions were correct), but the 0.00 recall means it still failed to capture any true Neutral samples. This confirms the class imbalance issue rather than a model capability limitation.

8.2.3 Logistic Regression
Class	Precision	Recall	F1-Score
Negative	0.85	0.76	0.80
Neutral	0.40	0.00	0.00
Positive	0.91	0.99	0.95
Overall Accuracy	—	—	89.92%

Key Observations: Logistic Regression achieved identical accuracy (89.92%) to Random Forest. Notably, it achieved the highest Neutral precision (0.40) among all BoW-based models, meaning 40% of its Neutral predictions were correct — but it still recalled 0.00 of the actual Neutral instances. This is a classic precision-recall tradeoff under severe class imbalance.

8.2.4 Gradient Boosting Classifier
Class	Precision	Recall	F1-Score
Negative	0.85	0.72	0.78
Neutral	1.00	0.00	0.00
Positive	0.90	0.99	0.94
Overall Accuracy	—	—	89.38%

Key Observations: Gradient Boosting achieved the lowest overall accuracy (89.38%), but achieved perfect Neutral precision (1.00) — every single Neutral prediction it made was correct, but it predicted the Neutral class so rarely that it still yielded 0.00 recall and F1. Negative recall also dropped to 0.72 (vs. 0.76 for other models), suggesting the model's decision boundary is less optimal for minority classes.

8.2.5 Random Forest with TF-IDF Vectorization
Class	Precision	Recall	F1-Score
Negative	0.85	0.76	0.80
Neutral	0.25	0.00	0.00
Positive	0.91	0.99	0.95
Overall Accuracy	—	—	89.93%

Key Observations: TF-IDF vectorization with Random Forest achieved the highest accuracy overall (89.93%) — a marginal but notable improvement over BoW representations. The TF-IDF weighting appropriately down-weights common filler terms, giving more discriminative power to content-bearing words. However, the Neutral class challenge persists regardless of vectorization method.

8.3 Comparative Accuracy Summary
Rank	Model	Vectorizer	Accuracy
1	Random Forest	TF-IDF	89.93%
2	Random Forest	BoW	89.92%
2	Logistic Regression	BoW	89.92%
4	SVC	BoW	89.91%
5	Gradient Boosting	BoW	89.38%

The accuracy range across all models is extremely narrow (89.38%–89.93%), suggesting the primary bottleneck is not the choice of algorithm but the inherent data imbalance in the Neutral class. All models converge to similar performance bounds, implying that addressing class imbalance would yield far greater gains than further algorithm tuning.
9. Graphical & Visual Results
The following visualizations were generated during the analysis. Each chart is described with its analytical significance:

9.1 Star Rating Donut Chart (Plotly Express)
Type: Interactive Donut/Pie Chart | Library: Plotly Express | Variable: Rate (1–5 stars)
Finding: The chart dramatically illustrated that approximately 60% of all Flipkart reviewers awarded the maximum 5-star rating, confirming a strong positive bias in the dataset. The remaining 40% was distributed across 1–4 stars, with 4-star being the next largest segment.

9.2 Sentiment Distribution Bar Chart (Seaborn countplot)
Type: Vertical Bar Chart | Library: Seaborn | Variable: Sentiment
Finding: The countplot revealed the magnitude of class imbalance — the Positive class contained the overwhelming majority of labeled samples, followed by Negative, with Neutral being extremely underrepresented. This directly explains why all models achieve near-zero performance on the Neutral class.

9.3 Product Price Distribution Histogram
Type: Histogram | Library: Seaborn | Variable: product_price
Finding: The price distribution was right-skewed, indicating that the majority of reviewed products are in lower price brackets, with a long tail of higher-priced items. This is consistent with the broad e-commerce category coverage on Flipkart.

9.4 Review Word Cloud
Type: Word Cloud | Library: WordCloud, NLTK | Input: Cleaned & Stemmed Review Text
Finding: The word cloud provided a visual vocabulary signature of the review corpus. Dominant terms included product-specific adjectives ("good", "nice", "quality"), product category words, and general satisfaction descriptors. Negative terms were noticeably smaller, confirming the positive sentiment skew observed in the ratings distribution.

9.5 VADER Polarity Score Bar Chart
Type: Bar Chart | Library: Matplotlib | Variables: Positive sum (x), Negative sum (y), Neutral sum (z), Polarity (x–y)
Finding: The bar chart visually confirmed the VADER aggregate finding — the Neutral bar was the tallest, followed by Positive, with Negative being the shortest. The Polarity bar (x–y) was positive, indicating an overall favorable sentiment direction in the corpus.

9.6 Word Frequency Histogram
Type: Histogram | Library: Matplotlib | Variable: Feature counts from CountVectorizer
Finding: The histogram (bins=50, range 0–5000) exhibited a sharp Zipfian distribution — the vast majority of features appeared very rarely (frequency near 0–100), while a tiny fraction appeared thousands of times. This confirmed high sparsity and motivated the use of max_features=10,000 as a vocabulary cap.

9.7 Confusion Matrix Heatmaps (Seaborn)
Type: Heatmap | Library: Seaborn | Models: All four classifiers
Finding: The confusion matrices for all models showed a consistent pattern — the large majority of predictions fell into the Positive class (high true-positive mass in the positive quadrant). The Neutral class cells showed near-zero or zero values for true positives, with most Neutral samples misclassified as Positive or Negative. The Negative class showed moderate true-positive performance, with some confusion with the Positive class.
10. Detailed Insights
10.1 Class Imbalance Is the Critical Bottleneck
The most significant finding across all experiments is that every model failed completely on Neutral sentiment classification, achieving F1-scores of exactly 0.00. This is not an algorithmic limitation — it is a data problem. The Neutral class is severely underrepresented in the labeled dataset (df2['Sentiment']). When models train on imbalanced data, they learn to ignore minority classes because doing so maximally preserves overall accuracy. The 89.9% accuracy is therefore misleading — a naive model that always predicts "Positive" would achieve comparable accuracy.

10.2 Positive Sentiment Dominates (60% Five-Star Reviews)
The finding that 60% of reviewers rated products 5 out of 5 has strong business implications. It suggests Flipkart customers are generally highly satisfied. However, this also means the model is heavily biased toward predicting Positive — a prediction strategy that inflates accuracy metrics without providing nuanced intelligence about dissatisfied or neutral customers.

10.3 Neutral Language Pervades Reviews
VADER's aggregate analysis confirmed that the summed Neutral polarity score was the highest among all three dimensions. This makes semantic sense — many product descriptions in reviews are factual and objective ("the phone has a 48MP camera, battery lasts 2 days") without expressing emotional sentiment. VADER scores such language as neutral, and when summed across thousands of reviews, neutral language dominates. This insight suggests that a purely VADER-based labeling approach may not capture true emotional sentiment for model training.

10.4 TF-IDF Marginally Outperforms BoW
TF-IDF vectorization (Random Forest: 89.93%) marginally outperformed Bag-of-Words (89.92%), confirming that down-weighting highly frequent but low-information terms (stop words that survived cleaning, repeated filler phrases) provides a small discriminative advantage. The margin is narrow because stop-word removal and stemming in preprocessing already reduce the dominance of such terms.

10.5 Gradient Boosting Shows Different Error Profile
While Gradient Boosting achieved the lowest accuracy (89.38%), it uniquely achieved 1.00 Neutral precision — meaning when it did predict Neutral, it was always correct. This indicates it learned a highly conservative Neutral decision boundary. This makes Gradient Boosting a candidate for high-precision neutral detection applications where false positives are costly, even at the expense of recall.

10.6 Negative Class Classification Is Reasonably Strong
All models achieved Negative class precision of 0.85 and recall of 0.72–0.76, yielding F1-scores of 0.78–0.80. This is noteworthy given the presumed minority status of Negative reviews. The relatively strong Negative performance (compared to the absent Neutral performance) suggests the vocabulary of negative reviews is more distinctively different from positive reviews than neutral reviews are.

10.7 Vocabulary Analysis: Zipfian Distribution
The feature frequency histogram revealed a classic power-law distribution. A large number of hapax legomena (words appearing only once) were detected. This confirms that: (1) the review vocabulary is extremely diverse; (2) a capped vocabulary (max_features=10,000) captures the most statistically useful terms; (3) high matrix sparsity is expected and appropriate for this type of text data.
11. Conclusion
This project successfully implemented a complete NLP pipeline for sentiment analysis of Flipkart customer reviews, from raw data loading through to model evaluation and comparison. The key achievements and findings are summarized below:

Finding	Detail
Overall Accuracy	All models achieved ~89.4%–89.9% accuracy
Best Model	Random Forest with TF-IDF (89.93%)
Positive Class	Strongly classified — F1: 0.95 across all models
Negative Class	Reasonably classified — F1: 0.78–0.80
Neutral Class	Failed completely — F1: 0.00 across all models
Root Cause	Severe class imbalance — Neutral is highly underrepresented
Customer Sentiment	Generally positive — 60% gave 5-star ratings
VADER Finding	Neutral language dominates the corpus by aggregate score

The project demonstrates that while machine learning models can achieve high aggregate accuracy on sentiment classification tasks, meaningful per-class performance requires addressing class imbalance. The models are not truly "90% accurate" at understanding customer sentiment — they are predominantly predicting the majority class. A balanced evaluation using macro-averaged F1-score would more honestly reflect model performance, yielding considerably lower scores than the reported accuracy.
From a business perspective, the project confirms that Flipkart customers are overwhelmingly positive, VADER-scored language is primarily neutral (factual), and negative sentiments — while a minority — are detectable with moderate reliability. These insights can directly inform product recommendation systems, review summarization tools, and customer service prioritization engines.
12. Further Improvements & Recommendations
12.1 Address Class Imbalance (Critical Priority)
•	SMOTE (Synthetic Minority Oversampling Technique): Synthetically generate Neutral class samples to balance the training set.
•	Class Weighting: Use class_weight='balanced' in Scikit-learn classifiers to penalize misclassification of minority classes.
•	Undersampling: Reduce Positive class samples to match minority class distribution.
•	Threshold Tuning: Adjust decision thresholds for Neutral classification using ROC analysis.

12.2 Improve Sentiment Labeling
•	Revisit VADER labeling: Use compound score thresholding (e.g., compound > 0.05 → Positive, < -0.05 → Negative, else → Neutral) to generate more balanced and semantically meaningful labels.
•	Manual annotation: Incorporate human-labeled ground truth for a subset of reviews to reduce VADER's limitation with domain-specific language.
•	Use star ratings as proxy labels: Map 4–5 stars to Positive, 3 stars to Neutral, 1–2 stars to Negative for a cleaner class distribution.

12.3 Advanced Deep Learning Models
•	BERT / RoBERTa: Fine-tune pre-trained transformer models on the review dataset. These models capture contextual semantics far better than BoW or TF-IDF representations.
•	LSTM / BiLSTM: Implement sequence-aware recurrent networks to capture word order and context, which BoW approaches discard.
•	DistilBERT: A lighter transformer option suitable for deployment, achieving near-BERT performance with significantly lower computational cost.

12.4 Feature Engineering Enhancements
•	N-gram features: Add bigrams and trigrams to CountVectorizer (ngram_range=(1,3)) to capture multi-word expressions (e.g., "not good", "very bad").
•	Sentiment lexicons: Integrate domain-specific e-commerce sentiment lexicons.
•	POS tagging: Use part-of-speech tags to weight adjectives and adverbs more heavily in the feature representation.

12.5 Hyperparameter Optimization
•	Grid Search / Random Search: Systematically tune n_estimators, max_depth (Random Forest), C, kernel (SVC), and learning_rate (Gradient Boosting).
•	Bayesian Optimization: Use Optuna or Hyperopt for efficient hyperparameter search.
•	Cross-validation: Replace single train-test split with k-fold cross-validation (k=5 or 10) for more robust performance estimates.

12.6 Evaluation Metric Improvements
•	Macro F1-Score: Replace accuracy as the primary metric to fairly evaluate per-class performance under imbalance.
•	ROC-AUC curves: Plot per-class ROC curves for probabilistic evaluation.
•	Cohen's Kappa: Measure inter-class agreement to assess model reliability beyond chance.

12.7 Deployment Considerations
•	REST API: Deploy the best model as a Flask/FastAPI REST endpoint for real-time review classification.
•	MLflow: Implement experiment tracking to log all model runs, hyperparameters, and metrics.
•	Streaming Integration: Connect to Flipkart review streams for real-time sentiment monitoring dashboards.
13. Tools

Python	Version 3.x — Primary programming language
Pandas	Data loading, manipulation, and preprocessing
NumPy	Numerical operations and array manipulation
Scikit-learn	ML models, vectorization, train-test split, metrics
NLTK	VADER sentiment scoring, stop-words, tokenization, stemming
TensorFlow / Keras	Deep learning framework (imported; ready for LSTM/CNN extensions)
Matplotlib	Static charts — bar charts, histograms
Seaborn	Statistical visualizations — countplots, heatmaps
Plotly Express	Interactive charts — donut/pie rating distribution
WordCloud	Word cloud visualization from review vocabulary
Dataset	Dataset-SA.csv — Flipkart customer product reviews


— End of Report —
