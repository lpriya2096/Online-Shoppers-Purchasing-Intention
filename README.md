# Online Shopping Purchase Prediction

## a. Problem statement
Online retailers want to know which website sessions end in a purchase. Using browsing behaviour (pages visited, time spent, bounce/exit rates, page value, visitor type, month, etc.), this project builds five classification models to predict whether a session results in a purchase (`Revenue`), compares them on six evaluation metrics, and serves them through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository, *Online Shoppers Purchasing Intention Dataset* (ID 468).
- **Size:** 12,330 sessions, 17 input features, 1 target (`Revenue`).
- **Features:** 10 numeric (Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay) and 7 categorical (Month, VisitorType, Weekend, OperatingSystems, Browser, Region, TrafficType).
- **Target:** binary. 10,422 sessions without a purchase (84.5%) and 1,908 with a purchase (15.5%), so the classes are imbalanced.
- **Quality checks:** no missing values; 125 duplicate rows were found and kept, since identical sessions can legitimately occur.
- **Split:** 80/20 stratified split (`random_state=42`): 9,864 training rows and 2,466 test rows (`test_data.csv`).
- **Preprocessing:** numeric features standardised, categorical features one-hot encoded, both inside a scikit-learn `Pipeline`, so the transformers are fitted on training data only.

## c. GitHub repository
https://github.com/lpriya2096/Online-Shoppers-Purchasing-Intention

Live Streamlit app: <paste your streamlit.app link here>

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8990 | 0.9172 | 0.7111 | 0.5864 | 0.6428 | 0.5883 |
| KNN | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Support Vector Machine | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.74) but it catches only 36% of real purchases. With 15.5% buyers and the default 0.5 threshold it plays safe and predicts "no purchase" often. AUC (0.88) shows the underlying ranking is good; the threshold is what limits recall. |
| Decision Tree | Best accuracy, AUC, F1 and MCC. Depth 5 lets it capture non-linear rules (such as high PageValues leading to a purchase) that a linear model misses, and it recalls 59% of buyers with precision still 0.71. |
| KNN | Weakest of the four usable models: lowest AUC (0.76) and MCC (0.45). With k=5 the probabilities are coarse, and the many one-hot columns dilute the distances, so neighbours are not very informative. Recall is only 38%. |
| Naive Bayes | Accuracy collapses to 0.27 because it predicts "purchase" for about 87% of sessions. Recall is 0.97 but precision (0.17) is about the base rate, i.e. it is barely better than guessing. The cause is the ~60 one-hot columns: Gaussian NB treats each 0/1 column as an independent bell curve, and these columns are neither Gaussian nor independent (dummies from one feature are mutually exclusive). Their tiny variances give extreme likelihoods that swamp the numeric evidence. |
| Support Vector Machine | Second-best on MCC (0.51) and accuracy (0.89), with better recall than Logistic Regression and KNN. Still misses over half of buyers at the default settings. |
| **Overall winner** | **Decision Tree.** It leads on accuracy (0.899), AUC (0.917), F1 (0.643) and MCC (0.588), and is second on precision. Because only 15.5% of sessions are purchases, accuracy alone is misleading, so MCC and F1 were used to decide. |

### Limitations
All models use default 0.5 thresholds and light hyperparameters without tuning; class weights or threshold tuning would likely raise recall for Logistic Regression, KNN and SVM.

## How to run
```
pip install -r requirements.txt
streamlit run app.py
```
Upload `test_data.csv` in the app, pick a model, and the app shows the six metrics, the confusion matrix and the formula for each metric.
