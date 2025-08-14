# Week 1 – Sentiment Classification with Traditional ML

## 📌 Objective
The goal of this project is to build and evaluate multiple text classification models on a balanced dataset, comparing architectures such as Logistic Regression, XGBoost, LightGBM, and CatBoost.  

We apply consistent text preprocessing (including TF–IDF feature extraction) and model training to identify the best-performing approach based on validation F1 score.  
The final output includes performance metrics, code for reproducibility, and a baseline for future model improvements.

## 🧩 Skills & Concepts
- **Text Preprocessing:** tokenization, lemmatization, stopword removal, number replacement 
- **Feature Engineering:** TF-IDF vectorization for sparse text features
- **Modeling:**: Logistic Regression, XGBoost, CatBoost, LightGBM
- **Hyperparameter Tuning:** Optuna integration
- **Evaluation Metrics:** Accuracy, precision, recall, F1, AUROC, AUPRC
- **Experiment Tracking:**: MLFlow for logging metrics, parameters, and artifacts 

## 📦 Dataset
- **Source:** [Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
- **Size:** 560k training examples, 140k validation samples, and 400k test samples
- **Preprocessing:** I preprocess with both the NLTK and SpaCy packages to get a feel for both, but modeling is only done with the NLTK datasets. I tokenize, lemmatize, remove stopwords, lowercase, etc. I also replace all numbers with `<NUM>`. 

Below are some example reviews from the dataset: 

*Positive Sentiment:* 'I recommend this for anyone that is interested in writing graphic novel and/or drawing for comics. This one is truly legendary.'

*Negative Sentiment:* 'I have not been satisfied with this water heater. When they say it is for 1 application, they mean it. If you are taking a shower, and someone washes their hands at a sink, the shower will get cold. Also, you basically have to turn the water on at almost full blast for the heater to kick on. If the pressure is too low, it will just click the igniter without lighting. The warmth of the water seems to fluctuate for no reason. It will be hot for a few minutes and then get colder. I usually have to adjust the temperature in the shower at least once per shower. I live in a 2-story house with the heater in the basement. I know two other people with this heater that live in 1-story houses and do not have my issues. Keep in mind that it will be at least $50 for the parts for the re-routing of the pipes. Plan on the work taking at least 1 day, maybe 2. Also note that it takes an extra 10 seconds to get warm water over a water heater tank. This should be expected though.'

## 📂 Project Structure
- `src/` – Core source code, preprocessing utils, and MLFlow logging functions
- `notebooks/` – Prototyping and data exploration  
- `artifacts/` – Contains metrics, AUROC/AUPRC curves, and confusion matrices for final models on train, val, and test sets
- `best_model/` - Contains metrics and artifacts for the best overall performing model 
- `preprocessing.py` - Script to run the data preprocessing and save the files
- `main.py` – Script to run models and log metrics
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week1 python=3.10 -y
conda activate week1

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 main.py
```

## 📊 Results
Below are the performance metrics for each of the best chosen models from the 4 different architectures. The best model was chosen based on validation F1 score since the dataset is perfectly balanced and both recall and precision are important. This best model was an XGBoost model. 

| Model    | Accuracy | Precision | Recall | Specificity | F1    | AUROC  | AUPRC  |
|----------|----------|-----------|--------|-------------|-------|--------|--------|
| LR       | 0.810    | 0.810     | **0.811**  | 0.809       | 0.810 | 0.893  | 0.892  |
| **XGBoost**  | **0.812**    | **0.815**     | 0.807  | **0.817**       | **0.811** | **0.896**  | **0.896**  |
| LightGBM | 0.782    | 0.789     | 0.770  | 0.794       | 0.780 | 0.868  | 0.866  |
| CatBoost | 0.782    | 0.792     | 0.766  | 0.799       | 0.779 | 0.867  | 0.865  |

Below is the confusion matrix of the chosen XGBoost model on the test set. 

![XGBoost Confusion Matrix](artifacts/xgb/test_confusion_matrix.png)

## 📌 Key Takeaways
- **XGBoost emerged as the top-performing model** across most evaluated metrics, achieving the highest accuracy (0.812), precision (0.815), specificity (0.817), AUROC (0.896), and AUPRC (0.896).  
- **Logistic Regression** closely followed XGBoost in performance, with the highest recall (0.811) and strong scores across all other metrics, indicating it remains a solid and interpretable baseline.  
- **LightGBM and CatBoost** underperformed relative to XGBoost and Logistic Regression, with ~3% lower accuracy and noticeably lower AUROC and AUPRC values, suggesting weaker generalization to the test set.  
- The **dataset being perfectly balanced** allowed F1 score to serve as a reliable metric for model selection, balancing the trade-off between recall and precision.  
- The **performance gap between XGBoost and the other gradient boosting models** (LightGBM, CatBoost) may point to differences in how each algorithm handled feature interactions, regularization, or hyperparameter tuning in this specific task.
- **TF-IDF vectorization** proved effective in converting raw text into meaningful numerical features. The results indicate that gradient boosting models—particularly XGBoost—were able to leverage the sparse, high-dimensional feature space better than other architectures, extracting more nuanced patterns from term-weighted representations.

## 🧠 Engineering Notes
- **Experiment Tracking**: All experiments were tracked using **MLflow**, with a dedicated experiment name and local tracking URI. This ensured reproducibility and made it easy to compare performance across multiple model architectures and hyperparameter configurations.
- **Model Selection Strategy**: The best model for each architecture was chosen based on **validation F1 score**, as the dataset was perfectly balanced and both recall and precision were equally important. This avoided bias toward models that might overfit to either precision or recall.
- **Vectorization Approach**: Text preprocessing was performed using **TF-IDF** vectorization, producing a high-dimensional sparse feature matrix. This representation worked well with linear models and tree-based boosting methods, but performance differences emerged based on each model's handling of sparsity and feature interactions.
- **Hyperparameter Tuning**: Hyperparameters were optimized using a held-out validation set within MLflow runs. For tree-based models, tuning focused on learning rate, maximum depth, subsampling ratios, and regularization parameters.
- **Model Comparison**: Logistic Regression provided a strong, interpretable baseline with competitive performance. However, **XGBoost consistently outperformed** other architectures, suggesting its regularization scheme and handling of sparse features aligned best with the TF-IDF representation.

## 🗺️ Next Steps
- Refine the TF-IDF feature space by experimenting with higher n-gram ranges and character-level features to capture subtler textual patterns.  
- Conduct additional hyperparameter tuning of the XGBoost model to push F1 and AUROC performance further.  
- Explore ensemble methods that combine predictions from multiple architectures (e.g., LR + XGB) to boost robustness.  
- Use SHAP values to analyze feature importance and identify the most influential terms contributing to predictions.  
- Compare TF-IDF results against alternative text representations, such as pretrained word embeddings or contextual embeddings.  
- Package the best-performing model with its preprocessing pipeline for deployment, ensuring reproducibility and scalability.  

## 🔗 References
- Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft) – Ch. 1–3
- Bird, S., Klein, E., & Loper, E. *NLTK Book* – Ch. 1–3: https://www.nltk.org/book/
- scikit-learn TF–IDF: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- spaCy 101: https://spacy.io/usage/spacy-101


