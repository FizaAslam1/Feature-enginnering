# 🚀 Feature Engineering with Scikit-Learn (Titanic Case Study)

This project demonstrates **feature engineering and model building** using the Titanic dataset with and without **scikit-learn Pipelines**.  
It covers preprocessing (imputation, encoding), model training, pipeline creation, model saving (pickling), and prediction workflows.

---



---

## 📌 Topics Covered

- **Intro to Feature Engineering**
- **Scikit-Learn Pipelines**
- **Example through Titanic Dataset**
- **Train/Test Split**
- **Applying Imputation**
- **Pickling the Model**
- **Pipeline Strategy**
- **Creating Pipelines (`Pipeline` vs `make_pipeline`)**
- **Exploring Pipelines**
- **Cross Validation with Pipelines**

---

## ⚡ Key Learnings

1. **Without Pipelines**  
   - Manual handling of missing values with `SimpleImputer`.  
   - Encoding categorical features (`Sex`, `Embarked`) with `OneHotEncoder`.  
   - Training classifier and saving components separately (`clf.pkl`, encoders).  

2. **With Pipelines**  
   - Combined preprocessing + model into a single workflow.  
   - Much cleaner and prevents **data leakage**.  
   - Saved complete pipeline as `pipe.pkl`.  

3. **Model Persistence**  
   - Pickled both individual components and the full pipeline.  
   - Demonstrated prediction workflows with both approaches.  

---

## 🔮 Usage

### 1️⃣ Train & Save Pipeline
Run the **`withpipeline.ipynb`** notebook to train the model and save the pipeline as `pipe.pkl`.

### 2️⃣ Predict Using Pipeline
Load the pipeline and directly predict on raw data:

```python
import pickle

# Load pipeline
pipe = pickle.load(open("module/pipe.pkl", "rb"))

# Predict on new data
sample = [[22, 1, 0, 7.25, 'male', 'S']]  # Example input
print(pipe.predict(sample))
import pickle

clf = pickle.load(open("module/clf.pkl", "rb"))
ohe_sex = pickle.load(open("module/ohe_sex.pkl", "rb"))
ohe_embarked = pickle.load(open("module/ohe_embarked.pkl", "rb"))

# Preprocess manually before passing to classifier

