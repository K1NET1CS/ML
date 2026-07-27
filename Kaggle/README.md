# KAGGLE COMPETITION MODEL DESCRIPTIONS
## Titanic_ModelEnsemble0
**[Jupyter Notebook](./Titanic_ModelEnsemble0.ipynb)**

This Project implements an **Ensembling Strategy** combining multiple Decison Tree based models

Model Performance - Accuracy Score: 0.77511

**Crucial Steps** :
1. Dropped Outlier training examples (10 examples) from dataset using Tukey Method : Dropped only those with more than half parametrs having outlier values. Embarked has only 2 missing values - filled by mode ("S")

2. SNS plotted Correlation Matrix to check correlation of parametrs with Survival - Only Fare has a significant Fare among considered numeric parameters but negative or small positive correlated features might also have subsets which are highly correlated but the overall feature does not appear well correlated and should be analysed
 
3. Numerical Data Analysis
   
   * Parch catplot reveals smaller families (slightly low for alone) tend to have higher survival chances
      
     *Created a new Family_Size parameter*

   * SibSp catplot reverals >=3 SibSp examples have lower survival chances
  
     *Hence did not drop after create a new Family_Size parameter*
   
   * Age kdeplot reveals higher chances of survival than not in very low age (~<=5) groups
     
     *Imputed carefully in accordance with high (magnitude) correlated parameters such as SibSp,Parch,Pclass instead of dropping*
   
   * Fare displot reveals very skewed distribution
     
     *Fixed using Log Scale*

4. Categorical Data Analysis
   * Sex barplot reveals much higher survival chance of females than males
   
   * Pclass catplot reveals survival chances in order 1st class > 2nd class > 3rd class
   
   * Extracted Title from Name and further classified into (integer) classes based on frequency

   * Created new parameter Family_Size = 1 + SibSp + Parch and further classified into bins on due to frequency trends

   * Extracted Cabin Class/Labelled X for numeric Cabin numbers

   * Categorised Ticket by Ticket Prefix

**This results in a 67 feature dataset**

5. Ensemble Model
  Trained an ensemble model of SupportVectorClassifier(SVC),DecisionTreeClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier

Hyperparameter tuning for each model was done using GridSearchCV on a range of parameter values

Finally, trained a voting classifier using these estimators

---

## Titanic_ModelEnsemble1
**[Jupyter Notebook](./Titanic_ModelEnsemble1.ipynb)**

This project implements an **Advanced Feature Engineering & Ensembling Strategy** combining regularized linear models and gradient-boosted decision trees tuned with Optuna.

Model Performance - Accuracy Score: *[Insert your final score here, e.g., 0.79425]*

**Crucial Steps** :

1. **Multicollinearity & Feature Dimension Control**
   * Identified high linear correlation between $\text{SibSp}$ and $\text{Parch}$ with $\text{FamilySize}$.
   * Evaluated $\text{FamilySize}$ non-monotonic ($U$-shaped) trends: solo travelers and large families ($\text{FamilySize} > 4$) had low survival rates, whereas small families ($2 \le \text{FamilySize} \le 4$) had significantly higher survival rates.
   * Created a binary $\text{IsAlone}$ flag and dropped redundant $\text{SibSp}$ and $\text{Parch}$ columns to keep feature dimensions optimal ($\approx 24$ total features) while avoiding the curse of dimensionality.

2. **Skewness & Missing Data Handling**
   * **$\text{Fare}$**: Applied log-transformation using $f(x) = \log(1 + x)$ (`np.log1p`) to resolve severe right-skewness and eliminate extreme outliers before imputation.
   * **$\text{Cabin}$**: Dropped raw high-cardinality strings and retained either deck-level dummy encodings or a binary $\text{HasCabin}$ flag (since 1st-class top-deck passengers had drastically higher survival rates).

3. **Advanced MICE Imputation for $\text{Age}$**
   * Leveraged passenger $\text{Title}$ (extracted from $\text{Name}$) along with $\text{Pclass}$, $\text{Sex}$, $\text{SibSp}$, $\text{Parch}$, and log-scaled $\text{Fare}$.
   * Temporarily ordinal-encoded $\text{Title}$ to run **MICE (`IterativeImputer`)** with an `ExtraTreesRegressor` estimator, preserving the distinct age distribution bump for infants/children without flattening it into a simple global median.
   * Constructed an explicit binary indicator $\text{IsChild} = \mathbb{I}(\text{Age} < 12)$ following imputation.

4. **Categorical Encoding**
   * Replaced temporary ordinal mappings with **One-Hot Encoding** for $\text{Title}$, $\text{Embarked}$, and $\text{Deck}$ features to avoid imposing false numerical rankings on linear estimators.

5. **Optuna Hyperparameter Tuning & Stacking Ensemble**
   * Used **Stratified 5-Fold Cross-Validation** to preserve target class ratios across evaluation splits.
   * Automated hyperparameter optimization using **Optuna** for three distinct model families: **Logistic Regression**, **Random Forest**, and **XGBoost Classifier**.
   * Constructed a **Soft Voting Classifier** and a **Stacking Classifier** (using $\text{LogisticRegression}$ as a meta-learner) to combine out-of-fold probability predictions from all tuned base models.
