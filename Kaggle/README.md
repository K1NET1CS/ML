# KAGGLE COMPETITION MODEL DESCRIPTIONS
## Titanic_ModelEnsemble
**[Jupyter Notebook](./Titanic_ModelEnsemble.ipynb)**

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
   

   
   


