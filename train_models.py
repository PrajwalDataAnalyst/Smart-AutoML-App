from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
import numpy as np

def train_and_evaluate_models(df, features, target, task_type):
    # Split features and target
    X = df[features]
    y = df[target]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipelines for numeric and categorical data
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Define models based on task type
    if task_type == 'regression':
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Support Vector Regressor": SVR(),
            "KNN Regressor": KNeighborsRegressor(),
            "XGBoost Regressor": XGBRegressor(verbosity=0),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=42)
        }
        scoring = 'r2'
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Support Vector Classifier": SVC(),
            "KNN Classifier": KNeighborsClassifier(),
            "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            "AdaBoost Classifier": AdaBoostClassifier(random_state=42)
        }
        scoring = 'accuracy'

    best_score = -np.inf
    best_model = None
    best_model_name = ""
    train_score = None

    # Train, cross-validate and select best model
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring)
            avg_cv_score = np.mean(cv_scores)

            pipeline.fit(X_train, y_train)
            current_train_score = pipeline.score(X_train, y_train)

            print(f"📊 {name}: CV {scoring} = {avg_cv_score:.4f}, Train {scoring} = {current_train_score:.4f}")

            if avg_cv_score > best_score:
                best_score = avg_cv_score
                best_model = pipeline
                best_model_name = name
                train_score = current_train_score
        except Exception as e:
            print(f"⚠️ Skipping {name} due to error: {e}")

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📈 Training {scoring} Score: {train_score:.4f}")
    print(f"📉 Validation CV {scoring} Score: {best_score:.4f}\n")

    return best_model, best_model_name, train_score, best_score
