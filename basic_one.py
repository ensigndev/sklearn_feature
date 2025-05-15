from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# Step 1: Load dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 2: Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k=4)  # top 2 features
X_new = selector.fit_transform(X, y)

# Step 3: Check selected features
selected_features = selector.get_support(indices=True)
print("Selected feature names:", X.columns[selected_features].tolist())
