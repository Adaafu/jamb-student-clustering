import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('jamb_exam_results.csv')

keep_cols = [
    'Study_Hours_Per_Week',
    'Attendance_Rate',
    'Teacher_Quality',
    'Assignments_Completed',
    'School_Type',
    'School_Location',
    'Extra_Tutorials',
    'Socioeconomic_Status',
    'IT_Knowledge',
    'Parent_Education_Level',
    'Parent_Involvement'
]

df = df[keep_cols].copy()

numerical_cols = [
    'Study_Hours_Per_Week',
    'Attendance_Rate',
    'Teacher_Quality',
    'Assignments_Completed'
]

categorical_cols = [
    'School_Type',
    'School_Location',
    'Extra_Tutorials',
    'Socioeconomic_Status',
    'IT_Knowledge',
    'Parent_Education_Level',
    'Parent_Involvement'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ]), categorical_cols)
    ])

X = preprocessor.fit_transform(df)

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    n_init=30,
    max_iter=800,
    random_state=42
)

kmeans.fit(X)

df['Cluster'] = kmeans.labels_

print("\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())

print("\nNumerical averages per cluster:")
print(df.groupby('Cluster')[numerical_cols].mean().round(2))

print("\nKey categorical distributions per cluster:")
for cluster in sorted(df['Cluster'].unique()):
    print(f"\n=== Cluster {cluster}  ({len(df[df['Cluster']==cluster])} students) ===")
    for col in ['Socioeconomic_Status', 'Parent_Involvement', 'Parent_Education_Level', 'Extra_Tutorials']:
        if col in df.columns:
            vc = df[df['Cluster']==cluster][col].value_counts(normalize=True).mul(100).round(1)
            print(f"{col}:")
            print(vc.head(4))

joblib.dump(kmeans, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
