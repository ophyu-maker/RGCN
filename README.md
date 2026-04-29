# RGCN
Using RGCN model to explore drugs related side effects 

This project uses a Relational Graph Convolutional Network (RGCN) to predict potential drug side effects based on drug–indication–side effect relationships.

## Project Overview
- Built a heterogeneous graph with drug, indication, and side effect nodes
- Trained an RGCN model for link prediction / side effect prediction
- Evaluated performance using ROC-AUC, PR-AUC, F1, precision, and recall
- Built a Streamlit demo app for interactive prediction

## Tools
Python, PyTorch, DGL/PyTorch Geometric, pandas, scikit-learn, Streamlit

## Limitations
This project is for educational and exploratory ML purposes only and is not intended for medical decision-making.

## Team & Contributions

This was a group project completed as part of an Applied Machine Learning course.

My contributions included:
- Data filtering & feature engineering
- Building the RGCN model & workflow
- Creating training progress and evaluation visualizations
- Final presentation & ppt

## Data

This project uses the publicly available SIDER dataset, which contains marketed drugs, known side effects, and MedDRA terminology. The dataset includes relationships between drugs, clinical indications, and reported adverse drug reactions.

During exploratory data analysis, some side effects and clinical indications appeared very frequently. This indicates that one drug may be associated with multiple side effects and multiple clinical indications. Similarly, the same side effect may appear across multiple drugs. Because of this relational structure, the data is well-suited for a graph-based machine learning approach.

## Methodology

###0. Model Development Pipeline

![Project Pipeline](images/pipeline_1.jpeg)

### 1. Data Cleaning and Filtering

The raw SIDER files were first merged using a common `stitch_id` to connect drug names, clinical indications, and side effects. After merging, duplicate rows and missing values were removed.

The text fields were standardized by converting drug names, indication names, side effect names, and MedDRA types to lowercase. The dataset was then filtered to keep only `PT` records, which represent MedDRA Preferred Terms. This helped make the side effect labels more consistent and meaningful.

Rare side effects appearing fewer than 20 times were removed. This step helped reduce extreme label sparsity because very rare side effects may not provide enough examples for the model to learn reliable patterns. After cleaning, the final columns used for modeling were:

- `drug_name`
- `indication_name`
- `side_effect_name`

The cleaned sample was saved and used as the input for graph construction and model training.

### 2. Feature Engineering and Graph Construction

The cleaned dataset was transformed into graph-ready format. First, all unique drug names, indication names, and side effect names were mapped to numeric IDs. Reverse mappings were also created so that model outputs could later be converted from numeric IDs back into readable names.

Each record was represented as a triple:

```text
drug + indication + side effect



## Final Presentation

[View Final Presentation PDF](Presentation/SideEffectsproject_GithubUpload.pdf)
