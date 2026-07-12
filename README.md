# 🍷 Wine Quality Prediction with Automated MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ElasticNet-F7931E?style=for-the-badge&logo=scikitlearn)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI/CD-2088FF?style=for-the-badge&logo=githubactions)
![Conda](https://img.shields.io/badge/Environment-Conda-44A833?style=for-the-badge&logo=anaconda)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

### Automated Machine Learning Experiment Tracking using MLflow, GitHub Actions, and Scikit-Learn

A reproducible MLOps project that automates machine learning training, experiment tracking, and Continuous Integration (CI) using **MLflow Projects**, **GitHub Actions**, and **ElasticNet Regression** for predicting wine quality based on physicochemical characteristics.

</div>

---

# 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Background](#-background)
- [Business Understanding](#-business-understanding)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Machine Learning Lifecycle](#-machine-learning-lifecycle)
- [MLOps Architecture](#-mlops-architecture)
- [Dataset Overview](#-dataset-overview)
- [Repository Structure](#-repository-structure)
- [Technology Stack](#-technology-stack)
- [Workflow Pipeline](#-workflow-pipeline)
- [Model Development](#-model-development)
- [Experiment Tracking](#-experiment-tracking)
- [Continuous Integration](#-continuous-integration)
- [Project Components](#-project-components)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Generated Artifacts](#-generated-artifacts)
- [Reproducibility](#-reproducibility)
- [Future Work](#-future-work)
- [Author](#-author)
- [License](#-license)

---

# 🚀 Project Overview

Machine Learning projects often emphasize model development while overlooking one of the most critical aspects of production-ready AI systems: **reproducibility and automation**.

A machine learning model that performs well locally may become difficult to reproduce, evaluate, maintain, or deploy without a standardized workflow.

This repository demonstrates how modern MLOps practices can automate the machine learning lifecycle by integrating:

- MLflow Projects
- MLflow Tracking
- GitHub Actions
- Scikit-Learn
- Conda Environment Management

Instead of focusing solely on predictive performance, this project highlights how experiments can be executed consistently, automatically tracked, and reproduced across different environments.

The project trains an ElasticNet Regression model to predict wine quality using physicochemical laboratory measurements while automatically logging experiment metadata, evaluation metrics, and trained models.

---

# 🌍 Background

Quality assessment is one of the most important stages in wine production.

Traditionally, wine quality is determined through sensory evaluation conducted by professional wine tasters (sommeliers). Although highly valuable, this process introduces several practical limitations.

### Manual Quality Assessment Challenges

- Subjective evaluation
- Human inconsistency
- High operational cost
- Time-consuming inspection
- Limited scalability

As production volume increases, manually evaluating every production batch becomes increasingly inefficient.

Modern wineries routinely collect laboratory measurements describing the physicochemical properties of each wine sample.

Examples include:

- Alcohol concentration
- Acidity
- Sulphates
- Density
- Chlorides
- Residual sugar
- pH

These numerical measurements create an opportunity to estimate wine quality using supervised machine learning before manual inspection.

---

# 💼 Business Understanding

Predicting wine quality using machine learning offers several practical benefits throughout the production pipeline.

Instead of relying exclusively on manual tasting, wineries can use predictive analytics as an early quality estimation system.

Potential applications include:

- Production quality monitoring
- Batch prioritization
- Laboratory decision support
- Manufacturing consistency
- Production optimization
- Cost reduction

Machine learning is not intended to replace expert tasters but rather to provide an objective decision-support tool capable of processing thousands of samples efficiently.

---

# ❓ Problem Statement

Wine manufacturers collect extensive physicochemical measurements for every production batch.

However, these measurements alone do not directly indicate product quality.

The primary challenge addressed in this repository is:

> **How can physicochemical laboratory measurements be transformed into an automated machine learning workflow capable of predicting wine quality while ensuring reproducibility, experiment traceability, and continuous integration?**

Unlike conventional machine learning projects, this repository also addresses operational challenges including:

- Experiment reproducibility
- Automated training
- Model tracking
- Version control
- Continuous Integration (CI)

---

# 🎯 Project Objectives

This project aims to:

- Develop a supervised regression model for wine quality prediction.
- Automate model training using MLflow Projects.
- Record every experiment using MLflow Tracking.
- Store trained model artifacts automatically.
- Evaluate regression performance using multiple metrics.
- Execute experiments reproducibly across environments.
- Automate training through GitHub Actions.
- Demonstrate an end-to-end MLOps workflow suitable for machine learning experimentation.

---

# 🔄 Machine Learning Lifecycle

The repository follows a simplified machine learning lifecycle.

```text
                  Wine Dataset
                        │
                        ▼
                Data Loading
                        │
                        ▼
              Train/Test Split
                        │
                        ▼
             ElasticNet Training
                        │
                        ▼
             Model Evaluation
                        │
                        ▼
           MLflow Experiment Log
                        │
                        ▼
            Save Model Artifact
                        │
                        ▼
             GitHub Actions CI
                        │
                        ▼
         Automated Experiment Run
```

Unlike traditional machine learning notebooks, every stage of this workflow can be reproduced automatically.

---

# ⚙️ MLOps Architecture

The overall project architecture integrates machine learning with automation tools.

```text
              Git Push / Pull Request
                       │
                       ▼
             GitHub Actions Workflow
                       │
                       ▼
               Setup Python 3.12
                       │
                       ▼
             Install Dependencies
                       │
                       ▼
              Execute MLflow Project
                       │
                       ▼
                   train.py
                       │
                       ▼
             Train ElasticNet Model
                       │
                       ▼
          Evaluate Regression Metrics
                       │
                       ▼
              MLflow Tracking Server
                       │
                       ▼
              Store Model Artifact
                       │
                       ▼
             Generate mlruns Folder
                       │
                       ▼
              Commit Back to GitHub
```

This automated architecture minimizes manual intervention while preserving complete experiment history.

---

# 📊 Dataset Overview

This project uses the **Wine Quality Dataset**, a widely recognized benchmark dataset for supervised machine learning.

The dataset contains physicochemical laboratory measurements collected from Portuguese "Vinho Verde" wines.

Each observation represents a single wine sample.

The prediction target is:

```text
quality
```

which represents the sensory quality score assigned by wine experts.

---

# 📑 Input Features

Each sample contains several numerical physicochemical measurements.

Examples include:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

These variables become the input features (`X`) used during model training.

The response variable (`y`) is:

```text
quality
```

---

# 🧠 Why ElasticNet?

This project uses **ElasticNet Regression**, a regularized linear regression algorithm that combines the strengths of both Ridge and Lasso regression.

ElasticNet introduces two complementary regularization techniques:

### L1 Regularization (Lasso)

L1 regularization encourages sparsity by shrinking less informative coefficients toward zero.

Benefits include:

- Feature selection
- Reduced model complexity
- Improved interpretability

---

### L2 Regularization (Ridge)

L2 regularization distributes coefficient weights more smoothly across correlated variables.

Benefits include:

- Reduced overfitting
- Better handling of multicollinearity
- Improved model stability

---

### Why ElasticNet Instead of Ordinary Linear Regression?

Physicochemical variables often exhibit substantial correlation.

Examples include relationships between:

- Density and Alcohol
- Acidity and pH
- Sulphates and Sulfur Dioxide

ElasticNet balances feature selection and coefficient regularization, making it a suitable baseline regression algorithm for datasets with correlated predictors.

---

# 🧪 Model Training Workflow

The training process implemented in `train.py` follows these sequential steps:

1. Load the wine quality dataset.
2. Split the dataset into training and testing subsets.
3. Separate predictor variables (`X`) and target variable (`y`).
4. Initialize the ElasticNet Regression model.
5. Train the model using the training dataset.
6. Predict wine quality for the testing dataset.
7. Evaluate prediction performance.
8. Log parameters, metrics, and trained model using MLflow.
9. Store experiment artifacts automatically.

This workflow is executed locally or automatically through GitHub Actions.

---

# 📈 Model Evaluation

Model performance is evaluated using three standard regression metrics provided by Scikit-Learn.

These metrics offer complementary perspectives on prediction accuracy and enable objective comparison across multiple experiment runs.

---

## Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

```text
RMSE = √( Σ(y - ŷ)² / n )
```

Characteristics:

- Penalizes larger prediction errors.
- Sensitive to outliers.
- Frequently used as the primary regression metric.

Lower RMSE indicates better predictive performance.

---

## Mean Absolute Error (MAE)

MAE computes the average absolute difference between predicted and actual values.

```text
MAE = Σ|y - ŷ| / n
```

Characteristics:

- Easy to interpret.
- Less sensitive to outliers than RMSE.
- Represents the average prediction error.

Lower MAE indicates more accurate predictions.

---

## R² Score (Coefficient of Determination)

The R² Score measures how much of the variance in the target variable is explained by the model.

```text
R² = 1 - (RSS / TSS)
```

Characteristics:

- Values closer to 1 indicate stronger explanatory power.
- Useful for comparing regression models.
- Indicates how well the model captures relationships within the data.

Higher R² indicates better model fit.

---

# 📦 MLflow Experiment Tracking

One of the key objectives of this repository is to demonstrate reproducible machine learning experiments using **MLflow Tracking**.

Instead of recording experiment results manually, every training run is automatically logged.

The following information is stored for each experiment:

- Hyperparameters
- Evaluation metrics
- Trained model artifacts
- Run metadata
- Timestamp
- Experiment history

This allows every experiment to be reproduced and compared in a structured manner.

---

## Logged Parameters

The project records the primary ElasticNet hyperparameters:

```python
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
```

These parameters determine the amount and type of regularization applied during training.

---

## Logged Metrics

Three evaluation metrics are automatically recorded.

```python
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("r2", r2)
```

Every execution creates a new experiment run, making it possible to compare different parameter configurations over time.

---

## Logged Model

After training, the model is stored as an MLflow artifact.

```python
mlflow.sklearn.log_model(
    lr,
    "model"
)
```

Saving the trained model ensures that it can later be loaded for inference, validation, or deployment without retraining.

---

# 📁 MLflow Artifact Structure

After training, MLflow automatically generates an experiment directory.

A simplified structure is shown below.

```text
mlruns/
│
├── experiment_id/
│   ├── run_id/
│   │   ├── artifacts/
│   │   │   └── model/
│   │   ├── metrics/
│   │   ├── params/
│   │   ├── tags/
│   │   └── meta.yaml
```

This directory preserves the complete history of each experiment, including parameters, metrics, metadata, and trained model artifacts.

---

# 🚀 MLflow Projects

Rather than executing the training script directly, this repository leverages **MLflow Projects** to standardize experiment execution.

The project configuration is defined in the `MLproject` file.

```text
MLproject
```

This configuration specifies:

- Entry point
- Input parameters
- Environment definition
- Training command

As a result, every experiment follows the same execution procedure regardless of the operating system or development environment.

---

## MLproject Entry Point

The primary entry point is defined as:

```yaml
entry_points:
  main:
```

The training command executes:

```bash
python train.py {alpha} {l1_ratio} {dataset}
```

This allows experiments to be launched simply by running:

```bash
mlflow run Project --env-manager=local
```

without manually specifying each training step.

---

# 🌱 Conda Environment

Project dependencies are managed using a dedicated Conda environment.

The configuration is stored in:

```text
conda.yaml
```

The environment includes:

- Python
- NumPy
- Pandas
- Scikit-Learn
- MLflow

Environment management ensures that experiments remain reproducible across different machines and operating systems.

---

# ⚡ Continuous Integration (CI)

Continuous Integration is implemented using **GitHub Actions**.

Every push or pull request targeting the `master` branch automatically triggers the workflow.

The workflow eliminates repetitive manual tasks and guarantees that the training pipeline remains executable after each code update.

---

# 🔄 CI Pipeline

The GitHub Actions workflow performs the following sequence automatically.

```text
Push / Pull Request
        │
        ▼
Checkout Repository
        │
        ▼
Setup Python 3.12
        │
        ▼
Install Dependencies
        │
        ▼
Run MLflow Project
        │
        ▼
Execute train.py
        │
        ▼
Generate mlruns/
        │
        ▼
Commit Experiment
        │
        ▼
Push Results
```

This automation provides a reproducible machine learning workflow with minimal manual intervention.

---

# 🤖 GitHub Actions Workflow

The CI configuration performs several automated tasks.

## Repository Checkout

The latest repository version is downloaded into the runner environment.

```yaml
actions/checkout@v3
```

---

## Python Environment

A clean Python environment is created.

```yaml
Python 3.12.7
```

This guarantees consistency across all workflow executions.

---

## Dependency Installation

Before training begins, the required packages are installed automatically.

The workflow upgrades `pip` and installs MLflow.

```bash
pip install mlflow
```

Any dependencies defined within the MLflow Project are subsequently resolved through the Conda environment.

---

## Automated Training

Instead of executing Python scripts manually, GitHub Actions launches the MLflow Project.

```bash
mlflow run Project --env-manager=local
```

This command automatically:

- Creates the execution environment
- Reads project configuration
- Executes the defined entry point
- Runs model training
- Logs experiment outputs

---

## Experiment Persistence

After execution, the generated `mlruns` directory is added to the repository.

```bash
git add mlruns/
```

The workflow then commits and pushes the updated experiment history back to the remote repository.

This approach enables experiment tracking to evolve alongside the source code.

---

# 📂 Repository Structure

The repository is organized to clearly separate source code, project configuration, datasets, experiment artifacts, and automation workflows.

```text
Wine-Quality-MLOps/
│
├── .github/
│   └── workflows/
│       └── main.yaml                 # GitHub Actions CI workflow
│
├── Project/
│   ├── MLproject                     # MLflow Project configuration
│   ├── conda.yaml                    # Reproducible environment specification
│   ├── train.py                      # Model training script
│   └── wine-quality.csv              # Wine Quality dataset
│
├── mlruns/                           # MLflow experiment tracking (generated)
│
├── README.md
│
└── requirements.txt (optional)
```

This modular organization follows common practices in machine learning engineering projects, making the repository easier to maintain, extend, and reproduce.

---

# 📄 Project Components

## `train.py`

The central script responsible for model training and experiment logging.

Main responsibilities:

- Load dataset
- Split training and testing data
- Initialize ElasticNet Regression
- Train the regression model
- Generate predictions
- Evaluate model performance
- Log metrics to MLflow
- Save trained model artifact

---

## `MLproject`

The `MLproject` file defines how experiments should be executed.

It specifies:

- Project entry point
- Required parameters
- Environment definition
- Execution command

Using MLflow Projects standardizes experiment execution and reduces configuration differences across development environments.

---

## `conda.yaml`

The Conda configuration file defines all dependencies required to reproduce the project.

Benefits include:

- Consistent environments
- Dependency isolation
- Reproducibility
- Simplified onboarding for collaborators

---

## `wine-quality.csv`

The input dataset used for supervised learning.

The dataset contains physicochemical measurements collected from Portuguese *Vinho Verde* wines and serves as the foundation for regression modeling.

---

## `main.yaml`

Defines the GitHub Actions workflow responsible for Continuous Integration (CI).

Every eligible Git push automatically triggers the complete machine learning pipeline without requiring manual execution.

---

## `mlruns/`

This directory is automatically generated by MLflow.

It stores:

- Experiment metadata
- Logged parameters
- Evaluation metrics
- Trained model artifacts
- Run history

The `mlruns` directory enables reproducibility and experiment comparison across multiple executions.

---

# 🛠 Technology Stack

This project integrates multiple technologies commonly used in modern machine learning engineering.

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.12 |
| Machine Learning | Scikit-Learn |
| Regression Algorithm | ElasticNet |
| Experiment Tracking | MLflow |
| Project Management | MLflow Projects |
| Environment Management | Conda |
| Continuous Integration | GitHub Actions |
| Version Control | Git |
| Repository Hosting | GitHub |

Each component contributes to a reproducible and automated machine learning workflow.

---

# 📚 Python Libraries

The project utilizes several widely adopted Python libraries.

| Library | Purpose |
|----------|---------|
| Pandas | Data loading and manipulation |
| NumPy | Numerical computation |
| Scikit-Learn | Regression modeling and evaluation |
| MLflow | Experiment tracking and model logging |

These libraries provide the core functionality required for data processing, model development, evaluation, and experiment management.

---

# ⚙️ Environment Variables

The GitHub Actions workflow defines several environment variables to simplify project configuration.

| Variable | Description |
|----------|-------------|
| `CSV_URL` | Dataset location |
| `TARGET_VAR` | Prediction target (`quality`) |
| `ALPHA` | ElasticNet alpha parameter |
| `L1_RATIO` | ElasticNet l1_ratio parameter |
| `RANDOM_STATE` | Random seed for reproducibility |

Centralizing these values improves readability and simplifies future configuration changes.

---

# ▶️ Installation

Clone the repository.

```bash
git clone https://github.com/<username>/<repository>.git
```

Navigate to the project directory.

```bash
cd <repository>
```

Create the Conda environment.

```bash
conda env create -f Project/conda.yaml
```

Activate the environment.

```bash
conda activate wine-quality-prediction
```

Alternatively, install the required packages manually.

```bash
pip install pandas numpy scikit-learn mlflow
```

---

# 🚀 Running the Project

## Option 1 — Execute as an MLflow Project (Recommended)

The recommended approach is to execute the project using MLflow Projects.

```bash
mlflow run Project --env-manager=local
```

This command automatically:

- Creates the execution environment
- Reads the `MLproject` configuration
- Passes the required parameters
- Executes the training script
- Logs experiment information
- Stores model artifacts

This method provides the highest level of reproducibility.

---

## Option 2 — Execute the Training Script Directly

The training script can also be executed manually.

```bash
python Project/train.py 0.5 0.5 Project/wine-quality.csv
```

Where:

- `0.5` = ElasticNet alpha
- `0.5` = ElasticNet l1_ratio
- `Project/wine-quality.csv` = dataset path

This option is useful for quick experimentation or debugging.

---

# 📊 Viewing Experiment Results

After training completes, experiment results are available through the MLflow user interface.

Start the MLflow Tracking UI.

```bash
mlflow ui
```

By default, the interface is accessible at:

```text
http://127.0.0.1:5000
```

The dashboard provides an organized overview of:

- Experiment runs
- Logged parameters
- Evaluation metrics
- Model artifacts
- Run comparison
- Training history

This interface simplifies experiment management and supports systematic model development.

---

# 📁 Generated Artifacts

Each successful training run produces several outputs.

| Artifact | Description |
|----------|-------------|
| Trained Model | Serialized ElasticNet model |
| Parameters | Alpha and L1 Ratio |
| Metrics | RMSE, MAE, R² |
| Run Metadata | Experiment information |
| Model Directory | Stored within `mlruns/` |

These artifacts allow experiments to be reproduced, compared, and reused without retraining from scratch.

---

# 🔁 Reproducibility

Reproducibility is a fundamental principle of MLOps.

This project promotes reproducible machine learning through:

- MLflow Projects
- Conda environment specification
- Version-controlled source code
- Automated CI execution
- Logged parameters
- Logged metrics
- Persisted model artifacts

Because every experiment is fully documented, the same workflow can be executed repeatedly with consistent results, provided the dataset and configuration remain unchanged.

---

# 💼 Business Impact

Although this project is developed as a machine learning and MLOps demonstration, the workflow reflects challenges commonly encountered in real-world production environments.

By integrating automated model training, experiment tracking, and continuous integration, organizations can significantly improve the efficiency and reliability of their machine learning lifecycle.

Potential business benefits include:

- Faster experimentation cycles
- Consistent model development process
- Reduced manual intervention
- Improved model reproducibility
- Better collaboration among data teams
- Centralized experiment documentation
- Easier model comparison
- Reduced operational risk

Instead of treating machine learning as a one-time activity, this repository demonstrates how it can become a repeatable engineering workflow.

---

# 📊 Potential Use Cases

The architecture presented in this repository can be adapted to numerous supervised learning applications beyond wine quality prediction.

Examples include:

### Manufacturing

- Product quality prediction
- Defect detection
- Production monitoring
- Process optimization

### Agriculture

- Crop quality prediction
- Soil analysis
- Yield estimation

### Healthcare

- Disease risk prediction
- Patient outcome estimation
- Clinical decision support

### Finance

- Credit scoring
- Loan default prediction
- Risk assessment

### Retail

- Customer satisfaction prediction
- Product quality assessment
- Demand forecasting

The workflow remains applicable because the underlying MLOps principles are domain-independent.

---

# 🔍 Project Highlights

This repository demonstrates several practical machine learning engineering concepts.

## Machine Learning

- Supervised Learning
- Regression
- ElasticNet Regression
- Train/Test Split
- Model Evaluation

---

## MLOps

- MLflow Projects
- MLflow Tracking
- Model Artifact Logging
- Experiment Management
- Reproducible Environments

---

## DevOps

- GitHub Actions
- Continuous Integration
- Automated Training
- Source Control
- Workflow Automation

---

## Software Engineering

- Modular project structure
- Environment management
- Version control
- Configuration management
- Reproducible execution

---

# 📖 What I Learned

This project provided hands-on experience in building a reproducible machine learning workflow rather than focusing solely on predictive performance.

Key takeaways include:

- Structuring machine learning projects for maintainability.
- Managing dependencies using Conda environments.
- Standardizing experiment execution with MLflow Projects.
- Tracking parameters, metrics, and model artifacts using MLflow.
- Automating machine learning workflows with GitHub Actions.
- Integrating software engineering practices into machine learning development.

These skills are fundamental for transitioning from experimental notebooks to production-oriented machine learning systems.

---

# 🎯 Future Improvements

Although the current implementation provides a complete automated training workflow, several enhancements can further improve the project.

## Model Development

Future experiments may compare ElasticNet against additional regression algorithms such as:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost
- LightGBM
- CatBoost
- Support Vector Regression

This would enable systematic benchmarking across multiple models.

---

## Hyperparameter Optimization

Current hyperparameters are manually specified.

Future work could integrate automated optimization techniques, including:

- Grid Search
- Random Search
- Bayesian Optimization
- Optuna

Automated tuning would improve model performance while reducing manual experimentation.

---

## Model Registry

Currently, trained models are stored as MLflow artifacts.

Future iterations could leverage the **MLflow Model Registry** to manage:

- Model versioning
- Stage transitions (Staging, Production, Archived)
- Deployment-ready model governance

This would strengthen the project's production-readiness.

---

## Automated Testing

The repository currently validates that the training workflow executes successfully.

Future improvements may include:

- Unit tests
- Integration tests
- Data validation
- Model validation
- Performance regression tests

These additions would further improve reliability and maintainability.

---

## Deployment

Once the model has been thoroughly evaluated, it can be deployed through various serving solutions, such as:

- FastAPI
- Flask
- Streamlit
- Gradio
- Docker
- Kubernetes

This would enable real-time prediction services for downstream applications.

---

# 🤝 Contributing

Contributions are welcome and greatly appreciated.

Potential contributions include:

- Additional regression algorithms
- Improved documentation
- Code optimization
- Workflow enhancements
- Model comparison
- Hyperparameter optimization
- Better visualizations
- Extended experiment tracking

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

Constructive feedback, bug reports, and feature suggestions are always welcome.

---

# 📚 References

1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling Wine Preferences by Data Mining from Physicochemical Properties*. Decision Support Systems, 47(4), 547–553.

2. Scikit-Learn Documentation  
   https://scikit-learn.org/

3. MLflow Documentation  
   https://mlflow.org/

4. GitHub Actions Documentation  
   https://docs.github.com/actions

---

# 👨‍💻 Author

## Ahmad Latif

Machine Learning • Data Science • MLOps

I am passionate about building reliable, reproducible, and scalable machine learning systems by combining data science with software engineering and automation practices.

My areas of interest include:

- Machine Learning
- MLOps
- Data Science
- Data Engineering
- Artificial Intelligence
- Business Intelligence
- Cloud Computing

Feel free to connect or explore my other projects.

**GitHub**

```text
https://github.com/ahmdltf
```

**LinkedIn**

```text
https://www.linkedin.com/in/ahmad-latif-aulia-rahman/
```

---

# 📄 License

This project is released under the **MIT License**.

You are free to use, modify, distribute, and extend this project in accordance with the terms of the license.

---

# 🙏 Acknowledgements

This project builds upon the open-source ecosystem that supports reproducible machine learning research and engineering.

Special thanks to:

- The Python Community
- Pandas Development Team
- NumPy Development Team
- Scikit-Learn Contributors
- MLflow Contributors
- GitHub Actions Team
- The UCI Machine Learning Repository
- Google Colab and Jupyter Community

Their tools and resources have enabled the development of accessible, reproducible, and collaborative machine learning workflows.

---

# 📌 Project Status

> **Current Status:** Completed (MLOps Training & Experiment Tracking Pipeline)

This repository demonstrates an end-to-end automated machine learning experimentation workflow, integrating model training, experiment tracking, environment management, and continuous integration.

It serves as a practical example of how MLOps principles can be applied to build reproducible and maintainable machine learning projects.

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a star!

**Reproducible experiments are the foundation of trustworthy Machine Learning.**

Thank you for visiting this repository.

</div>
