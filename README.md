
# SONAR Mine Prediction

This project aims to classify sonar signals as either mines or rocks using machine learning techniques. The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/mayurdalvi/sonar-mine-dataset).

## Table of Contents

1. [Installation and Dependencies](#installation-and-dependencies)
2. [Usage Instructions](#usage-instructions)
3. [Project Structure](#project-structure)
4. [Dataset Information](#dataset-information)
5. [Methodology](#methodology)
6. [Results and Evaluation](#results-and-evaluation)
7. [License](#license)


## Installation and Dependencies

To run this project, you'll need the following dependencies:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage Instructions

To use this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sonar-mine-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd sonar-mine-prediction
    ```
3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4. Open the `SONAR_Mine_prediction.ipynb` notebook and run the cells sequentially.

## Project Structure

The project directory contains the following files:

- `SONAR_Mine_prediction.ipynb`: Jupyter Notebook containing the code for the project.
- `README.md`: This README file.

## Dataset Information

The dataset used in this project is the Sonar Mines vs. Rocks dataset. The dataset contains 208 samples, each with 60 features representing sonar signal frequencies. Each sample is labeled as either a mine (M) or a rock (R).

**Features:**
- 60 numerical features representing the energy of the sonar signal at different frequencies.
- 1 target feature indicating whether the sample is a mine (M) or a rock (R).

## Methodology

The project follows these steps:

1. **Importing Libraries:**
    - Import necessary libraries such as pandas, numpy, scikit-learn, matplotlib, and seaborn.

2. **Loading the Dataset:**
    - Load the dataset from the provided CSV file.
    - Display basic information and statistics about the dataset.

3. **Exploratory Data Analysis (EDA):**
    - Visualize the distribution of the classes (mine vs. rock).
    - Generate correlation matrices and pair plots to understand relationships between features.

4. **Data Preprocessing:**
    - Handle missing values if any.
    - Encode the target variable (M/R) to numerical values.
    - Split the dataset into training and testing sets.

5. **Feature Engineering:**
    - Standardize or normalize the features as needed.

6. **Model Training:**
    - Train the Logistic regression model on Training dataset.

7. **Model Evaluation:**
    - Evaluate the model using training and testing accuracy score.

## Results and Evaluation
The model testing gave the following metrics:
- Accuracy on training data: 0.8342245989304813
- Accuracy on test data: 0.7619047619047619
  (which indicates that the model is a good fit for the given dataset.)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
