# CounterACT: Counterfactual ACTion Rule Mining Approach

## Experiment Reproduction

To reproduce the experiments and obtain the results presented in our study, please follow these steps:

1. **Run Experiment Script**: Execute the `runexp.py` script to perform the experiments. This process will generate and store all the resulting data in the `data` directory in CSV format.

2. **Research Questions (RQs) Analysis**: For each research question, execute the respective `rq1.py`, `rq2.py`, or `rq3.py` script. These scripts will compute and present the results for each RQ.

3. **Precision and Recall Calculation**: Use the `precision_recall.py` script to calculate precision and recall metrics at both the release and commit levels. Set the parameter `commit=True` for commit level calculations.

## Results Directory

Sample results for all three RQs are available in the `results` directory for reference and validation of the experiment outcomes.

## Understanding CounterACT

CounterACT is designed to propose maintainable and achievable action plans for each file within a project. These plans are derived from a counterfactual action rule mining approach, tailored to guide practitioners in improving project outcomes.

### Example Usage of CounterACT

#### Step 1: Select Actionable Features

- **Feature Identification**: The Hedge function identifies features with significant variance in historical data. Users can select the top-M features to consider for planning, where M is user-defined. In our case study, we selected the top 5 features.
- **Data Balancing with SMOTE**: After identifying the actionable features, we use the Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset, ensuring that the model is not biased towards the majority class.

#### Step 2: Data Preparation with Entropy Discretizer

- **Entropy Discretization**: Before mining action rules, we apply an entropy discretizer to convert continuous features into discrete intervals. This step helps in capturing the non-linear relationships between features and improves the interpretability of the action rules.

#### Step 3: Mine Action Rules

- **Action Rule Mining**: This process generates action rules based on the data between the previous and current release of the project, allowing the creation of plans grounded in historical records. CounterACT focuses on actionable features identified in Step 1.

#### Step 4: Generate and Select Plans

- **Plan Recommendation and Selection**: CounterACT recommends plans using the action rules mined from the historical data. The plan selection process is based on the effectiveness and success of similar actions in historical data, ensuring that the proposed plans are not only actionable but also proven to be successful in past implementations.
