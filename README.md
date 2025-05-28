Predicting the Sales of Products of a Retail Chain using Supervised Learning

Project Background and Objectives
The project, titled "Predicting the Sales of Products of a Retail Chain using Supervised Learning," focuses on forecasting future sales for a large Indian retail chain with stores in Maharashtra, Telangana, and Kerala. The primary objective is to improve inventory management, reduce operational costs, and enhance strategic planning by leveraging accurate sales predictions. This is crucial for minimizing stockouts, reducing excess inventory, and supporting data-driven decision-making in the retail sector.

Methodology and Implementation
The project adopted the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, which structured the workflow into several key phases:

Business Understanding
The initial phase identified the business need for precise sales forecasting. Accurate predictions are essential for optimizing inventory levels, reducing costs associated with overstocking or understocking, and enabling better strategic planning. The retail chain stocks products across categories such as Fast Moving Consumer Goods (FMCG), eatables/perishables, and others, making sales prediction a complex but vital task.

Data Understanding and Preparation
The project utilized four datasets to build the predictive model:
Train Data: Included columns such as date, product_identifier, department_identifier, category_of_product, outlet, state, and sales (the target variable, representing the number of sales for a product).
Test Data: Contained similar features (id, date, product_identifier, department_identifier, category_of_product, outlet, state) but lacked the sales column, used for final predictions.
Product Prices: Provided details like outlet, product_identifier, week_id, and sell_price, indicating the price at which a product was sold in a specific outlet and week.
Date to Week ID Map: Mapped date to week_id, facilitating temporal analysis.

Key features analyzed included:
date: Observation date for sales data.
product_identifier: Unique identifier for each product.
department_identifier: ID for a specific department within a store.
category_of_product: Product category (e.g., FMCG, eatables/perishables, others).
outlet: Store ID.
state: State name (Maharashtra, Telangana, Kerala).
sales: Number of sales (target for training).
week_id: Unique identifier for a week.
sell_price: Selling price of a product in a specific outlet and week.

The preprocessing steps:
Data Merging: Datasets were merged into a single dataframe using common keys (date, product_identifier, week_id, outlet) to create a comprehensive training set, ensuring all relevant information was combined for analysis.
Exploratory Data Analysis (EDA): Involved datatype conversion, feature extraction from dates (e.g., extracting the month from the date column), and correlation analysis to identify relationships between variables. For instance, high correlation was observed between product_identifier and department_identifier, which was noted but both retained for their individual importance.
Data Cleaning and Transformation: The date column was converted to datetime format, and the month was extracted as a new feature, replacing the original date column. Features were ensured to be in appropriate data types (categorical and numerical). No missing values were found in the dataset, simplifying the cleaning process.
Feature Selection: Irrelevant or redundant features were dropped based on correlation analysis. The final features for modeling included product_identifier, department_identifier, category_of_product, outlet, state, week_id, sell_price, and month, with sales as the target variable.
Data Splitting: The dataset was split into training (70%) and testing (30%) sets to ensure robust model evaluation and prevent overfitting.

Modelling
Several supervised learning models were implemented to predict sales, reflecting a comprehensive approach to model selection:
Linear Regression
Logistic Regression
Decision Tree Regressor
Random Forest Regressor
K-Nearest Neighbors (KNN) Regressor
Gaussian Naive Bayes
Each model was trained on the training set and evaluated on the testing set. The Random Forest Regressor emerged as the most effective model due to its lowest Root Mean Squared Error (RMSE), attributed to its ability to handle non-linear relationships and its robustness against overfitting, making it suitable for real-world applications.

Evaluation
The models were evaluated using the following metrics:
Root Mean Squared Error (RMSE): Measures the square root of the average squared differences between predicted and actual values, emphasizing larger errors.
Mean Absolute Error (MAE): Measures the average absolute differences between predicted and actual values, providing a linear score of error.
R² Score: Indicates the proportion of variance in the dependent variable that is predictable from the independent variables, with higher values indicating better fit.
The performance comparison showed:
Random Forest Regressor had the lowest RMSE, making it the preferred choice due to its scalability, stability, and suitability for real-world use.
KNN also showed a competitive RMSE but was less preferred compared to Random Forest due to its computational intensity and sensitivity to data scaling.
Other models, such as Linear Regression, Decision Tree, Logistic Regression, and Gaussian Naive Bayes, had higher RMSE values, with Gaussian Naive Bayes performing significantly worse, likely due to its assumption of feature independence, which may not hold for sales data.

Final Model and Prediction
The Random Forest Regressor was selected as the final model and applied to the test dataset for sales prediction. The test data underwent the same preprocessing steps (merging, date conversion, feature encoding, etc.) to ensure consistency. Predicted sales values were rounded and converted to integers for clarity and practical use, then added back to the test dataset for final output.
Key Technologies Used
The project was executed using Python, leveraging several key libraries:
Pandas: For data manipulation and analysis, particularly for merging datasets and handling dataframes.
NumPy: For numerical computations and array operations.
Matplotlib and Seaborn: For data visualization, used in EDA to plot correlations and distributions.
Scikit-learn: For machine learning tasks, including model implementation (e.g., Random Forest, KNN), data preprocessing (e.g., StandardScaler for numerical features), and evaluation metrics (e.g., RMSE, MAE, R²).
The environment for execution was Google Colab, providing a cloud-based platform for coding, running, and sharing the notebook, which facilitated collaboration and access to computational resources.
Results and Implications
The project successfully developed a sales prediction model, with the Random Forest Regressor demonstrating the best performance. This model can be integrated into the retail chain's inventory management system to forecast sales accurately, reducing costs associated with overstocking or stockouts, and supporting strategic planning. The use of multiple models allowed for a comparative analysis, ensuring the chosen model was robust and reliable.
This detailed analysis ensures all aspects of the project code are covered, providing a comprehensive understanding for stakeholders and aligning with the style of the provided project overview.

