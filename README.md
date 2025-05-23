# Property Assessment

## Case Overview

The **Cook County Assessor’s Office (CCAO)** is committed to delivering fair and transparent valuations for all residential and commercial properties. Prior to 2018, the valuation models of properties were inaccurate and lacked transparency. As part of the CCAO data science team, we aim to develop an accurate machine learning model to predict the values of over 1.8 million properties in Cook County, the second most populous county in the U.S. 

## Methodology

To accurately predict residential property values in Cook County, we use a structured modeling approach grounded in interpretability and predictive performance. Our process begins with **simple linear regression**, a transparent and interpretable method that establishes a baseline by estimating the relationship between property characteristics and their sale prices. Dealing with missing values in numeric columns, we filled them with their respective mode, rather than mean or median, to maintain consistency with categorical treatment. As for categorical values, we also used mode and these variables were converted to factor types to prepare for modeling.

To prevent overfitting and improve generalization to unseen properties, we apply **10-fold cross-validation**. We chose this method since our dataset is quite large. Although Leave-one-out cross-validation(LOOCV) can cover more training samples, it is computationally intensive, and predictions could be more variable. Moreover, including more observations, 10-fold cross-validation performs better than validation sets, which only estimate half of the observations. In our model, the target variable is “sale_price”, char_site , char_cnst_qlty,  econ_tax_rate, econ_midincome, meta_deed_type, meta_certified_est_bldg, meta_certified_est_land, and char_beds are our predictors in the first phase, selected manually by reading the description of the code book.

As model complexity grows with more variables, we incorporate **Lasso regression** to resolve our problem. Lasso helps by shrinking less important coefficients toward zero, effectively performing variable selection and simplifying the model. This makes the model more interpretable while maintaining strong predictive accuracy.
To further refine the set of predictors, we explore forward stepwise selection, which begins with a minimal model and iteratively adds variables that improve model performance the most. This method allows us to build a parsimonious model that balances accuracy and complexity.

We used the **forward stepwise** procedure that evaluates each model through **10-fold cross-validation**. First, we start with a very simple model that includes no variables, just an intercept to represent the average sale price. At each step, we evaluate all remaining predictor variables by adding them one at a time to the existing model and fitting a linear regression for each. For each model, we calculate how well it performs using an **80/20 split—80% of the data for training and 20% for validation**. Next, we chose the variable that could reduce the most in MSE and added it to the model. 

This stepwise process will continue until adding more variables no longer helps the improvement of the model. We chose this approach to avoid overfitting and improve the model’s ability to generalize to new data. As more variables were added, the MSE dropped significantly at first. However, the improvement becomes smaller after each step. The best model was chosen at the point where MSE was lowest, which is Step 6, with the following variables: sale_price ~ meta_certified_est_land + econ_midincome + char_beds + meta_deed_type + econ_tax_rate + char_site. This model gains a good balance between accuracy and model simplicity.

## Conclusion

The assessed_value value file contains exactly 10,000 rows, which corresponds to a property from the predict_property_data file. It includes two columns that show the following: pid – a unique property identification number ranging from 1 to 10,000 and the assessed_value – the predicted market value, shown as a numerical variable, rounded to 2 decimal places. 
Based on the results of the Lasso regression, we observed the following summary statistics for the assessed property values: The **minimum assessed value is $1,000**, the **maximum is $7,900,986**, and the **mean observed value is approximately $323,248**. The **median assessed value is $252,163**, and this means that half of the properties are valued below this point. 

The **first quartile (Q1) is $161,867** and the **third quartile (Q3) is $380,622**, and this shows that **50% of the assessed properties fall between these two values**. These values show a right-skewed distribution, which would make sense since the results describe the real estate markets, where a small number of high-value properties significantly increases the mean. 
To put this into perspective, we can compare the predictions with the distribution of sales prices in the training dataset (historic_property_data), and we observe the following summary statistics for the sale prices: The minimum sale price was around $850, the maximum sale price was around $12,700,000, and the mean observed sale price is approximately $296,804. The median sale price is $222,000, and this shows that half of the sale price is below this point. 

Overall, this comparison suggests that the Lasso regression model produced a distribution of assessed values that is loosely consistent with the historical data, while also including safeguards, like setting a $1,000 minimum, to improve the spread of the predictions.

