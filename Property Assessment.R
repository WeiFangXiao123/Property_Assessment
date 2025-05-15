# Load packages
library(tidyverse)
library(dplyr)
library(broom)
library(readr)
library(caret)
library(magrittr)
library(glmnet)


# Read the data
historic <- read.csv("historic_property_data.csv", na.strings = c("", "NA"))
predict <- read.csv("predict_property_data.csv", na.strings = c("", "NA"))

# Linear regression and 10-fold cross-validation
# Mode function
calculate_mode <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}

# (1) Fill missing values - Numeric columns
numeric_columns <- sapply(historic, is.numeric)
for (col in names(historic)[numeric_columns]) {
  historic[[col]][is.na(historic[[col]])] <- calculate_mode(historic[[col]])
}

# (2) Fill missing values - Categorical columns
categorical_columns <- sapply(historic, function(col) is.character(col) || is.factor(col))
for (col in names(historic)[categorical_columns]) {
  mode_value <- calculate_mode(historic[[col]])
  historic[[col]][is.na(historic[[col]])] <- mode_value
  historic[[col]] <- as.factor(historic[[col]])
}

print(colSums(is.na(historic)))
write.csv(historic, "processed_historic_property_data.csv", row.names = FALSE)

# 10-Fold Cross-Validation
set.seed(123)
train_control <- trainControl(method = "cv", number = 10)

model <- train(
  sale_price ~ char_site + char_cnst_qlty + econ_tax_rate + econ_midincome + meta_deed_type + meta_certified_est_bldg + meta_certified_est_land + char_beds,
  data = historic,
  method = "lm",
  trControl = train_control
)
print(model)


#Lasso Regression
# Load 1st Dataset 
getwd()
dir.create("~/Final_Project_2_Clean")
setwd("~/Final_Project_2_Clean")
historic <- read.csv("historic_property_data.csv", na.strings = c("", "NA"))
predict <- read.csv("predict_property_data.csv", na.strings = c("", "NA"))
head(historic)

# Preparing the Data 
historic <- historic[!is.na(historic$sale_price), ]

vars_to_use <- c("sale_price", "char_site", "char_cnst_qlty", "econ_tax_rate",
                 "econ_midincome", "meta_deed_type", "meta_certified_est_bldg",
                 "meta_certified_est_land", "char_beds")

historic_model <- historic[, vars_to_use]
historic_model <- na.omit(historic_model)

categorical_cols <- sapply(historic_model, is.character)
historic_model[categorical_cols] <- lapply(historic_model[categorical_cols], as.factor)

x <- model.matrix(sale_price ~ ., data = historic_model)[, -1]
y <- historic_model$sale_price

# Fit Lasso Model Using CV
set.seed(123)
lasso_cv <- cv.glmnet(x, y, alpha = 1, nfolds = 10)
plot(lasso_cv)
best_lambda <-lasso_cv$lambda.min
lambda_1se <- lasso_cv$lambda.1se

cat("Best lambda (min):", best_lambda, "\n")
cat("Lambda (1SE rule):", lambda_1se, "\n")

# Extract Coefficients 
lasso_model <- glmnet(x, y, alpha=1, lambda = best_lambda)
coef(lasso_model)

# Predict Using Property Data 
head(predict)

calculate_mode <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}

for (col in names(predict)) {
  if (is.numeric(predict[[col]])) {
    predict[[col]][is.na(predict[[col]])] <- calculate_mode(predict[[col]])
  } else {
    predict[[col]][is.na(predict[[col]])] <- calculate_mode(predict[[col]])
    predict[[col]] <- as.factor(predict[[col]])
  }
}
# Match dummy variables to training data 
x_pred <- model.matrix(~., data = predict)[, -1]
missing_cols <- setdiff(colnames(x), colnames(x_pred))
for (col in missing_cols) {
  x_pred <- cbind(x_pred, setNames(data.frame(0), col))
}
x_pred <- x_pred[, colnames(x)]

# Predict 
predictions <- predict(lasso_model, newx = x_pred)

# Results 
output <- data.frame(pid = predict$pid, sale_price = as.vector(predictions))
output$assessed_value <- ifelse(output$sale_price < 1000, 1000, output$sale_price)
output$assessed_value <- round(output$assessed_value, 2)
output <- output[, c("pid", "assessed_value")]
write.csv(output, "assessed_value.csv", row.names = FALSE)

# View summary statistics
summary(output$assessed_value)



#Forward Stepwise
# Step 1: Read Data & Impute Missing Values

# Define a custom mode function
calculate_mode <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}

# Read Dataset
historic <- read.csv("historic_property_data.csv", na.strings = c("", "NA"))

# Impute missing numeric values using the mode
for (col in names(historic)[sapply(historic, is.numeric)]) {
  historic[[col]][is.na(historic[[col]])] <- calculate_mode(historic[[col]])
}

# Impute missing categorical values using the mode and convert to factor
for (col in names(historic)[sapply(historic, function(x) is.character(x) || is.factor(x))]) {
  mode_val <- calculate_mode(historic[[col]])
  historic[[col]][is.na(historic[[col]])] <- mode_val
  historic[[col]] <- as.factor(historic[[col]])
}

# Define variables
vars <- c("char_site", "char_cnst_qlty", "econ_tax_rate", "econ_midincome",
          "meta_deed_type", "meta_certified_est_land", "char_beds")
outcome <- "sale_price"


# Filter only relevant columns for model training
clean_data <- historic[, c(outcome, vars)]
for (col in names(clean_data)) {
  if (is.character(clean_data[[col]])) {
    clean_data[[col]] <- as.factor(clean_data[[col]])
  }
}
clean_data <- clean_data[!is.na(clean_data$sale_price), ]

# Step 2: Define Cross-Validated MSE Function

# GLM model and calculates 10-fold CV MSE.
cv_fun <- function(formula, data) {
  set.seed(123)
  n <- nrow(data)
  idx <- sample(1:n, size = round(0.8 * n))
  train_data <- data[idx, ]
  valid_data <- data[-idx, ]
  
  tryCatch({
    model <- glm(formula, data = train_data)
    pred <- predict(model, newdata = valid_data)
    actual <- valid_data[[outcome]]
    return(mean((actual - pred)^2))
  }, error = function(e) {
    message("⚠️ Model failed (val set): ", deparse(formula))
    return(Inf)
  })
}
# Step 3: Forward Stepwise Regression

# To store formulas at each step
formulas <- list()

# To store corresponding MSEs
mse_list <- c()

# Null Model
formulas[[1]] <- as.formula(paste(outcome, "~ 1"))
mse_list[1] <- cv_fun(formulas[[1]], data = clean_data)

# Initialize variable
used_vars <- c()
available_vars <- vars

# Forward Selection Loop
for (step in 1:length(vars)) {
  candidate_formulas <- list()
  candidate_mse <- c()
  
  for (var in available_vars) {
    formula_text <- paste(outcome, "~", paste(c(used_vars, var), collapse = " + "))
    f <- as.formula(formula_text)
    candidate_formulas[[var]] <- f
    candidate_mse[var] <- cv_fun(f, data = clean_data)
    cat(paste0("Try: ", formula_text, " → MSE = ", round(candidate_mse[var], 2), "\n"))
  }
  
  # Select the variable that gives the lowest MSE this round
  best_var <- names(which.min(candidate_mse))
  best_formula <- candidate_formulas[[best_var]]
  best_mse <- candidate_mse[best_var]
  
  # Store results from this step
  formulas[[step + 1]] <- best_formula
  mse_list[step + 1] <- best_mse
  used_vars <- c(used_vars, best_var)
  available_vars <- setdiff(available_vars, best_var)
  
  # Report best result at current step
  cat(paste0("✅ Step ", step, ": +", best_var, " → MSE = ", round(best_mse, 2), "\n\n"))}

# Step 4: Output Best Model and MSE
best_step <- which.min(mse_list)
cat("The Best model is Step", best_step - 1, ":\n")
print(formulas[[best_step]])
cat("Best MSE = ", round(mse_list[best_step], 2), "\n")
