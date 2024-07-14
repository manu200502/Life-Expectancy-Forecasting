library(psych)
library(ggplot2)
library(DataExplorer)
library(lmtest) # autocorrelation
library(Metrics) # loss/cost function
library(caret)
library(dplyr)
library(car)
library(glmnet)
library(MASS)

# Load the data
data1 <- Life_Expectancy_Data
names(data1)
dim(data1)

# Perform EDA
# Summary and Missingness analysis
summary(data1)
is.na(data1)
plot_missing(data1)

# Understand distributions and correlations
plot_histogram(data1)
plot_density(data1)
plot_correlation(data1)

# Exclude 'Country', 'Continent', and 'Year' columns
x <- data1[, !(names(data1) %in% c("Country", "Continent", "Year"))]

# Manually encode the 'Status' column: 0 for Developing, 1 for Developed
x$Status <- ifelse(x$Status == "Developing", 0, 1)

# Convert Status to a numeric column
x$Status <- as.numeric(x$Status)

boxplot(x)

set.seed(1234)
mixed<-x[order(runif(2461)),]
training<-mixed[1:1723,]
testing<-mixed[1724:2461,]

# Check Assumptions Before Fitting the Model

# 1. Linearity
pairs.panels(training[, c("Life_expectancy", "Status", "Adult_Mortality", "infant_deaths", "Alcohol", "BMI", "under_five_deaths", "HIV.AIDS", "GDP", "thinness..1.19.years")])

# 2. Homoscedasticity
# Fit a preliminary model for assumption checks
fullmodel <- lm(Life_expectancy ~ ., data = training)
summary(fullmodel)

lm_step<-stepAIC(fullmodel,direction = "backward")

# Residual plot
plot(fullmodel, which = 1)

# 3. Multivariate Normality
# Q-Q plot
plot(fullmodel, which = 2)

# 4. Lack of Multicollinearity
# Calculate VIF
vif_values <- vif(fullmodel)
cat("VIF values:\n")
print(vif_values)

# 5. Outlier Check
# Cook's distance
plot(fullmodel, which = 4)

# Leverage plot
plot(fullmodel, which = 5)

#Multi Linear Regression Model
mlr1<- lm(Life_expectancy~Status+Adult_Mortality+Measles+Alcohol+BMI+HIV.AIDS,data=training)
summary(mlr1)

lm_step1=stepAIC(mlr1,direction='backward')

# Predict and evaluate on test data
fullmodel_pred <- predict(fullmodel, newdata = testing)
mlr_rdspend_pred <- predict(mlr1, newdata = testing)

# Calculate performance metrics
fullmodel_r2 <- summary(fullmodel)$r.squared
fullmodel_test_r2 <- cor(testing$Life_expectancy, fullmodel_pred)^2

mlr_rdspend_r2 <- summary(mlr1)$r.squared
mlr_rdspend_test_r2 <- cor(testing$Life_expectancy, mlr_rdspend_pred)^2

# Compare R-squared values
cat("Full Model - Train R2:", fullmodel_r2, "Test R2:", fullmodel_test_r2, "\n")

cat("mlr_Rdspend Model - Train R2:", mlr_rdspend_r2, "Test R2:", mlr_rdspend_test_r2, "\n")

# Define lambda sequence
lambda <- 10^seq(10, -2, length = 100)
print(lambda)

# Prepare data
x1 <- as.matrix(x[, !(names(x) %in% c("Life_expectancy", "percentage_expenditure","infant_deaths","under_five deaths", "GDP","thinness..1.19.years", "thinness.5.9.years"))])
y1 <- x$Life_expectancy

# Split the data into training and validation
set.seed(567)
part <- sample(2, nrow(x), replace = TRUE, prob = c(0.7, 0.3))
x_train <- x1[part == 1,]
x_cv <- x1[part == 2,]
y_train <- y1[part == 1]
y_cv <- y1[part == 2]

print(head(x_train))

# Perform Ridge regression
ridge_reg <- glmnet(x_train, y_train, alpha = 0, lambda = lambda)
summary(ridge_reg)

# Find the best lambda via cross-validation (Train to test and test to train) via K-fold
ridge_reg_cv <- cv.glmnet(x_train, y_train, alpha = 0)
best_lambda_ridge <- ridge_reg_cv$lambda.min
print(best_lambda_ridge)

# Predict on the validation set
ridge_pred <- predict(ridge_reg, s = best_lambda_ridge, newx = x_cv)

# Calculate mean squared error for Ridge
mse_ridge <- mean((y_cv - ridge_pred)^2)
print(paste("Ridge Regression Mean Squared Error:", mse_ridge))

# Perform Lasso regression
lasso_reg <- glmnet(x_train, y_train, alpha = 1, lambda = lambda)
summary(lasso_reg)

# Find the best lambda via cross-validation (Train to test and test to train) via K-fold
lasso_reg_cv <- cv.glmnet(x_train, y_train, alpha = 1)
best_lambda_lasso <- lasso_reg_cv$lambda.min
print(best_lambda_lasso)

# Predict on the validation set
lasso_pred <- predict(lasso_reg, s = best_lambda_lasso, newx = x_cv)

# Calculate mean squared error for Lasso
mse_lasso <- mean((y_cv - lasso_pred)^2)
print(paste("Lasso Regression Mean Squared Error:", mse_lasso))
