# Load, summarize, and clean data
admission = read.csv("Admission_Predict_Ver1.1.csv")
names(admission)
summary(admission)
admission$Serial.No. = NULL
dim(admission)
na.omit(admission)
attach(admission)

# Create a qualitative variable from Chance.of.Admit
# 0 represents not admitted, 1 represents admitted
# Based on the median value for Chance.of.Admit
admission.median = median(admission$Chance.of.Admit)
admit01 = rep(0, nrow(admission))
admit01[admission$Chance.of.Admit > admission.median] = 1
sum(admit01)

# Split into training and testing sets
admission01 = data.frame(admission, admit01)
set.seed(1)
train = sample(500, 400)
admission.tr = admission01[train,]
admission.ts = admission01[-train,]
summary(admission.tr)
summary(admission.ts)

# Linear Regression on Chance.of.Admit in testing set
lm.admission.tr = lm(Chance.of.Admit~. - admit01, data = admission.tr)
lm.admission.tr
summary(lm.admission.tr)
# GRE.Score, TOEFL.score, LOR, CGPA, and Research are significant influencers

# Diagnostic Plots for Linear Regression
par(mfrow = c(2, 2))
plot(lm.admission.tr)
# The Residuals vs. Fitted Plot resembles a linear relationship, meaning the data is simulated in a way that meets the regression assumptions very well.
# The Normal Q-Q Plot demonstrates that the data generally follows a straight line well, with the exception of the leftmost points.  With this, we have concluded that the residuals are normally distributed.
# The Scale Location Plot shows that the residuals are in fact equally spread along the ranges of predictors.  Therefore, we can make the assumption of equal variance.
# The Residuals vs. Leverage Plot, we can barely see the Cook's Distance lines because all cases are well within them.  Therefore, we have no extreme cases of outliers to disregard.

# Influential Observations by the Hat Statistic
par(mfrow = c(1, 1))
hat.lm = lm.influence(lm.admission.tr)$hat
plot(hat.lm, pch = "$", main = "Influential Observations by the Hat Statistic")
abline(h = 2*(2+1)/nrow(admission.tr), col = "purple")
sum(hat.lm > 2*(2+1)/nrow(admission.tr))
# 253 training observations are potential outliers/influential observations

# Cook's Distance
par(mfrow = c(1, 1))
cd.lm = cooks.distance(lm.admission.tr)
par(mfrow = c(1,1))
plot(cd.lm, pch = "$", main = "Influential Observations by Cook's Distance")
abline(h = 4*mean(cd.lm), col="purple")
sum(cd.lm > 4*mean(cd.lm))
abline(h = 1, col = "red")
sum(cd.lm > 1)
# Based on Rule 1, 27 training observations are potential outliers/influential observations.
# Based on Rule 2, none of the observations are potential outliers.

# Producing histograms to observe distributions of variables
hist(admission.tr$GRE.Score, main = "Distribution of GRE Scores", xlab = "GRE Score")
hist(admission.tr$CGPA, main = "Distribution of College GPAs", xlab = "College GPAs")
hist(admission.tr$TOEFL.Score, main = "Distribution of TOEFL Scores", xlab = "TOEFL Scores")
hist(admission.tr$LOR, main = "Distribution of Letter of Recommendation Strengths", xlab = "Letter of Recommendations Strengths")
# GRE Score, College GPA, TOEFL Score, and Letter of Recommendation Strength appear to have seemingly even distributions

# Checking skewness
install.packages("e1071")
library(e1071)
skewness(admission.tr$GRE.Score)
skewness(admission.tr$CGPA)
skewness(admission.tr$TOEFL.Score)
skewness(admission.tr$LOR)
# GRE Score, College GPA, TOEFL Score, and Letter of Recommendation Strength are approximately symmetric

# Based on these tests, the linear model does a good job predicting the Chance.of.Admit.

# Confusion matrix
admission.prob = predict(lm.admission.tr, newdata = admission.ts, type = "response")
admission.pred = rep("No", nrow(admission.ts))
admission.pred[admission.prob > admission.median] = "Yes"
table(admission.pred, admission.ts$admit01)
# Confusion matrix demonstrates accuracy of the model based on 75% Chance.of.Admit.
((9 + 2) / 100)
# Here, the error rate is 11.0%

# Logistic Regression to predict admit01
lr.admissions = glm(admit01~.-Chance.of.Admit, data = admission.tr, family = binomial)
summary(lr.admissions)
# GRE.Score, CGPA, and Research are significant predictors in the Logistic Regression model.

lr.prob.tr = predict(lr.admissions, type = "response")
lr.pred.tr = rep(0,nrow(admission.tr))
lr.pred.tr[lr.prob.tr > 0.5] = 1
mean(lr.pred.tr != admission.tr$admit01)
# 12.0% error rate for the training set on LR

lr.prob.ts = predict(lr.admissions, admission.ts, type = "response")
lr.pred.ts = rep(0, nrow(admission.ts))
lr.pred.ts[lr.prob.ts > 0.5] = 1
mean(lr.pred.ts != admission.ts$admit01)
# 11.0% error rate for the test set on LR

# Linear Discriminant Analysis (LDA) Model to predict admit01
library(MASS)
lda.admissions = lda(admit01 ~ .-Chance.of.Admit, data = admission.tr)
lda.admissions
lda.pred.tr = predict(lda.admissions)
mean(lda.pred.tr$class != admission.tr$admit01)
# The training error rate is 12.25% on LDA

lda.pred.ts = predict(lda.admissions, newdata = admission.ts)
mean(lda.pred.ts$class != admission.ts$admit01)
# The test error rate is 11.0% on LDA

# Quadratic Discriminant Analysis (QDA) Model to predict admit01
qda.admissions = qda(admit01 ~ .-Chance.of.Admit, data = admission.tr)
qda.admissions
qda.pred.tr = predict(qda.admissions)
mean(qda.pred.tr$class != admission.tr$admit01)
# The training error rate is 12.5% on QDA

qda.pred.ts = predict(qda.admissions, newdata = admission.ts)
mean(qda.pred.ts$class != admission.ts$admit01)
# The test error rate is 12.0% on QDA

# Comparing model accuracies
# Linear Regression
AIC(lm.admission.tr) # Akaike information criterion
BIC(lm.admission.tr) # Bayesian information criterion

# Logistic Regression
AIC(lr.admissions) # Akaike information criterion
BIC(lr.admissions) # Bayesian information criterion

# The Linear Regression model demonstrates the best accuracy since it has the minimum values for these.  Thus, we will use this to make predictions.

# Remove insignificant predictors
lm.admission.predictions = lm(Chance.of.Admit ~ . - admit01 - SOP - University.Rating, data = admission01)
summary(lm.admission.predictions)

# Making predictions
predict(lm.admission.predictions, CGPA = c(7.0), interval = "confidence")[1:10,]

savehistory()
save.image()
