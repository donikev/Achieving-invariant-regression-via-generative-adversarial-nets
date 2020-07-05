### Example where residuals do not all have the same distribution

##Preamble
library("InvariantCausalPrediction")

## Model Setup
set.seed(2)
# Observations per environment
J <- 1000
# Number of environments
e <- 10
# Number of total observations
n <- e*J
# Number of predictor variables
p <- 1000
# Label Experiment
ExpInd <- c(rep(1,J),rep(2,(J*(e-1))))
#Label the environment
Env <- rep(1:e,each = J)
# Colors
cols <- c(rgb(0.5,0.5,1,0.3),rgb(0.5,0.5,0.5,0.05),rgb(1,0.5,0.5,0.3))
# Labels
labs <- c("e = 1",paste("e = 2, ... ,", e),paste("e =",e))

## Data generation
# Generate observational data
X <- matrix(rnorm(n*p),nrow=n)
# Generate perturbed data
X[ExpInd==2,] <- sweep(X[ExpInd==2,],2,rnorm(p),FUN = "*")
X[Env == (e-1),4] <- X[Env == (e-1),4]+0.5*rnorm(J)
# Generate base environment
X[,2] <- 2*X[,1] + X[,4] + 0.1 * rnorm(n)
X[,3] <- 1/5*X[,1] + X[,2] + 0.1 * rnorm(n)
Y <- 3*X[,3] + X[,4] + 0.1 * rnorm(n)
X[,5] <- Y + 0.05*rnorm(n)
#Shift in only one environment, last one, other shif possibilities: 0.1*rexp(J) or simply +0.5
X[Env == e,5] <- Y[Env == e] + 0.5*rnorm(J)

##Estimation
# Estimate coefficient on all the data across all environments
fit <- lm(Y ~ X)
summary(fit)
coef <- fit$coefficients

## Plotting
par(mfrow = c(1,2), oma = c(0, 0, 0, 0)) # 0, 0, 2, 0
# Get residuals for all environments + plot
resid <- fit$residuals
MSE <- sum(resid^2)/n
MSE_last <- sum((resid[Env == e])^2)/J
plot(resid[Env == 1], col=rgb(0.5,0.5,1,0.3), pch=20, xaxt='n', ylim = c(-0.7,0.7), main = "linear regression", ylab = expression(epsilon^e), xlab = paste("MSE =",round(MSE,5), "\n MSE_",e," = ",round(MSE_last,5)))
for (i in 2:(e-1)) {
  points(resid[Env==i], col=rgb(0.5,0.5,0.5,0.05), pch=20)
}
points(resid[Env == e], col=rgb(1,0.5,0.5,0.3), pch=20)
legend("topright",labs,col = cols,pch = 20)
# Plot with true residuals
true_resid <- Y-(3*X[,3] + X[,4])
true_MSE <- sum(true_resid^2)/n
true_MSE_last <- sum((true_resid[Env == e])^2)/J
plot(true_resid[1:J], col=rgb(0.5,0.5,1,0.3), pch=20, xaxt='n', ylim = c(-0.7,0.7), main = "causal model", ylab = expression(epsilon^e), xlab = paste("MSE =",round(true_MSE,5), "\n MSE_", e, " = ", round(true_MSE_last,5)))
for (i in 2:(e-1)) {
  points(true_resid[((i-1)*J+1):(i*J)], col=rgb(0.5,0.5,0.5,0.05), pch=20)
}
points(true_resid[(((e-1)*J)+1):n], col=rgb(1,0.5,0.5,0.3), pch=20)
legend("topright",labs,col = cols,pch = 20)
#mtext(paste("Residuals"), outer = T, cex = 1.5)
# QQ plot
qqnorm(resid/sqrt(var(resid)), ylim = c(-5,5), pch = 20, col=adjustcolor(4,alpha.f=0.5),main = "Std residuals from linear regression")
abline(0,1,lty = 1,lwd = 2)
qqnorm(true_resid/sqrt(var(true_resid)),  ylim = c(-5,5), pch = 20, col=adjustcolor(4,alpha.f=0.5),main = "Std residuals under causal model")
abline(0,1,lty = 1,lwd = 2)
par(mfrow = c(1,1))

## ICP
# Test with ICP function: note that X[,5] in this case is rejected as causal variable
icp <- ICP(X,Y,ExpInd)
print(icp)
plot(icp)

