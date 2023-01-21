setwd("~/Desktop/rotation/ESNN/data")

######## method 1: mvsusie
#set.seed(2023)
N = 5000
P = 200
true_eff <- 5
X = matrix(sample(c(0, 1, 2), size=N*P, replace=T), nrow=N, ncol=P)
beta1 = beta2 = beta3 = beta4 = beta5 = beta6 = rep(0, P)
beta1[1:true_eff] = runif(true_eff)
beta2[1:true_eff] = runif(true_eff)
beta3[1:true_eff] = runif(true_eff)
beta4[1:true_eff] = runif(true_eff)
beta5[1:true_eff] = runif(true_eff)
beta6[1:true_eff] = runif(true_eff)
y1 = X %*% beta1 + rnorm(N)
y2 = X %*% beta2 + rnorm(N)
y3 = X %*% beta3 + rnorm(N)
y4 = X %*% beta4 + rnorm(N)
y5 = X %*% beta5 + rnorm(N)
y6 = X %*% beta6 + rnorm(N)

Y <- cbind(y1, y2, y3, y4, y5, y6)

## mvsusie test
#remotes::install_github("stephenslab/mvsusieR")
library(mvsusieR)

prior = create_mixture_prior(R=6)
fit = mvsusie(X,Y,prior_variance = prior)
unlist(fit$sets$cs)
##

write.table(X, file = "ESNN_sim_X.txt", row.names = F, col.names = F, quote = F)
write.table(Y, file = "ESNN_sim_Y.txt", row.names = F, col.names = F, quote = F)

####### test for unvariate susie
library(susieR)

x <- matrix(rnorm(5000,0, 1))
temp <- rnorm(5000, 0, 0.1)
y <- cos(x) * 0.8 + temp
model <- susie(x,y,L = 1)
model
plot(x,y)
lines(x, predict(model, x), col = 'red')

## q1: why susie can get results given non-linear transformation

####### 


######## method 2: VIMCO
library(mvtnorm)

set.seed(20132014)
n = 5000
p = 200
K = 8
X   = rmvnorm(n, mean=rep(0, p))
sigma.beta = rep(1, K)
bet = matrix(0, nrow = p, ncol = K)
lambda = 0.05
eta = rbinom(p, 1, lambda)
alpha = 1
gam = matrix(rbinom(p*K, 1, alpha), ncol=K)
for (k in 1:K){
  bet[, k] = rnorm(p, mean = 0, sd = sigma.beta[k]) * gam[,k] * eta 
}

sigma = diag(rep(1, K))
lp  = X %*% bet
sigma.e = diag(sqrt(diag(var(lp)))) %*% sigma %*% diag(sqrt(diag(var(lp))))
err = rmvnorm(n, rep(0, K), sigma.e)
Y   = lp + err

which(eta != 0)

prior = create_mixture_prior(R=8)
fit = mvsusie(X,Y,prior_variance = prior)
unlist(fit$sets$cs)

write.table(X, file = "ESNN_sim_X.txt", row.names = F, col.names = F, quote = F)
write.table(Y, file = "ESNN_sim_Y.txt", row.names = F, col.names = F, quote = F)

### non-linearity test: mvsusie method
N = 5000
P = 200
true_eff <- 5
X = matrix(sample(c(0, 1, 2), size=N*P, replace=T), nrow=N, ncol=P)
beta1 = beta2 = beta3 = beta4 = beta5 = beta6 = rep(0, P)
beta1[1:true_eff] = runif(true_eff)
beta2[1:true_eff] = runif(true_eff)
beta3[1:true_eff] = runif(true_eff)
beta4[1:true_eff] = runif(true_eff)
beta5[1:true_eff] = runif(true_eff)
beta6[1:true_eff] = runif(true_eff)
y1 = cos(X-1) %*% beta1 + rnorm(N, 0, 1)
y2 = -cos(X-1) %*% beta2 + rnorm(N, 0, 1)
y3 = cos(X-1) %*% beta3 + rnorm(N, 0, 1)
y4 = -cos(X-1) %*% beta4 + rnorm(N, 0, 1)
y5 = cos(X-1) %*% beta5 + rnorm(N, 0, 1)
y6 = -cos(X-1) %*% beta6 + rnorm(N, 0, 1)

Y <- cbind(y1, y2, y3, y4, y5, y6)

plot(X[,1],Y[,1])
plot(X[,1],Y[,2])
plot(X[,2],Y[,1])

prior = create_mixture_prior(R=6)
fit = mvsusie(X,Y,prior_variance = prior)
unlist(fit$sets$cs)

write.table(X, file = "ESNN_sim_X.txt", row.names = F, col.names = F, quote = F)
write.table(Y, file = "ESNN_sim_Y.txt", row.names = F, col.names = F, quote = F)

### non-linearity test: VIMCO method
n = 5000
p = 2
K = 3
X   = rmvnorm(n, mean=rep(0, p))
sigma.beta = rep(1, K)
bet = matrix(0, nrow = p, ncol = K)
lambda = 0.5
eta = rbinom(p, 1, lambda)
alpha = 1
gam = matrix(rbinom(p*K, 1, alpha), ncol=K)
for (k in 1:K){
  bet[, k] = rnorm(p, mean = 0, sd = sigma.beta[k]) * gam[,k] * eta 
}

sigma = diag(rep(1, K))
lp  = cos(X) %*% bet
sigma.e = diag(sqrt(diag(var(lp)))) %*% sigma %*% diag(sqrt(diag(var(lp)))) * 0.5
err = rmvnorm(n, rep(0, K), sigma.e)
Y   = lp + err

which(eta != 0)

plot(X[,1], Y[,1])
plot(X[,1], Y[,2])
plot(X[,1], Y[,3])
plot(X[,2], Y[,1])
plot(X[,2], Y[,2])
plot(X[,2], Y[,3])

prior = create_mixture_prior(R=3)
fit = mvsusie(X,Y,prior_variance = prior)
unlist(fit$sets$cs)

write.table(X, file = "ESNN_sim_X.txt", row.names = F, col.names = F, quote = F)
write.table(Y, file = "ESNN_sim_Y.txt", row.names = F, col.names = F, quote = F)
