################################################################################################
## Bivariate distribution of NARCCAP ECP2 (NCEP driven) temperature and precipitation
################################################################################################

load("/Users/wkleiber/Documents/NARCCAP/ECP2_ncep.save")

attach(ECP2_ncep)

library(MASS)
library(fields)

tokeep1 <- seq(1,dim(lon)[1],by=3)
tokeep2 <- seq(1,dim(lat)[2],by=3)

tas <- tas[tokeep1,tokeep2,]
pr <- pr[tokeep1,tokeep2,]
lon <- lon[tokeep1,tokeep2]
lat <- lat[tokeep1,tokeep2]
mo <- months
yr <- years
nyr <- length(unique(yr))

detach(ECP2_ncep)
rm(tokeep1,tokeep2,ECP2_ncep)

################################################################################################
## More setup -- average DJF
################################################################################################

## Average temperature and precipitation values within DJF
uyr <- unique(yr)
tas.a <- array(dim=c(dim(lat),length(uyr)))
for(i in 1:nyr){
  tas.a[,,i] <- apply(tas[,,yr==uyr[i]][,,c(12,1,2)],c(1,2),mean)
}
pr.a <- array(dim=c(dim(lat),length(uyr)))
for(i in 1:nyr){
  pr.a[,,i] <- apply(pr[,,yr==uyr[i]][,,c(12,1,2)],c(1,2),mean)
}
pr.a <- pr.a^(1/3) # take cube root after averaging DJF

## Take out effect of latitude in DJF
tas.c <- array(lm(c(tas.a)~c(rep(lat,nyr)))$resid,dim=dim(tas.a))
pr.c <- array(lm(c(pr.a)~c(rep(lat,nyr)))$resid,dim=dim(pr.a))

## Center variables by spatially varying month specific mean
tas.c <- tas.c - array(apply(tas.c,c(1,2),mean),dim=dim(tas.c))
pr.c <- pr.c - array(apply(pr.c,c(1,2),mean),dim=dim(pr.c))

rm(i,mo,pr,pr.a,tas,tas.a,yr)

n <- prod(dim(lat)) # number of locations 

loc <- cbind(c(lon),c(lat))

p <- 2 # number of processes

dist.mat <- rdist.earth(loc,miles=F)

## Put samples in p-variate observation matrix
obsmat <- array(dim=c(p,n,nyr))
obsmat[1,,] <- tas.c
obsmat[2,,] <- pr.c

################################################################################################
## Do estimation of sample matrix
################################################################################################

SigmaEst  <- matrix(nc=p*n,nr=p*n,0)
# note SigmaEst[1,1] = t(loc1), SigmaEst[2,2] = p(loc1), blocked by location, not variable

for(time in 1:nyr){
  obsvec <- as.vector(obsmat[,,time])

  ## Calculate empirical cross-covariance matrix
  Sigma <- outer(obsvec,obsvec)
  SigmaEst <- SigmaEst + Sigma
}
SigmaEst <- SigmaEst/nyr
rm(time,obsvec)

################################################################################################
## Use CV to find best smoothing parameter for beta function (Gaussian kernel)
################################################################################################

LAMBDA <- seq(20,40,by=2)
CV <- rep(0,length(LAMBDA))

K <- function(loc.index,distances,l){
  exp(-((distances[loc.index,-loc.index]/l)^2) / 2)
}
K2 <- function(loc.index,distances,l){
  exp(-(distances[loc.index,-loc.index]/l)^2)
}

W <- function(loc.index,distances,lambda){
  K(loc.index,distances,lambda) * K(loc.index,distances,lambda) /
    sqrt(sum(K2(loc.index,distances,lambda)) * sum(K2(loc.index,distances,lambda)))
}

A <- matrix(nr=n,nc=n-1)

for(i in 1:length(LAMBDA)){
  for(s in 1:n){
    A[s,] <- W(s,dist.mat,LAMBDA[i])
  }
  for(s in 1:n){
    for(time in 1:nyr){
      CV[i] <- CV[i] + sum((A[s,] %*% c(obsmat[1,-s,time]*obsmat[2,-s,time]) -
                            obsmat[1,s,time]*obsmat[2,s,time])^2)
    }
  }
  print(CV[i])
}

################################################################################################
## Find smoothed cross-covariance matrix with R code
################################################################################################

library(Matrix)

SigmaBeta <- matrix(nc=p*n,nr=p*n,0)
# SigmaBeta is blocked with first nxn block being (z_1,z_1), the upper right block is
# (z_1,z_2), so there are four large blocks, not n small 2x2 blocks

lambda <- 800

K <- function(k1,l){
  exp(-((dist.mat[k1,]/l)^2) / 2)
}
K2 <- function(k1,l){
  exp(-(dist.mat[k1,]/l)^2)
}

temp.denom <- NULL
for(s in 1:n){
  temp.denom[s] <- sum(K2(s,lambda))^(1/2)
}
denom <- outer(temp.denom,temp.denom)
rm(temp.denom)

for(time in 1:nyr){
  M <- array(dim=c(n,n,p)) # holds K(s_n1 - s_n2)^1/2 * Z_p(s_n2)
  for(s in 1:n){
    for(i in 1:p){
      M[s,,i] <- K(s,lambda)*obsmat[i,,time]
    }
  }
  MtM <- array(dim=c(n,n,p^2))
  for(i in 1:p){
    for(j in i:p){
      MtM[,,i+j] <- M[,,i] %*% t(M[,,j])
    }
  }
  ## Calculate kernel smoothed cross-covariance matrix
  Sigma <- matrix(nc=p*n,nr=p*n)
  for(s in 1:n){
    for(t in s:n){
      for(i in 1:p){
        for(j in i:p){
          Sigma[s+n*(i-1),t+n*(j-1)] <- MtM[s,t,i+j]/denom[s,t]
        }
      }
    }
  }

  SigmaBeta <- SigmaBeta + Sigma
  
  print(time)
}
SigmaBeta <- SigmaBeta/nyr
rm(M,MtM)

SigmaBeta[1:n,1:n] <- as.matrix(forceSymmetric(SigmaBeta[1:n,1:n]))
SigmaBeta[1:n,(n+1):(p*n)] <- as.matrix(forceSymmetric(SigmaBeta[1:n,(n+1):(p*n)]))
SigmaBeta[(n+1):(p*n),(n+1):(p*n)] <-
  as.matrix(forceSymmetric(SigmaBeta[(n+1):(p*n),(n+1):(p*n)]))
SigmaBeta[(n+1):(p*n),1:n] <- t(SigmaBeta[1:n,(n+1):(p*n)])

# For correlation matrix
Sigma11cor <- Sigma12cor <- Sigma21cor <- Sigma22cor <- matrix(nc=n,nr=n)
Sigma11 <- SigmaBeta[1:n,1:n]
Sigma12 <- SigmaBeta[1:n,(n+1):(p*n)]
Sigma21 <- SigmaBeta[(n+1):(p*n),1:n]
Sigma22 <- SigmaBeta[(n+1):(p*n),(n+1):(p*n)]
for(s in 1:n){
  for(t in 1:n){
    Sigma11cor[s,t] <- Sigma11[s,t]/sqrt(Sigma11[s,t]*Sigma11[s,t])
    Sigma12cor[s,t] <- Sigma12[s,t]/sqrt(Sigma11[s,t]*Sigma22[s,t])
    Sigma21cor[s,t] <- Sigma21[s,t]/sqrt(Sigma22[s,t]*Sigma11[s,t])
    Sigma22cor[s,t] <- Sigma22[s,t]/sqrt(Sigma22[s,t]*Sigma22[s,t])
  }
}

SigmaBetaCor <- rbind(cbind(Sigma11cor,Sigma12cor),cbind(Sigma21cor,Sigma22cor))
rm(Sigma11,Sigma12,Sigma21,Sigma22,Sigma11cor,Sigma12cor,Sigma21cor,Sigma22cor)

################################################################################################
## Get smoothed Guillot matrix
################################################################################################

## Guillot is smoothing of empirical covariance matrix
# NOTE: Guillot is blocked by p and not n
SigmaGuil <- matrix(nr=n*p,nc=n*p,0)

lambdaG <- 800

temp.denom <- temp.numer1 <- temp.numer2 <- NULL
for(s in 1:n){
  temp.denom[s] <- sum(K2(s,lambdaG))
}
denom <- outer(temp.denom,temp.denom)

for(time in 1:nyr){
  for(i in 1:p){
    for(j in 1:p){
      for(s in 1:n){
        temp.numer1[s] <- sum(K2(s,lambdaG)*obsmat[i,,time])
        temp.numer2[s] <- sum(K2(s,lambdaG)*obsmat[j,,time])
      }
      SigmaGuil[((j-1)*n+1):((j-1)*n+n),((i-1)*n+1):((i-1)*n+n)] <-
        SigmaGuil[((j-1)*n+1):((j-1)*n+n),((i-1)*n+1):((i-1)*n+n)] +
          outer(temp.numer1,temp.numer2)
    }
  }
  print(time)
}
for(i in 1:p){
  for(j in 1:p){
    SigmaGuil[((j-1)*n+1):((j-1)*n+n),((i-1)*n+1):((i-1)*n+n)] <-
      SigmaGuil[((j-1)*n+1):((j-1)*n+n),((i-1)*n+1):((i-1)*n+n)] / denom
  }
}
rm(temp.denom,temp.numer1,temp.numer2,i,j,time,denom)
SigmaGuil <- SigmaGuil/nyr

################################################################################################
## Nonstationary bivariate Matern with
## constant smoothnesses of 2, fixed globally estimated scale, local variances
################################################################################################

M <- function(Q,nu=2){
  Q[Q==0] <- 1e-10
  Q^nu * besselK(x=Q,nu=nu)
}

##
## Global estimate of scale, scale matrix is diag(A^2), and below the estimate is for A
##

LS.G <- function(try){
  # proposed covariance matrix
  print(try)
  SigmaTry <- matrix(nc=n*p,nr=n*p)
  SigmaTry[1:n,1:n] <- try[1]^2*M(Q=dist.mat/try[3],nu=2) / try[3]^2
  SigmaTry[(n+1):(p*n),(n+1):(p*n)] <-
    try[2]^2*M(Q=dist.mat/try[3],nu=2) / try[3]^2
  SigmaTry[1:n,(n+1):(p*n)] <- SigmaBetaCor[1:n,(n+1):(p*n)]*
      try[1]*try[2]*M(Q=dist.mat/try[3],nu=2) / try[3]^2
  SigmaTry[(n+1):(p*n),1:n] <- t(SigmaTry[1:n,(n+1):(p*n)])
  # sum up weighted least squares distances between empirical
  # covariance matrix and viewed covariance matrix
  LSsum <- sum((SigmaGuil - SigmaTry)^2)
  LSsum
}

best <- optim(par=c(700,50,700),fn=LS.G,lower=c(1,1,1),
                upper=c(2000,2000,2000),method="L-BFGS-B")

rm(CV,LAMBDA,LS.G,tas.c,pr.c)

A <- 902.52442

##
## Estimating nonstationarity parameters at training locations
##

MMat <- M(Q=dist.mat/A,nu=2) / A^2

LS.N <- function(try){
  # proposed covariance matrix
  SigmaTry <- matrix(nc=n*p,nr=n*p)
  SigmaTry[1:n,1:n] <- try[1]^2*MMat
  SigmaTry[(n+1):(p*n),(n+1):(p*n)] <- try[2]^2*MMat
  SigmaTry[1:n,(n+1):(p*n)] <- SigmaBetaCor[1:n,(n+1):(p*n)]*try[1]*try[2]*MMat
  SigmaTry[(n+1):(p*n),1:n] <- t(SigmaTry[1:n,(n+1):(p*n)])
  # sum up weighted least squares distances between empirical
  # covariance matrix and viewed covariance matrix
  LSsum <- sum((W*(SigmaGuil - SigmaTry))^2)
  LSsum
}

write.table(SigmaBeta,file="NARCCAPNSMMFitFull2SigmaBeta.txt")
write.table(SigmaBetaCor,file="NARCCAPNSMMFitFull2SigmaBetaCor.txt")

VWLS <- matrix(nr=n,nc=3) # v1, v2, minimizing value
for(s in 1:n){
  Kmatrix <- outer(K2(s,lambda),K2(s,lambda)) # note we square below in the Frobenius norm

  W <- matrix(nr=n*p,nc=n*p)
  W[1:n,1:n] <- W[1:n,(n+1):(n*p)] <-
    W[(n+1):(n*p),1:n] <- W[(n+1):(n*p),(n+1):(n*p)] <- Kmatrix

  best <- optim(par=c(700,50),fn=LS.N,lower=c(1,1),
                  upper=c(2000,2000),method="L-BFGS-B")
  VWLS[s,1:2] <- best$par
  VWLS[s,3] <- best$val
  if((s %% 20) == 0){
    write.table(VWLS,"NARCCAPNSMMFitFull2VWLS.txt")
    print("saved")
  }
  print(s)
}

################################################################################################
## Plots
################################################################################################

M <- function(Q,nu=2){
  Q[Q==0] <- 1e-10
  Q^nu * besselK(x=Q,nu=nu)
}

load("/Users/wkleiber/Documents/NARCCAP/ECP2_ncep.save")

attach(ECP2_ncep)

library(MASS)
library(fields)

tokeep1 <- seq(1,dim(lon)[1],by=3)
tokeep2 <- seq(1,dim(lat)[2],by=3)

lon <- lon[tokeep1,tokeep2]
lat <- lat[tokeep1,tokeep2]
mo <- months
yr <- years
nyr <- length(unique(yr))

detach(ECP2_ncep)
rm(tokeep1,tokeep2,ECP2_ncep)
n <- prod(dim(lat)) # number of locations 
loc <- cbind(c(lon),c(lat))
p <- 2 # number of processes

A <- 902.5244

SigmaBetaCor <-
  read.table("/Users/wkleiber/Documents/NSMM/code/NARCCAPNSMMFitFull2SigmaBetaCor.txt")
SigmaBetaCor <- as.matrix(SigmaBetaCor)
cc <- diag(SigmaBetaCor[1:n,(n+1):(p*n)])
VWLS <- read.table("/Users/wkleiber/Documents/NSMM/code/NARCCAPNSMMFitFull2VWLS.txt")

## Just cross-correlation
image.plot(lon,lat,matrix(cc,nc=ncol(lon),nr=nrow(lon)),zlim=c(-0.25,0.44),
           main="Cross-Correlation",ylab="Latitude",xlab="Longitude")
US(add=T);world(add=T)
## Just SDs
par(mfrow=c(1,2),mar=c(5,4,4,7))
image.plot(lon,lat,matrix(sqrt(2)*VWLS[,1]/A,nc=ncol(lon),nr=nrow(lon)),
           main="Temperature",ylab="Latitude",xlab="Longitude")
US(add=T);world(add=T)
image.plot(lon,lat,matrix(sqrt(2)*VWLS[,2]/A,nc=ncol(lon),nr=nrow(lon)),
           main="Precipitation",ylab="Latitude",xlab="Longitude")
US(add=T);world(add=T)

## Altogether
par(mfrow=c(1,3),mar=c(5,4,4,6)) # note, converting local sigma^2/A^2 to actual SD
# where gamma(2)/2^(1-2) = 2 needs to be multiplied through before converting to SD
# VAWLS[,1] is sigma, A is A, so var = (sigma^2 / A^2) * (gamma(2)/2^(1-2))
# implies SD = (sigma/A) * sqrt(2)
image.plot(lon,lat,matrix(sqrt(2)*VWLS[,1]/A,nc=ncol(lon),nr=nrow(lon)),
           main="(a)",ylab="Latitude",xlab="Longitude")
US(add=T);world(add=T)
image.plot(lon,lat,matrix(sqrt(2)*VWLS[,2]/A,nc=ncol(lon),nr=nrow(lon)),
           main="(b)",ylab="Latitude",xlab="Longitude")
US(add=T);world(add=T)
image.plot(lon,lat,matrix(cc,nc=ncol(lon),nr=nrow(lon)),zlim=c(-0.25,0.44),
           main="(c)",ylab="Latitude",xlab="Longitude")
US(add=T);world(add=T)
