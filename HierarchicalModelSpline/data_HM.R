#options(warn=-1)
# This file creates the dataset to be used in sampling the Hierarchical Model
#==============================================================================
#======== Loading Libraries =======
require(orthopolynom,warn.conflicts = FALSE)
require(FNN,warn.conflicts = FALSE)
require(ggplot2,warn.conflicts = FALSE)
library(grid)
require(mclust,warn.conflicts = FALSE)
require(mvtnorm,warn.conflicts = FALSE)
require(plyr,warn.conflicts = FALSE)
require(reshape2,warn.conflicts = FALSE)
require(signal,warn.conflicts = FALSE)
require(msm,warn.conflicts = FALSE)
require(pracma,warn.conflicts = FALSE)
require(truncnorm,warn.conflicts = FALSE)
require(MASS,warn.conflicts = FALSE)
require(rhdf5,warn.conflicts = FALSE)
require(fda,warn.conflicts = FALSE)
library(MCMCpack,warn.conflicts = FALSE)
#--------------- Notes --------
# There should be no pm data with just one value. This is pm could be both present or
# both missing.

N_tot   <- 10000 # rough estimate on the number of stars. more will be added to improve computing time
npfld   <- 30   # Number of MPI processes to compute filed parameters
N_th    <- 12   # Number of threads. Missing and Non-missing will be integer multiples of this number.
wr      <- 2    # Walkers ratio




dir <- "/pcdisk/boneym5/jolivares/Data"
dir_f <- "/pcdisk/boneym5/jolivares/hmStarClusters/Analysis/Functions"
# Directory where output data will be stored
dir_out <- file.path(dir,paste("Boneym_10G_0.75_",format(N_tot,scientific=T,digits=1),sep=''))
dir.create(dir_out,showWarnings=F)
dir_graphs <- file.path(dir_out,'Graphs')
dir.create(dir_graphs,showWarnings=F)
dir.create(file.path(dir_out,'Field'),showWarnings=F)

# Directory where input data is stored
dataset     <- "/pcdisk/kool5/scratch-lsb/Data_lsb"

ffield      <- file.path(dataset,'All_unique.RData')
# fmembers    <- file.path(dataset,'Sarro_members.RData')

fsele          <- file.path(dir_out,'Field/Selection.h5') #Stars which will be used to compute field parameters
fpar_pm        <- file.path(dir_out,'Field/FieldParameters_PM.h5')
fpar_phot      <- file.path(dir_out,'Field/FieldParameters_Phot.h5')

#------------- FILES to be created ---------
finit  <- file.path(dir_out,'Initial.h5') # initial solution and size of parmeters
fdata  <- file.path(dir_out,'Dataset.h5')
#===================== Names =================================
names.sarro  <- c("pmRA","pmDEC","K","J","H","i_K","Y_J")
names.usarro <- c("pmRA","pmDEC","K","J","H","i","Y")

names        <- c("pmRA","pmDEC","i_K","Y","J","H","K")
names.uncert <- c("pmRA","pmDEC",  "i","Y","J","H","K")

# names.selec        <- c("pmRA","pmDEC","i_K","Y_J","J","H","K","i")
# names.selec.uncert <- c("pmRA","pmDEC",  "i",  "Y","J","H","K","i")

clr_par   <- 'i_K'
fit_var   <- c('Y','J','H','K')
# clr_var   <- 1  # index of the color variable

# Uncertainty transformation matrix
A <- diag(rep(1,7))
A[3,7]     <- -1
#===================== Constants ==================
#Values that will replace NANs
x_NA       <- 1000
u_NA       <- 1000
#==========Photometry==================================
stp_int    <- 10         # Number of intervals to compute color integral
rg_clr     <- c(0.8,8)    # Color index range. 

ngauss_clr <- 5          # Number of gaussians for color distribution
#Polynomial fit to isochrone
K          <- 9       # degree +1
#++++++++++++++++++++++++ SPLINE ++++++++++++++++++++++
polytype   <- "Spline"
spl_deg    <- 3  # Cubic Spline
# knots      <- c(2.91641529,  3.10518381,  3.25445115,  3.41755774,  3.80668357)
knots      <- c(3.10518381,  3.25445115,  3.41755774,  3.80668357,5.0)
knots      <- c(rg_clr[1],knots,rg_clr[2])
ext_knots  <- c(rep(knots[1],spl_deg),knots,rep(tail(knots,1),spl_deg))
#++++++++++++++++++++++++ LAGUERRE ROTATED ++++++++++++
# polytype   <- "ChebRot"
# x0         <- 3.5#c(0,0,0,6)#3.395939
# y0         <- 13#c(0,0,0,16)#12.31455
# omega      <- 0#c(0,0,0,1.54)

#---- to control the initial fit use one of the following criteria
p_p        <- 0.05 # percentage of points in last part of the sequence to extend it.
p_pe       <- 0.05

p_tresh    <- 0.75  # Probability used to define members 
p_field    <- 0.75 # Probsbility treshold for selecting field objects
#------------------------
dm         <- 0.75  # binaries displacement in magnitudes
rwdb       <- c(0.1,0.1,0.1,0.1) 
mbu        <- c(0.2,0.2,0.2,0.2)
mbl        <- c(0.2,0.2,0.2,0.2)
bnr        <- c(7  ,7  ,7  ,7.0) # remove outliers
#=============Proper Motions===============================
max.gpm   <- 5        # Max number of multivariate normals for cluster
max.gclr  <- 8


###################################### DATA ###########################
######################## MEMBERS ######################################
# load(fmembers)
# stop()
# #=============== Treshold to members =================
# indicator <- which(pro_members > p_tresh)
# pro_members <- pro_members[indicator]
# members   <- members[indicator,]
# u_members <- u_members[indicator,]
# cat("Members with probability grater than",p_tresh," : ",dim(members)[1],"\n")

# # Select desired quantities
# members    <- members[,names.selec]
# u_members  <- u_members[,names.selec.uncert]
# #--------------------------------------------
# indicator   <- is.na(members) | is.na(u_members)
# members[indicator]   <- NA
# u_members[indicator] <- NA
# cat("Initial members: ",dim(members)[1],"\n")

# # #=============== Remove high uncertainty candidates =================
# indicator <- which((u_members[,names.uncert[1]] < 30 &
#                     u_members[,names.uncert[2]] < 30)) 
#                     # u_members[,names.uncert[3]] < 10  &
#                     # u_members[,names.uncert[4]] < 10  &
#                     # u_members[,names.uncert[5]] < 10  &
#                     # u_members[,names.uncert[6]] < 10  &
#                     # u_members[,names.uncert[7]] < 10 
#                      # ))
# pro_members <- pro_members[indicator]
# members     <- members[indicator,]
# u_members   <- u_members[indicator,]
# cat("Members with uncertainty in pm less than 30 mas/yr: ",dim(members)[1],"\n")
# # Remove stars with color index less or grater than color range. IF IT IS PRESENT!
# indicator   <- which(!is.na(members[,clr_par])) # Observations with color index
# nonvalid    <- ((members[indicator,clr_par]< rg_clr[1]) | (members[indicator,clr_par]> rg_clr[2]))
# if (any(nonvalid)){
# members     <- members[-c(indicator[nonvalid]),]
# u_members   <- u_members[-c(indicator[nonvalid]),]
# pro_members <- pro_members[-c(indicator[nonvalid])]
# }
# cat("Members inside color range ",clr_par," [",rg_clr,"] : ",dim(members)[1],"\n")
# #========== REMOVING ===========
# #Finding duplicates
# # valid       <- !duplicated(members)
# # pro_members <- pro_members[valid]
# # members     <- members[valid,]
# # u_members   <- u_members[valid,]

# if (anyDuplicated(members) != 0) stop('Duplicated entries in members')
# N_mem <- dim(members)[1]
# cat("Total number of members: ",N_mem,"\n")


# fsavmem <- file.path("/pcdisk/boneym5/jolivares/Data/Boneym_10G_1e+04/MemFit.h5")
# file.remove(fsavmem)
# h5createFile(file=fsavmem)
# h5write(t(members),     file=fsavmem,"mem")
# h5write(t(u_members),   file=fsavmem,"u_mem")


######################### FIELD ############################
load(ffield)
########################################################################################################
# Remove stars with color index less or grater than color range. IF IT IS PRESENT!
indicator  <- which(!is.na(nm_obs[,"Y_J"])) # Observations with color index
nonvalid   <- ((nm_obs[indicator,"Y_J"]< -1) | (nm_obs[indicator,"Y_J"]> 2))
if (any(nonvalid)){
nm_obs     <- nm_obs[-c(indicator[nonvalid]),]
u_nm_obs   <- u_nm_obs[-c(indicator[nonvalid]),]
pro_nm     <- pro_nm[-c(indicator[nonvalid])]
}
cat("Candidates inside color range Y-J [",-1,2,"]: ",dim(nm_obs)[1],"\n")

nm_obs <- cbind(nm_obs,"Y"= nm_obs[,"Y_J"]+ nm_obs[,"J"])

# Select desired quantities
nm_obs   <- nm_obs[,names]
u_nm_obs <- u_nm_obs[,names.uncert]


#===========================================================
# if obs is missing uncertainty too viceversa
indicator   <- is.na(nm_obs) | is.na(u_nm_obs)
#---------- if K is missing i-K too -----
# indicator[,clr_par] <- indicator[,clr_par] | indicator[,clr_mag]

nm_obs[indicator]   <- NA
u_nm_obs[indicator] <- NA
#------------------------
####################################################################
cat("Initial candidates: ",dim(nm_obs),"\n")
#=============== Remove high uncertainty candidates =================
indicator <- which((u_nm_obs[,names.uncert[1]] < 30 &
                    u_nm_obs[,names.uncert[2]] < 30))
                    # u_nm_obs[,names.uncert[3]] < 5  &
                    # u_nm_obs[,names.uncert[4]] < 5  &
                    # u_nm_obs[,names.uncert[5]] < 5  &
                    # u_nm_obs[,names.uncert[6]] < 5  &
                    # u_nm_obs[,names.uncert[7]] < 5 
                     # ))
pro_nm    <- pro_nm[indicator]
nm_obs    <- nm_obs[indicator,]
u_nm_obs  <- u_nm_obs[indicator,]
cat("Candidates with uncertainty in pm less than 30 mas/yr: ",dim(nm_obs)[1],"\n")
# #--remove candidates with high uncertainties
# valid    <- which(!(rowSums(abs(u_nm_obs[,c(names[3:5],names[7])]/nm_obs[,c(names[3:5],names[7])]) > 0.9,na.rm=T) > 0))
# nm_obs   <- nm_obs[valid,]
# u_nm_obs <- u_nm_obs[valid,]
# pro_nm   <- pro_nm[valid]
# cat("Candidates with relative uncertainties less than 90%: ",dim(nm_obs)[1],"\n")
#--remove duplicates in non-members
# valid    <- !duplicated(nm_obs)
# nm_obs   <- nm_obs[valid,]
# u_nm_obs <- u_nm_obs[valid,]
# pro_nm   <- pro_nm[valid]
#===================================
# Remove stars with color index less or grater than color range. IF IT IS PRESENT!
indicator  <- which(!is.na(nm_obs[,clr_par])) # Observations with color index
nonvalid   <- ((nm_obs[indicator,clr_par]< rg_clr[1]) | (nm_obs[indicator,clr_par]> rg_clr[2]))
if (any(nonvalid)){
nm_obs     <- nm_obs[-c(indicator[nonvalid]),]
u_nm_obs   <- u_nm_obs[-c(indicator[nonvalid]),]
pro_nm     <- pro_nm[-c(indicator[nonvalid])]
}
cat("Candidates inside color range ",clr_par," [",rg_clr,"]: ",dim(nm_obs)[1],"\n")


########################################################################################################
#---remove stars with missing pm-------
indicator <- is.na(nm_obs)*1
valid     <- !(indicator[,1]==1 | indicator[,2]==1)
pro_nm    <- pro_nm[valid]
nm_obs    <- nm_obs[valid,]
u_nm_obs  <- u_nm_obs[valid,]  
cat("Candidates with two pm entries: ",dim(nm_obs)[1],"\n")
#---remove stars with missing phot, at least two entries present
# (one could be tractable but functions in python (e.g. np.linalg.inv) doesnt work)
indicator <- is.na(nm_obs)*1
valid     <- !(rowSums(indicator[,3:7])==5 | rowSums(indicator[,3:7])==4 )
pro_nm    <- pro_nm[valid]
nm_obs    <- nm_obs[valid,]
u_nm_obs  <- u_nm_obs[valid,]  
cat("Candidates with at least two photometric entries: ",dim(nm_obs)[1],"\n")


#----- select field objects to construct field parameters --------------
N_mfl          <- npfld*floor((N_tot-sum(pro_nm >  p_field))/npfld) # Number of field stars to compute field
cat("Selecting ",N_mfl," non members with probability below ",p_field, " for field parameters ... \n")
idx_selec      <- which(pro_nm <= p_field)
idx_selec      <- idx_selec[order(pro_nm[idx_selec],decreasing=T)][1:N_mfl]
pro_selec      <- pro_nm[idx_selec]
nm_selec       <- nm_obs[idx_selec,]
u_nm_selec     <- u_nm_obs[idx_selec,]
cat("Searching duplicates in previous selection ...\n")
if (anyDuplicated(nm_selec) != 0) stop('Duplicated entries in non members. Remove them!')
cat("Writting selection ...\n")
garbage <- file.remove(fsele)
garbage <- h5createFile(file=fsele)
h5write(t(nm_selec),     file=fsele,"nonmem")
h5write(t(u_nm_selec),   file=fsele,"u_nonmem")
h5write(t(pro_selec), file=fsele,"pro_nonmem")
h5write(A,             file=fsele,"A")
# stop("Compute field parameters!!!")

####################### Selection using iter6 probability ############################
cat("Selecting members...\n")
idx_members <- which(pro_nm >  p_tresh)
members     <- nm_obs[idx_members,]
pro_members <- pro_nm[idx_members]
u_members   <- u_nm_obs[idx_members,]
N_mem       <- length(pro_members)
cat("Members with probability grater than ",p_tresh," : ",N_mem,"\n")

cat("Selecting non members ... \n")
idx_field   <- which(pro_nm <= p_tresh)
pro_nm      <- pro_nm[idx_field]
nm_obs      <- nm_obs[idx_field,]
u_nm_obs    <- u_nm_obs[idx_field,]
cat("Non members with probability lower than ",p_tresh, " : ",length(pro_nm),"\n")

N_nm     <- N_tot-N_mem
idx_cand <- order(pro_nm,decreasing=T)[1:N_nm]
cat("Candidates with probability lower than",p_tresh," : ",N_nm,"\n")
################## TOTAL DATA ###################################
cat("Mixing members with candidates.\n")
#============MIXING MEMBERS AND NON_MEMBERS AND FIND DUPLICATES ==============
pro_obs      <- c(pro_members,pro_nm[idx_cand])
x_obs        <- rbind(members,nm_obs[idx_cand,])
uncert_x_obs <- rbind(u_members,u_nm_obs[idx_cand,])

dup_x_obs <- duplicated(x_obs)
if (any(dup_x_obs)!= 0){
  print(which(dup_x_obs))
  stop('Duplicated entries in mixed dataset!')
  }

#--------------- Add stars to complete multithreading processes ------
####### Index of missing data #############
# calculamos la matriz indicador
# cat("Computing missing indicator matrix ...\n")
index_full_obs    <- which(rowSums(is.na(x_obs[,names])*1) == 0, arr.ind = TRUE)
index_miss_obs    <- which(rowSums(is.na(x_obs[,names])*1) != 0, arr.ind = TRUE)
N_full            <- length(index_full_obs)
N_miss            <- length(index_miss_obs)
idx_field_dif     <- (1:length(pro_nm))[-idx_cand]

idfo        <- idx_field_dif[which(rowSums(is.na(nm_obs[idx_field_dif,names])*1) == 0, arr.ind = TRUE)]
idmo        <- idx_field_dif[which(rowSums(is.na(nm_obs[idx_field_dif,names])*1) != 0, arr.ind = TRUE)]

cat("Adding candidates to improve MPI/multithreading performance ...\n")
i <- 1
while (N_miss%%N_th != 0){
  pro_obs         <- c(pro_obs,pro_nm[idmo[i]])
  x_obs           <- rbind(x_obs,nm_obs[idmo[i],])
  uncert_x_obs    <- rbind(uncert_x_obs,u_nm_obs[idmo[i],])
  index_miss_obs  <- which(rowSums(is.na(x_obs)*1) != 0, arr.ind = TRUE)
  N_miss          <- length(index_miss_obs)
  i <- i +1 
}

i <- 1
while (N_full%%N_th != 0){
  pro_obs         <- c(pro_obs,pro_nm[idfo[i]])
  x_obs           <- rbind(x_obs,nm_obs[idfo[i],])
  uncert_x_obs    <- rbind(uncert_x_obs,u_nm_obs[idfo[i],])
  index_full_obs  <- which(rowSums(is.na(x_obs)*1) == 0, arr.ind = TRUE)
  N_full          <- length(index_full_obs)
  i <- i +1 
}
i <- NULL

#------------------------------------------------------------
pro_obs_ext      <- pro_obs
x_obs_ext        <- x_obs
uncert_x_obs_ext <- uncert_x_obs

cat("Selecting quantities: ",names,"\n")

pro_obs          <- pro_obs
x_obs            <- x_obs[,names]
uncert_x_obs     <- uncert_x_obs[,names.uncert]

cat("Checking for duplicates ...\n")
#Finding duplicates
dup_x_obs <- duplicated(x_obs)
if (any(dup_x_obs)!= 0){
  print(which(dup_x_obs))
  stop('Duplicated entries')
}
cat("Total number of stars: ",dim(x_obs)[1],"\n")
#----- select incomplete members --------------
cat("Computing priors using known members.\n")
###################### FINDING INITIAL VALUES ##############################
#===================== PROPER MOTIONs=============================
#============================================================================

#===================== PHOTOMETRY =============================
#---------- Functions ------------
transrot  <- function(x,y,x0,y0,omega){
  xr <- (x - x0)*cos(omega)-(y - y0)*sin(omega)
  yr <- (x - x0)*sin(omega)+(y - y0)*cos(omega)
  return(cbind(xr,yr))
}
clr2lam <- function(x,rg){(2*(x - rg[1])/(rg[2]-rg[1]))-1}
lam2clr <- function(x,rg){(0.5*(x +1)*(rg[2]-rg[1]))+rg[1]}
coeffpoly <- function(x,y,ncoeff,rg){
  d <- ncoeff -1

  if (polytype== "Laguerre") {
    cpoly <- polynomial.functions(laguerre.polynomials(d,normalized=F))
  }
  if (polytype== "Chebyshev"){ 
    cpoly <- polynomial.functions(chebyshev.t.polynomials(d,normalized=F))
    x  <- clr2lam(x,rg)
  }
  if (polytype== "ChebRot"){
    cpoly <- polynomial.functions(chebyshev.t.polynomials(d,normalized=F))
    xr <- transrot(xn,yn,x0,y0,omega)[,1]
    y  <- transrot(xn,yn,x0,y0,omega)[,2]
    print(c(min(xr),max(xr)))
    x  <- clr2lam(xr,rg_rot)
  }
  if (polytype== "Spline") {
    bs <- create.bspline.basis(rangeval=rg,breaks=knots,norder= 4)
    coeff <- as.vector(lm(y ~ 0+eval.basis(x,bs))$coef)
    return(coeff)
  }
  s <- matrix(NA, nrow=length(x),ncol=ncoeff)
  for (i in 1:ncoeff){s[,i] <- cpoly[[i]](x)}
  coeff <- as.vector(lm(y ~ 0+s)$coef)
  return(coeff)
}
evalpoly <- function(clr,rg,coef){
  d <- length(coef)-1
  
  if (polytype== "Laguerre") {
    cpoly <- polynomial.functions(laguerre.polynomials(d,normalized=F))
    x  <- clr 
  }
  if (polytype== "Chebyshev"){
    cpoly <- polynomial.functions(chebyshev.t.polynomials(d,normalized=F))
    x  <- clr2lam(clr,rg)  
  }
  if (polytype== "Spline") {
    bs <- create.bspline.basis(rangeval=rg,breaks=knots,norder=spl_deg+1)
    s <- eval.basis(clr,bs)
    return(s%*%coef)
  }
  if (polytype== "ChebRot") {
    cpoly <- polynomial.functions(chebyshev.t.polynomials(d,normalized=F))
    f <- function (z,xo,x0,theta,cff){
      temp <- rep(NA,d+1)
      for (i in 1:(d+1)) temp[i] <- cpoly[[i]](z)
      yr  <- temp%*%cff
      return((x0-xo) + lam2clr(z,rg_rot)*cos(theta)+yr*sin(theta))
    }
    xr <- rep(NA,length(clr))
    for (i in 1:length(clr)) xr[i] <- uniroot(f, c(-1,1), tol = 0.0001, xo=clr[i],x0=x0,theta=omega,cff=coef,extendInt="yes")$root
    s <- matrix(NA, nrow=length(clr),ncol=d+1)
    for (i in 1:(d+1)){s[,i] <- cpoly[[i]](xr)}
      yr  <- as.numeric(s%*%coef)

    pdf(file=file.path(dir_graphs,'fit.pdf'))
    xs   <- seq(-1,1,length.out=100)
    s <- matrix(NA, nrow=length(xs),ncol=d+1)
    for (i in 1:(d+1)){s[,i] <- cpoly[[i]](xs)}
    ys <- s%*%coef
    plot(xs,ys,type='l')
    points(xr,yr)
    dev.off()
    return(yr*cos(omega)-lam2clr(xr,rg_rot)*sin(omega)+y0)
  }
  s <- matrix(NA, nrow=length(x),ncol=d+1)
  for (i in 1:(d+1)){s[,i] <- cpoly[[i]](x)}
  return(s%*%coef)
}
#---------- Data to obtain coefficients.
pdf(file=file.path(dir_graphs,'BIC_Photometry_fit.pdf'))
coefs <- matrix(0,nrow=4,ncol=K)
singlesMask <- matrix(FALSE, nrow=N_mem,ncol=4,dimnames=list(NULL,fit_var))
xs   <- seq(rg_clr[1],rg_clr[2],length.out=100)
# rg_rot <- c(transrot(0.8,8,x0,y0,omega)[,1],transrot(8,22,x0,y0,omega)[,1])

for ( j in 1:4){
  pdf(file=file.path(dir_graphs,'BIC_Photometry_fit.pdf'))
  mask  <- which(!is.na(members[,fit_var[j]]) & !is.na(members[,clr_par]))
  y <- members[mask,fit_var[j]]
  x <- members[mask,clr_par]
  ord   <- order(x)
  x     <- x[ord]
  y     <- y[ord]
  # remove outliers
  lfit <- lm(y ~ x)
  # plot(xn,yn,ylim=c(max(yn),min(yn)),col='blue',type='p')
  # lines(xn,lfit$fitted.values,col='red')
  mask <- rep(TRUE, length(x))
  mask <-  abs(y - lfit$fitted.values) < bnr[j]
  xn <- x[mask]
  yn <- y[mask]
  idd <- duplicated(xn)
  if (sum(idd) != 0){
  xn <- xn[!idd]
  yn <- yn[!idd]
  }
  #---------------------- INITIAL FIT --------------------------
  # ------ extend sequence using a linear fit to the last part of the sequence
  x_end <- tail(xn,floor(p_p*length(xn)))
  y_end <- tail(yn,floor(p_p*length(xn)))
  out_x <- seq(max(x_end),rg_clr[2],length.out=floor(p_pe*length(xn)))
  coef  <- as.vector(lm(y_end ~ x_end)$coef)
  out_y <- coef[1]+coef[2]*out_x
  plot(x,y,xlim=c(rg_clr),ylim=c(max(out_y),min(y)),col='blue')
  points(x_end,y_end,col='red')
  points(out_x,out_y,col='green')
  #------re do fit with extended points--------
  yn <- c(yn, out_y)
  xn <- c(xn, out_x)
  #------- remove binaries -----
  for (h in 3:1){
  plot(xn,yn,ylim=c(max(y),min(y)),xlim=rg_clr,col='blue',type='p')
  bsr  <- evalpoly(xn,rg_clr,coeffpoly(xn,yn,K,rg_clr))
  lines(xn,bsr,col='red')
  mask <- rep(TRUE, length(xn))
  mask <- (yn > (bsr - (rwdb[j]*h + mbu[j]))) & (yn < (bsr + (rwdb[j]*h + mbl[j]))) | ( xn > 3.0 & xn < 3.4)
  xn <- xn[mask]
  yn <- yn[mask]
  }
  #-------------------------------------------------------------
  singlesMask[,fit_var[j]] <- members[,fit_var[j]]%in%yn
  plot(x,y,ylim=c(max(y),min(y)),xlim=rg_clr,col='black',xlab=clr_par,ylab=fit_var[j])
  coefs[j,] <- coeffpoly(xn,yn,K,rg_clr)
  lines(xs,evalpoly(xs,rg_clr,coefs[j,]),col="red")
  # for ( g in 1:20) lines(xs,evalpoly(xs,rg_clr,mvrnorm(1,coefs[j,],diag(seq(1e-2,1e-1,length.out=K)))),col="orange")
}
graphics.off()
singlesMask <- apply(singlesMask[,fit_var],1,any)

# fmem <- file.path(dir_out,'MemFit.h5') 
# garbage <- file.remove(fmem)
# garbage <- h5createFile(file=fmem)
# h5write(t(members),     file=fmem,"mem")
# h5write(ext_knots,     file=fmem,"knots")
# h5write(t(coefs),     file=fmem,"coeffs")
# stop()

# pdf(file=file.path(dir_graphs,'BIC_Photometry_fit_sarro_members.pdf'))
# for ( j in 1:4){
#   y <- mem_srr[,fit_var[j]]
#   x <- mem_srr[,clr_par]
#   ord   <- order(x)
#   xn     <- x[ord]
#   yn     <- y[ord]
#   # remove outliers
#   lfit <- lm(yn ~ xn)
#   mask <- rep(TRUE, length(xn))
#   mask <-  abs(yn - lfit$fitted.values) < 7
#   xn <- xn[mask]
#   yn <- yn[mask]
#   xf   <- seq(rg_clr[1],rg_clr[2],length.out=500)
#   colors <- rainbow(8)

#   #------- remove binaries -----
#   for (h in 5:0){
#   coef <- as.vector(lm(yn ~ evalpoly(xn,13,rg_clr))$coef)
#   bsr  <- evalpoly(xn,rg_clr,coef)
#   mask <- rep(TRUE, length(xn))
#   mask <-  (yn < (bsr + (0.1*h + mbu))) & (yn > (bsr - (0.1*h + mbl)))
#   xn <- xn[mask]
#   yn <- yn[mask]
#   }
#   plot(xn,yn,ylim=c(max(y),min(y)),col='black',pch='.',xlab=clr_par,ylab=fit_var[j])

#   for (K in 7:13){
#   coefsb <- as.vector(lm(yn ~ evalpoly(xn,K,rg_clr))$coef)
#   lines(xf,evalpoly(xf,rg_clr,coefsb),col=colors[K-6])
#   lines(xf,evalpoly(xf,rg_clr,coefsb),col=colors[K-6])
#   # coefsb <- lm(yn ~ stats::poly(xn,K))
#   # lines(xn,predict(coefsb),col=colors[K-6])
#   }
#   legend(6,13,seq(7,13),lty=c(1,1),lwd=c(2.5,2.5),col=colors)
#   contour(kde2d(xn,yn,h=c(0.1,0.2),n=100),add=T,col="black")
# }
# graphics.off()
#---------impute missing values ---------
mem_imp   <- members
u_mem_imp <- u_members
for (j in 3:7){
    c_temp <- which(!is.na(mem_imp[,j]) & !is.na(u_mem_imp[,j]))
    i_temp <- which( is.na(mem_imp[,j]))
    dist   <-  rep(NA,length(c_temp))
for (i in 1:length(i_temp)){
  for (k in 1:length(c_temp)) 
  dist[k]     <- sqrt(sum((mem_imp[i_temp[i],3:7]-mem_imp[c_temp[k],3:7])**2,na.rm=T))

  u_mem_imp[i_temp[i],j] <- u_mem_imp[c_temp[which.min(dist)],j]
  mem_imp[i_temp[i],j]   <- mem_imp[c_temp[which.min(dist)],j]+rnorm(1,0,u_mem_imp[i_temp[i],j])
}
}
#============================================================================
dfclrmm <- data.frame(mem_imp[,clr_par])
colnames(dfclrmm) <- clr_par
pdf(file=file.path(dir_graphs,'BIC_Color_fit.pdf'))
source(file.path(dir_f,'p_mg.R'))
bic_clr <- rep(NA,max.gclr)
abs      <- seq(rg_clr[1],rg_clr[2],length.out=100)
for (i in 1:max.gclr){
  fit    <- Mclust(dfclrmm,i,"V")
  bic_clr[i] <- fit$BIC
  theta_clr  <- as.vector(fit$parameters$pro)
  mu_clr     <- as.vector(fit$parameters$mean)
  vr_clr     <- as.vector(fit$parameters$variance$sigmasq)
  df_clr     <- data.frame(x=abs,y=p_mg(abs,theta_clr,mu_clr,vr_clr,rg_clr))
  print(ggplot(df_clr)+
  geom_path(aes(x,y),colour="black")+
  xlab(clr_par)+
  ylab("Density")+
  stat_density(data=dfclrmm,aes_string(clr_par),colour='blue',alpha=0,adjust=1)+
  coord_cartesian(xlim=rg_clr)+
  ggtitle('Members')+
  theme_bw())
}
print(ggplot(data.frame(x=1:max.gclr,y=bic_clr))+
  geom_line(aes(x,y))+
  scale_x_discrete()+
  xlab("Number of gaussians")+
  ylab("BIC")+
  theme_bw())
graphics.off()
fit.clr    <- Mclust(dfclrmm,ngauss_clr,"V")$parameters
theta_clr  <- as.vector(fit.clr$pro)
mu_clr     <- as.vector(fit.clr$mean)
vr_clr     <- as.vector(fit.clr$variance$sigmasq)
#---sorted gaussians
ord_clr    <- order(mu_clr)
theta_clr  <- theta_clr[ord_clr]
mu_clr     <- mu_clr[ord_clr]
vr_clr     <- vr_clr[ord_clr]

dfpmPs <- data.frame(members[singlesMask,c(1,2)])
pdf(file=file.path(dir_graphs,'BIC_PM_Ps_fit.pdf'))
source(file.path(dir_f,'plotcov.R'))
bic_pm <- rep(NA,max.gpm)
for (i in 2:max.gpm){
  fit        <- Mclust(dfpmPs,i,"VVV")
  bic_pm[i] <- fit$BIC
  theta_pm   <- fit$parameters$pro
  mu_pm      <- fit$parameters$mean
  sigma_pm   <- fit$parameters$variance$sigma

  df4PM      <- pltcovnm(t(mu_pm),sigma_pm)

  plt <- ggplot(dfpmPs, aes(pmRA,pmDEC)) + 
  geom_point(color="blue") +
  geom_path(data=df4PM,aes(x,y,group=g),size=0.5)+
  xlab('R.A. [mas/yr]')+
  ylab('Dec. [mas/yr]')+
  ggtitle('Proper motions')+
  theme_bw()
  print(plt)

  data_sim_pm <- members[1,1:2]
  for (j in 1:i) data_sim_pm <- rbind(data_sim_pm,
                mvrnorm(floor(theta_pm[j]*N_mem),mu_pm[,j],sigma_pm[,,j]))
  data_sim_pm  <- data_sim_pm[-c(1),]
  mockpm <- data.frame(data_sim_pm)

  plt <- ggplot(dfpmPs) + 
  stat_density(aes(pmRA,colour="real"),alpha=0,adjust=0.75) +
  stat_density(data=mockpm,aes(pmRA,colour="mock"),alpha=0,adjust=0.75)+
  scale_colour_manual(name="",values=c("real"="blue","mock"="black")) + 
  xlab('R.A. [mas/yr]')+
  ggtitle('Proper motion density')+
  theme_bw()
  print(plt)

  plt <- ggplot(dfpmPs) + 
  stat_density(aes(pmDEC,colour="real"),alpha=0,adjust=0.75) +
  stat_density(data=mockpm,aes(pmDEC,colour="mock"),alpha=0,adjust=0.75)+
  scale_colour_manual(name="",values=c("real"="blue","mock"="black")) + 
  xlab('Dec. [mas/yr]')+
  ggtitle('Proper motion density')+
  theme_bw()
  print(plt)
}
print(ggplot(data.frame(x=1:max.gpm,y=bic_pm))+
  geom_line(aes(x,y))+
  scale_x_discrete()+
  xlab("Number of gaussians")+
  ylab("BIC")+
  theme_bw())
graphics.off()

dfpmBs <- data.frame(members[!singlesMask,c(1,2)])
pdf(file=file.path(dir_graphs,'BIC_PM_Bs_fit.pdf'))
source(file.path(dir_f,'plotcov.R'))
bic_pm <- rep(NA,max.gpm)
for (i in 2:max.gpm){
  fit        <- Mclust(dfpmBs,i,"VVV")
  bic_pm[i] <- fit$BIC
  theta_pm   <- fit$parameters$pro
  mu_pm      <- fit$parameters$mean
  sigma_pm   <- fit$parameters$variance$sigma

  df4PM      <- pltcovnm(t(mu_pm),sigma_pm)

  plt <- ggplot(dfpmBs, aes(pmRA,pmDEC)) + 
  geom_point(color="blue") +
  geom_path(data=df4PM,aes(x,y,group=g),size=0.5)+
  xlab('R.A. [mas/yr]')+
  ylab('Dec. [mas/yr]')+
  ggtitle('Proper motions')+
  theme_bw()
  print(plt)

  data_sim_pm <- members[1,1:2]
  for (j in 1:i) data_sim_pm <- rbind(data_sim_pm,
                mvrnorm(floor(theta_pm[j]*N_mem),mu_pm[,j],sigma_pm[,,j]))
  data_sim_pm  <- data_sim_pm[-c(1),]
  mockpm <- data.frame(data_sim_pm)

  plt <- ggplot(dfpmBs) + 
  stat_density(aes(pmRA,colour="real"),alpha=0,adjust=0.75) +
  stat_density(data=mockpm,aes(pmRA,colour="mock"),alpha=0,adjust=0.75)+
  scale_colour_manual(name="",values=c("real"="blue","mock"="black")) + 
  xlab('R.A. [mas/yr]')+
  ggtitle('Proper motion density')+
  theme_bw()
  print(plt)

  plt <- ggplot(dfpmBs) + 
  stat_density(aes(pmDEC,colour="real"),alpha=0,adjust=0.75) +
  stat_density(data=mockpm,aes(pmDEC,colour="mock"),alpha=0,adjust=0.75)+
  scale_colour_manual(name="",values=c("real"="blue","mock"="black")) + 
  xlab('Dec. [mas/yr]')+
  ggtitle('Proper motion density')+
  theme_bw()
  print(plt)
}
print(ggplot(data.frame(x=1:max.gpm,y=bic_pm))+
  geom_line(aes(x,y))+
  scale_x_discrete()+
  xlab("Number of gaussians")+
  ylab("BIC")+
  theme_bw())
graphics.off()
#----------------Initial Conditions ------------------------------
cm_par   <- Mclust(members[singlesMask,c(1,2)],2,"VVV")$parameters
theta_Ps <- cm_par$pro
# mu_pm    <- t(cm_par$mean)
mu_pm    <- c(mean(cm_par$mean[1,]),mean(cm_par$mean[2,]))
sigma_pm <- aperm(cm_par$variance$sigma,c(3,2,1))
tu_sg_pm  <- matrix(NA,2,3)
for (i in 1:2){
  tu_sg_pm[i,1] <- sigma_pm[i,1,1]
  tu_sg_pm[i,2] <- sigma_pm[i,1,2]
  tu_sg_pm[i,3] <- sigma_pm[i,2,2]
}
#----------------Initial Conditions ------------------------------
cm_par       <- Mclust(members[!singlesMask,c(1,2)],2,"VVV")$parameters
theta_Bs     <- cm_par$pro
# mu_pm4bin    <- t(cm_par$mean)
mu_pm4bin    <- c(mean(cm_par$mean[1,]),mean(cm_par$mean[2,]))
sigma_pm4bin <- aperm(cm_par$variance$sigma,c(3,2,1))
tu_sg_pm4bin  <- matrix(NA,2,3)
for (i in 1:2){
  tu_sg_pm4bin[i,1] <- sigma_pm4bin[i,1,1]
  tu_sg_pm4bin[i,2] <- sigma_pm4bin[i,1,2]
  tu_sg_pm4bin[i,3] <- sigma_pm4bin[i,2,2]
}
#===============================================================================
#=============== Load Field parameters ========
cat("Loading field parameters ...\n")
# This must include the field parameters of the Gaussian mixture
M_pm        <-       h5read(fpar_pm, "M_pm",bit64conversion="int")
theta_nm_pm <-       h5read(fpar_pm, "theta_nm_pm")
mu_nm_pm    <-     t(h5read(fpar_pm, "mu_nm_pm"))
sigma_nm_pm <- aperm(h5read(fpar_pm, "sigma_nm_pm"),c(3,2,1))
mima_nm_pm  <-     t(h5read(fpar_pm, "mima_nm_pm"))
cte_nm_pm   <-       h5read(fpar_pm, "cte_nm_pm")


M_phot        <-       h5read(fpar_phot, "M_phot",bit64conversion="int")
theta_nm_phot <-       h5read(fpar_phot, "theta_nm_phot")
mu_nm_phot    <-     t(h5read(fpar_phot, "mu_nm_phot"))
sigma_nm_phot <- aperm(h5read(fpar_phot, "sigma_nm_phot"),c(3,2,1))

rownames(mu_nm_pm) <- names[1:2]
rownames(mu_nm_phot) <- names[3:7]
dimnames(sigma_nm_pm) <- list(names[1:2],names[1:2],NULL)
dimnames(sigma_nm_phot) <- list(names[3:7],names[3:7],NULL)
#------------ check imput -----
for ( i in 1:dim(sigma_nm_pm)[3]) chol(sigma_nm_pm[,,i])
for ( i in 1:dim(sigma_nm_phot)[3]) chol(sigma_nm_phot[,,i])
####### Index of missing data #############
# calculamos la matriz indicador
index_fully_obs    <- which(rowSums(is.na(x_obs)*1) == 0, arr.ind = TRUE)
N_fully_obs        <- length(index_fully_obs)
index_miss_obs     <- which(rowSums(is.na(x_obs)*1) != 0, arr.ind = TRUE)
indicator          <- is.na(x_obs[!complete.cases(x_obs),])*1

#############################################
#=====================================================
# ################################ MIsssing data ##################
old_u_x_obs <- uncert_x_obs
x_obs_old   <- x_obs
N_potential <- dim(x_obs)[1]
# change NaN to u_NA and x_NA values
  for(i in 1:length(index_miss_obs)){ 
    uncert_x_obs[index_miss_obs[i],which(indicator[i,]==1)] <- u_NA
           x_obs[index_miss_obs[i],which(indicator[i,]==1)] <- x_NA
  }
rownames(x_obs) <- c()   
rownames(uncert_x_obs) <- c()   
#############################################
cat("Computing field likelihood for each star in the data set ...\n")
######## Photometric uncertainties  #########
sigma_obs <- array(dim=c(N_potential,7,7),dimnames=list(NULL,names,names))

for(r in 1:N_potential){
  #------- covariance matrix of data ---------
  sigma_obs[r,,] <- A%*%diag(uncert_x_obs[r,]^2)%*%t(A)
}
#==============Compute likelihood of beeing field ========================
llnm <- rep(NA,N_potential)
for(r in 1:N_fully_obs){
  lp_pm   <- rep(NA,M_pm)

  lp_pm[1] <- log(theta_nm_pm[1])+cte_nm_pm
  
  for (i in 1:(M_pm-1)) {
    x_comp     <- x_obs[index_fully_obs[r],names[1:2]]
    mu_comp    <- mu_nm_pm[names[1:2],i]
    # sigma_comp <- sigma_nm_pm[names[1:2],names[1:2],i]
    sigma_comp <- sigma_nm_pm[names[1:2],names[1:2],i]+sigma_obs[index_fully_obs[r],names[1:2],names[1:2]]
    lp_pm[i+1] <- log(theta_nm_pm[i+1])+ dmvnorm(x_comp,mu_comp,sigma_comp,log=T)
  }
  cte <- mean(lp_pm)
  slp_pm <- cte+ log(sum(exp(lp_pm-cte)))

  lp_phot <- rep(NA,M_phot)
  for (i in 1:M_phot) {
    x_comp     <- x_obs[index_fully_obs[r],names[3:7]]
    mu_comp    <- mu_nm_phot[names[3:7],i]
    # sigma_comp <- sigma_nm_phot[names[3:7],names[3:7],i]
    sigma_comp <- sigma_nm_phot[names[3:7],names[3:7],i]+sigma_obs[index_fully_obs[r],names[3:7],names[3:7]]
    # sigma_comp <- t(chol(sigma_comp))%*%chol(sigma_comp)
    lp_phot[i] <- log(theta_nm_phot[i])+ dmvnorm(x_comp,mu_comp,sigma_comp,log=T)
  }
  cte <- max(lp_phot)-700
  slp_phot <- cte + log(sum(exp(lp_phot-cte)))
  llnm[index_fully_obs[r]]<- exp(slp_phot)*exp(slp_pm)
}


for(r in 1:(N_potential-N_fully_obs)){
  lp_pm   <- rep(NA,M_pm)
  lp_pm[1] <- log(theta_nm_pm[1])+cte_nm_pm

  for (i in 1:(M_pm-1)){
    x_miss     <- x_obs[index_miss_obs[r],names[1:2]]
    mu_miss    <- mu_nm_pm[names[1:2],i]
    # sigma_miss <- sigma_nm_pm[names[1:2],names[1:2],i]
    sigma_miss <- sigma_nm_pm[names[1:2],names[1:2],i]+sigma_obs[index_miss_obs[r],names[1:2],names[1:2]]

    lp_pm[i+1]   <- log(theta_nm_pm[i+1])+ dmvnorm(x_miss,mu_miss,sigma_miss,log=T)
  }
  cte <- mean(lp_pm)
  slp_pm <- cte+ log(sum(exp(lp_pm-cte)))
  
  lp_phot <- rep(NA,M_phot)
  for (i in 1:M_phot){ 
    x_miss     <- x_obs[index_miss_obs[r],names[3:7]][!indicator[r,names[3:7]]]
    mu_miss    <- mu_nm_phot[!indicator[r,names[3:7]],i]
    # sigma_miss <- sigma_nm_phot[!indicator[r,names[3:7]],!indicator[r,names[3:7]],i]
    sigma_miss <- (sigma_nm_phot[!indicator[r,names[3:7]],!indicator[r,names[3:7]],i]+
                   sigma_obs[index_miss_obs[r],names[3:7],names[3:7]][!indicator[r,names[3:7]],!indicator[r,names[3:7]]])
    
    # sigma_miss <- t(chol(sigma_miss))%*%chol(sigma_miss)
 
    lp_phot[i]   <- log(theta_nm_phot[i])+dmvnorm(x_miss,mu_miss,sigma_miss,log=T)
  }
  cte <- max(lp_phot)-700
  slp_phot <- cte + log(sum(exp(lp_phot-cte)))
  llnm[index_miss_obs[r]]<- exp(slp_phot)*exp(slp_pm)
}
#------------------
if (min(llnm) == 0) stop("Field Likelihood contains zeros!\n Remove outliers or modify field model")
#--------- total data -------------------
u_phot <- array(NA,dim=c(N_potential,15))
u_pm   <- array(NA,dim=c(N_potential,2))
id <- which(lower.tri(matrix(,5,5),diag = T) == TRUE, arr.ind=T)
for(r in 1:N_potential){
  u_phot[r,] <- sigma_obs[r,names[3:7],names[3:7]][id]
  u_pm[r,]   <- uncert_x_obs[r,c(1:2)]^2
}
obs_T <- cbind(llnm,x_obs,u_pm,u_phot)

cat("Data set succsesfully assembled. Its dimensions are:\n")
print(dim(obs_T))
###################### END of DATA ################################           
cat("Asembling initial solution ...\n")  
############################ INIT,MINS,MAXS for initial exploration PSO###############
#--------Setup initial parameters ----------
szs_parms <- rbind(c(2,0),c(2,1),c(4,0),c(5,0),c(5,0),c(1,2),c(2,3),c(1,2),c(2,3),c(4,K),c(15,0))#,c(5,0))
#The previous line contains the sizes of the parameters.
# there are 4 fractions, 2 fractions for gaussian mixture, 3 means and 3 variances for gaussian mixture,
#one array of [2,2] for proper motions in the principal curve, 2 cholesky decompositions of the inverse of 
#covariance matrix for pm, one matrix
# for proper motion of binaries and 2 chol decomp of inverse of cov-matrix for proper motion of binaries 
# and one matrix of K times O for variances of Chebyshev coefficients.
# 15 entries of covariance matrix for cluster's photometry
# 5 entries of alpha parameter of mcsn

#================ Fractions ======================
Bf <- 0.2   #Binary fraction prior
theta    <- c(N_nm,N_mem)/(N_tot)
theta_PB <- c(1-Bf,Bf)
#==========

init  <- c(theta[1],theta_PB[1],theta_Ps[1],theta_Bs[1],theta_clr[1:4],mu_clr,vr_clr,
           as.vector(t(mu_pm)),as.vector(t(tu_sg_pm)),as.vector(t(mu_pm4bin)),
           as.vector(t(tu_sg_pm4bin)),as.vector(t(coefs)),rep(1e-3,15))#,rep(0,5))
#----------
npars <- length(init)
mins  <- rep(NA,npars)
maxs  <- rep(NA,npars)
hlow  <- rep(-1e6,npars)
hupp  <- rep(1e6,npars)
width <- rep(0.0001,npars) #Not used
#----- Fractions------
mins[1] <- theta[1]-0.1*theta[1]
maxs[1] <- theta[1]+0.1*theta[1]
mins[2] <- 0.6
maxs[2] <- 1.0
#------- PM Fractions ------
idi <- 3
idf <- idi+1
mins[idi:idf] <- 0#init[idi:idf]-0.1*abs(init[idi:idf])
maxs[idi:idf] <- init[idi:idf]+0.1*abs(init[idi:idf])
hlow[idi:idf] <- 0
hupp[idi:idf] <- 1
##############  COLOR DISTRIBUTION ################################
idi <- 5
idf <- idi+3
#------- Color Fractions ------
mins[idi:idf] <- 0#init[idi:idf]-0.1*abs(init[idi:idf])
maxs[idi:idf] <- init[idi:idf]+0.1*abs(init[idi:idf])
hlow[idi:idf] <- 0
hupp[idi:idf] <- 1
#------ Color means ----
idi <- 9
idf <- idi+4
mins[idi:idf] <- init[idi:idf]-0.1*abs(init[idi:idf])
maxs[idi:idf] <- init[idi:idf]+0.1*abs(init[idi:idf])
#------ Color variances ----
idi <- 14
idf <- idi+4
mins[idi:idf] <- init[idi:idf]-0.1*abs(init[idi:idf])
maxs[idi:idf] <- init[idi:idf]+0.1*abs(init[idi:idf])
hlow[idi:idf] <- 1e-5
########################### PROPER MOTIONS #########################################
#------ PM means ----
idi <- 19
idf <- idi+1
mins[idi:idf]  = init[idi:idf]-0.1*abs(init[idi:idf])
maxs[idi:idf]  = init[idi:idf]+0.1*abs(init[idi:idf]) 
idi <- 27
idf <- idi+1
mins[idi:idf]  = init[idi:idf]-0.1*abs(init[idi:idf])
maxs[idi:idf]  = init[idi:idf]+0.1*abs(init[idi:idf]) 
#------ PM sigmas ----
idi <- 21
idf <- idi + 5
mins[idi:idf]  <- init[idi:idf]-0.1*abs(init[idi:idf])
maxs[idi:idf]  <- init[idi:idf]+0.1*abs(init[idi:idf])
hlow[idi:idf][c(1,3,4,6)] <- 0 #diagonal of covariance matrix
idi <- 29
idf <- idi + 5
mins[idi:idf]  = init[idi:idf]-0.1*abs(init[idi:idf])
maxs[idi:idf]  = init[idi:idf]+0.1*abs(init[idi:idf])
hlow[idi:idf][c(1,3,4,6)] <- 0 #diagonal of covariance matrix
##################### ISOCHRONE COEFFICIENTS ####################
idi <- 35
idf <- (idi+(4*K)-1)
factcoef <- rep(seq(5e-3,5e-2,length.out=K),4)
mins[idi:idf]  = init[idi:idf]-factcoef*abs(init[idi:idf])
maxs[idi:idf]  = init[idi:idf]+factcoef*abs(init[idi:idf])
#------------- Covariance of cluster's photometry ---------
idi <- idf +1
idf <- idi -1 + 15
mins[idi:idf] <-  -1e-2
maxs[idi:idf] <-   1e-2
mins[idi:idf][c(1,6,10,13,15)] <- 1e-4
maxs[idi:idf][c(1,6,10,13,15)] <- 1e-2
hlow[idi:idf][c(1,6,10,13,15)] <- 1e-10 #diagonal of covariance matrix
#-------------------------- alpha parameters of csn
# idi <- idf +1
# idf <- idi -1 + 5
# mins[idi:idf] <-  -1
# maxs[idi:idf] <-   1
#------ Check values -----
if (any(maxs <= mins)){ stop("ERROR in mins-maxs")}
if (any(mins < hlow)) { stop("ERROR in hlow")}
if (any(maxs > hupp)) { stop("ERROR in hupp")}

#==================================================================== 
######################### HYPERPRIORS ###############################
cat("Establishing hyperpriors ...\n")
#======= FRACTIONS ===============
# The probabilities are drown from a dirichlet distribution with the following hyperparameters
alpha        <- c(theta[1],theta[2])*1e1
alpha_PB     <- c(8,2)
alpha_PM     <- c(2,8)
theta_hyp    <- cbind(alpha,alpha_PB)
#======= Proper Motions ================
fit.ppmm     <- mvn('XXX',members[,c(1,2)])$parameters
mean_pm_hyp  <- as.vector(fit.ppmm$mean)
sigma_pm_hyp <- as.matrix(fit.ppmm$variance$Sigma)
nu_hyp       <- 2
A_pm         <- rep(1e5,2)
#======= Color distribution ========
alpha_mag    <- rep(1,5)
vr_clr_hyp   <- 10   # Scale of half-cauchy distribution for variances
mu_vr_clr    <- 1e-5 # center of the half-cauchy distribution
#======= Coefficients ===============
mu_coefs     <- coefs
# vr_coefs_hyp <- seq(1,0.01,length.out=K) # Scale of the normal ditribution for coefficients.
vr_coefs_hyp <- rep(1,K) # Scale of the normal ditribution for coefficients.
#=======  Isochrone width ============
A_phot       <- rep(1e1,5) 
#======= CSN ================
vr_alpha_hyp <- rep(10,5)
############################Saving########################

cat("Writing data and related constants ...\n")
#-------------- Data and Constants ----------
garbage <- file.remove(fdata)
garbage <- h5createFile(file=fdata)
garbage <- h5createGroup(file=fdata,"Data")
h5write(t(indicator),    file=fdata,"Data/indicator")
h5write(t(obs_T),        file=fdata,"Data/observations")
h5write(index_fully_obs, file=fdata,"Data/index_full")
h5write(index_miss_obs,  file=fdata,"Data/index_miss")

garbage <- h5createGroup(file=fdata,"Constants")
h5write(theta_hyp,     file=fdata,"Constants/theta_hyp")
h5write(mean_pm_hyp,   file=fdata,"Constants/mean_pm_hyp")
h5write(sigma_pm_hyp,  file=fdata,"Constants/sigma_pm_hyp")
h5write(rg_clr,        file=fdata,"Constants/rg_clr")
h5write(alpha_mag,     file=fdata,"Constants/alpha_mag")
h5write(vr_clr_hyp,    file=fdata,"Constants/vr_clr_hyp")
h5write(mu_vr_clr,     file=fdata,"Constants/mu_vr_clr_hyp")
h5write(stp_int,       file=fdata,"Constants/stp_int")
h5write(K,             file=fdata,"Constants/K")
h5write(t(mu_coefs),   file=fdata,"Constants/mu_coefs")
h5write(A_pm,          file=fdata,"Constants/A_pm")
h5write(A_phot,        file=fdata,"Constants/A_phot")
h5write(vr_coefs_hyp,  file=fdata,"Constants/vr_coefs_hyp")
h5write(nu_hyp,        file=fdata,"Constants/nu_hyp")
h5write(vr_alpha_hyp,  file=fdata,"Constants/vr_alpha_hyp")
h5write(ext_knots,     file=fdata,"Constants/ext_knots")
h5write(spl_deg,       file=fdata,"Constants/spl_deg")
h5write(alpha_PM,      file=fdata,"Constants/alpha_PM")
h5write(exp(cte_nm_pm),file=fdata,"Constants/cte_unipm")


####################### CREATE SAMPLES ####################
cat("Writing initial solution ...\n")
#-----Intitial solution --------
nrep <- npars*wr      #number of replications for samples of priors
data <- matrix(NA, nrep,npars)
for (j in 1:npars) data[,j] <- runif(nrep,mins[j],maxs[j])
#-------------------------------
params  <- cbind(init,mins,maxs,width)
garbage <- file.remove(finit)
garbage <- h5createFile(file=finit)
h5write(t(params),     file=finit,"params")
h5write(t(data),       file=finit,"positions")
h5write(t(szs_parms),  file=finit,"szs_parms")
########################################################################################
cat("Generating graphics ...\n")
#====================================================================  
#Return nans to observations
x_obs <- x_obs_old
uncert_x_obs <- old_u_x_obs
#-----------------
li.pm.ra <- c(-35,65)
li.pm.de <- c(-90,10)
li.mag   <- matrix(NA,4,2)
li.mag[2,] <- c(20,8)
li.mag[1,] <- c(24,8)
li.mag[3,] <- c(20,8)
li.mag[4,] <- c(20,8)
li.clr   <- rg_clr  # limit to plots
limit    <- rbind(li.pm.ra,li.pm.de)
nlam     <- 100
lambdas  <- seq(rg_clr[1],rg_clr[2],length.out=nlam)
#------------
groups   <- rbind(c(clr_par,fit_var[1]),c(clr_par,fit_var[2]),c(clr_par,fit_var[3]),c(clr_par,fit_var[4]),c("pmRA","pmDEC"))
lnames   <- c('Field','Sarro')
alnames  <- c('Field'=0.2,
              'Stauffer'=0.7,
              'Sarro'=0.7,
              'Samples'=0.5,
              'pm 1'=0.5,
              'pm 2'=0.5,
              'err_bar'=0.5,
              'Members'=0.8)
shpnames <- c('Field'=16,
              'Stauffer'=15,
              'Sarro'=17,
              'Samples'=0,
              'pm 1'=0,
              'pm 2'=0,
              'Members'=17)
clnames  <- c('Field'='grey60',
              'Stauffer'="dodgerblue",
              'Sarro'='dodgerblue4',
              'Samples'='orange',
              'pm 1'='springgreen3',
              'pm 2'='springgreen',
              'Ps 1'='springgreen3',
              'Ps 2'='springgreen',
              'Bs 1'='magenta3',
              'Bs 2'='magenta',
              'Ps'='springgreen',
              'Bs'='magenta',
              'Model'='black',
              'HM'='black',
              'All'='cyan',
              'Sarro_missing'='navyblue',
              'Members'='dodgerblue4')
sqnames  <- c('Ps'=2,
              'Bs'=3,
              'Field'=1,
              'Samples'=1,
              'Ps 1'=2,
              'Bs 1'=4,
              'Ps 2'=2,
              'Bs 2'=4,
              'Members'=1)
szsnames <- c('Field'=1,
              'Stauffer'=1.5,
              'Sarro'=1.5,
              'Samples'=0.5,
              'pm 1'=0.5,
              'pm 2'=0.5,
              'Model'=0.5,
              'err_bar'=0.1,
              'Members'=1.5)
pnames <- c('Field fraction','Pc fraction','PS 1 fraction','BS 1 fraction',
            'Color fraction 1','Color fraction 2','Color fraction 3','Color fraction 4',
            'Mean color 1','Mean color 2','Mean color 3','Mean color 4','Mean color 5',
            'Variance color 1','Variance color 2','Variance color 3','Variance color 4','Varianc color 5',
            'Mean PM[1,1]','Mean PM [1,2]',
            # 'Mean PM [2,1]','Mean PM [2,2]',
            'Variance PM[1,1]','Variance PM[1,2]','Variance PM[1,3]',
            'Variance PM[2,1]','Variance PM[2,2]','Variance PM[2,3]',
            'Mean PM Bs[1,1]','Mean PM Bs[1,2]',
            # 'Mean PM Bs[2,1]','Mean PM Bs[2,2]',
            'Variance PM Bs[1,1]','Variance PM Bs[1,2]','Variance PM Bs[1,3]',
            'Variance PM Bs[2,1]','Variance PM Bs[2,2]','Variance PM Bs[2,3]',
            'Coefficient [1,1]','Coefficient [1,2]','Coefficient [1,3]',
            'Coefficient [1,4]','Coefficient [1,5]','Coefficient [1,6]',
            'Coefficient [1,7]','Coefficient [1,8]','Coefficient [1,9]',
            # 'Coefficient [1,10]','Coefficient [1,11]',#'Coefficient [1,12]',
            # 'Coefficient [1,13]',
            'Coefficient [2,1]','Coefficient [2,2]','Coefficient [2,3]',
            'Coefficient [2,4]','Coefficient [2,5]','Coefficient [2,6]',
            'Coefficient [2,7]','Coefficient [2,8]','Coefficient [2,9]',
            # 'Coefficient [2,10]','Coefficient [2,11]',#'Coefficient [2,12]',
            # 'Coefficient [2,13]',
            'Coefficient [3,1]','Coefficient [3,2]','Coefficient [3,3]',
            'Coefficient [3,4]','Coefficient [3,5]','Coefficient [3,6]',
            'Coefficient [3,7]','Coefficient [3,8]','Coefficient [3,9]',
            # 'Coefficient [3,10]','Coefficient [3,11]',#'Coefficient [3,12]',
            # 'Coefficient [3,13]',
            'Coefficient [4,1]','Coefficient [4,2]','Coefficient [4,3]',
            'Coefficient [4,4]','Coefficient [4,5]','Coefficient [4,6]',
            'Coefficient [4,7]','Coefficient [4,8]','Coefficient [4,9]',
            # 'Coefficient [4,10]','Coefficient [4,11]',#'Coefficient [4,12]',
            # 'Coefficient [4,13]',
            'phot_cov[1]','phot_cov[2]','phot_cov[3]','phot_cov[4]','phot_cov[5]',
            'phot_cov[6]','phot_cov[7]','phot_cov[8]','phot_cov[9]','phot_cov[10]',
            'phot_cov[11]','phot_cov[12]','phot_cov[13]','phot_cov[14]','phot_cov[15]')
            #'alpha[1]','alpha[2]','alpha[3]','alpha[4]','alpha[5]')
#=====================================================================
upphot <- 0.1
ratio  <- 0.2
rnames <- rep('o',N_potential)
rnames[pro_obs<p_tresh] <- "Field"
rnames[pro_obs>p_tresh] <- "Sarro"
############################## dataframes ####################
# df_stff <- as.data.frame(mem_stf)
df_mem  <- as.data.frame(members)
df_mmis <- as.data.frame(mem_imp)
df_allm <- as.data.frame(x_obs[which(pro_obs > p_tresh),])
df      <- as.data.frame(cbind(x_obs,"pro"=pro_obs,"fll"=llnm))
dfe     <- as.data.frame(cbind(x_obs_ext,"pro"=pro_obs_ext,"fll"=llnm))
udf     <- as.data.frame(uncert_x_obs)



#=========================== FUNCTIONS =========================================
source(file.path(dir_f,'plotcov.R'))
source(file.path(dir_f,'p_mg.R'))
source(file.path(dir_f,'multiplot.R'))
#============= Selection ==========================
df_selec<- as.data.frame(nm_selec)
df_cand <- as.data.frame(nm_obs[idx_cand,])
nsa     <- length(pro_selec)
#------------------ Generate sinthetic data using parameters of field -------------
data_sim_phot <- nm_obs[1,names[3:7]]
for (i in 1:M_phot) data_sim_phot <- rbind(data_sim_phot,
                mvrnorm(floor(theta_nm_phot[i]*2*nsa),
                    mu_nm_phot[,i],
                    sigma_nm_phot[,,i]))
data_sim_phot  <- data_sim_phot[-c(1),]
data_sim_phot  <- data_sim_phot[sample.int(dim(data_sim_phot)[1],size=nsa),]

data_sim_pm <- nm_obs[1,names[1:2]]
for (i in 1:(M_pm-1)) data_sim_pm <- rbind(data_sim_pm,
                mvrnorm(floor(theta_nm_pm[i+1]*2*nsa),
                    mu_nm_pm[,i],
                    sigma_nm_pm[,,i]))
data_sim_pm <- rbind(data_sim_pm,
                cbind(runif(floor(theta_nm_pm[1]*2*nsa),min=mima_nm_pm[1,1],max=mima_nm_pm[1,2]),
                      runif(floor(theta_nm_pm[1]*2*nsa),min=mima_nm_pm[2,1],max=mima_nm_pm[2,2])))
data_sim_pm  <- data_sim_pm[-c(1),]
data_sim_pm  <- data_sim_pm[sample.int(dim(data_sim_pm)[1],size=nsa),]
data_sim  <- cbind(data_sim_pm,data_sim_phot)

df_st     <- as.data.frame(data_sim)

#------------- Extra Graphs -------------------

graphs   <- rbind(c("H_K","K"),c("Y_J","K"))

df_cand$H_K <- df_cand$H - df_cand$K
df_cand$Y_J <- df_cand$Y - df_cand$J

df_mem$Y_J <- df_mem$Y - df_mem$J
df_mem$H_K <- df_mem$H - df_mem$K

df_selec$H_K <- df_selec$H - df_selec$K
df_selec$Y_J <- df_selec$Y - df_selec$J

df_st$H_K <- df_st$H - df_st$K
df_st$Y_J <- df_st$Y - df_st$J
#============================= Kolmogorov-Smirnov 2sample test===========
cat("KS two samples test for proper motion.\n")
dkspm <- ks.test(nm_selec[,c(1,2)],data_sim_pm)$statistic
d_t  <- 1.95*sqrt(2/nsa)
cat("At the significance level of alpha=0.001 the rejection treshold is ",d_t,"\n")
cat("The statistic for proper motion is D = ",dkspm,"\n")
cat("The null hypothesis is rejected?", dkspm > d_t,"\n")

source(file.path(dir_f,'plot_selection.R'), print.eval=TRUE)
# source(file.path(dir_f,'plot_field.R'), echo=TRUE)
#===============================================================================
cat("Plotting data ...\n")
source(file.path(dir_f,'plot_data.R'), print.eval=TRUE)
####################### PLOT SAMPLES ################
cat("Plotting initial solutions ...\n")
source(file.path(dir_f,'plot_samples.R'))
##################### PLOT priors #########################
cat("Plotting priors ...\n")
source(file.path(dir_f,'plot_priors.R'), print.eval=TRUE)
save.image(file=file.path(dir_graphs,'Dataset.RData'))
cat("The total number of parameters is ", length(init),"\n")
cat("The total number of complete sources is ", N_fully_obs,"\n")
cat("The total number of incomplete sources is ", N_potential-N_fully_obs,"\n")
cat("Use nodes of MPI, and multiprocessing threads accordingly\n")
