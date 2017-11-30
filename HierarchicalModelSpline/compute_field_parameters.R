#==============================================================================
#======== Loading Libraries =======
require(mclust)
# require(pmclust)
require(mix)
require(ggplot2)

# Numbers of stars to compute field parameters
N_nm <- 4e4

dir       <- "/pcdisk/boneym5/jolivares"
dir_f     <- "/pcdisk/boneym5/jolivares/m45py/Analysis/Functions"
# Directory where output data will be stored
dir_out   <- file.path(dir,"Data")
ffield    <- file.path(dir_out,"FieldParametersNonMiss-3e+05-28G.RData")
name_plot_ori <- paste('Probability-3e+05-28G-ori.pdf',sep='')
name_plot_sim <- paste('Probability-3e+05-28G-sim.pdf',sep='')
fnonmem   <- file.path(dir_out,"FieldSources.RData")
dir.create(dir_out,showWarnings=F)

# Directory where input data is stored
dataset <- "/pcdisk/kool5/scratch-lsb/Data_lsb"
fincomplete <- file.path(dataset,'incomplete.RData')
fpreds <- file.path(dataset,'full-preds.RData')
fiter6 <- file.path(dataset,'iter6.RData')
fmembers <- file.path(dataset,'miembros.RData')
funique <- file.path(dataset,'All_unique.RData')
#===================== Names =================================
names.sarro  <- c("pmRA","pmDEC","K","J","H","i_K","Y-J")
names.stf    <- c("pmRA","pmDEC","K","J","H","r_K","i_K","Y-J")
names.out    <- c("pmRA","pmDEC","J","H","K","Y","i_K","Y-J","J-K","Y-H")
names.data   <- c("source_id","pmRA","pmDEC","K","J","H","i_K","Y-J","class","prob")
names.missing<- c("pmRA","pmDEC","K","J","H","i_K","Y-J")
#===================== Constants ==================
N_mm <- 1741  # 1741 Number of known members in data set

#============= Field ===============================
span.gauss <- c(4:6)# Max number of multivariate normals for field
rg_clr <- c(0,10)
clr_par  <- 'i_K'
#=================================
####################################################################
# attach(fpreds,name='probs')
# pro_all <- p.m.miss
# detach('probs')
# attach(fincomplete,name='missing')
# all     <- missing
# u_all   <- uncert
# detach('missing')
# colnames(all)       <- names.sarro
# colnames(u_all)     <- c("pmRA","pmDEC","K","J","H","i","Y")
# Y_all   <- all[,c("Y-J")]  +  all[,c("J")]
# all     <- cbind(all,"Y"=Y_all)

# i_all   <- all[,c("i_K")]  +  all[,c("K")]
# all     <- cbind(all,"i"=i_all)

# JK_all  <- all[,c("J")]    -  all[,c("K")]
# all     <- cbind(all,"J-K"=JK_all)

# YH_all  <- all[,c("Y")]    -  all[,c("H")]
# all     <- cbind(all,"Y-H"=YH_all)

# YK_all  <- all[,c("Y")]    -  all[,c("K")]
# all     <- cbind(all,"Y-K"=YK_all)

# cat("Total data set: ",dim(all),"\n")
# #--remove duplicates in non-members
# valid     <- !duplicated(all)
# all       <- all[valid,]
# uncert_all<- u_all[valid]
# pro_all   <- pro_all[valid]
# save(all,pro_all,uncert_all,file=funique)
# cat("Total data set without duplicateds: ",dim(all)[1],"\n")
# stop()
################################################
#======== LOAD PREVIOUSLY KNOWN MEMBERS ================
attach(fmembers,name='sfl')    # Obs of Stauffer list. 
members_1 <- as.matrix(miembros)
detach('sfl')
 attach(fiter6,name='members_sarro')
 members_2      <- ts1.members
 detach('members_sarro')
#------ Establish names of datasets-------
colnames(members_1) <- names.stf
colnames(members_2) <- names.sarro
#========== Y-J to Y ======================
Y_mem_1      <- members_1[,c("Y-J")]  +  members_1[,c("J")]
Y_mem_2      <- members_2[,c("Y-J")]  +  members_2[,c("J")]

members_1  <- cbind(members_1,"Y"=Y_mem_1)
members_2  <- cbind(members_2,"Y"=Y_mem_2)
#========== J-K  ======================
JK_mem_1      <- members_1[,c("J")]  -  members_1[,c("K")]
JK_mem_2      <- members_2[,c("J")]  -  members_2[,c("K")]

members_1  <- cbind(members_1,"J-K"=JK_mem_1)
members_2  <- cbind(members_2,"J-K"=JK_mem_2)
#========== Y-H  ======================
YH_mem_1      <- members_1[,c("Y")]  -  members_1[,c("H")]
YH_mem_2      <- members_2[,c("Y")]  -  members_2[,c("H")]

members_1  <- cbind(members_1,"Y-H"=YH_mem_1)
members_2  <- cbind(members_2,"Y-H"=YH_mem_2)

#========== Y-K ======================
YK_mem_1      <- members_1[,c("Y")]  -  members_1[,c("K")]
YK_mem_2      <- members_2[,c("Y")]  -  members_2[,c("K")]

members_1  <- cbind(members_1,"Y-K"=YH_mem_1)
members_2  <- cbind(members_2,"Y-K"=YH_mem_2)

# Select desired quantities
members_1  <- members_1[,names.out]
members_2  <- members_2[,names.out]

#===============MIX======================
mem      <- rbind(members_1,members_2)
mem      <- mem[!duplicated(mem),]
###########################################################
#=============== all data =================================
load(funique)
all     <- all[,names.out]
##################### Members missing data ################
indx_mm_miss <- which(pro_all > 0.75)
mem_all      <- all[indx_mm_miss,]
pro_mem_all  <- pro_all[indx_mm_miss]
valid   <- !(mem_all[,1]%in%mem[,1] & 
             mem_all[,2]%in%mem[,2] &
             mem_all[,3]%in%mem[,3] &
             mem_all[,4]%in%mem[,4] &
             mem_all[,5]%in%mem[,5] &
             mem_all[,6]%in%mem[,6] &
             mem_all[,7]%in%mem[,7])
mem_inc      <- mem_all[valid,]
pro_mem_inc  <- pro_mem_all[valid]
###############################################################
#--remove members in non-members
valid    <- which(pro_all <= 0.75)
nm_obs   <- all[valid,]
pro_nm   <- pro_all[valid]
cat("Non-members: ",dim(nm_obs)[1],"\n")
#---remove stars with missing pm-------
indicator <- is.na(nm_obs)*1
valid     <- !(indicator[,1]==1 | indicator[,2]==1)
pro_nm    <- pro_nm[valid]
nm_obs    <- nm_obs[valid,]
indicator <- NULL 
cat("Non-members with two pm entries: ",dim(nm_obs)[1],"\n")
#---remove stars with missing phot, at least two entries present
#(one could be tractable but functions in python (e.g. np.linalg.inv) dont work)
indicator <- is.na(nm_obs)*1
valid     <- !(rowSums(indicator[,3:7])==5 | rowSums(indicator[,3:7])==4 )
pro_nm    <- pro_nm[valid]
nm_obs    <- nm_obs[valid,]
cat("Non-members with at least two photometric entries: ",dim(nm_obs)[1],"\n")
#-----------------------------------------------------------------------------
# Remove stars with color index less or grater than color range. IF IT IS PRESENT!
indicator  <- which(!is.na(nm_obs[,clr_par])) # Observations with color index
nonvalid   <- ((nm_obs[indicator,clr_par]< rg_clr[1]) | (nm_obs[indicator,clr_par]> rg_clr[2]))
nm_obs     <- nm_obs[-c(indicator[nonvalid]),]
pro_nm     <- pro_nm[-c(indicator[nonvalid])]
cat("Non-members inside color range: ",dim(nm_obs)[1],"\n")

nonmem   <- cbind(nm_obs,"pro"=pro_nm)

#----- select field acording to probability
# ord_nm     <- order(pro_nm,decreasing=T)[1:N_nm]
# pro_samp   <- pro_nm[ord_nm]
# nm_samp    <- nm_obs[ord_nm,]
# cat("The probability threshold is: ",min(pro_nm),"\n")

#----- select randomly -----
sam      <- sample.int(dim(nm_obs)[1],N_nm)
pro_samp <- pro_nm[sam]
nm_samp  <- nm_obs[sam,]
################################################################
load(ffield)
names.field <- c("pmRA","pmDEC","J","H","K","Y","i_K")
colnames(mu_nm) <- names.field
dimnames(sigma_nm) <- list(names.field,names.field,NULL)
source(file.path(dir_f,'plotcov.R'))


######################### FIELD ############################
#=============== Compute Field parameters using pmclust ========
# cnm      <- pmclust(nm,K=20)
# comm.print(cnm)
# save(cnm,file=file.path(dir_out,"FieldParameters.RData"))
#=============== Compute PM Field parameters using mclust ======== 
# fpmp         <- Mclust(nm_samp[,c(1,2)],30,"VVV")
# theta_nm_pm  <- fpmp$parameters$pro
# M_pm         <- length(theta_nm_pm)
# mu_nm_pm     <- t(fpmp$parameters$mean)
# sigma_nm_pm  <- fpmp$parameters$variance$sigma 
# print(fpmp$BIC)
# save(M_pm,theta_nm_pm,mu_nm_pm,sigma_nm_pm,file=fpar_pm)
# cat("The optimal number of gaussians for proper motion data is: ",M_pm,"\n")
#=============== Compute non-missing Field photometry parameters using mclust ======== 
# ind_comp        <- which(rowSums(is.na(nm_samp[,3:7])*1) == 0)
# fphotp_c        <- Mclust(nm_samp[ind_comp,3:7],1:5,"VVV")
# theta_nm_phot_c <- fphotp_c$parameters$pro
# M_phot_c        <- length(theta_nm_phot_c)
# mu_nm_phot_c    <- t(fphotp_c$parameters$mean)
# sigma_nm_phot_c <- fphotp_c$parameters$variance$sigma 
# cat("The optimal number of gaussians for complete photometric data is: ",M_phot_c,"\n")
# print(fphotp_c$BIC)
# # #=============== Compute Field parameters using MclustMissing ========
# source("GMM-Missing.R")
# fphotp        <- MclustMissing(nm_samp[,3:7],4:8,tol=1e-4,modelNames="VVV",maxiter=1000)
# theta_nm_phot <- fphotp$parameters$pro
# M_phot        <- length(theta_nm_phot)
# mu_nm_phot    <- t(fphotp$parameters$mean)
# sigma_nm_phot <- fphotp$parameters$variance$sigma
# cat("The optimal number of gaussians for missing photometric data is: ",M_phot,"\n")

source(file.path(dir_f,'plotcov.R'))
pdf(file=file.path(dir_out,name_plot_ori))
source(file.path(dir_f,'plot_field.R'))
dev.off()

require(MASS)

# compute the number of complete sources
N_nm_c <- sum(rowSums(is.na(nm_obs)) == 0)

# Create simulated dataset with the same number of complete sources
n <- rep(NA,M)
data_sim <- nm_samp[1,1:7]
for (i in 1:M) data_sim <- rbind(data_sim,
								mvrnorm(floor(theta_nm[i]*N_nm_c),
										mu_nm[i,],
										sigma_nm[,,i]))

nm_obs     <- data_sim[-c(1),]

source(file.path(dir_f,'plotcov.R'))
pdf(file=file.path(dir_out,name_plot_sim))
source(file.path(dir_f,'plot_field.R'))
dev.off()

# save(M_pm,M_phot,
# 	theta_nm_pm,theta_nm_phot,
# 	mu_nm_pm,mu_nm_phot,
# 	sigma_nm_pm,sigma_nm_phot,
# 	file=file.path(dir_out,fparameters))

