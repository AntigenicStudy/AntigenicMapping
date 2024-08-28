#!/usr/bin/env Rscript --vanilla
# rm(list=ls()); Sys.setenv(SLURM_ARRAY_TASK_ID=1, SLURM_CPUS_ON_NODE=1)

rm(list=ls());

library(argparse)
library(parallel)
library(tidyverse); theme_set(theme_minimal())
library(seqinr)

rawdir = ""
setwd(rawdir)
getwd()

parser = ArgumentParser(description = "fit substitution effect sizes using nnls")
parser$add_argument('train.stem')
parser$add_argument('-test.stem')
parser$add_argument('-combine.stem')
parser$add_argument('-outroot', default = "substitution_model")

idx_tarSeason = 6;

tarSeason_vec = c(
  "2005N", "2005S", "2006N", "2006S", "2007N", "2007S", 
  "2008N", "2009S", "2010N", "2010S", "2011N", "2011S" # "2009N", 
  )

tarSeason = tarSeason_vec[idx_tarSeason];

nm_strain = paste(rawdir, "00-RawData", tarSeason, sep = .Platform$file.sep)

nm_trainProp = "h3-train"
nm_testProp = "h3-test"
nm_combineProp = "h3-combine"

inputArg = parser$parse_args( c(
  paste(nm_strain, nm_trainProp, sep = .Platform$file.sep),
  '-test.stem',
  paste(nm_strain, nm_testProp, sep = .Platform$file.sep),
  '-combine.stem',
  paste(nm_strain, nm_combineProp, sep = .Platform$file.sep)
  ) )

write_csv_mkdir = function(x, outstem) {
  outfile = paste0(outstem, ".csv")
  dir.create(dirname(outfile), recursive = T)
  write_csv(x, outfile)
  }


# Train ......
aa = paste0(inputArg$train.stem, '.fa') %>% read.fasta
Dist = paste0(inputArg$train.stem, '.txt') %>% 
  read.table(row.names = names(aa)) %>%
  dist(method = "euclidean") %>%
  as.matrix

#   compute genetic feature changes
#   ...............................

# cluster positions with exactly the same information
subClust = data.frame(
    position = seq_along(aa[[1]]), 
    pattern = aa %>%    
        do.call(what = rbind) %>%
        apply(2, paste, collapse = ";")
    ) %>%
    group_by(pattern) %>%
    summarize(pos = list(position)) %>%
    mutate(clustNum = seq_along(pos))

# remove clusters with no diversity
subClust = subClust %>% filter( 
    (strsplit(subClust$pattern %>% as.character, ";") %>% sapply(function(a){ unique(a) %>% length })) > 1 
    ) %>%
    select(-pattern) %>%
    # representative position to use when extracting features
    # in pair-wise comparison
    mutate(pos1 = sapply(pos, function(a) a[1]))

# compute pair-wise changes
library(Matrix)

getFeatLong = function(aa){           
  names(aa) %>% combn(          
    2,        
    function(x) {           
      tibble(         
        virus = x,      
        serum = rev(x),           
        sub = mapply(         
          function(virus, serum){         
            featVirus = aa[[virus]][subClust$pos1]; # AA positions with genetic diversity       
            featSerum = aa[[serum]][subClust$pos1];         
            changed = (featVirus != featSerum);   
            # store substitution such that inverse substitutions degenerate to the same feature       
            paste0(         
              ifelse(featSerum[changed] < featVirus[changed], featSerum[changed], featVirus[changed]),  
              ifelse(featSerum[changed] >= featVirus[changed], featSerum[changed], featVirus[changed]),     
              subClust$clustNum[changed]    
              )
            },  
          virus = x,    
          serum = rev(x),
          SIMPLIFY = FALSE
        )   
      )
    }, simplify = F
  ) %>% do.call(what = rbind)
}
    
computeChange = function(aa) {      
  # make genetic feature matrix   
  uniqueFeatures = featLong$sub %>% unlist %>% unique;  
  iFeat = seq_along(uniqueFeatures);
  names(iFeat) = uniqueFeatures;
  features = sparseMatrix(
    i = rep(seq_along(featLong$sub), sapply(featLong$sub, length)),
    j = iFeat[unlist(featLong$sub)], 
    dims = c(nrow(featLong), length(uniqueFeatures))
    );
  colnames(features) = uniqueFeatures;
  features
}


featLong = getFeatLong(aa);
features = computeChange(aa);
    

# substitutions found in the training set
subTrain = colnames(features)[ (features %>% colSums) >= 2 ]

# store antigenic distance between pairs
distTrain = featLong %>%
  select(virus, serum) %>% mutate(
    D = apply(featLong, 1, function(x){ Dist[x[['virus']], x[['serum']]] })
    )



#   Fit substitution effects with nnls
#   ........................

A.make = function(feat, d){     
  out = feat[  , intersect(subTrain, colnames(feat)) ]  
  moreFeat = setdiff(subTrain, colnames(feat))    
  out.more = Matrix(0, nrow(out), length(moreFeat), sparse = T)
  colnames(out.more) = moreFeat   
  out = cbind(out, out.more)  
  out[ , subTrain] %>% cbind(
    lapply(dNames$virus, function(x) d$virus == x ) %>% 
    do.call(what = cbind)
  ) %>% cbind(
    lapply(dNames$serum, function(x) d$serum == x ) %>%
    do.call(what = cbind)
  )
}



  # strain names in the training set
  dNames = lapply(featLong %>% select(virus, serum), unique)

  # create objects to conform with my re-derived casting
  # of the quadratic problem in Neher,2016 in which
  # Q = (t(A) %*% A) + M
  # q = -HA + [ lambda, 0, 0 ]
  # where M is the diagonal matrix of 0, delta, gamma
  # ....................................................

  H = matrix(distTrain$D, nrow=1) # vector of measured titers

  A = A.make(features, distTrain)

  M = Matrix(0, ncol(A), ncol(A), sparse = T)
  diag(M) = c(    
    rep(0  , length(subTrain)),
    rep(0.6, length(dNames$virus)), # kappa = 0.6
    rep(1.2, length(dNames$serum))  # delta = 1.2
    )
  Q = (t(A) %*% A) + M

  q = (- H %*% A) + c(      
    rep(3, length(subTrain)),      # lambda = 3
    rep(0, length(dNames$virus) + length(dNames$serum))
    )

  library(lsei)
  # Lawson-Hanson implementation of an algorithm for 
  # non-negative least-squares
  # : allowing the combination of non-negative and non-positive constraints
  # : solves min || Ax − b ||2 with the constraint x ≥ 0
  fit = pnnqp(    
    q = Q %>% as.matrix,
    p = t(q) %>% as.matrix
    )
  
  est = split(
    fit$x, 
    c(
      rep('d', length(subTrain)),
      rep('v', length(dNames$virus)),
      rep('p', length(dNames$serum))
      )
    )
  names(est$d) = subTrain
  names(est$v) = dNames$virus
  names(est$p) = dNames$serum

  D_ab = A[, seq_along(subTrain)] %*% est$d;
  avidity_vec = A[, (length(subTrain) + 1):(length(subTrain) + length(dNames$virus))] %*% est$v;
  potency_vec = A[, (length(subTrain) + length(dNames$virus) + 1):ncol(A)] %*% est$p;
  
  predictDist = D_ab + avidity_vec + potency_vec;      
  
  #  predictDist = A %*% do.call(c, est)

  
  # write predictions
  df_distTrain = distTrain %>%   
    rename(Observed = D) %>% mutate(
      Predicted = as.numeric(predictDist),
      Dab = as.numeric(D_ab),
      avidity = as.numeric(avidity_vec),
      potency = as.numeric(potency_vec)
      ) 
  
  df_distTrain %>% write_csv_mkdir(file.path(inputArg$outroot, tarSeason, "distTrain_Dab"))  # nmFold,
  
  save(
    df_distTrain,
    file = paste(getwd(), inputArg$outroot, tarSeason, "distTrain_Dab.Rdata", sep = .Platform$file.sep)  # nmFold,
    )

  ## write GLM fitting parameters ------------------------------------

  data.frame(Feature = names(est$d), effect = est$d) %>%
    mutate(clustNum = gsub("[^0-9]", "", Feature) %>% as.integer) %>%
    left_join(
      subClust %>%
      select(-pos1) %>%
      unnest(cols = pos)
      , by = "clustNum"
    ) %>%
    arrange(desc(effect)) %>%
    write_csv(file.path(inputArg$outroot, tarSeason, "d_Dab.csv"))  # nmFold,

  data.frame(virus = names(est$v), avidity = est$v) %>%
    # arrange(desc(avidity)) %>%
    write_csv(file.path(inputArg$outroot, tarSeason, "v_Dab.csv")) # nmFold,

  data.frame(serum = names(est$p), potency = est$p) %>%
    # arrange(desc(potency)) %>%
    write_csv(file.path(inputArg$outroot, tarSeason, "p_Dab.csv")) # nmFold,

  
  
  
  
  
  
  
  
  # Test set ......

  # if there is no test set
  if(is.null(inputArg$combine.stem)) {   
    message("No test set specified. Exiting...")
    q(save = "no")
  }

  aa = paste0(inputArg$combine.stem, '.fa') %>% read.fasta
  Dist = paste0(inputArg$combine.stem, '.txt') %>% 
    read.table(row.names = names(aa)) %>%
    dist(method = "euclidean") %>%
    as.matrix
  
  featLong = getFeatLong(aa)
  features = computeChange(aa)
  
  # store antigenic distance between pairs
  distTest = featLong %>% 
    select(virus, serum) %>% mutate(
      D = apply(featLong, 1, function(x){ Dist[x[['virus']], x[['serum']]] })
      )

  # make predictions
  A.test = A.make(features, distTest)
  
  D_ab = A.test[, seq_along(subTrain)] %*% est$d;
  avidity_vec = A.test[, (length(subTrain) + 1):(length(subTrain) + length(dNames$virus))] %*% est$v;
  potency_vec = A.test[, (length(subTrain) + length(dNames$virus) + 1):ncol(A.test)] %*% est$p;
  
  predictDist = D_ab + avidity_vec + potency_vec;
  
#  predictDist = A.test %*% do.call(c,est)

  # average distance between any pairs due to virus/serum specific effects
  # baseDist = est[c('v','p')] %>% sapply(mean) %>% sum
  # write predictions
  df_distTest = distTest %>%  
    rename(Observed = D) %>% mutate(
      Predicted = as.numeric(predictDist),
      Dab = as.numeric(D_ab),
      avidity = as.numeric(avidity_vec),
      potency = as.numeric(potency_vec)
      ) # %>%  # + baseDist
  
  df_distTest %>% write_csv(file.path(inputArg$outroot, tarSeason, "distTest_Dab.csv")) # nmFold,
  
  save(
    df_distTest,
    file = paste(getwd(), inputArg$outroot, tarSeason, "distTest_Dab.Rdata", sep = .Platform$file.sep)  # nmFold,
    )
  


