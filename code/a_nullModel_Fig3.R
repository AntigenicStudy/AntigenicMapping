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
parser$add_argument('-newSeason.stem')
parser$add_argument('-lastSeason.stem')
parser$add_argument('-history.stem')

parser$add_argument('-outroot', default = "substitution_model")

idx_tarSeason = 1;

tarSeason_vec = c(
  "2005N", "2005S", "2006N", "2006S", "2007N", "2007S", 
  "2008N", "2009S", "2010N", "2010S", "2011N", "2011S" # "2009N", 
  )

tarSeason = tarSeason_vec[idx_tarSeason];

nm_strain = paste(rawdir, "00-RawData", tarSeason, sep = .Platform$file.sep)

nm_trainProp = "h3-train"
nm_testProp = "h3-test"
nm_combineProp = "h3-combine"

nm_newSeason  = "h3-newSeason"
nm_lastSeason = "h3-lastSeason"
nm_history = "h3-history"


inputArg = parser$parse_args( c(
  paste(nm_strain, nm_trainProp, sep = .Platform$file.sep),
  '-test.stem',
  paste(nm_strain, nm_testProp, sep = .Platform$file.sep),
  '-combine.stem',
  paste(nm_strain, nm_combineProp, sep = .Platform$file.sep),
  '-newSeason.stem',
  paste(nm_strain, nm_newSeason, sep = .Platform$file.sep),
  '-lastSeason.stem',
  paste(nm_strain, nm_lastSeason, sep = .Platform$file.sep),
  '-history.stem',
  paste(nm_strain, nm_history, sep = .Platform$file.sep)
  ) )

write_csv_mkdir = function(x, outstem) {
  outfile = paste0(outstem, ".csv")
  dir.create(dirname(outfile), recursive = T)
  write_csv(x, outfile)
  }


## New season: Seq, coordinates ------------------------------

aa_newSeason    = paste0(inputArg$newSeason.stem, '.fa' ) %>% read.fasta
coord_newSeason_data = read_delim(
  paste0(inputArg$newSeason.stem, '.txt'),
  col_types = cols(
    Coord1 = col_double(), 
    Coord2 = col_double()
    ),
  delim = " "
  ) %>% mutate(   
    ag.name = names(aa_newSeason)
    )
  

## Old season: Seq, coordinates ------------------------------

aa_lastSeason = paste0(inputArg$lastSeason.stem, '.fa') %>% read.fasta
coord_lastSeason_data = read_delim(
  paste0(inputArg$lastSeason.stem, '.txt'),
  col_types = cols(
    Coord1 = col_double(), 
    Coord2 = col_double()
  ),
  delim = " "
  ) %>% mutate(
    ag.name = names(aa_lastSeason)
    )


coord_newSeason_null2D = sapply(
  coord_newSeason_data$ag.name,
  function(newSeason_agName, df_lastSeason, aa_lastSeason, aa_newSeason) {
    hamm_vec = sapply(
      df_lastSeason$ag.name,
      function(lastSeason_agName, aa_lastSeason, aa_newSeason) {
        oldVirus = aa_lastSeason[[lastSeason_agName]];
        newVirus = aa_newSeason[[newSeason_agName]];
        
        changed = (newVirus != oldVirus);
        return(length( which(changed) ))
      },
      aa_lastSeason = aa_lastSeason,
      aa_newSeason = aa_newSeason
    )
    
    rIdx_tar = unname(which(hamm_vec == min(hamm_vec)));
    # return(rIdx_tar)
    
    coord_1D = c( mean(df_lastSeason$Coord1[rIdx_tar]), mean(df_lastSeason$Coord2[rIdx_tar]) );
    
    # coord_1D = c( df_lastSeason$Coord1[sample(rIdx_tar, 1)], df_lastSeason$Coord2[sample(rIdx_tar, 1)] );
    return(coord_1D)
  },
  df_lastSeason = coord_lastSeason_data,
  aa_lastSeason = aa_lastSeason,
  aa_newSeason  = aa_newSeason  
  )


## data.frame for nowcast and history viruses

df_Ag_coord_newSeason <- tibble(
  ag.name = coord_newSeason_data$ag.name,
  Coord1_raw = coord_newSeason_data$Coord1,
  Coord2_raw = coord_newSeason_data$Coord2,
  Coord1_null = coord_newSeason_null2D[1, ],
  Coord2_null = coord_newSeason_null2D[2, ] 
  )

n_Ag_newSeason = nrow(df_Ag_coord_newSeason)
  
aa_history = paste0(inputArg$history.stem, '.fa') %>% read.fasta

df_Ag_coord_history = read_delim(
  paste0(inputArg$history.stem, '.txt'),
  col_types = cols(
    Coord1 = col_double(), 
    Coord2 = col_double()
    ),
  delim = " " 
  ) %>% mutate( 
    ag.name = names(aa_history)
    )

n_Ag_history = nrow(df_Ag_coord_history)

distTest_old2newVirus <- tibble(
  oldVirus = character(),
  newVirus = character(),
  Observed = double(),
  Predicted = double()
  )


Get_EucDist <- function(Coord1_new, Coord2_new, Coord1_old, Coord2_old) {
  return( sqrt( (Coord1_new - Coord1_old)^2 + (Coord2_new - Coord2_old)^2 ) )
}



for (newVirus_ID in 1:n_Ag_newSeason) { 
  for (oldVirus_ID in 1:n_Ag_history) {
    
    Observed = Get_EucDist(
      df_Ag_coord_newSeason$Coord1_raw[newVirus_ID],
      df_Ag_coord_newSeason$Coord2_raw[newVirus_ID],
      df_Ag_coord_history$Coord1[oldVirus_ID],
      df_Ag_coord_history$Coord2[oldVirus_ID]
      )  
    
    Predicted = Get_EucDist(
      df_Ag_coord_newSeason$Coord1_null[newVirus_ID],
      df_Ag_coord_newSeason$Coord2_null[newVirus_ID],
      df_Ag_coord_history$Coord1[oldVirus_ID],
      df_Ag_coord_history$Coord2[oldVirus_ID]
      )
    
    distTest_old2newVirus <- bind_rows(
      distTest_old2newVirus,
      tibble(
        oldVirus = df_Ag_coord_history$ag.name[oldVirus_ID],
        newVirus = df_Ag_coord_newSeason$ag.name[newVirus_ID],
        Observed = Observed,
        Predicted = Predicted       
        )
      )
  }
}



distTest_old2newVirus %>% write_csv_mkdir(
  file.path(inputArg$outroot, tarSeason, "distTest_old2newVirus_Null_meanLoc")
  )

save(
  distTest_old2newVirus,
  file = paste(getwd(), inputArg$outroot, tarSeason, "distTest_old2newVirus_Null_meanLoc.Rdata", sep = .Platform$file.sep)  
  )





