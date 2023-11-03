########################################################################################################################
# Initialize: Libraries, functions, parameter
########################################################################################################################

rm(list = ls())


# --- Libraries, settings, functions -----------------------------------------------------------------------------------

library(tidyverse)
source("./code/init.R")
source("./code/utils_plots.R")


# --- Parameter --------------------------------------------------------------------------------------------------------

# Plot
PLOT = TRUE

# Constants
TARGET_TYPES = c("REGR", "CLASS", "MULTICLASS")
MISSPCT_THRESHOLD = 0.95
TOOMANY_THRESHOLD = 5


########################################################################################################################
# ETL
########################################################################################################################

# --- Read data and adapt  ---------------------------------------------------------------------------------------------

# Read data and adapt to be more readable
df_orig = read_csv(paste0(DATALOC, "hour.csv")) %>%
  mutate(season = recode(season, "1" = "1_winter", "2" = "2_spring", "3" = "3_summer", "4" = "4_fall"),         
         yr = recode(yr, "0" = "2011", "1" = "2012"),
         workingday = recode(workingday, "0" = "No", "1" = "Yes"),
         weathersit = recode(weathersit, "1" = "1_clear", "2" = "2_misty", "3" = "3_light rain",
                                         "4" = "4_heavy rain")) %>% 
  mutate(weekday = str_c(as.character(weekday), str_sub(lubridate::wday(dteday, label = TRUE), 1, 3), sep = "_"),
         mnth = str_pad(as.character(mnth), 2, pad = "0"),
         hr = str_pad(as.character(hr), 2, pad = "0")) %>% 
  mutate(temp = temp * 47 - 8,
         atemp = atemp * 66 - 16,
         windspeed = windspeed * 67) %>%
  mutate(kaggle_fold = if_else(lubridate::day(dteday) >= 20, "test", "train"))

# Create some artifacts helping to illustrate important concepts
set.seed(42)
df_orig = df_orig %>% 
  mutate(high_card = as.character(hum),  # high cardinality categorical variable
         weathersit = na_if(weathersit, "4_heavy rain"),  # some informative missings
         holiday = if_else(sample(10, nrow(df_orig), TRUE) == 1, NA_real_,
                           holiday),  # some random missings
         windspeed = na_if(windspeed, 0))  # 0 equals missing
summary(as.factor(df_orig$holiday))    

# Create artificial targets
df_orig = df_orig %>%
  mutate(cnt_REGR = log(cnt + 1),
         cnt_CLASS = factor(cut(cnt, breaks = c(-Inf, quantile(cnt, 0.8), Inf), 
                                labels = c("0_low", "1_high"))),
         cnt_MULTICLASS = factor(cut(cnt, breaks = c(-Inf, quantile(df_orig$cnt, c(0.8, 0.95)), Inf),
                                     labels = c("0_low", "1_high", "2_very_high"))))

# Check some stuff
skip = function() {
  
  map_chr(df_orig, class)
  df_orig %>% my_summary()
  
  par(mfrow = c(1, 3))
  hist(df_orig$cnt, breaks = 50)
  plot(ecdf(df_orig$cnt))
  hist(log(df_orig$cnt), breaks = 50)
  par(mfrow = c(1,1))
  
  #fig, ax = plt.subplots(1,1)
  #up.plot_feature_target(ax, df_orig["windspeed"], df_orig["cnt_CLASS"])
}

# "Save" orignial data
write_csv(df_orig, paste0(DATALOC, "df_orig.csv"))
df = df_orig


# --- Get metadata information -----------------------------------------------------------------------------------------

df_meta = readxl::read_excel(paste0(DATALOC, "datamodel_bikeshare.xlsx"), skip = 1)

# Check difference of metainfo to data
setdiff(colnames(df), df_meta$variable)
setdiff(df_meta %>% filter(category == "orig") %>% pull("variable"), colnames(df))

# Subset on data that is "ready" to get processed
df_meta_sub = df_meta %>% filter(status %in% c("ready"))


# Feature engineering -------------------------------------------------------------------------------------------------

df$day_of_month = str_pad(lubridate::day(df$dteday), 2, pad = "0")

# Check metadata again
setdiff(df_meta_sub$variable, colnames(df))


# --- Define train/test/util-fold --------------------------------------------------------------------------------------
set.seed(42)
df = df %>% mutate(fold = factor(if_else(kaggle_fold == "train",
                                  if_else(sample(10, nrow(.), TRUE) == 1, "util", "train"),
                                  kaggle_fold)))
summary(as.factor(df$fold))


########################################################################################################################
# Numeric features: Explore and adapt
########################################################################################################################

# --- Define numeric features ------------------------------------------------------------------------------------------

nume = df_meta_sub %>% filter(type == "nume")  %>% pull("variable")
df[nume] = map(df[nume], ~ as.numeric(.))
my_summary(df[nume])




# --- Missings + Outliers + Skewness -----------------------------------------------------------------------------------

# Remove features with too many missings
misspct = round(map_dbl(df[nume], ~ mean(is.na(.))), 3)
sort(misspct, TRUE)
(remove = names(misspct[misspct > 0.99]))
nume = setdiff(nume, remove)

# Plot untransformed features -> check for outliers and skewness
my_summary(df[nume])
start = Sys.time()
for (TARGET_TYPE in TARGET_TYPES) {
  if (PLOT) {
    plots = map(nume, ~ plot_feature_target(df, feature_name = .x, target_name = paste0("cnt_", TARGET_TYPE), 
                                            title = .x))
    ggsave(paste0(PLOTLOC, "1__distr_nume_orig__", TARGET_TYPE, ".pdf"), 
           arrange_plots(plots, n_cols = 4, n_rows = 2), width = 18, height = 12)
  }
}
print(Sys.time() - start)

# Winsorize
df[nume] = map_df(df[nume], ~ winsorize(., 0.01, 0.99))

# Log-Transform
tolog = c("hum")
df[paste0(tolog,"_LOG")] = map(df[tolog], ~ {if(min(., na.rm=TRUE) == 0) log(. + min(., na.rm=TRUE)) else log(.)})
nume = map_chr(nume, ~ ifelse(. %in% tolog, paste0(.,"_LOG"), .)) #adapt metadata (keep order)


# --- Create categorical (binned) equivalents for all numeric features (for linear models to mimic non-linearity) ------

# Bin variables
nume_binned = paste0(nume,"_BINNED")
df[nume_binned] = map_df(df[nume], ~ bin(.))
  
# Convert missings to own level ("(Missing)")
df[nume_binned] = map_df(df[nume_binned], ~ replace_na(., "(Missing)"))  # fct_explicit_na
my_summary(df[nume_binned])

# Get binned variables with just 1 bin (removed later)
(onebin = nume_binned[map_lgl(df[nume_binned], ~ length(unique(.)) == 1)])


# --- Final feature information ----------------------------------------------------------------------------------------

for (TARGET_TYPE in TARGET_TYPES) {
  #TARGET_TYPE = "CLASS"
  
  # Univariate correlation with target
  corr_nume = unlist(map(c(rbind(nume, nume_binned)), 
                         ~ feature_target_correlation(df, 
                                                      feature_name = .x, target_name = paste0("cnt_", TARGET_TYPE),
                                                      calc_cramersv = FALSE)))
  print(corr_nume[order(corr_nume, decreasing = TRUE)])
  
  # Plot
  if (PLOT) {
    plots = map(c(rbind(nume, nume_binned)), 
                ~ plot_feature_target(df, 
                                      feature_name = .x, 
                                      target_name = paste0("cnt_", TARGET_TYPE), 
                                      title = paste0(.x, " (Corr: ",format(corr_nume[.x], digits = 2), ")")))
    # add_miss_info=True if feature in nume else False))
    ggsave(paste0(PLOTLOC, "1__distr_nume__", TARGET_TYPE, ".pdf"),
           arrange_plots(plots, n_cols = 4, n_rows = 2), width = 18, height = 12)
  }
}


# --- Removing variables (inlcuding correlation analysis) --------------------------------------------------------------

# Remove leakage features
remove = c("xxx", "xxx")
nume = setdiff(nume, remove)

# Remove highly/perfectly correlated (the ones with less NA!)
my_summary(df[nume])
ggsave(paste0(PLOTLOC, "1__corr_nume.pdf"), 
       plot_corr(df[nume], method = "spearman", cutoff = 0),
       width = 9, height = 9)
remove = c("atemp") #put at xxx the variables to remove
nume = setdiff(nume, remove) #remove
nume_binned = setdiff(nume_binned, paste0(remove,"_BINNED")) #keep "binned" version in sync


# --- Detect data drift (time/fold depedency of features) --------------------------------------------------------------

# HINT: In case of having a detailed date variable, this can be used as regression target here as well!

# Univariate variable correlation
corr_nume_fold = unlist(map(nume, 
                       ~ feature_target_correlation(df, feature_name = .x, target_name = "fold", 
                                                    calc_cramersv = FALSE)))
print(corr_nume_fold[order(corr_nume_fold, decreasing = TRUE)])

# Plot: only variables with with highest correlation
nume_toplot = names(corr_nume_fold)[corr_nume_fold >= 0.05]
plots = map(nume_toplot, 
            ~ plot_feature_target(df,
                                  feature_name = .x, 
                                  target_name = "fold", 
                                  title = paste0(.x, " (CORR: ",format(corr_nume_fold[.x], digits = 2), ")")))
ggsave(paste0(PLOTLOC, "1__distr_nume_folddep.pdf"),
       arrange_plots(plots, n_cols = 4, n_rows = 2), width = 18, height = 12)


# --- Create missing indicator and impute feature missings--------------------------------------------------------------

(miss = nume[map_lgl(df[nume], ~ any(is.na(.)))])
df[paste0("MISS_",miss)] = map(df[miss], ~ ifelse(is.na(.x), "No", "Yes"))
df[paste0("MISS_",miss)] %>% my_summary()

# Impute missings
df[miss] = map(df[miss], ~ impute(., type = "median"))
df[nume] %>% my_summary()


########################################################################################################################
# Categorical features: Explore and adapt
########################################################################################################################

# --- Define categorical features --------------------------------------------------------------------------------------

cate = df_meta_sub %>% filter(type == "cate") %>% pull("variable")
df[cate] = map(df[cate], ~ as.character(.)) 
my_summary(df[cate])


# Handling factor values ----------------------------------------------------------------------------------------------

# Map missings to own level
df[cate] = map_df(df[cate], ~ replace_na(., "(Missing)"))
my_summary(df[cate])

# Create ordinal and binary-encoded features
# ATTENTION: Usually this processing needs special adaption depending on the data
ordi = c("hr", "day_of_month", "mnth", "yr")
df[paste0(ordi, "_ENCODED")] = map(df[ordi], ~ as.numeric(.))
yesno = c("workingday", paste0("MISS_", miss))  # binary features
df[paste0(yesno, "_ENCODED")] = map(df[yesno], ~ as.numeric(recode(., "No" = "0", "Yes" = "1")))

# Create target-encoded features for nominal variables
nomi = setdiff(cate, c(ordi, yesno))
df[paste0(nomi, "_ENCODED")] = 
    map(nomi, ~ (df %>% filter(fold == "util") %>% 
                    group_by_at(.x) %>% summarise(mean_target = mean(.data[["cnt_REGR"]])) %>% 
                    ungroup() %>% arrange(desc(mean_target)) %>% 
                    right_join(df[.x] %>% mutate(n = row_number())) %>% 
                    arrange(n) %>% .$mean_target))

# Create compact covariates for "too many members" columns 
(levinfo = map_int(df[nomi], ~ length(unique(.))) %>% .[order(., decreasing = TRUE)]) #number of levels
(toomany = names(levinfo)[which(levinfo > TOOMANY_THRESHOLD)])
(toomany = setdiff(toomany, c("hr", "mnth", "weekday"))) #set exception for important variables
df[toomany] = map(df[toomany], ~ fct_lump(., TOOMANY_THRESHOLD, other_level = "_OTHER_")) #collapse


# --- Final variable information ---------------------------------------------------------------------------------------
options(dplyr.summarise.inform = FALSE)
for (TARGET_TYPE in TARGET_TYPES) {

  # Univariate correlation with target
  corr_cate = unlist(map(c(cate, paste0("MISS_", miss)), 
                         ~ feature_target_correlation(df, 
                                                      feature_name = .x, target_name = paste0("cnt_", TARGET_TYPE),
                                                      calc_cramersv = FALSE)))
  print(corr_cate[order(corr_cate, decreasing = TRUE)])
  
  # Plot
  if (PLOT) {
    plots = map(c(cate, paste0("MISS_", miss)), 
                ~ plot_feature_target(df, 
                                      feature_name = .x, 
                                      target_name = paste0("cnt_", TARGET_TYPE), 
                                      title = paste0(.x, " (Corr: ",format(corr_cate[.x], digits = 2), ")")))
    ggsave(paste0(PLOTLOC, "1__distr_cate__", TARGET_TYPE, ".pdf"),
           arrange_plots(plots, n_cols = 3, n_rows = 2), width = 18, height = 12)
  }
}


# Removing variables ----------------------------------------------------------------------------------------------

# Remove leakage variables
remove = c("xxx", "xxx")
cate = setdiff(cate, remove)
toomany = setdiff(toomany, remove)

# Remove highly/perfectly correlated (the ones with less levels!)
ggsave(paste0(PLOTLOC, "1__corr_cate.pdf"), 
       plot_corr(df[c(cate, paste0("MISS_", miss))], method = "cramersv", cutoff = 0),
       width = 9, height = 9)
remove = c("xxx") #put at xxx the variables to remove
cate = setdiff(cate, remove) #remove


# Time/fold depedency --------------------------------------------------------------------------------------------

# HINT: In case of having a detailed date variable, this can be used as regression target here as well!
# Univariate variable importance
corr_cate_fold = unlist(map(c(cate, paste0("MISS_", miss)), 
                            ~ feature_target_correlation(df, feature_name = .x, target_name = "fold")))
print(corr_cate_fold[order(corr_cate_fold, decreasing = TRUE)])

# Plot: only variables with with highest importance
cate_toplot = names(corr_cate_fold)[corr_cate_fold >= 0.5]
plots = map(cate_toplot, ~ plot_feature_target(df,
                                               feature_name = .x, 
                                               target_name = "fold", 
                                               title = paste0(.x, " (Corr: ",format(corr_cate_fold[.x], 
                                                                                  digits = 2), ")")))
ggsave(paste0(PLOTLOC, "1__distr_cate_folddep.pdf"),
       arrange_plots(plots, n_cols = 4, n_rows = 2), width = 18, height = 12)



########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Define final features --------------------------------------------------------------------------------------------

# Standard: can be used together by all algorithms
nume_standard = c(nume, paste0(toomany, "_ENCODED"))
cate_standard = c(cate, paste0("MISS_", miss))

# Binned: can be used by elasticnet (without any numeric features) to mimic non-linear numeric effects
features_binned = c(setdiff(paste0(nume, "_BINNED"), onebin), cate)

# Encoded: can be used as complete feature-set for deeplearning (as bad with one-hot encoding)
# or lightgbm (with additionally denoting encoded features as "categorical")
features_encoded = unique(c(nume, paste0(cate, "_ENCODED"), paste0("MISS_", miss, "_ENCODED")))

# Check again
all_features = unique(c(nume_standard, cate_standard, features_binned, features_encoded))
setdiff(all_features, colnames(df))
setdiff(colnames(df), all_features)


# --- Remove "burned" data -----------------------------------------------------------------------------------------------

df = df %>% filter(fold != 'util')


# --- Save image -------------------------------------------------------------------------------------------------------

# Clean up
graphics.off()
rm(df_orig)

# Serialize
save("df", "nume_standard", "cate_standard", "features_binned", "features_encoded",
     file = paste0(DATALOC, "1_explore.Rdata"))

