########################################################################################################################
# Initialize: Libraries, functions, parameter
########################################################################################################################

rm(list = ls())


# --- Libraries --------------------------------------------------------------------------------------------------------

library(tidyverse)
library(lubridate)
library(gridExtra)

# Custom
options(box.path = getwd())
box::use(
  init = ./init,
  up = ./utils_plots, # box::reload(up)
)


# --- Parameter --------------------------------------------------------------------------------------------------------

# Plot
PLOT = TRUE

# Constants
TARGET_TYPES = c("REGR", "CLASS", "MULTICLASS")
MISSPCT_THRESHOLD = 0.95
VARPERF_THRESHOLD_DATADRIFT = 0.53
TOOMANY_THRESHOLD = 5
'
# Adapt some default parameter different for target types -> probably also different for a new use-case
# color = switch(TYPE, "CLASS" = twocol, "REGR" = hexcol, "MULTICLASS" = threecol) #probably need to change MULTICLASS opt
# cutoff = switch(TYPE, "CLASS" = 0.1, "REGR"  = 0.9, "MULTICLASS" = 0.9) #need to adapt
# ylim = switch(TYPE, "CLASS" = NULL, "REGR"  = c(0,2.5e2), "MULTICLASS" = NULL) #need to adapt in regression case
# min_width = switch(TYPE, "CLASS" = 0, "REGR"  = 0, "MULTICLASS" = 0.2) # need to adapt in multiclass case
'

########################################################################################################################
# ETL
########################################################################################################################

# --- Read data and adapt  ---------------------------------------------------------------------------------------------

# Read data and adapt to be more readable
df_orig = read_csv(paste0(init$DATALOC, "hour.csv")) %>%
  mutate(season = recode(season, "1" = "1_winter", "2" = "2_spring", "3" = "3_summer", "4" = "4_fall"),         
         yr = recode(yr, "0" = "2011", "1" = "2012"),
         workingday = recode(workingday, "0" = "No", "1" = "Yes"),
         weathersit = recode(weathersit, "1" = "1_clear", "2" = "2_misty", "3" = "3_light rain",
                                         "4" = "4_heavy rain")) %>% 
  mutate(weekday = str_c(as.character(weekday), str_sub(wday(dteday, label = TRUE), 1, 3), sep = "_"),
         mnth = str_pad(as.character(mnth), 2, pad = "0"),
         hr = str_pad(as.character(hr), 2, pad = "0")) %>% 
  mutate(temp = temp * 47 - 8,
         atemp = atemp * 66 - 16,
         windspeed = windspeed * 67) %>%
  mutate(kaggle_fold = if_else(day(dteday) >= 20, "test", "train"))

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
         cnt_CLASS = cut(cnt, breaks = c(-Inf, quantile(cnt, 0.8), Inf), 
                              labels = c("0_low", "1_high")),
         cnt_MULTICLASS = cut(cnt, breaks = c(-Inf, quantile(df_orig$cnt, c(0.8, 0.95)), Inf),
                                   labels = c("0_low", "1_high", "2_very_high")))

# Check some stuff
skip = function() {
  
  map_chr(df_orig, class)
  df_orig %>% mutate(across(where(is.character), as.factor))  %>% summary()
  
  par(mfrow = c(1, 3))
  hist(df_orig$cnt, breaks = 50)
  plot(ecdf(df_orig$cnt))
  hist(log(df_orig$cnt), breaks = 50)
  par(mfrow = c(1,1))
  
  #fig, ax = plt.subplots(1,1)
  #up.plot_feature_target(ax, df_orig["windspeed"], df_orig["cnt_CLASS"])
}

# "Save" orignial data
write_csv(df_orig, paste0(init$DATALOC, "df_orig.csv"))
df = df_orig


# --- Get metadata information -----------------------------------------------------------------------------------------

df_meta = readxl::read_excel(paste0(init$DATALOC, "datamodel_bikeshare.xlsx"), skip = 1)

# Check difference of metainfo to data
setdiff(colnames(df), df_meta$variable)
setdiff(df_meta %>% filter(category == "orig") %>% .$variable, colnames(df))

# Subset on data that is "ready" to get processed
df_meta_sub = df_meta %>% filter(status %in% c("ready"))


# Feature engineering -------------------------------------------------------------------------------------------------

df$day_of_month = str_pad(day(df$dteday), 2, pad = "0")

# Check metadata again
setdiff(df_meta_sub$variable, colnames(df))


# --- Define train/test/util-fold --------------------------------------------------------------------------------------
set.seed(42)
df = df %>% mutate(fold = if_else(kaggle_fold == "train",
                                  if_else(sample(10, nrow(.), TRUE) == 1, "util", "train"),
                                  kaggle_fold
))
summary(as.factor(df$fold))


########################################################################################################################
# Numeric features: Explore and adapt
########################################################################################################################

# --- Define numeric features ------------------------------------------------------------------------------------------

nume = df_meta_sub %>% filter(type == "nume")  %>% .$variable
df[nume] = map(df[nume], ~ as.numeric(.))
summary(df[nume])


# --- Missings + Outliers + Skewness -----------------------------------------------------------------------------------

# Remove features with too many missings
misspct = round(map_dbl(df[nume], ~ mean(is.na(.))), 3)
sort(misspct, TRUE)
(remove = names(misspct[misspct > 0.99]))
nume = setdiff(nume, remove)

# Plot untransformed features -> check for outliers and skewness
summary(df[nume])
start = Sys.time()
if (PLOT) {
  plots = map(nume, 
              ~ up$get_plot_nume_CLASS(df[c(., "cnt_CLASS")], 
                                       feature_name = .x, 
                                       target_name = "cnt_CLASS", 
                                       title = .x
  ))
  ggsave(paste0(init$PLOTLOC, "distr_nume.pdf"), 
         up$arrange_plots(plots, ncol = 4, nrow = 2), width = 18, height = 12)
}
print(Sys.time() - start)

# Winsorize
df[nume] = map_df(df[nume], ~ up$winsorize(., 0.01, 0.99))

# Log-Transform
tolog = c("hum")
df[paste0(tolog,"_LOG")] = map(df[tolog], ~ {if(min(., na.rm=TRUE) == 0) log(. + min(., na.rm=TRUE)) else log(.)})
nume = map_chr(nume, ~ ifelse(. %in% tolog, paste0(.,"_LOG"), .)) #adapt metadata (keep order)


# --- Create categorical (binned) equivalents for all numeric features (for linear models to mimic non-linearity) ------

nume_binned = paste0(nume,"_BINNED")
df[nume_binned] = map_df(df[nume], ~ {
  as.character(cut(., unique(quantile(., seq(0, 1, 0.2), na.rm = TRUE)), include.lowest = TRUE))  
})

# Convert missings to own level ("(Missing)")
df[nume_binned] = map_df(df[nume_binned], ~ replace_na(., "(Missing)"))  # fct_explicit_na
summary(map_df(df[nume_binned], ~ as.factor(.)))

# Get binned variables with just 1 bin (removed later)
(onebin = nume_binned[map_lgl(df[nume_binned], ~ length(unique(.)) == 1)])


# --- Final feature information ----------------------------------------------------------------------------------------
'
# Univariate variable performances
varperf_nume = df[nume + nume_BINNED].swifter.progress_bar(False).apply(
    lambda x: (up.variable_performance(feature=x, 
                                       target=df["cnt_" + TARGET_TYPE],
                                       splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                       scorer=up.D_SCORER[TARGET_TYPE]["spear" if TARGET_TYPE == "REGR" 
                                                                       else "auc"])))
print(varperf_nume.sort_values(ascending=False))
# RRRR
(varimp_metr = (filterVarImp(map_df(df[metr], ~ impute(.)), df$target, nonpara = TRUE) %>% rowMeans() %>% 
                  .[order(., decreasing = TRUE)] %>% round(2)))

# Plot
if PLOT:
    _ = up.plot_l_calls(pdf_path=f"{s.PLOTLOC}1__distr_nume__{TARGET_TYPE}.pdf",
                        l_calls=[(up.plot_feature_target,
                                  dict(feature=df[feature], target=df["cnt_" + TARGET_TYPE],
                                       title=f"{feature} (VI:{varperf_nume[feature]: 0.2f})",
                                       regplot_type="lowess",
                                       add_miss_info=True if feature in nume else False))
                                 for feature in up.interleave(nume, nume_BINNED)])

'
# --- Removing variables (inlcuding correlation analysis) --------------------------------------------------------------

# Remove leakage features
remove = c("xxx", "xxx")
nume = setdiff(nume, "xxx")


# Remove highly/perfectly correlated (the ones with less NA!)
# Plot correlation
#get_plot_corr(df, method, absolute=True, cutoff=None):
  df_tmp = df[nume]
  method = "spearman"
  absolute = TRUE
  cutoff = 0.1
  # Check for mixed types
  
  count_numeric_types = sum(map_lgl(df_tmp, ~ is.numeric(.x)))
  if (!(count_numeric_types %in% c(0, dim(df_tmp)[2]))) {stop("Mixed types")}
  
  # Nume case
  if (count_numeric_types != 0) {
    if (!(method %in% c("pearson", "spearman"))) {stop("False method for numeric values: Choose pearson or spearman")}
    df_corr = as_tibble(cor(df_tmp, method = method, use = "pairwise.complete.obs"))
    suffix = paste0(" (", map_chr(df_tmp, ~ as.character(round(mean(is.na(.)) * 100, 1))), " %NA)")
  }
  '
  # Cate case
  else:
  '
  
  # Add info to names
  colnames(df_tmp) = paste0(colnames(df_tmp), suffix)
  
  # Absolute trafo
  if (absolute) {df_corr = map_df(df_corr, ~ abs(.))}  
  
  # Filter out rows or cols below cutoff and then fill diagonal
  for (i in seq_along(df_corr)) {df_corr[i,i] = 0}
  if (!is.null(cutoff)) {
    b_cutoff = map_lgl(df_corr, ~ (max(abs(.)) > cutoff)) %>% as.logical()
    df_corr = df_corr[b_cutoff, b_cutoff]
  }
  for (i in seq_along(df_corr)) {df_corr[i,i] = 1}
  
  # Cluster df_corr
  '
  tmp_order = linkage(1 - np.triu(df_corr),
                      method="average", optimal_ordering=False)[:, :2].flatten().astype(int)
  new_order = df_corr.columns.values[tmp_order[tmp_order < len(df_corr)]]
  df_corr = df_corr.loc[new_order, new_order]
  '
  
  
  
  # Plot
  sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="Reds" if absolute else "BLues",
              xticklabels=True, yticklabels=True, ax=ax)
  ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0)
  ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
  ax.set_title(("Absolute " if absolute else "") + method.upper() + " Correlation" +
               (" (cutoff: " + str(cutoff) + ")" if cutoff is not None else ""))

  return df_corr




map_lgl(df[nume], ~ is.numeric(.))
is.numeric()
m.corr = abs(cor(df.plot[vars], method = tolower(method), use = "pairwise.complete.obs"))
'
summary(df[metr])
plot = get_plot_corr(df, input_type = "metr", vars = metr, missinfo = misspct, cutoff = cutoff)
ggsave(paste0(plotloc, TYPE, "_corr_metr.pdf"), plot, width = 9, height = 9)
remove = c("xxx") #put at xxx the variables to remove
metr = setdiff(metr, remove) #remove
metr_binned = setdiff(metr_binned, paste0(remove,"_BINNED_")) #keep "binned" version in sync
'



# --- Detect data drift (time/fold depedency of features) --------------------------------------------------------------

# HINT: In case of having a detailed date variable, this can be used as regression target here as well!

# Univariate variable importance
'
df$fold_test = factor(ifelse(df$fold == "test", "Y", "N"))
(varimp_metr_fold = filterVarImp(df[metr], df$fold_test, nonpara = TRUE) %>% rowMeans() %>%
    .[order(., decreasing = TRUE)] %>% round(2))

# Plot: only variables with with highest importance
metr_toprint = names(varimp_metr_fold)[varimp_metr_fold >= 0.52]
options(warn = -1)
plots = get_plot_distr_metr(df, metr_toprint, color = c("blue","red"), target_name = "fold_test", 
                            missinfo = misspct, varimpinfo = varimp_metr_fold, ylim = ylim)
ggsave(paste0(plotloc, TYPE, "_distr_metr_final_folddependency.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       width = 18, height = 12)
options(warn = 0)
'

# --- Create missing indicator and impute feature missings--------------------------------------------------------------

(miss = nume[map_lgl(df[nume], ~ any(is.na(.)))])
df[paste0("MISS_",miss)] = map(df[miss], ~ ifelse(is.na(.x), "No", "Yes"))
summary(map_df(df[paste0("MISS_",miss)], ~ as.factor(.)))

# Impute missings
'
df[miss] = map(df[miss], ~ impute(., type = "random"))
summary(df[metr]) 
'

########################################################################################################################
# Categorical features: Explore and adapt
########################################################################################################################

# --- Define categorical features --------------------------------------------------------------------------------------

cate = df_meta_sub %>% filter(type == "cate") %>% .$variable
df[cate] = map(df[cate], ~ as.character(.)) 
summary(map_df(df[cate], ~ as.factor(.)))


# Handling factor values ----------------------------------------------------------------------------------------------

# Map missings to own level
df[cate] = map_df(df[cate], ~ replace_na(., "(Missing)"))
summary(map_df(df[cate], ~ as.factor(.)))

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
                    group_by_at(.x) %>% summarise(mean_target = mean(.data[[target]])) %>% 
                    ungroup() %>% arrange(desc(mean_target)) %>% 
                    right_join(df[.x] %>% mutate(n = row_number())) %>% 
                    arrange(n) %>% .$mean_target))


# Create compact covariates for "too many members" columns 
topn_toomany = 10
(levinfo = map_int(df[nomi], ~ length(levels(.))) %>% .[order(., decreasing = TRUE)]) #number of levels
(toomany = names(levinfo)[which(levinfo > topn_toomany)])
(toomany = setdiff(toomany, c("xxx"))) #set exception for important variables
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(., topn_toomany, other_level = "_OTHER_")) #collapse
nomi = map_chr(nomi, ~ ifelse(. %in% toomany, paste0(.,"_OTHER_"), .)) #adapt metadata (keep order)
summary(df[nomi], topn_toomany + 2)

'
# Univariate variable importance
(varimp_nomi = filterVarImp(df[nomi], df$target, nonpara = TRUE) %>% rowMeans() %>% 
    .[order(., decreasing = TRUE)] %>% round(2))
'


GO HEEEEEEEEEEEEEEEEERE
# Check
plots = suppressMessages(get_plot_distr_nomi(df, nomi, color = color, varimpinfo = varimp_nomi, inner_barplot = TRUE,
                                             min_width = min_width, ylim = ylim))
ggsave(paste0(plotloc,TYPE,"_distr_nomi.pdf"), marrangeGrob(plots, ncol = 3, nrow = 3), 
       width = 18, height = 12)

library(magrittr)
df  %>% use_series("temp")

# Removing variables ----------------------------------------------------------------------------------------------

# Remove Self-features
if (TYPE == "CLASS") nomi = setdiff(nomi, "boat_OTHER_")
if (TYPE %in% c("REGR","MULTICLASS")) nomi = setdiff(nomi, "xxx")

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!) 
plot = get_plot_corr(df, input_type = "nomi",  vars = setdiff(nomi, paste0("MISS_",miss)), cutoff = cutoff)
ggsave(paste0(plotloc,TYPE,"_corr_nomi.pdf"), plot, width = 9, height = 9)
if (TYPE %in% c("REGR","MULTICLASS")) {
  plot = get_plot_corr(df, input_type = "nomi",  vars = paste0("MISS_",miss), cutoff = 0.98)
  ggsave(paste0(plotloc,TYPE,"_corr_nomi_MISS.pdf"), plot, width = 9, height = 9)
  nomi = setdiff(nomi, c("MISS_BsmtFin_SF_2","MISS_BsmtFin_SF_1","MISS_second_Flr_SF","MISS_Misc_Val_LOG_",
                        "MISS_Mas_Vnr_Area","MISS_Garage_Yr_Blt","MISS_Garage_Area","MISS_Total_Bsmt_SF"))
}




# Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance
(varimp_nomi_fold = filterVarImp(df[nomi], df$fold_test, nonpara = TRUE) %>% rowMeans() %>% 
   .[order(., decreasing = TRUE)] %>% round(2))

# Plot (Hint: one might want to filter just on variable importance with highest importance)
nomi_toprint = names(varimp_nomi_fold)[varimp_nomi_fold >= 0.52]
plots = get_plot_distr_nomi(df, nomi_toprint, color = c("blue","red"), target_name = "fold_test", inner_barplot = FALSE,
                            varimpinfo = varimp_nomi_fold, ylim = ylim)
ggsave(paste0(plotloc,TYPE,"_distr_nomi_folddependency.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), 
       width = 18, height = 12)




#######################################################################################################################-
#|||| Prepare final data ||||----
#######################################################################################################################-

# Define final features ----------------------------------------------------------------------------------------

features = c(metr, nomi)
formula = as.formula(paste("target", "~ -1 + ", paste(features, collapse = " + ")))
features_binned = c(metr_binned, setdiff(nomi, paste0("MISS_",miss))) #do not need indicators if binned variables
formula_binned = as.formula(paste("target", "~ ", paste(features_binned, collapse = " + ")))

# Check
summary(df[features])
setdiff(features, colnames(df))
summary(df[features_binned])
setdiff(features_binned, colnames(df))




# Save image ----------------------------------------------------------------------------------------------------------
rm(df.orig, plots, plots1, plots2)
save.image(paste0(TYPE,"_1_explore.rdata"))



