########################################################################################################################
# Libraries
########################################################################################################################

library(tidyverse)

#box::use(
#  stats[quantile],
#  ggplot2[...],
#  gridExtra[marrangeGrob]
#)
#library(tidyverse)

# 
# COLORBLIND = c(
#   "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161",
#   "#fbafe4", "#949494", "#ece133", "#56b4e9"
# )
# theme_my = theme_bw() + theme(plot.title = element_text(hjust = 0.5))

########################################################################################################################
# General Functions
########################################################################################################################

# --- General ----------------------------------------------------------------------------------------------------------

debug_test = function(a=1, b=1) {
  print("start")
  a=2
  browser()
  if (TRUE) {
    print("start of if")
    b=2
    print("end of if")
  }
  print("end")
}


########################################################################################################################
# Explore
########################################################################################################################

# --- Non-plots --------------------------------------------------------------------------------------------------------

# Summary
my_summary = function(df) {
  df %>% mutate(across(where(is.character), as.factor)) %>% summary()
}

# Winsorize
winsorize = function(variable, lower = NULL, upper = NULL) {
  if (!is.null(lower)) {
    q_lower = quantile(variable, lower, na.rm = TRUE)
    variable[variable < q_lower] = q_lower
  }
  if (!is.null(upper)) {
    q_upper = quantile(variable, upper, na.rm = TRUE)
    variable[variable > q_upper] = q_upper
  }
  variable
}

arrange_plots = function(grobs, ncol, nrow, ...) {
  gridExtra::marrangeGrob(grobs, ncol, nrow, 
               layout_matrix = t(matrix(seq_len(nrow * ncol), ncol, nrow)),
               ...)
}

# plot distribution
plot_nume_CLASS = function(df, feature_name, target_name,
                               n_bins = 20, color=COLORBLIND,
                               title=NULL, inner_ratio=0.2) {
  df_plot = df[c(feature_name, target_name)]
  df_plot[[target_name]] = as.character(df_plot[[target_name]])
  unique(df_plot[[target_name]])
  summary(df_plot)
  p = ggplot(data = df_plot, 
             aes_string(x = feature_name)) +
      geom_histogram(aes_string(y = "..density..",
                                fill = target_name, 
                                color = target_name),
                     bins = n_bins, 
                     position = "identity") +
      geom_density(aes_string(color = target_name)) +
      scale_fill_manual(values = alpha(color, .4), name = target_name) + 
      scale_color_manual(values = color, name = target_name) +
      theme_my +
      #guides(fill = guide_legend(reverse = TRUE), color = guide_legend(reverse = TRUE))
      labs(title = title)

  # Get underlying data for max of y-value and range of x-value
  #tmp = ggplot_build(p)
      
  # Inner boxplot
  p.inner = ggplot(data = df_plot, 
                   aes_string(x = target_name, 
                              y = feature_name)) +
    geom_boxplot(aes_string(fill = target_name)) +
    stat_summary(fun = mean, geom = "point", shape = 4) +
    scale_x_discrete(limits = rev(sort(unique(df_plot[[target_name]])))) +
    scale_y_continuous(limits = c(min(layer_data(p)$xmin), max(layer_data(p)$xmax))) +
    scale_fill_manual(values = color, name = target_name) +
    coord_flip() +
    theme_void() +
    theme(legend.position = "none")

  # Put all together
  p = p + 
    scale_y_continuous(limits = c(-layer_scales(p)$y$get_limits()[[2]] * inner_ratio, NA)) +
    theme_my +
    theme(legend.position = c(0.95, 0.95), legend.justification = c("right", "top")) +
    annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0)
  p
}


plot_corr = function(df, method, absolute = TRUE, cutoff = None) {
  #df = df[nume]; method = "spearman"; absolute = TRUE; cutoff = 0.1;  text_color = "white"
  
  # Check for mixed types
  count_numeric_types = sum(map_lgl(df, ~ is.numeric(.x)))
  if (!(count_numeric_types %in% c(0, dim(df)[2]))) {stop("Mixed types")}
  
  
  if (count_numeric_types != 0) {
    # Nume case
    if (!(method %in% c("pearson", "spearman"))) {stop("False method for numeric values: Choose pearson or spearman")}
    df_corr = as_tibble(cor(df, method = method, use = "pairwise.complete.obs"))
    suffix = paste0(" (", map_chr(df, ~ as.character(round(mean(is.na(.)) * 100, 1))), " %NA)")
  } else {
    # Cate case
    if (!(method %in% c("cramersv"))) {stop("False method for categorical values: Choose cramersv")}
    k = dim(df)[2]
    df_corr = data.frame(matrix(0, k, k)) %>% magrittr::set_colnames(colnames(df))
    for (i in 1:(k-1)) {
      for (j in (i+1):k) {
        #print(paste0(i,"...", j))
        df_corr[i, j] = i*j # adapt with cramersv
        df_corr[j, i] = df_corr[i, j]
      }
    }
    suffix = paste0(" (", + map_int(df, ~length(unique(.))), ")")
    
  }
  
  # Add info to names
  colnames(df_corr) = paste0(colnames(df_corr), suffix)
  
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
  #library(corrplot)
  #new_order = corrMatOrder(df_corr %>% as.matrix() , order = "hclust")
  #df_corr = df_corr[new_order, new_order]
  
  # Plot
  df_plot = df_corr %>% 
    mutate(rowvar = colnames(df_corr)) %>% 
    pivot_longer(cols = colnames(df_corr), names_to = "colvar", values_to = "corr") %>% 
    mutate(textcolor = as.factor(ifelse(abs(corr) > 0.5, 2, 1)))
  p = ggplot(data = df_plot, 
             mapping = aes(x = rowvar, y = colvar)) +
    geom_tile(aes(fill = corr)) + 
    geom_text(aes(label = round(corr, 2), 
                  colour = textcolor)) +
    scale_fill_gradient(low = "white", high = "darkred") +
    scale_x_discrete(limits = colnames(df_corr)) +
    scale_y_discrete(limits = rev(colnames(df_corr))) +
    scale_color_manual(values = c("black", "white")) +
    guides(color = "none") +
    labs(title = paste0("abs. ", toupper(method)," Correlation", " (cutoff: ", cutoff, ")"), 
         fill = "", x = "", y = "") +
    theme_my + theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
  p
}





