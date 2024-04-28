########################################################################################################################
# Libraries, Parameter
########################################################################################################################

library(dplyr)
library(ggplot2)

# Theme
theme_up = theme_bw() + theme(plot.title = element_text(hjust = 0.5), 
                              plot.subtitle = element_text(hjust = 0.5),
                              plot.caption = element_text(hjust = 0))

# Colors
COLORDEFAULT = c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
  "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
#color = COLORMANY; barplot(1:length(color), col = color)


########################################################################################################################
# General Functions
########################################################################################################################

# --- General ----------------------------------------------------------------------------------------------------------

debug_test = function(a = 1, b = 1) {
  print("start")
  a = 2
  browser()
  if (TRUE) {
    print("start of if")
    b = 2
    print("end of if")
  }
  print("end")
}


########################################################################################################################
# Explore
########################################################################################################################

# --- Non-plots --------------------------------------------------------------------------------------------------------

#' Standard summary with characters as factors
#'
#' @param data Data frame for summary.
#' @return Summary of the data frame with characters converted to factors.
#' @examples
#' my_summary(data.frame(a = c("x", "y", "z"), b = c(1, 2, 3)))
my_summary = function(df) {
  df %>% mutate(across(where(is.character), as.factor)) %>% summary()
}


#' Calculate Cramers V between two vectors
#'
#' @param x Variable 1.
#' @param y Variable 2.
#'
#' @return Cramer's V value.
#'
cramersv = function(x, y) {
  result = sqrt(suppressWarnings(chisq.test(x, y, correct = FALSE)$statistic) / 
                  (length(x) * (min(length(unique(x)), length(unique(y))) - 1)))
  names(result) = "cramersv"
  result
}


#' Percentage format
#'
#' @param x Numeric vector to be formatted as a percentage.
#' @param n Number of decimal places (Default = 2).
#' @return Formatted percentage string.
#' @examples
#' format_pct(0.75) # Returns "75.00%"
percent_format = function(x, digits = 2) {
  trimws(paste0(format(100 * x, digits = digits, nsmall = digits), "%"))
}


#' Winsorize on either or both edges of numeric variable
#'
#' @param variable Numeric vector to winsorize.
#' @param lower Lower quantile (optional).
#' @param upper Upper quantile (optional).
#'
#' @return Winsorized vector.
#'
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


#' Impute
#'
#' @param variable Numeric vector to impute.
#' @param type Imputation type, "median", "random", or "zero" (Default = "median").
#'
#' @return Imputed vector.
#'
impute = function(variable, type = "median") {
  i_na = which(is.na(variable))
  if (length(i_na)) {
    variable[i_na] = switch(type, 
                            "median" = median(variable[-i_na], na.rm = TRUE),
                            "random" = sample(variable[-i_na], length(i_na) , replace = TRUE) ,
                            "zero" = 0)
  }
  variable 
}


#' Bin
#'
#' @param x Numeric vector to be binned.
#' @param k Number of bins (Default = 5).
#' @return Character vector representing the bins.
#' @examples
#' bin(c(1, 2, 3, 4, 5, 6), k = 3) # Returns c(q1 [1,2.67], q1 [1,2.67], q2 (2.67,4.33],
#' q2 (2.67,4.33], q3 (4.33,6], q3 (4.33,6])
bin = function(x, k = 5) {
  x_binned = cut(x, unique(quantile(x, seq(0, 1, 1/k), na.rm = TRUE)), include.lowest = TRUE)
  levels(x_binned) = paste(paste0("q", 1:length(levels(x_binned))), levels(x_binned))
  as.character(x_binned)
}


#' Helper function: Arrange plots by row first
#'
#' @param grobs List of plots to be arranged.
#' @param n_cols Number of columns in the arrangement (Default = 1).
#' @param n_rows Number of rows in the arrangement (Default = 1).
#' @return Arranged plots (by row first, then columns).
#' @examples
#' plots <- list(ggplot(), ggplot(), ggplot())
#' arrange_plots(plots, n_cols = 2, n_rows = 2)
arrange_plots = function(grobs, n_cols = 1, n_rows = 1, ...) {
  gridExtra::marrangeGrob(grobs, ncol = n_cols, nrow = n_rows, 
               layout_matrix = t(matrix(seq_len(n_rows * n_cols), n_cols, n_rows)),
               ...)
}


#' Feature-Target Correlation
#'
#' @param df Data frame containing the feature and target variable.
#' @param feature_name Name of the feature variable.
#' @param target_name Name of the target variable.
#' @param feature_type Explicitly define type of the feature variable, "nume" or "cate" (optional).
#' @param target_type Explicitly define of the target variable , "REGR", "CLASS", or "MULTICLASS" (optional)).
#' @param calc_cramersv Logical, whether to calculate CramersV for categorical features (Default = FALSE).
#'
#' @return Feature-target correlation value.
#'
feature_target_correlation = function(df, feature_name, target_name, feature_type = NULL, target_type = NULL, 
                                      calc_cramersv = FALSE) {
  # Determine feature and target type
  if (is.null(feature_type)) {
    feature_type = if (is.numeric(df[[feature_name]])) "nume" else "cate"
  }
  if (is.null(target_type)) {
    target_type = if (is.numeric(df[[target_name]])) "REGR" else {
      if (length(unique(df[[target_name]])) == 2) "CLASS" else "MULTICLASS"
    }
  }
  
  # Calc correlation
  if (feature_type == "nume") {
    if (target_type == "REGR") {
      # Spearman correlation
      correlation = df[c(feature_name, target_name)] %>% drop_na() %>% 
        cor(method = "spearman") %>% .[1,2] %>% abs()
    } else {
      if (!calc_cramersv) {
        # Spearman correlation of feature with feature-mean of target_class 
        correlation = df[c(feature_name, target_name)] %>% drop_na() %>% 
        group_by_at(target_name) %>% mutate(mean_per_class = mean(.data[[feature_name]])) %>% ungroup() %>% 
        select_at(c(feature_name, "mean_per_class")) %>% 
        cor(method = "spearman") %>% .[1,2] %>% abs
      } else {
        # CramersV of binned feature with target
        correlation = df[c(feature_name, target_name)] %>% drop_na() %>% 
          mutate(!!feature_name := as.character(cut(.data[[feature_name]], 
                                                   unique(quantile(.data[[feature_name]], seq(0, 1, 0.2), 
                                                                   na.rm = TRUE)), 
                                                   include.lowest = TRUE))) %>% 
          (function(x) cramersv(x[[feature_name]], x[[target_name]]))
      }
    }                
  } else {
    if (target_type == "REGR") {
      if (!calc_cramersv) {
        # Spearman correlation of target with target-mean of feature-level 
        correlation = df[c(feature_name, target_name)] %>% drop_na() %>% 
          group_by_at(feature_name) %>% mutate(mean_per_class = mean(.data[[target_name]])) %>% ungroup() %>% 
          select_at(c(target_name, "mean_per_class")) %>% 
          cor(method = "spearman") %>% .[1,2] %>% abs
      } else {
        # CramersV of binned target with feature
        correlation = df[c(feature_name, target_name)] %>% drop_na() %>% 
          mutate(!!target_name := as.character(cut(.data[[target_name]], 
                                                    unique(quantile(.data[[target_name]], seq(0, 1, 0.2), 
                                                                    na.rm = TRUE)), 
                                                    include.lowest = TRUE))) %>% 
          (function(x) cramersv(x[[feature_name]], x[[target_name]]))
      }
    } else {
      # CramersV
      correlation = df[c(feature_name, target_name)] %>% drop_na() %>% 
        (function(x) cramersv(x[[feature_name]], x[[target_name]]))
    }
  }
  names(correlation) = feature_name
  correlation
}


#' Helper: Adapt Feature-Target Data by removing missings and print message in case of missings
#'
#' @param df Data frame containing the features and target variable.
#' @param feature_name Name of the feature variable.
#' @param target_name Name of the target variable.
#' @param verbose Logical, whether to print progress messages (Default = TRUE).
#'
#' @return List containing the adapted data frame and percentage of missing feature values.
#'
helper_adapt_feature_target = function(df, feature_name, target_name, verbose = TRUE) {
  if (verbose) {print(paste0("Plotting ", feature_name, " vs. ", target_name))}
  n_target_miss = sum(is.na(df[[target_name]]))
  if (n_target_miss) {print(paste0("ATTENTION: ", n_target_miss, " records removed due to missing target!"))}
  n_feature_miss = sum(is.na(df[[feature_name]]))
  if (n_feature_miss) {print(paste0("ATTENTION: ", n_feature_miss, " records removed due to missing feature!"))}
  pct_miss_feature = 100 * n_feature_miss / nrow(df)
  df_plot = df %>% filter((!is.na(.data[[feature_name]]) & (!is.na(.data[[target_name]])))) 
  return(list(df_plot = df_plot, pct_miss_feature = pct_miss_feature))
}


#' Helper: Inner Barplot
#'
#' @param p Original ggplot object.
#' @param df Data frame containing the features and target variable.
#' @param feature_name Name of the feature variable.
#' @param inner_ratio Ratio of inner plot size to the original plot size (Default = 0.2).
#' @param coord_flip Logical, whether to flip coordinates in the inner plot (Default = FALSE).
#'
#' @return Modified ggplot object with an inner barplot.
#'
helper_inner_barplot = function(p, df, feature_name, inner_ratio = 0.2, coord_flip = FALSE) {
  df_plot = df %>% group_by_at(feature_name) %>% summarise(n = n()) %>% mutate(pct = n/sum(n)) %>% ungroup()
  p_inner = ggplot(data = df_plot, 
                   mapping = aes(x = .data[[feature_name]], y = pct)) +
    geom_bar(stat = "identity", fill = "lightgrey", colour = "black", width = 0.9) +
    coord_flip() +
    theme_void()
  
  # Put all together
  p_build = ggplot_build(p)
  x_range = p_build$layout$panel_params[[1]]$x.range
  x_breaks = as.numeric(p_build$layout$panel_params[[1]]$x$get_labels())
  if (coord_flip) {
    p = p + 
      scale_y_continuous(limits = c(x_range[1] - inner_ratio*(x_range[2] - x_range[1]), NA),
                         breaks = x_breaks[between(x_breaks, x_range[1], x_range[2])]) +
      theme_up +
      annotation_custom(ggplotGrob(p_inner), 
                        xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = -0.05*(x_range[2] - x_range[1])) +
      geom_hline(yintercept = 0, colour = "black")
  } else {
    p = p + 
      scale_x_continuous(limits = c(x_range[1] - inner_ratio*(x_range[2] - x_range[1]), NA),
                         breaks = x_breaks[between(x_breaks, x_range[1], x_range[2])]) +
      theme_up +
      annotation_custom(ggplotGrob(p_inner), 
                        ymin = -Inf, ymax = Inf, xmin = -Inf, xmax = x_range[1] - 0.05*(x_range[2] - x_range[1])) +
      geom_vline(xintercept = x_range[1], colour = "black")
  }
  p
}


#' Helper: Calculate Bar-Box Width
#'
#' @param df Data frame containing the features and target variable.
#' @param feature_name Name of the feature variable.
#' @param target_name Name of the target variable.
#' @param min_width Minimum width for the bar-box (Default = 0.2).
#'
#' @return Data frame with calculated widths and formatted feature names.
#'
helper_calc_barboxwidth = function(df, feature_name, target_name, min_width = 0.2) {

  df %>% group_by_at(c(feature_name, target_name)) %>% summarise(n = n()) %>% ungroup() %>% 
    complete(.data[[feature_name]], .data[[target_name]], fill = list(n = 0)) %>% 
    group_by_at(feature_name) %>% mutate(y = n / sum(n)) %>% ungroup() %>%  # target percentage (per feature value)
    left_join(df %>% group_by_at(feature_name) %>% summarise(n = n()) %>% 
                ungroup() %>% mutate(pct = n / sum(n), w = n / max(n)) %>% 
                select(-n),
              by = feature_name) %>%  # feature percentage
    mutate(w = ifelse(w < min_width, min_width, w)) %>%   # possibly adapt w to min_width
    mutate(!!paste0(feature_name, "_fmt") := 
             paste0(.data[[feature_name]], " (", percent_format(pct, 1), ")")) # format feature_name

}


#' Plot Categorical Feature against Binary Target
#'
#' @param df Data frame containing the feature and target variable.
#' @param feature_name Name of the feature variable.
#' @param target_name Name of the target variable.
#' @param target_category Specific target category to plot, for binary classification (optional).
#' @param multiclass_target Logical, indicating whether the target is multiclass (Default = FALSE).
#' @param title Custom title for the plot (optional).
#' @param add_miss_info Logical, whether to add information about missing values (Default = FALSE).
#' @param add_legend Logical, whether to add a legend, for multiclass target (Default = FALSE).
#' @param inner_ratio Ratio of inner plot size to the original plot size (Default = 0.2).
#' @param min_width Minimum width for the bar-box (Default = 0.2).
#' @param add_refline Logical, whether to add reference lines (Default = TRUE).
#' @param color Vector of colors for the plot (Default = COLORDEFAULT).
#' @param alpha Transparency level for colors (Default = 0.5).
#' @param verbose Logical, whether to print progress messages (Default = TRUE).
#' @param ... Additional arguments to be passed to the function.
#'
#' @return ggplot object displaying the relationship between the feature and the target.
#'
plot_cate_CLASS = function(df, feature_name, target_name,
                           target_category = NULL, multiclass_target = FALSE,
                           title = NULL,
                           add_miss_info = FALSE, add_legend = FALSE,
                           inner_ratio = 0.2, 
                           min_width = 0.2, 
                           add_refline = TRUE,
                           color = COLORDEFAULT,
                           alpha = 0.5,
                           verbose = TRUE,
                           ...) {

  # Adapt df
  l_tmp = helper_adapt_feature_target(df[c(feature_name, target_name)], feature_name, target_name, verbose)
  df_plot = l_tmp$df_plot
  df_plot[[target_name]] = as.character(df_plot[[target_name]])
  pct_miss_feature = l_tmp$pct_miss_feature
  
  # Add title
  if (is.null(title)) {title = feature_name}
  
  # Get minority class
  if (!multiclass_target) {
    if (is.null(target_category)) {
      target_category = names(sort(summary(factor(df_plot[[target_name]])))[1])
    }
  }
  
  # Prepare
  df_ggplot = helper_calc_barboxwidth(df_plot, feature_name, target_name, min_width = min_width)
  
  # Reduce to target_category
  if (!multiclass_target) {
    df_ggplot = df_ggplot %>% filter(.data[[target_name]] == target_category)
  }
  
  # Barplot
  p = ggplot(data = df_ggplot,
             mapping = aes(x = .data[[paste0(feature_name, "_fmt")]], y = .data[["y"]], fill = .data[[target_name]])) +
    geom_bar(stat = "identity", 
             position = if (!multiclass_target) "stack" else position_fill(reverse = TRUE),
             width = 0.9 * df_ggplot$w, color = "black",
             show.legend = if (multiclass_target) TRUE else FALSE) +
    scale_fill_manual(values = alpha(color, alpha)) +
    labs(title = title, y = paste0("avg(", target_name, ")")) +
    coord_flip() +
    theme_up
  
  # Adapt axis label
  if (add_miss_info) {
    p = p + xlab(paste0(p$labels$x, " (", percent_format(pct_miss_feature), " NA)"))
  } else {
    p = p + xlab("")
  }
  
  # Reflines
  if (add_refline) {
    df_refs = df_plot %>% group_by_at(target_name) %>% summarise(n = n()) %>% ungroup %>% 
      mutate(pct = n / sum(n), cumpct = cumsum(n)/sum(n))
    if (!multiclass_target) {
      refs = df_refs %>% filter(.data[[target_name]] == target_category) %>% pull(pct)
    } else {
      refs = df_refs %>% pull(cumpct)
      refs = rev(refs)[-1]
    }
    p = p + geom_hline(yintercept = refs, size = 0.5, colour = "black", linetype = "dashed")
  }
  
  # Add inner Barplot
  p = helper_inner_barplot(p, df_plot, feature_name, inner_ratio = inner_ratio, coord_flip = TRUE)
  p
}


#' Plot Categorical Feature against Multiclass Target
#'
#' @param ... Arguments passed to `plot_cate_CLASS`.
#'
#' @return ggplot object displaying the relationship between the feature and the target (multiclass version).
#'
plot_cate_MULTICLASS = function(...) {
  plot_cate_CLASS(multiclass_target = TRUE, ...)
}


#' Plot Categorical Features against Regression Target
#'
#' @param df Data frame containing the features and target variable.
#' @param feature_name Name of the feature variable.
#' @param target_name Name of the target variable.
#' @param title Custom title for the plot (optional).
#' @param add_miss_info Logical, whether to add information about missing values (Default = FALSE).
#' @param inner_ratio Ratio of inner plot size to the original plot size (Default = 0.2).
#' @param min_width Minimum width for the bar-box (Default = 0.2).
#' @param add_violin Logical, wether to add a violin plot (Default = FALSE).
#' @param add_refline Logical, whether to add reference lines (Default = TRUE).
#' @param col_color Name of grouping variable defining color of points (optional).
#' @param color Vector of colors for the plot (Default = COLORDEFAULT).
#' @param alpha Transparency level for colors (Default = 0.5).
#' @param verbose Logical, whether to print progress messages (Default = TRUE).
#' @param ... Additional arguments to be passed to the function.
#'
#' @return ggplot object displaying the relationship between the feature and the target.
#'
plot_cate_REGR = function(df, feature_name, target_name,
                          title = NULL,
                          add_miss_info = FALSE,
                          inner_ratio = 0.2, 
                          min_width = 0.2, 
                          add_violin = FALSE,
                          add_refline = TRUE,
                          col_color = NULL,
                          color = COLORDEFAULT,
                          alpha = 0.5,
                          verbose = TRUE,
                          ...) {
  
  # Adapt df
  cols = c(feature_name, target_name)
  if (!is.null(col_color)) {
    if (col_color != feature_name) cols = c(cols, col_color)
  }
  l_tmp = helper_adapt_feature_target(df[cols], feature_name, target_name, verbose)
  df_plot = l_tmp$df_plot
  pct_miss_feature = l_tmp$pct_miss_feature
  
  # Add title
  if (is.null(title)) {title = feature_name}
  
  # Prepare
  df_plot = df_plot %>% 
    left_join(helper_calc_barboxwidth(df_plot %>% mutate(dummy = "dummy"), 
                                      feature_name, "dummy", min_width = min_width) %>% 
                select_at(c(feature_name, paste0(feature_name, "_fmt"))))
  
  # Boxplot
  p = ggplot(data = df_plot,
             mapping = aes(y = .data[[paste0(feature_name, "_fmt")]], x = .data[[target_name]])) +
    geom_boxplot(varwidth = TRUE) +
    stat_summary(fun = mean, geom = "point", shape = 4) +
    labs(title = title) +
    theme_up
  
  if (!is.null(col_color)) {
    p = p + geom_jitter(aes(color = .data[[col_color]])) + scale_color_manual(values = color)
  }
  
  if (add_violin) {
    p = p + geom_violin(fill = NA, color = "grey") 
  }
  
  # Adapt axis label
  if (add_miss_info) {
    p = p + ylab(paste0(p$labels$y, " (", percent_format(pct_miss_feature), " NA)"))
  } else {
    p = p + ylab("")
  }
  p
  
  # Reflines
  if (add_refline) {
    p = p + geom_vline(xintercept = mean(df_plot[[target_name]]), size = 0.5, colour = "black", linetype = "dashed")
  }
  
  # Add inner Barplot
  p = helper_inner_barplot(p, df_plot, feature_name, inner_ratio = inner_ratio)
  p
  
}
  

#' Plot Numeric Features against Binary Target
#'
#' @param df Data frame containing the features and target variable.
#' @param feature_name Name of the feature variable.
#' @param target_name Name of the target variable.
#' @param title Custom title for the plot (optional).
#' @param add_miss_info Logical, whether to add information about missing values (Default = TRUE).
#' @param add_legend Logical, whether to add a legend (Default = TRUE).
#' @param inner_ratio Ratio of inner plot size to the original plot size (Default = 0.2).
#' @param n_bins Number of bins for the histogram (Default = 20).
#' @param color Vector of colors for the plot (Default = COLORDEFAULT).
#' @param alpha Transparency level for colors (Default = 0.3).
#' @param verbose Logical, whether to print progress messages (Default = TRUE).
#' @param ... Additional arguments to be passed to the function.
#'
#' @return ggplot object displaying the distribution of the feature for a numeric target.
#'
plot_nume_CLASS = function(df, feature_name, target_name,
                           title = NULL,
                           add_miss_info = TRUE, add_legend = TRUE,
                           inner_ratio = 0.2, 
                           n_bins = 20, 
                           color = COLORDEFAULT,
                           alpha = 0.3,
                           verbose = TRUE,
                           ...) {
  
  # Adapt df
  l_tmp = helper_adapt_feature_target(df[c(feature_name, target_name)], feature_name, target_name, verbose)
  df_plot = l_tmp$df_plot
  df_plot[[target_name]] = as.character(df_plot[[target_name]])
  pct_miss_feature = l_tmp$pct_miss_feature
  
  # Add title
  if (is.null(title)) {title = feature_name}
  
  # Distribution plot
  p = ggplot(data = df_plot, 
             aes_string(x = feature_name)) +
      geom_histogram(aes_string(y = "..density..",
                                fill = target_name, 
                                color = target_name),
                     show.legend = add_legend,
                     bins = n_bins, 
                     position = "identity") +
      geom_density(aes_string(color = target_name)) +
      scale_fill_manual(values = alpha(color, alpha), name = target_name) + 
      scale_color_manual(values = color, name = target_name) +
      theme_up +
      #guides(fill = guide_legend(reverse = TRUE), color = guide_legend(reverse = TRUE))
      labs(title = title)
  if (add_miss_info) {p = p + xlab(paste0(p$labels$x, " (", percent_format(pct_miss_feature), " NA)"))}

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
    theme_up +
    theme(legend.position = c(0.95, 0.95), legend.justification = c("right", "top")) +
    annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0)
  p
}


#' Plot Numeric Features against Multiclass Target
#'
#' @param ... Arguments passed to `plot_nume_CLASS`.
#'
#' @return ggplot object displaying the distribution of the feature for a numeric target (multiclass version).
#'
plot_nume_MULTICLASS = function(...) {
  plot_nume_CLASS(...)
}


#' Plot Numeric Feature against Regression Target
#'
#' @param df Data frame containing the features and target variable.
#' @param feature_name Name of the feature variable.
#' @param target_name Name of the target variable.
#' @param title Custom title for the plot (optional).
#' @param add_regplot Logical, whether to add a spline regression fit (Default = TRUE).
#' @param color_regplot Color of regplot (Default = "red").
#' @param add_miss_info Logical, whether to add information about missing values (Default = TRUE).
#' @param add_legend Logical, whether to add a legend (Default = TRUE).
#' @param inner_ratio Ratio of inner plot size to the original plot size.
#' @param add_feature_distribution Logical, whether to add the feature distribution at bottom (Default = TRUE)
#' @param add_target_distribution Logical, whether to add the target distribution at left side (Default = TRUE).
#' @param n_bins Number of bins for feature and target distribution histogrem (Default = 20).
#' @param hex_plot Logical, whether to plot a hex_plot instead of points (Default = TRUE).
#' @param colormap Colormap to use for hexplot (Default = colorRampPalette(c("lightgrey", "blue", "yellow"))(100)).
#' @param add_contour Logical, whether to add a contour plot (Default = FALSE).
#' @param color_contour Color of contour lines (Default = "white")
#' @param col_color Color variable for additional coloring of points. 
#' @param color Vector of colors for the plot (Default = COLORDEFAULT).
#' @param verbose Logical, whether to print progress messages.
#' @param ... Additional arguments to be passed to the function.
#'
#' @return ggplot object displaying the relationship between the feature and the target for regression.
#'
plot_nume_REGR = function(df, feature_name, target_name,
                          title = NULL, 
                          add_regplot = TRUE, color_regplot = "red", add_miss_info = TRUE, add_legend = TRUE,
                          inner_ratio = 0.2, 
                          add_feature_distribution = TRUE, add_target_distribution = TRUE, n_bins = 20,
                          hex_plot = TRUE, colormap = colorRampPalette(c("lightgrey", "blue", "yellow"))(100), 
                          add_contour = FALSE, color_contour = "white",
                          col_color = NULL, color = COLORDEFAULT,
                          verbose = TRUE, ...) {
  
  # Adapt df
  cols = c(feature_name, target_name)
  if (!is.null(col_color)) {
    if (col_color != feature_name) cols = c(cols, col_color)
  }
  l_tmp = helper_adapt_feature_target(df[cols], feature_name, target_name, verbose)
  
  # Subset
  df_plot = l_tmp$df_plot
  
  # Calc feature missings
  pct_miss_feature = l_tmp$pct_miss_feature
  
  # Add title
  if (is.null(title)) {title = feature_name}
  
  # Heatmap
  p = ggplot(data = df_plot, 
             mapping = aes(x = .data[[feature_name]], 
                           y = .data[[target_name]]))
  if (hex_plot == TRUE) {
    p = p + geom_hex(show.legend = add_legend) +
      scale_fill_gradientn(colors = colormap)
  } else {
    p = p + geom_point(mapping = if (!is.null(col_color)) aes(color = .data[[col_color]]) else NULL) +
      scale_color_manual(values = color)
  }
  
  # Contour lines
  if (add_contour) {
    p = p + geom_density2d(color = color_contour)
  }
  
  # Ohter
  p = p + labs(title = title) + 
    theme_up 
  if (add_regplot) {p = p + geom_smooth(color = color_regplot, fill = color_regplot, level = 0.95, size = 1)}
  if (add_miss_info) {p = p + xlab(paste0(p$labels$x, " (", percent_format(pct_miss_feature), " NA)"))}
  
  # Get plot dimension
  p_build = ggplot_build(p)
  y_range = p_build$layout$panel_params[[1]]$y.range
  y_breaks = as.numeric(p_build$layout$panel_params[[1]]$y$get_labels())
  x_range = p_build$layout$panel_params[[1]]$x.range
  x_breaks = as.numeric(p_build$layout$panel_params[[1]]$x$get_labels())
  
  # Feature distribution
  if (add_feature_distribution) {
    
    # Inner Histogram
    p_inner = ggplot(data = df_plot, 
                     mapping = aes(x = .data[[feature_name]])) +
      geom_histogram(aes(y = ..density..), bins = n_bins, position = "identity", fill = "lightgrey", color = "black") +
      geom_density(color = "black") +
      theme_void()
    
    # Inner Boxplot
    p_innerinner = ggplot(data = df_plot, 
                          mapping = aes(x = '', y = .data[[feature_name]])) +
      geom_boxplot() +
      stat_summary(fun = mean, geom = "point", shape = 4) +
      coord_flip() +
      theme_void()
    
    # Put inner plots together
    p_inner = p_inner +
      scale_y_continuous(limits = c(-ggplot_build(p_inner)$layout$panel_params[[1]]$y.range[2]/2, NA)) +
      annotation_custom(ggplotGrob(p_innerinner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0) 
    
    # Put all together
    p = p + 
      scale_y_continuous(limits = c(y_range[1] - inner_ratio*(y_range[2] - y_range[1]), NA),
                         breaks = y_breaks[between(y_breaks, y_range[1], y_range[2])]) +
      theme(plot.title = element_text(hjust = 0.5)) + #default style here due to white spots
      annotation_custom(ggplotGrob(p_inner), xmin = x_range[1], xmax = Inf, ymin = -Inf, ymax = y_range[1])  +
      geom_hline(yintercept = y_range[1], color = "black")
  }
  
  # Target distribution
  if (add_target_distribution) {
    
    # Inner Histogram
    p_inner = ggplot(data = df_plot, 
                     mapping = aes(x = .data[[target_name]])) +
      geom_histogram(aes(y = ..density..), bins = n_bins, position = "identity", fill = "lightgrey", color = "black") +
      geom_density(color = "black") +
      theme_void() +
      coord_flip()
    
    # Inner Boxplot
    p_innerinner = ggplot(data = df_plot, 
                          mapping = aes(x = '', y = .data[[target_name]])) +
      geom_boxplot() +
      stat_summary(fun = mean, geom = "point", shape = 4) +
      theme_void()
    
    # Put inner plots together
    p_inner = p_inner +
      scale_y_continuous(limits = c(NA, -ggplot_build(p_inner)$layout$panel_params[[1]]$x.range[2]/2),
                         trans = "reverse") +
      annotation_custom(ggplotGrob(p_innerinner), xmin = -Inf, xmax = Inf, ymin = 0, ymax = Inf) 
    
    # Put all together
    p = p + 
      scale_x_continuous(limits = c(x_range[1] - inner_ratio*(x_range[2] - x_range[1]), NA),
                         breaks = x_breaks[between(x_breaks, x_range[1], x_range[2])]) +
      theme(plot.title = element_text(hjust = 0.5)) + #default style here due to white spots
      annotation_custom(ggplotGrob(p_inner), xmin = -Inf, xmax = x_range[1], ymin = y_range[1], ymax = Inf)  +
      geom_vline(xintercept = x_range[1], color = "black")
  }
  
  # Hide intersection
  if (add_feature_distribution & add_target_distribution) {
    p = p + annotate("rect", xmin = -Inf, xmax = x_range[1], ymin = -Inf, ymax = y_range[1], 
                     fill="white", color = "black")
  }
  
  p
}


#' Plot the Feature against Target 
#'
#' @param df Data frame containing the features and target variable.
#' @param feature_name Name of the feature variable.
#' @param target_name Name of the target variable.
#' @param feature_type Explicitly define type of the feature variable, "nume" or "cate" (otional).
#' @param target_type Explicitly define type of the target variable, "REGR", "CLASS", or "MULTICLASS" (optional).
#' @param ... Additional arguments to be passed to the specific plot function.
#'
#' @return ggplot object displaying the relationship between the feature and the target.
#'
plot_feature_target = function(df, feature_name, target_name, feature_type = NULL, target_type = NULL, ...) {
  
  # Determine feature and target type
  if (is.null(feature_type)) {
    feature_type = if (is.numeric(df[[feature_name]])) "nume" else "cate"
  }
  if (is.null(target_type)) {
    target_type = if (is.numeric(df[[target_name]])) "REGR" else {
      if (length(unique(df[[target_name]])) == 2) "CLASS" else "MULTICLASS"
    }
  }
  
  # Call plot functions
  if  (feature_type == "nume") {
    if (target_type == "CLASS") plot_nume_CLASS(df, feature_name, target_name, ...)
    else if (target_type == "MULTICLASS") plot_nume_MULTICLASS(df, feature_name, target_name, ...)
    else if (target_type == "REGR") plot_nume_REGR(df, feature_name, target_name, ...)
    else stop('Wrong TARGET_TYPE')
  } else {
    if (target_type == "CLASS") plot_cate_CLASS(df, feature_name, target_name, ...)
    else if (target_type == "MULTICLASS") plot_cate_MULTICLASS(df, feature_name, target_name, ...)
    else if (target_type == "REGR") plot_cate_REGR(df, feature_name, target_name, ...)
    else stop('Wrong TARGET_TYPE')
  }
}


#' Plot Correlation Matrix
#'
#' @param df Data frame containing the features.
#' @param method Method to calculate the correlation. Options: "pearson", "spearman", or "cramersv".
#' @param absolute Logical, indicating whether to plot the absolute values of the correlation (Default = TRUE).
#' @param cutoff Threshold for filtering out weak correlations. If NULL, no filtering is applied (optional).
#'
#' @return ggplot object displaying the correlation matrix.
#'
plot_corr = function(df, method, absolute = TRUE, cutoff = NULL) {
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
    for (i in 1:(k - 1)) {
      for (j in (i + 1):k) {
        #print(paste0(i,"...", j))
        df_corr[i, j] = cramersv(df[[i]],df[[j]])# adapt with cramersv
        df_corr[j, i] = df_corr[i, j]
      }
    }
    suffix = paste0(" (", map_int(df, ~length(unique(.))), ")")
  }
  
  # Add info to names
  colnames(df_corr) = paste0(colnames(df_corr), suffix)
  
  # Absolute trafo
  if (absolute) {df_corr = map_df(df_corr, ~ abs(.))}  
  
  # Filter out rows or cols below cutoff and then fill diagonal
  for (i in seq_along(df_corr)) {df_corr[i,i] = 0}
  if (!is.null(cutoff)) {
    b_cutoff = map_lgl(df_corr, ~ (max(abs(.), na.rm = TRUE) > cutoff)) %>% as.logical()
    df_corr = df_corr[b_cutoff, b_cutoff]
  }
  for (i in seq_along(df_corr)) {df_corr[i,i] = 1}
  
  # Cluster df_corr
  new_order = corrplot::corrMatOrder(df_corr %>% as.matrix() , order = "hclust")
  df_corr = df_corr[new_order, new_order]
  
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
    {if (absolute) scale_fill_gradient(low = "white", high = "darkred") else 
      scale_fill_gradient2(low = "darkblue", mid = "white", high = "darkred")} +
    scale_x_discrete(limits = colnames(df_corr)) +
    scale_y_discrete(limits = rev(colnames(df_corr))) +
    scale_color_manual(values = c("black", "white")) +
    guides(color = "none") +
    labs(title = paste0("abs. ", toupper(method)," Correlation", " (cutoff: ", cutoff, ")"), 
         fill = "", x = "", y = "") +
    theme_up + theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
}





