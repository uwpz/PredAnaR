########################################################################################################################
# Libraries
########################################################################################################################

box::use(
  stats[quantile],
  ggplot2[...],
  gridExtra[marrangeGrob]
)
#library(tidyverse)


COLORBLIND = c(
  "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161",
  "#fbafe4", "#949494", "#ece133", "#56b4e9"
)
theme_my = theme_bw() + theme(plot.title = element_text(hjust = 0.5))

########################################################################################################################
# General Functions
########################################################################################################################

# --- General ----------------------------------------------------------------------------------------------------------

debug_test = function(a=1, b=1) {
  print("start")
  a=2
  #browser()
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

## Winsorize
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
  marrangeGrob(grobs, ncol, nrow, 
               layout_matrix = t(matrix(seq_len(nrow * ncol), ncol, nrow)),
               ...)
}

get_plot_nume_CLASS = function(df, feature_name, target_name,
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