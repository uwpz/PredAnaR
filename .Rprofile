source("renv/activate.R")
options(vsc.plot = FALSE)
Sys.setlocale("LC_TIME", "English")

options(languageserver.formatting_style = function(options) {
    style <- styler::tidyverse_style(indent_by = options$tabSize)
    style$token$force_assignment_op <- NULL
    style
})
