library(dplyr)
library(tidyselect)
iris %>% select(where(is.numeric))

iris %>%
  mutate(across(where(is.double) & !c(Petal.Length, Petal.Width), round))

df = mtcars
df2 = mtcars %>% mutate_all(as.character)
 
var1 = "cyl"
value1 = 4
value1_char = "4"
var2 = "gear"
value2 = 3
var3 = "mpg"
newvar = "blub"
 
newvalue = "new"
twovars = c(var1, var2)
 
 
 
## BEST OPTIONS
df %>% select(!!newvar := cyl, !!"other_var" := var2)
df %>% select({{newvar}} := cyl, "other_var" := {{var2}})
df %>% rename(!!newvar := !!var1)
df %>% rename({{newvar}} := {{var1}})
df %>% filter(.data[[var1]] == value1)
df %>% mutate(!!newvar := .data[[var1]])
df %>% mutate({{newvar}} := .data[[var1]])
df %>% arrange_at(twovars)
df %>% arrange({{twovars}})
df %>% group_by_at(twovars) %>% summarize(!!newvar := mean(.data[[var3]])) %>% ungroup() %>% 
  spread_(var2, "blub") %>% gather_(var2, "blub", c("3","4","5"))