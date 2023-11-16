library(tidyverse)
library(gridExtra)
library(kableExtra)
library(knitr)


is_red = function(x) x == red_marble

get_samples = function(population, sample_size, number_of_samples = 1, func = NULL) {
  
  if (is.null(func)) {
    func = function(x) x
  }
  
  samples = tibble(sample_number = numeric(), sample_value = numeric())
  for (i in c(1:number_of_samples)) {
    sample_number = rep(i, each = sample_size)
    sample_value = func(sample(population, sample_size))
    samples = bind_rows(samples, tibble(sample_number, sample_value))
  }
  
  return(samples)
}

plot_samples_and_averages = function(samples, true_value) {
  
  x_min = floor(min(samples$sample_value))
  x_max = ceiling(max(samples$sample_value))

  x_breaks = seq(x_min, x_max, (x_max - x_min) / 10)
  
  sample_means = samples |>
    group_by(sample_number) |>
    summarise(average = mean(sample_value))
  
  # plotting samples as points and sample means 
  p1 = ggplot(samples, aes(x = sample_value, y = sample_number)) +
    geom_jitter(aes(color = factor(sample_value)), width = 0.2, height = 0) +
    geom_point(data = sample_means, aes(x = average, y = sample_number), color = "orange", size = 2) +
    geom_path(data = sample_means, aes(x = average, y = sample_number), color = "orange") +
    scale_color_manual(values = c("1" = "red", "0" = "blue")) +
    scale_y_continuous(breaks = seq(0, max(samples$sample_number), 5)) +
    labs(y = "Sample #", x = "Sample Value") +
    theme_minimal() + 
    theme(legend.position = "none")
  
  # Plotting histogram of sample means
  p2 = ggplot(sample_means, aes(x = average)) +
    geom_histogram(aes(y = after_stat(density)), fill = "orange", color = "white", breaks = seq(x_min, x_max, (x_max - x_min) / 50)) +
    geom_density(aes(y = after_stat(density)), color = "red") +
    scale_x_continuous(limits = c(x_min, x_max), breaks = seq(x_min, x_max, (x_max - x_min) / 10)) +
    geom_vline(xintercept = true_value, color = "blue", linetype = "dashed") +
    labs(x = "Sample mean", y = "Density", title = "Sampling Distribution of the Sample Means") +
    theme_minimal()
  
  grid.arrange(p1, p2, ncol = 1)
  
}


plot_samples = function(samples) {

  sample_means = samples |>
    group_by(sample_number) |>
    summarise(average = mean(sample_value))

  p1 = ggplot(samples, aes(x = sample_value, y = sample_number)) +
    geom_jitter(aes(color = factor(sample_value)), width = 0.2, height = 0) +
    geom_point(data = sample_means, aes(x = average, y = sample_number), color = "orange", size = 2) +
    geom_path(data = sample_means, aes(x = average, y = sample_number), color = "orange") +
    scale_color_manual(values = c("1" = "red", "0" = "blue")) +
    scale_y_continuous(breaks = seq(0, max(samples$sample_number), 5)) +
    labs(y = "Sample #", x = "Sample Value") +
    theme_minimal() + 
    theme(legend.position = "none")
  
  return(p1)
}

plot_sample_means = function(samples, size, true_prop, include_title = TRUE) {

  x_min = floor(min(samples$sample_value))
  x_max = ceiling(max(samples$sample_value))

  sample_means = samples |>
    group_by(sample_number) |>
    summarise(average = mean(sample_value))
  
  if (include_title) {
    title = str_glue("Distribution of the Sample Means (size={size})")
  } else {
    title = NULL
  }

  p = ggplot(sample_means, aes(x = average)) +
    geom_histogram(aes(y = after_stat(density)), fill = "orange", color = "white", breaks = seq(x_min, x_max, (x_max - x_min) / 100)) +
    geom_density(aes(y = after_stat(density)), color = "red") +
    geom_vline(xintercept = true_prop, color = "blue", linetype = "dashed") +
    scale_x_continuous(limits = c(x_min, x_max), breaks =  seq(x_min, x_max, (x_max - x_min) / 10)) +
    labs(x = "Sample mean", y = "Density", title = title) +
    # ylim(0, 15) +
    theme_minimal() + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  
  return(p)
}

standard_normal_curve = function() {
  x = seq(-4, 4, by=0.01)
  y = dnorm(x)
  df = tibble(x, y)
  
  ggplot(df, aes(x, y)) +
    geom_line(color="black") +
    stat_function(fun=dnorm, xlim=c(-4,4), fill="gray", geom="area") +
    scale_x_continuous(breaks = c(-4, -3, -2, -1, 0, 1, 2, 3, 4),
                       labels = c("-4σ", "-3σ", "-2σ", "-1σ", "0", "1σ", "2σ", "3σ", "4σ")) + 
    labs(x="", y="") +
    theme_minimal() + 
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y = element_blank())
  
}


normal_plot = function(mean = 0, sd = 1) {
  x = seq(mean - 4 * sd, mean + 4 * sd, by=0.001)
  y = dnorm(x, mean, sd)
  df = tibble(x, y)
  
  breaks = c(round(mean-4*sd, 2),
             round(mean-3*sd, 2),
             round(mean-2*sd, 2), 
             round(mean-sd, 2),
             round(mean, 2), 
             round(mean+sd, 2),
             round(mean+2*sd, 2),
             round(mean+3*sd, 2),
             round(mean+4*sd, 2))

  ggplot(df, aes(x, y)) +
    geom_line(color="black") + 
    scale_x_continuous(breaks = breaks) +
    stat_function(fun=function(x) dnorm(x, mean, sd),
                  xlim=c(mean-sd,mean), fill="dodgerblue3", geom="area") + 
    stat_function(fun=function(x) dnorm(x, mean, sd), 
                  xlim=c(mean,mean+sd), fill="dodgerblue3", geom="area") +
    stat_function(fun=function(x) dnorm(x, mean, sd),
                  xlim=c(mean-2*sd,mean-sd), fill="dodgerblue", geom="area") +
    stat_function(fun=function(x) dnorm(x, mean, sd),
                  xlim=c(mean+sd, mean+2*sd), fill="dodgerblue", geom="area") +
    stat_function(fun=function(x) dnorm(x, mean, sd),
                  xlim=c(mean-3*sd,mean-2*sd), fill="deepskyblue", geom="area") +
    stat_function(fun=function(x) dnorm(x, mean, sd),
                  xlim=c(mean+2*sd, mean+3*sd), fill="deepskyblue", geom="area") +
    annotate("text", x = c(mean - 0.5*sd, mean + 0.5*sd), y = 2, color = "white", 
             label = c("34.1%", "34.1%"), size = 5) +
    annotate("text", x = c(mean - 1.5*sd, (mean + 1.5*sd)), y = 1, color = "white", 
             label = c("13.6%", "13.6%"), size = 4) +
    annotate("text", x = c(mean - 2.25*sd, mean+2.25*sd), y = 0.25, color = "white", 
             label = c("2.1%", "2.1%"), size = 3) +
    labs(x="", y="") +
    theme_minimal()
}


critical_region_plot = function(mean = 0, sd = 1, alpha = 0.05, tailed = "two") {
  x = seq(mean - 4 * sd, mean + 4 * sd, by=0.001)
  y = dnorm(x, mean, sd)
  df = tibble(x, y)
  
  breaks = c(round(mean-4*sd, 2),
             round(mean-3*sd, 2),
             round(mean-2*sd, 2), 
             round(mean-sd, 2),
             round(mean, 2), 
             round(mean+sd, 2),
             round(mean+2*sd, 2),
             round(mean+3*sd, 2),
             round(mean+4*sd, 2))

  if (tailed == "two") {
    z_critical = qnorm(c(alpha / 2, 1 - alpha / 2))
  } else if (tailed == "left") {
    z_critical = qnorm(alpha)
  } else if (tailed == "right") {
    z_critical = qnorm(1 - alpha)
  }
  
  critical_values = mean + z_critical * sd

  ggplot(df, aes(x, y)) +
    geom_line(color="black") + 
    scale_x_continuous(breaks = breaks) +
    stat_function(fun=function(x) dnorm(x, mean, sd),
                  xlim=c(min(x), critical_values[1]), fill="red", geom="area") + 
    stat_function(fun=function(x) dnorm(x, mean, sd), 
                  xlim=c(critical_values[2], max(x)), fill="red", geom="area") +
    labs(x="", y="") +
    theme_minimal()
}


normal_dist_plot = function() {
  x = seq(-4, 4, by=0.01)
  y = dnorm(x)
  df = tibble(x, y)
  
  
  ggplot(df, aes(x, y)) +
    geom_line(color="black") +
    stat_function(fun=dnorm, xlim=c(-3,-2), fill="deepskyblue", geom="area") +
    stat_function(fun=dnorm, xlim=c(2,3), fill="deepskyblue", geom="area") +
    stat_function(fun=dnorm, xlim=c(-2,-1), fill="dodgerblue", geom="area") +
    stat_function(fun=dnorm, xlim=c(1,2), fill="dodgerblue", geom="area") +
    stat_function(fun=dnorm, xlim=c(-1,1), fill="dodgerblue3", geom="area") +
    annotate("text", x = c(-0.5, 0.5), y = 0.2, color = "white", 
             label = c("34.1%", "34.1%"), size = 5) +
    annotate("text", x = c(-1.5, 1.5), y = 0.05, color = "white", 
             label = c("13.6%", "13.6%"), size = 5) +
    annotate("text", x = c(-2.25, 2.25), y = 0.0125, color = "white", 
             label = c("2.1%", "2.1%"), size = 3) +
    geom_segment(aes(x = -3, y = 0, xend = -3, yend = dnorm(-3)), linetype="dashed") + 
    geom_segment(aes(x = -2, y = 0, xend = -2, yend = dnorm(-2)), linetype="dashed") +
    geom_segment(aes(x = -1, y = 0, xend = -1, yend = dnorm(-1)), linetype="dashed") +
    geom_segment(aes(x = 0, y = 0, xend = 0, yend = dnorm(0)), linetype="dashed") +
    geom_segment(aes(x = 1, y = 0, xend = 1, yend = dnorm(1)), linetype="dashed") +
    geom_segment(aes(x = 2, y = 0, xend = 2, yend = dnorm(2)), linetype="dashed") +
    geom_segment(aes(x = 3, y = 0, xend = 3, yend = dnorm(3)), linetype="dashed") +
    scale_x_continuous(breaks = c(-4, -3, -2, -1, 0, 1, 2, 3, 4),
                       labels = c("-4σ", "-3σ", "-2σ", "-1σ", "0", "1σ", "2σ", "3σ", "4σ")) + 
    labs(x="", y="") +
    theme_minimal() + 
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y = element_blank())
  
}


#----------------- confidence interval --------------------#
# confidence interval simulation
simulate_intervals = function(pop_mean = 0.65, 
                              pop_sd = 0.15,
                              number_of_samples = 100, 
                              confidence_level = 0.95) {

  alpha = 1 - confidence_level
  z = qnorm(1 - alpha / 2)

  included = tibble(lower = numeric(), upper = numeric())
  excluded = tibble(lower = numeric(), upper = numeric())

  n_included = number_of_samples * confidence_level
  n_excluded = number_of_samples * (1 - confidence_level)

  while (nrow(included) < n_included | nrow(excluded) < n_excluded){
    sample = rnorm(100, pop_mean, pop_sd)
    sample_mean = mean(sample)
    sample_sd = sd(sample)
    lower = sample_mean - z * sample_sd / sqrt(100)
    upper = sample_mean + z * sample_sd / sqrt(100)

    if (lower < pop_mean & upper > pop_mean) {
      if (nrow(included) < n_included){
        included = bind_rows(included, tibble(lower, upper))
      }
    } else {
      if (nrow(excluded) < n_excluded){
        excluded = bind_rows(excluded, tibble(lower, upper))
      }
    }
  }

  included = included |> 
    slice(1:n_included) |> 
    mutate(is_included = TRUE)
  excluded = excluded |> 
    slice(1:n_excluded) |> 
    mutate(is_included = FALSE)
  
  intervals = bind_rows(included, excluded) |> 
    sample_frac(1) |>
    mutate(sample_number = row_number()) |>
    select(sample_number, lower, upper, is_included)

  return(intervals)
}

# plot confidence intervals
plot_confidence_intervals = function(intervals, pop_mean) {
  
  ggplot(intervals, aes(x = sample_number, ymin = lower, ymax = upper)) +
    geom_pointrange(aes(y = (lower + upper) / 2, color = is_included),
                    fatten = 2,
                    position = position_dodge(0.5)) +
    scale_color_manual(values = c("TRUE" = "black", "FALSE" = "red")) + 
    geom_hline(yintercept = pop_mean, color = "blue", linetype = "dashed") +
    coord_flip() +
    # scale_y_continuous(breaks = seq(pop_mean - 3 * pop_sd, pop_mean + 3 * pop_sd, pop_sd)) +
    labs(
      x = "Sample #",
      y = "Value",
      title = "Confidence Intervals and True Population Mean"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
}
