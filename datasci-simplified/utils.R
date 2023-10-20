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

  # split x to 10 bins
  x_breaks = seq(x_min, x_max, (x_max - x_min) / 10)
  
  # Calculating average of red marbles in each sample
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
  
  # Combining the plots vertically
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
    title = str_glue("Sampling Distribution of the Sample Means (size={size})")
  } else {
    title = NULL
  }

  p = ggplot(sample_means, aes(x = average)) +
    geom_histogram(aes(y = after_stat(density)), fill = "orange", color = "white", breaks = seq(x_min, x_max, (x_max - x_min) / 50)) +
    geom_density(aes(y = after_stat(density)), color = "red") +
    geom_vline(xintercept = true_prop, color = "blue", linetype = "dashed") +
    scale_x_continuous(limits = c(x_min, x_max), breaks =  seq(x_min, x_max, (x_max - x_min) / 10)) +
    labs(x = "Sample mean", y = "Density", title = title) +
    ylim(0, 15) +
    theme_minimal() + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  
  return(p)
}