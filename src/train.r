library(jsonlite)
library(dplyr)

options(digits = 15)

data = read.csv("./data/calibration/inputs/20240928T014700--npz-calibration.csv")

# filter for greater than 3
df = data |> filter(compression_ratio > 3)

if (nrow(df) == 0) {
    stop("No data after filtering for compression_ratio > 3.")
}

compression_ratio = df$compression_ratio
entropy = df$entropy


deflate_nls_model = nls(entropy ~ a / (compression_ratio + b) + c * log(compression_ratio) + d * compression_ratio^2, 
                        data = df, 
                        start = list(a = 1, b = 1, c = 0, d = 0)
)

# predictions
df$predicted <- predict(deflate_nls_model, newdata = df)

calc_mse = function(obs, pred) {
         mean((obs - pred)^2)
}

mse = calc_mse(df$entropy, df$predicted)
stderr = summary(deflate_nls_model)$parameters[, "Std. Error"]
pcov = vcov(deflate_nls_model)

# model data
model_data = list(
    # for precision
    params = list(
        a = sprintf("%.15f", coef(deflate_nls_model)["a"]),
        b = sprintf("%.15f", coef(deflate_nls_model)["b"]),
        c = sprintf("%.15f", coef(deflate_nls_model)["c"]),
        d = sprintf("%.15f", coef(deflate_nls_model)["d"])
    ),
    "function" = "deflate_nls_model",
    fit_metadata = list(
        mse = mse,
        stderr = stderr,
        pcov = as.vector(pcov),
        p0 = c(1, 1, 0, 0)  # Removed trailing comma here
    )  
)  

print(model_data)

save_path = "./data/models/pretrained/deflate_nls_model-v1.json"
write_json(model_data, save_path , pretty = TRUE)
cat("Model saved to: ", save_path, "\n")