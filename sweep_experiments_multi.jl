include("run_experiment_multi.jl")
using CSV
using DataFrames
using Dates
using Random
using JSON

"""
Script de barrido (sweep) para DoME.

Qué guarda:
- metrics_YYYYmmdd_HHMMSS.csv : 1 fila por experimento (compacto)
- detailed_YYYYmmdd_HHMMSS.jsonl : (opcional) 1 JSON por experimento con expression + history + hiperparámetros
- predictions_YYYYmmdd_HHMMSS/*.csv : (opcional) 1 CSV por experimento con y_true/y_pred (raw + norm)

Notas:
- DoME se describe como determinista en el artículo, pero mantenemos `seed` y `Random.seed!` por trazabilidad.
- El split train/test es determinista: si no hay archivos train/test separados, hace hold-out temporal (primer 75% train, resto test).
"""

# -------------------------
# Configuración
# -------------------------
datasets_config = Dict(
    "ElectricDevices/LD2011_2014.txt" => Dict("target_col" => "MT_196", "windows" => [12, 24, 48]),
    "ETT/ETTh2.csv" => Dict("target_col" => "OT", "windows" => [12, 24, 48]),
    "ETT/ETTm1.csv" => Dict("target_col" => "OT", "windows" => [12, 24, 48]),
    "LCDS/LCD_USW00094789_2024.csv" => Dict("target_col" => "HourlyDryBulbTemperature", "windows" => [12, 24, 48])
)

normalization = "MaxMin"
model = :DoME

# Si realmente quieres repetir, sube esto (p.ej. 1:10). Si no, 1:1 suele bastar.
seeds = 1:10

# ⚠️ Recomendación CESGA: primero sweep sin preds, y luego segunda pasada solo a los mejores con preds.
SAVE_PREDICTIONS = false   # true => guarda *_predictions.csv (puede ocupar MUCHO si lo activas en todo el sweep)
SAVE_DETAILED    = true    # true => guarda JSONL con expression+history

# -------------------------
# Salida
# -------------------------
results_dir = "results"
isdir(results_dir) || mkdir(results_dir)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
metrics_file  = joinpath(results_dir, "metrics_$(timestamp).csv")
jsonl_file    = joinpath(results_dir, "detailed_$(timestamp).jsonl")

predictions_dir = nothing
if SAVE_PREDICTIONS
    predictions_dir = joinpath(results_dir, "predictions_$(timestamp)")
    isdir(predictions_dir) || mkdir(predictions_dir)
end

model_configs = modelConfigurations(model)

total_experiments = 0
for (_, cfg) in datasets_config
    total_experiments += length(cfg["windows"]) * length(model_configs) * length(seeds)
end

println("="^70)
println("SWEEP DoME")
println("="^70)
println("Total experiments: $total_experiments | configs=$(length(model_configs)) | seeds=$(length(seeds))")
println("CSV:   $metrics_file")
println("JSONL: $(SAVE_DETAILED ? jsonl_file : "(desactivado)")")
println("Preds: $(SAVE_PREDICTIONS ? string(predictions_dir) : "(desactivado)")")
println("="^70)

# -------------------------
# Loop
# -------------------------
experiment_counter = 0
start_time = time()

for (dataset, dataset_conf) in datasets_config
    for window in dataset_conf["windows"]
        for config_id in 1:length(model_configs)
            for seed in seeds
                experiment_counter += 1

                try
                    res_df, det, preds = run_experiment(
                        dataset=dataset,
                        window=window,
                        normalization=normalization,
                        model=model,
                        config_id=config_id,
                        seed=seed,
                        target_col=dataset_conf["target_col"],
                        save_predictions=SAVE_PREDICTIONS,
                        save_detailed=SAVE_DETAILED,
                        predictions_dir=predictions_dir,
                        verbose=false
                    )

                    # CSV (append)
                    if experiment_counter == 1
                        CSV.write(metrics_file, res_df)
                    else
                        CSV.write(metrics_file, res_df; append=true, writeheader=false)
                    end

                    # JSONL (append)
                    if SAVE_DETAILED && det !== nothing
                        open(jsonl_file, "a") do io
                            write(io, JSON.json(det))
                            write(io, "\n")
                        end
                    end

                    # Predicciones (1 CSV por experimento)
                    if SAVE_PREDICTIONS && preds !== nothing
                        dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")
                        pred_path = joinpath(
                            predictions_dir,
                            "$(dataset_name)_w$(window)_c$(config_id)_s$(seed)_predictions.csv"
                        )
                        CSV.write(pred_path, preds)
                    end

                catch e
                    ts_now = Dates.now()
                    dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")

                    # Error row con el MISMO schema que results_df
                    error_df = DataFrame(
                        dataset=[dataset],
                        dataset_name=[dataset_name],
                        window=[window],
                        normalization=[normalization],
                        model=[string(model)],
                        config_id=[config_id],
                        seed=[seed],

                        max_nodes=[missing],
                        min_improvement=[missing],
                        use_division=[missing],
                        strategy=[missing],

                        mse_initial=[missing],
                        mse_final_train=[missing],
                        mse_test_norm=[missing],
                        mse_test_raw=[missing],
                        improvement_pct=[missing],

                        iterations=[missing],
                        converged=[missing],
                        hit_max_iterations=[missing],
                        stop_reason=[missing],

                        train_samples=[missing],
                        test_samples=[missing],
                        features=[missing],

                        timestamp=[ts_now],
                        error=[string(e)]
                    )

                    if experiment_counter == 1
                        CSV.write(metrics_file, error_df)
                    else
                        CSV.write(metrics_file, error_df; append=true, writeheader=false)
                    end

                    if SAVE_DETAILED
                        open(jsonl_file, "a") do io
                            write(io, JSON.json(Dict(
                                "dataset" => dataset,
                                "dataset_name" => dataset_name,
                                "window" => window,
                                "config_id" => config_id,
                                "seed" => seed,
                                "error" => string(e),
                                "timestamp" => string(ts_now)
                            )))
                            write(io, "\n")
                        end
                    end
                end

                if experiment_counter % 50 == 0 || experiment_counter == total_experiments
                    elapsed = time() - start_time
                    pct = round(experiment_counter / total_experiments * 100, digits=2)
                    println("Progreso: $experiment_counter/$total_experiments ($pct%) | $(round(elapsed/60, digits=2)) min")
                end
            end
        end
    end
end

elapsed = time() - start_time
println("="^70)
println("SWEEP COMPLETADO | $(round(elapsed/60, digits=2)) min")
println("CSV:   $metrics_file")
println("JSONL: $(SAVE_DETAILED ? jsonl_file : "(desactivado)")")
println("Preds: $(SAVE_PREDICTIONS ? string(predictions_dir) : "(desactivado)")")
println("="^70)
