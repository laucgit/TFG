include("run_experiment_multi.jl")
using CSV
using DataFrames
using Dates
using Random

"""
Script de barrido de experimentos para DoME.

CORRECCIONES:
1. Sin readline() - compatible con batch/CESGA
2. Sin sleep(3) - innecesario en batch
3. seed=1:1 por defecto (DoME es determinista, repetir es redundante)
4. Usa load_data con lags del objetivo incluidos
5. error_df tiene la misma columna "error" que results_df (siempre presente,
   vacía si no hay error) → CSV append nunca tiene schema mismatch

CONFIGURACIONES:
- minimumReductionMSE: [1e-5, 1e-6, 1e-7, 1e-8, 1e-9] (5 valores)
- maxNumNodes: [5, 10, 20, 30, 50, 75, 100, 150, 200] (9 valores)
  (maxNodes=1 eliminado: árbol de 1 nodo = mean(y), Step! retorna false inmediatamente)
- useDivisionOperator: [false, true] (2 valores)
- strategy: [Strategy4, Strategy3] (2 valores)

Total configs: 9 × 5 × 2 × 2 = 180

Total experimentos: 4 datasets × 3 windows × 180 configs × 1 seed = 2,160
"""

# Definición de datasets
datasets_config = Dict(
    "ElectricDevices/LD2011_2014.txt" => Dict(
        "target_col" => "MT_196",
        "windows" => [12, 24, 48]
    ),
    "ETT/ETTh2.csv" => Dict(
        "target_col" => "OT",
        "windows" => [12, 24, 48]
    ),
    "ETT/ETTm1.csv" => Dict(
        "target_col" => "OT",
        "windows" => [12, 24, 48]
    ),
    "LCDS/LCD_USW00094789_2024.csv" => Dict(
        "target_col" => "HourlyDryBulbTemperature",
        "windows" => [12, 24, 48]
    )
)

normalization = "MaxMin"
model = :DoME

# DoME es determinista, no necesita repetir con distintas semillas
seeds = 1:1

# Obtener configuraciones
model_configs = modelConfigurations(model)
println("="^70)
println("BARRIDO DE EXPERIMENTOS - DOME")
println("="^70)
println("Total de configuraciones por modelo: $(length(model_configs))")
println("Seeds: $(length(seeds))")

# Calcular total
total_experiments = 0
for (dataset, config) in datasets_config
    total_experiments += length(config["windows"]) * length(model_configs) * length(seeds)
end

println("Total de experimentos: $total_experiments")
println("\nDesglose:")
println("  - Datasets: $(length(datasets_config))")
println("  - Windows promedio: 3")
println("  - Configuraciones: $(length(model_configs))")
println("  - Seeds: $(length(seeds)) (DoME es determinista)")
println("="^70)

# Inicio automático sin sleep (batch/CESGA)
println("\nIniciando barrido...")

results_dir = "results"
isdir(results_dir) || mkdir(results_dir)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
consolidated_file = joinpath(results_dir, "sweep_consolidated_$(timestamp).csv")

experiment_counter = 0
start_time = time()

for (dataset, dataset_conf) in datasets_config
    for window in dataset_conf["windows"]
        for (config_idx, config) in enumerate(model_configs)
            for seed in seeds
                experiment_counter += 1

                Random.seed!(seed)

                elapsed = time() - start_time
                avg_time = (experiment_counter > 1) ? elapsed / (experiment_counter - 1) : 0.0
                eta = avg_time * (total_experiments - experiment_counter)

                println("\n" * "="^70)
                println("EXPERIMENTO $experiment_counter/$total_experiments")
                println("Progreso: $(round(experiment_counter/total_experiments*100, digits=2))%")
                println("Tiempo transcurrido: $(round(elapsed/60, digits=2)) min")
                println("ETA: $(round(eta/60, digits=2)) min")
                println("="^70)
                println("Dataset: $dataset")
                println("Window: $window")
                println("Config: $config_idx/$(length(model_configs))")
                println("Seed: $seed")
                println("="^70)

                try
                    results_df, tree, history = run_experiment(
                        dataset=dataset,
                        window=window,
                        normalization=normalization,
                        model=model,
                        config_id=config_idx,
                        seed=seed,
                        target_col=dataset_conf["target_col"]
                    )

                    # results_df ya tiene columna "error" (vacía). Schema unificado. ✓
                    if experiment_counter == 1
                        CSV.write(consolidated_file, results_df)
                    else
                        CSV.write(consolidated_file, results_df, append=true)
                    end

                    println("\n✓ Experimento completado exitosamente")

                catch e
                    println("\n✗ ERROR en experimento:")
                    println("  Dataset: $dataset")
                    println("  Window: $window")
                    println("  Config: $config_idx")
                    println("  Seed: $seed")
                    println("  Error: $e")
                    showerror(stdout, e, catch_backtrace())
                    println()

                    # CORRECCIÓN: Mismo schema que results_df.
                    # Todas las columnas numéricas son missing, "error" tiene el mensaje.
                    error_df = DataFrame(
                        dataset = [dataset],
                        window = [window],
                        normalization = [normalization],
                        model = [string(model)],
                        config_id = [config_idx],
                        seed = [seed],
                        max_nodes = [missing],
                        min_improvement = [missing],
                        use_division = [missing],
                        strategy = [missing],
                        mse_initial = [missing],
                        mse_final_train = [missing],
                        mse_test = [missing],
                        improvement_pct = [missing],
                        iterations = [missing],
                        converged = [missing],
                        error = [string(e)]
                    )

                    if experiment_counter == 1
                        CSV.write(consolidated_file, error_df)
                    else
                        CSV.write(consolidated_file, error_df, append=true)
                    end
                end
            end
        end
    end
end

total_time = time() - start_time

println("\n" * "="^70)
println("BARRIDO COMPLETADO")
println("="^70)
println("Total de experimentos: $total_experiments")
println("Tiempo total: $(round(total_time/60, digits=2)) minutos")
println("Tiempo promedio por experimento: $(round(total_time/total_experiments, digits=2)) segundos")
println("\nResultados consolidados en:")
println("  $consolidated_file")
println("="^70)