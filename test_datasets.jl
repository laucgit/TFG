include("run_experiment_multi.jl")
using Statistics
using SymDoME
using Random

println("="^70)
println("TEST DE VALIDACIÓN - Todos los datasets del sweep")
println("="^70)
println("\nOBJETIVO: Verificar que DoME funciona en los 4 datasets antes de CESGA")
println("\nESTRATEGIA:")
println("  - Mismo conjunto de datasets y target_cols que el sweep")
println("  - Window=12 (el más pequeño, para rapidez)")
println("  - 2000 muestras train / 500 test (cap)")
println("  - 6 configuraciones por dataset")
println("  - Si un dataset falla, se sigue con los demás")
println("="^70)

# Mismo diccionario que en sweep_experiments_multi.jl
datasets_config = Dict(
    "ETT/ETTh2.csv" => "OT",
    "ETT/ETTm1.csv" => "OT",
    "ElectricDevices/LD2011_2014.txt" => "MT_196",
    "LCDS/LCD_USW00094789_2024.csv" => "HourlyDryBulbTemperature"
)

window = 12  # El más pequeño del sweep, suficiente para validar

test_configs = [
    Dict("maxNodes" => 5,   "minReduction" => 1e-4, "useDiv" => false, "strategy" => SymDoME.Strategy4),
    Dict("maxNodes" => 20,  "minReduction" => 1e-5, "useDiv" => false, "strategy" => SymDoME.Strategy4),
    Dict("maxNodes" => 30,  "minReduction" => 1e-6, "useDiv" => false, "strategy" => SymDoME.Strategy3),
    Dict("maxNodes" => 50,  "minReduction" => 1e-7, "useDiv" => true,  "strategy" => SymDoME.Strategy3),
    Dict("maxNodes" => 100, "minReduction" => 1e-8, "useDiv" => false, "strategy" => SymDoME.Strategy4),
    Dict("maxNodes" => 150, "minReduction" => 1e-9, "useDiv" => true,  "strategy" => SymDoME.Strategy3),
]

# Resumen global: un vector de (dataset_name, passed::Bool, best_improvement, best_config_idx, error_msg)
global_results = Tuple{String, Bool, Float64, Any, String}[]

for (dataset_name, target_col) in datasets_config

    println("\n" * "="^70)
    println("DATASET: $dataset_name  →  target: $target_col")
    println("="^70)

    try
        # Cargar datos
        Xtr, ytr, Xte, yte = load_dataset(
            dataset_name,
            data_dir="data",
            window=window,
            horizon=1,
            target_col=target_col,
            train_ratio=0.75
        )

        # Limitar muestras para que sea rápido
        n_samples = min(2000, size(Xtr, 1))
        Xtr = Xtr[1:n_samples, :]
        ytr = ytr[1:n_samples]

        n_test = min(500, size(Xte, 1))
        Xte = Xte[1:n_test, :]
        yte = yte[1:n_test]

        println("\n[DATOS] Xtr=$(size(Xtr)), Xte=$(size(Xte))")

        # Imputar (impute_nonfinite! viene de run_experiment_multi.jl)
        impute_nonfinite!(Xtr)
        impute_nonfinite!(ytr)
        impute_nonfinite!(Xte)
        impute_nonfinite!(yte)

        # Normalizar
        X_min = minimum(Xtr, dims=1)
        X_max = maximum(Xtr, dims=1)
        X_range = X_max .- X_min
        X_range[X_range .== 0] .= 1.0

        Xtr_norm = (Xtr .- X_min) ./ X_range
        Xte_norm = (Xte .- X_min) ./ X_range

        y_min = minimum(ytr)
        y_max = maximum(ytr)
        y_range = y_max - y_min
        if y_range == 0
            y_range = 1.0
        end

        ytr_norm = (ytr .- y_min) ./ y_range
        yte_norm = (yte .- y_min) ./ y_range

        println("  OK: Imputado y normalizado")

        # Looping configs
        dataset_passed = false
        best_improvement = 0.0
        best_config = nothing

        for (i, cfg) in enumerate(test_configs)
            println("\n  --- Config $i/$(length(test_configs)) ---")
            println("    maxNodes=$(cfg["maxNodes"])  minReduction=$(cfg["minReduction"])  useDiv=$(cfg["useDiv"])  strategy=$(cfg["strategy"])")

            try
                Random.seed!(42 + i)

                params = DoMEParams(
                    cfg["maxNodes"],
                    cfg["minReduction"],
                    cfg["useDiv"],
                    cfg["strategy"]
                )

                t_start = time()
                tree, history = train_dome(Xtr_norm, ytr_norm, params)
                t_elapsed = time() - t_start

                ŷ_test, mse_test = evaluate_dome(tree, Xte_norm, yte_norm)

                improvement = (1 - history[end]/history[1]) * 100

                println("    MSE train: $(round(history[1], digits=6)) → $(round(history[end], digits=6))")
                println("    MSE test:  $(round(mse_test, digits=6))")
                println("    Mejora:    $(round(improvement, digits=2))%  |  Iters: $(length(history))  |  Tiempo: $(round(t_elapsed, digits=1))s")

                if improvement > best_improvement
                    best_improvement = improvement
                    best_config = i
                end

                if improvement > 0.5
                    println("    PASO: MEJORA DETECTADA")
                    dataset_passed = true
                else
                    println("    FALLO: Sin mejora significativa")
                end

            catch e
                println("    ERROR: $e")
            end
        end

        # Resumen por dataset
        println("\n  " * "-"^66)
        if dataset_passed
            println("  PASO: $dataset_name  (mejor mejora: $(round(best_improvement, digits=2))% en config $best_config)")
        else
            println("  FALLO: $dataset_name  (ninguna config mejora > 0.5%)")
        end
        println("  " * "-"^66)

        push!(global_results, (dataset_name, dataset_passed, best_improvement, best_config, ""))

    catch e
        println("\n  ERROR AL CARGAR/PROCESAR $dataset_name:")
        println("    $e")
        showerror(stdout, e, catch_backtrace())
        push!(global_results, (dataset_name, false, 0.0, nothing, string(e)))
    end
end

# =============================================================================
# RESUMEN GLOBAL
# =============================================================================
println("\n" * "="^70)
println("RESUMEN GLOBAL")
println("="^70)

all_passed = true
for (name, passed, best_imp, best_cfg, err) in global_results
    if passed
        println("  PASO: $name  →  mejor mejora $(round(best_imp, digits=2))% (config $best_cfg)")
    elseif !isempty(err)
        println("  FALLO: $name  →  ERROR: $err")
        all_passed = false
    else
        println("  FALLO: $name  →  sin mejora significativa")
        all_passed = false
    end
end

println("="^70)
if all_passed
    println("\nTODOS LOS DATASETS PASARON")
    println("\n>>> LISTO PARA CESGA <<<")
else
    n_passed = count(r -> r[2], global_results)
    n_total  = length(global_results)
    println("\nRESULTADO MIXTO: $n_passed/$n_total datasets pasaron")
    println("\nLos que fallaron pueden ser:")
    println("  - Archivos que no existen en data/ (verificar rutas)")
    println("  - Datos sin señal simbólica clara (no es un bug, es del dataset)")
    println("  - Consultar con tu profesor antes de subir a CESGA")
end
println("="^70)