include("run_experiment_multi.jl")

# ==========================================
# CONFIGURACIÓN DEL BARRIDO
# ==========================================

# Lista de datasets a probar
datasets_config = [
    # ElectricDevices - archivo único con train/test por ratio
    Dict(
        "name" => "ElectricDevices/LD2011_2014_mini.txt",
        "target_col" => "MT_196",
        "train_ratio" => 0.75
    ),
    
    # CinCECGTorso - archivos separados _TRAIN y _TEST
    Dict(
        "name" => "CinCECGTorso",
        "target_col" => nothing,  # Usa última columna
        "train_ratio" => 0.75 # No se usa
    ),
    
    # CAMBIAR!!
    Dict(
        "name" => "PEMS_SF",
        "target_col" => 1,  # Primera columna de sensores
        "train_ratio" => 0.75  # No se usa
    )
]

# Parámetros de los experimentos
windows = [12, 24, 48]  # Diferentes tamaños de ventana
horizon = 1
max_nodes_list = [10, 15, 20]  # Diferentes complejidades del modelo
strategies = [:selective]  # Solo selective por ahora
min_improvements = [1e-5, 1e-6]  # Diferentes umbrales de mejora
seeds = 1:3  # Múltiples semillas para robustez

# ==========================================
# CÁLCULO Y RESUMEN
# ==========================================

total_experiments = length(datasets_config) * length(windows) * length(max_nodes_list) * 
                   length(strategies) * length(min_improvements) * length(seeds)

println("="^70)
println("INICIANDO BARRIDO DE EXPERIMENTOS MULTI-DATASET")
println("="^70)
println("Total de experimentos: $total_experiments")
println("\nDatasets:")
for (i, ds) in enumerate(datasets_config)
    println("  $i. $(ds["name"])")
    println("     Target: $(ds["target_col"] === nothing ? "última columna" : ds["target_col"])")
end
println("\nParámetros:")
println("  - Windows: $windows")
println("  - Max nodes: $max_nodes_list")
println("  - Strategies: $strategies")
println("  - Min improvements: $min_improvements")
println("  - Seeds: $seeds")
println("="^70)
println()

# ==========================================
# EJECUCIÓN DEL BARRIDO
# ==========================================

exp_counter = 1
successful_exps = 0
failed_exps = 0

for ds_config in datasets_config,
    window in windows,
    max_nodes in max_nodes_list,
    strategy in strategies,
    min_imp in min_improvements,
    seed in seeds
    
    global exp_counter, successful_exps, failed_exps
    
    dataset_name = ds_config["name"]
    target_col = ds_config["target_col"]
    train_ratio = ds_config["train_ratio"]
    
    # Crear ID único
    dataset_short = replace(split(dataset_name, "/")[end], r"\.(txt|csv|arff|ts)$" => "")
    exp_id = "sweep_$(dataset_short)_w$(window)_n$(max_nodes)_s$(seed)"
    
    println("\n" * "─"^70)
    println("Experimento $exp_counter de $total_experiments")
    println("ID: $exp_id")
    println("Parámetros:")
    println("  - dataset=$dataset_name")
    println("  - target_col=$target_col")
    println("  - window=$window")
    println("  - max_nodes=$max_nodes")
    println("  - strategy=$strategy")
    println("  - min_improvement=$min_imp")
    println("  - seed=$seed")
    println("─"^70)
    
    try
        run_experiment(
            dataset=dataset_name,
            target_col=target_col,
            window=window, 
            horizon=horizon, 
            max_nodes=max_nodes, 
            strategy=strategy, 
            min_improvement=min_imp, 
            seed=seed, 
            exp_id=exp_id,
            train_ratio=train_ratio
        )
        println("✓ Experimento $exp_id completado exitosamente")
        successful_exps += 1
    catch e
        println("✗ Error en experimento $exp_id:")
        println("   $e")
        if isdefined(Main, :stacktrace)
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end
        println("   Continuando con el siguiente experimento...")
        failed_exps += 1
    end
    
    exp_counter += 1
end

# ==========================================
# RESUMEN FINAL
# ==========================================

println("\n" * "="^70)
println("BARRIDO COMPLETADO")
println("="^70)
println("Total de experimentos: $(exp_counter - 1)")
println("  - Exitosos: $successful_exps")
println("  - Fallidos: $failed_exps")
println("\nResultados guardados en:")
println("  - Resúmenes CSV: results/*.csv")
println("  - Reportes TXT: results/*.txt")
println("  - Predicciones: results/*_predictions.csv")
println("="^70)

# ==========================================
# CONSOLIDACIÓN DE RESULTADOS
# ==========================================

try
    println("\nConsolidando resultados...")
    
    # Leer todos los CSV de resultados
    result_files = filter(f -> endswith(f, ".csv") && !contains(f, "predictions"), 
                         readdir("results", join=true))
    
    if !isempty(result_files)
        all_results = DataFrame()
        
        for file in result_files
            df = CSV.read(file, DataFrame)
            all_results = vcat(all_results, df)
        end
        
        # Guardar CSV consolidado
        consolidated_file = "results/sweep_consolidated_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
        CSV.write(consolidated_file, all_results)
        
        println("Resultados consolidados guardados en: $consolidated_file")
        
        # Estadísticas por dataset
        println("\n" * "="^70)
        println("ESTADÍSTICAS POR DATASET")
        println("="^70)
        
        for ds_name in unique(all_results.dataset)
            ds_results = filter(row -> row.dataset == ds_name, all_results)
            
            println("\nDataset: $ds_name")
            println("  Experimentos: $(nrow(ds_results))")
            println("  MSE test promedio: $(round(mean(ds_results.test_mse), digits=6))")
            println("  MSE test mínimo: $(round(minimum(ds_results.test_mse), digits=6))")
            println("  MSE test máximo: $(round(maximum(ds_results.test_mse), digits=6))")
            
            # Mejor configuración para este dataset
            best_idx = argmin(ds_results.test_mse)
            println("  Mejor configuración:")
            println("    - Window: $(ds_results.window[best_idx])")
            println("    - Max nodes: $(ds_results.max_nodes[best_idx])")
            println("    - Min improvement: $(ds_results.min_improvement[best_idx])")
            println("    - MSE test: $(round(ds_results.test_mse[best_idx], digits=6))")
        end
        
        # Estadísticas globales
        println("\n" * "="^70)
        println("ESTADÍSTICAS GLOBALES")
        println("="^70)
        println("MSE test promedio (todos): $(round(mean(all_results.test_mse), digits=6))")
        println("MSE test mínimo (todos): $(round(minimum(all_results.test_mse), digits=6))")
        println("MSE test máximo (todos): $(round(maximum(all_results.test_mse), digits=6))")
        
        # Mejor configuración global
        best_idx = argmin(all_results.test_mse)
        println("\nMejor configuración global:")
        println("  - Dataset: $(all_results.dataset[best_idx])")
        println("  - Window: $(all_results.window[best_idx])")
        println("  - Max nodes: $(all_results.max_nodes[best_idx])")
        println("  - Min improvement: $(all_results.min_improvement[best_idx])")
        println("  - MSE test: $(round(all_results.test_mse[best_idx], digits=6))")
        println("="^70)
    end
catch e
    println("Advertencia: Error al consolidar resultados: $e")
end