include("run_experiment.jl")

# Configuración del sweep
dataset = "data/ElectricDevices/LD2011_2014_mini.txt"
target_col = "MT_196"  # Columna a predecir
windows = [12, 24, 48]  # Diferentes tamaños de ventana
horizon = 1
max_nodes_list = [10, 15, 20]  # Diferentes complejidades del modelo
strategies = [:selective]  # Solo selective por ahora (exhaustive si está disponible)
min_improvements = [1e-5, 1e-6]  # Diferentes umbrales de mejora
seeds = 1:3  # Múltiples semillas para robustez

# Calcular total de experimentos
total_experiments = length(windows) * length(max_nodes_list) * length(strategies) * 
                   length(min_improvements) * length(seeds)

println("="^70)
println("INICIANDO BARRIDO DE EXPERIMENTOS")
println("="^70)
println("Total de experimentos: $total_experiments")
println("\nConfiguración:")
println("  - Dataset: $dataset")
println("  - Target column: $target_col")
println("  - Windows: $windows")
println("  - Max nodes: $max_nodes_list")
println("  - Strategies: $strategies")
println("  - Min improvements: $min_improvements")
println("  - Seeds: $seeds")
println("="^70)
println()

exp_counter = 1
successful_exps = 0
failed_exps = 0

for window in windows,
    max_nodes in max_nodes_list,
    strategy in strategies,
    min_imp in min_improvements,
    seed in seeds
    
    global exp_counter, successful_exps, failed_exps
    
    exp_id = "sweep_$(lpad(exp_counter, 3, '0'))"
    
    println("\n" * "─"^70)
    println("Experimento $exp_counter de $total_experiments")
    println("ID: $exp_id")
    println("Parámetros:")
    println("  - window=$window")
    println("  - max_nodes=$max_nodes")
    println("  - strategy=$strategy")
    println("  - min_improvement=$min_imp")
    println("  - seed=$seed")
    println("─"^70)
    
    try
        run_experiment(
            dataset=dataset,
            target_col=target_col,
            window=window, 
            horizon=horizon, 
            max_nodes=max_nodes, 
            strategy=strategy, 
            min_improvement=min_imp, 
            seed=seed, 
            exp_id=exp_id
        )
        println("Experimento $exp_id completado exitosamente")
        successful_exps += 1
    catch e
        println("Error en experimento $exp_id:")
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

# Opcional: Consolidar todos los resultados en un solo CSV
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
        
        # Mostrar estadísticas básicas
        println("\nEstadísticas del barrido:")
        println("  - MSE test promedio: $(round(mean(all_results.test_mse), digits=6))")
        println("  - MSE test mínimo: $(round(minimum(all_results.test_mse), digits=6))")
        println("  - MSE test máximo: $(round(maximum(all_results.test_mse), digits=6))")
        
        # Mejor configuración
        best_idx = argmin(all_results.test_mse)
        println("\nMejor configuración:")
        println("  - Window: $(all_results.window[best_idx])")
        println("  - Max nodes: $(all_results.max_nodes[best_idx])")
        println("  - Min improvement: $(all_results.min_improvement[best_idx])")
        println("  - MSE test: $(round(all_results.test_mse[best_idx], digits=6))")
    end
catch e
    println("Advertencia: Error al consolidar resultados: $e")
end