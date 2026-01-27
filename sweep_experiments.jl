include("run_experiment.jl")

# Configuración del sweep
dataset = "data/ElectricDevices/LD2011_2014_mini.txt"  # Ruta correcta al dataset
windows = [12, 24]  # Diferentes tamaños de ventana
horizon = 1
max_nodes_list = [10, 15]  # Diferentes complejidades del modelo
strategies = [:selective, :exhaustive]  # Diferentes estrategias (aunque no se usan actualmente)
min_improvements = [1e-5, 1e-6]  # Diferentes umbrales de mejora
seeds = 1:3  # Múltiples semillas para robustez (reducido a 3 para pruebas rápidas)

# Calcular total de experimentos
total_experiments = length(windows) * length(max_nodes_list) * length(strategies) * 
                   length(min_improvements) * length(seeds)

println("="^70)
println("INICIANDO BARRIDO DE EXPERIMENTOS")
println("="^70)
println("Total de experimentos: $total_experiments")
println("Configuración:")
println("  - Windows: $windows")
println("  - Max nodes: $max_nodes_list")
println("  - Strategies: $strategies")
println("  - Min improvements: $min_improvements")
println("  - Seeds: $seeds")
println("="^70)
println()

exp_counter = 1

for window in windows,
    max_nodes in max_nodes_list,
    strategy in strategies,
    min_imp in min_improvements,
    seed in seeds
    
    global exp_counter  # CRÍTICO: declarar como global para modificarlo dentro del bucle
    
    exp_id = "exp_$(lpad(exp_counter, 3, '0'))"  # Formato: exp_001, exp_002, etc.
    
    println("\n" * "─"^70)
    println("Experimento $exp_counter de $total_experiments")
    println("ID: $exp_id")
    println("Parámetros: window=$window, max_nodes=$max_nodes, strategy=$strategy")
    println("            min_improvement=$min_imp, seed=$seed")
    println("─"^70)
    
    try
        run_experiment(
            dataset=dataset, 
            window=window, 
            horizon=horizon, 
            max_nodes=max_nodes, 
            strategy=strategy, 
            min_improvement=min_imp, 
            seed=seed, 
            exp_id=exp_id
        )
        println("✓ Experimento $exp_id completado exitosamente")
    catch e
        println("Error en experimento $exp_id:")
        println("   $e")
        println("   Continuando con el siguiente experimento...")
    end
    
    exp_counter += 1
end

println("\n" * "="^70)
println("BARRIDO COMPLETADO")
println("="^70)
println("Total de experimentos ejecutados: $(exp_counter - 1)")
println("Resultados guardados en:")
println("  - Binarios: results/raw/")
println("  - CSVs: resultados_experimentos/")
println("="^70)