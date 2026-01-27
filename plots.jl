using Serialization
using Statistics
using Plots
using DataFrames

println("="^70)
println("GENERANDO VISUALIZACIONES")
println("="^70)

# Crear directorio para plots
mkpath("plots")

# Cargar resultados con manejo de errores
results_dir = "results/raw"
if !isdir(results_dir)
    println("❌ Error: No existe el directorio $results_dir")
    println("   Ejecuta primero run_experiment.jl o sweep_experiments.jl")
    exit(1)
end

# Filtrar solo archivos .jls
files = filter(f -> endswith(f, ".jls"), readdir(results_dir, join=true))

if isempty(files)
    println("❌ No se encontraron archivos .jls en $results_dir")
    println("   Ejecuta primero run_experiment.jl o sweep_experiments.jl")
    exit(1)
end

println("Archivos .jls encontrados: $(length(files))")

# Cargar resultados
results = []
for (i, file) in enumerate(files)
    try
        r = deserialize(file)
        push!(results, r)
        println("✓ [$i/$(length(files))] Cargado: $(basename(file))")
    catch e
        println("❌ [$i/$(length(files))] Error cargando $(basename(file)): $e")
    end
end

if isempty(results)
    println("❌ No se pudieron cargar resultados válidos")
    exit(1)
end

println("\n✓ Total de resultados cargados: $(length(results))\n")

# Extraer datos
train_mse = [r["train_mse_final"] for r in results]
test_mse = [r["test_mse"] for r in results]
nodes = [r["max_nodes"] for r in results]
windows = [r["window"] for r in results]
strategies = [String(r["strategy"]) for r in results]
min_improvements = [r["min_improvement"] for r in results]
seeds = [get(r, "seed", 1) for r in results]  # Por si no existe el campo

# ============================================================
# PLOT 1: Complejidad (nodos) vs Error de Test
# ============================================================
println("📊 Generando Plot 1: Complejidad vs Error...")

p1 = scatter(
    nodes,
    test_mse,
    xlabel = "Número máximo de nodos",
    ylabel = "MSE en test",
    title = "Complejidad del Modelo vs Error de Test",
    legend = false,
    markersize = 6,
    markercolor = :blue,
    markerstrokewidth = 1,
    markerstrokecolor = :darkblue,
    grid = true
)

savefig(p1, "plots/01_complejidad_vs_error.png")
println("✓ Guardado: plots/01_complejidad_vs_error.png")

# ============================================================
# PLOT 2: Tamaño de ventana vs Error de Test
# ============================================================
println("📊 Generando Plot 2: Ventana vs Error...")

p2 = scatter(
    windows,
    test_mse,
    xlabel = "Tamaño de ventana temporal",
    ylabel = "MSE en test",
    title = "Tamaño de Ventana vs Error de Test",
    legend = false,
    markersize = 6,
    markercolor = :green,
    markerstrokewidth = 1,
    markerstrokecolor = :darkgreen,
    grid = true
)

savefig(p2, "plots/02_ventana_vs_error.png")
println("✓ Guardado: plots/02_ventana_vs_error.png")

# ============================================================
# PLOT 3: Error de Entrenamiento vs Error de Test
# ============================================================
println("📊 Generando Plot 3: Train vs Test MSE...")

p3 = scatter(
    train_mse,
    test_mse,
    xlabel = "MSE en entrenamiento",
    ylabel = "MSE en test",
    title = "Error de Entrenamiento vs Error de Test",
    legend = false,
    markersize = 6,
    markercolor = :red,
    markerstrokewidth = 1,
    markerstrokecolor = :darkred,
    grid = true
)
# Añadir línea diagonal (donde train_mse = test_mse)
plot!(p3, [minimum([train_mse; test_mse]), maximum([train_mse; test_mse])],
          [minimum([train_mse; test_mse]), maximum([train_mse; test_mse])],
          linestyle=:dash, linecolor=:black, label="Train = Test", legend=:bottomright)

savefig(p3, "plots/03_train_vs_test.png")
println("✓ Guardado: plots/03_train_vs_test.png")

# ============================================================
# PLOT 4: Boxplot por número de nodos
# ============================================================
if length(unique(nodes)) > 1
    println("📊 Generando Plot 4: Boxplot por nodos...")
    
    p4 = boxplot(
        string.(nodes),
        test_mse,
        xlabel = "Número máximo de nodos",
        ylabel = "MSE en test",
        title = "Distribución de Error por Complejidad",
        legend = false,
        fillcolor = :lightblue,
        linecolor = :blue
    )
    
    savefig(p4, "plots/04_boxplot_nodes.png")
    println("✓ Guardado: plots/04_boxplot_nodes.png")
end

# ============================================================
# PLOT 5: Boxplot por tamaño de ventana
# ============================================================
if length(unique(windows)) > 1
    println("📊 Generando Plot 5: Boxplot por ventana...")
    
    p5 = boxplot(
        string.(windows),
        test_mse,
        xlabel = "Tamaño de ventana",
        ylabel = "MSE en test",
        title = "Distribución de Error por Tamaño de Ventana",
        legend = false,
        fillcolor = :lightgreen,
        linecolor = :green
    )
    
    savefig(p5, "plots/05_boxplot_windows.png")
    println("✓ Guardado: plots/05_boxplot_windows.png")
end

# ============================================================
# PLOT 6: Comparación por estrategia
# ============================================================
if length(unique(strategies)) > 1
    println("📊 Generando Plot 6: Comparación por estrategia...")
    
    p6 = boxplot(
        strategies,
        test_mse,
        xlabel = "Estrategia",
        ylabel = "MSE en test",
        title = "Comparación de Estrategias",
        legend = false,
        fillcolor = :lightyellow,
        linecolor = :orange
    )
    
    savefig(p6, "plots/06_boxplot_strategies.png")
    println("✓ Guardado: plots/06_boxplot_strategies.png")
end

# ============================================================
# PLOT 7: Heatmap (si hay suficientes combinaciones)
# ============================================================
if length(unique(nodes)) > 1 && length(unique(windows)) > 1
    println("📊 Generando Plot 7: Heatmap...")
    
    # Crear matriz para el heatmap
    unique_nodes = sort(unique(nodes))
    unique_windows = sort(unique(windows))
    
    heatmap_data = zeros(length(unique_windows), length(unique_nodes))
    heatmap_counts = zeros(Int, length(unique_windows), length(unique_nodes))
    
    for i in 1:length(results)
        w_idx = findfirst(==(windows[i]), unique_windows)
        n_idx = findfirst(==(nodes[i]), unique_nodes)
        heatmap_data[w_idx, n_idx] += test_mse[i]
        heatmap_counts[w_idx, n_idx] += 1
    end
    
    # Promediar si hay múltiples experimentos con los mismos parámetros
    heatmap_data = heatmap_data ./ max.(heatmap_counts, 1)
    
    p7 = heatmap(
        unique_nodes,
        unique_windows,
        heatmap_data,
        xlabel = "Número máximo de nodos",
        ylabel = "Tamaño de ventana",
        title = "MSE Test: Ventana × Nodos",
        color = :viridis,
        colorbar_title = "MSE"
    )
    
    savefig(p7, "plots/07_heatmap_window_nodes.png")
    println("✓ Guardado: plots/07_heatmap_window_nodes.png")
end

# ============================================================
# PLOT 8: Grid de plots combinado
# ============================================================
println("📊 Generando Plot 8: Grid combinado...")

# Recrear plots para el grid
sp1 = scatter(nodes, test_mse, xlabel="Max Nodes", ylabel="Test MSE", 
              legend=false, markersize=4, title="Nodes vs Error")
              
sp2 = scatter(windows, test_mse, xlabel="Window", ylabel="Test MSE",
              legend=false, markersize=4, title="Window vs Error")
              
sp3 = scatter(train_mse, test_mse, xlabel="Train MSE", ylabel="Test MSE",
              legend=false, markersize=4, title="Train vs Test")

sp4 = histogram(test_mse, xlabel="Test MSE", ylabel="Frecuencia",
                legend=false, bins=20, title="Distribución Test MSE")

p8 = plot(sp1, sp2, sp3, sp4, layout=(2,2), size=(1000, 800),
          plot_title="Resumen de Resultados DoME")

savefig(p8, "plots/08_combined_grid.png")
println("✓ Guardado: plots/08_combined_grid.png")

# ============================================================
# RESUMEN
# ============================================================
println("\n" * "="^70)
println("VISUALIZACIONES COMPLETADAS")
println("="^70)
println("Archivos generados en el directorio 'plots/':")
println("  1. 01_complejidad_vs_error.png - Nodos vs MSE test")
println("  2. 02_ventana_vs_error.png - Ventana vs MSE test")
println("  3. 03_train_vs_test.png - MSE train vs MSE test")
println("  4. 04_boxplot_nodes.png - Distribución por nodos")
println("  5. 05_boxplot_windows.png - Distribución por ventana")
println("  6. 06_boxplot_strategies.png - Comparación estrategias")
println("  7. 07_heatmap_window_nodes.png - Heatmap ventana × nodos")
println("  8. 08_combined_grid.png - Grid resumen")
println("="^70)

println("\n✓ Todas las visualizaciones se han generado exitosamente")
println("  Puedes verlas en el directorio: plots/")