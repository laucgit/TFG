"""
Script maestro para generar TODAS las visualizaciones de un experimento DoME

Genera:
    - 9 gráficas de rendimiento y análisis de errores
    - 4 gráficas de análisis del árbol y complejidad
    - 2 reportes detallados en texto

Total: 13 gráficas + 2 reportes

Uso:
    julia generate_all_plots.jl [exp_id]

Ejemplo:
    julia generate_all_plots.jl exp_mt196_corrected
"""

println("="^80)
println("GENERADOR MAESTRO DE VISUALIZACIONES DoME")
println("="^80)

# Obtener ID del experimento
exp_id = length(ARGS) >= 1 ? ARGS[1] : "exp_mt196_corrected"

println("\n Experimento: $exp_id")
println("Fecha: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Verificar que existen los archivos necesarios
predictions_file = "results/$(exp_id)_predictions.csv"
report_file = "results/$(exp_id).txt"

if !isfile(predictions_file)
        println("\nERROR: No se encontró $predictions_file")
        println("   Ejecuta primero: julia run_experiment.jl")
        exit(1)
end

if !isfile(report_file)
        println("\nERROR: No se encontró $report_file")
        println("   Ejecuta primero: julia run_experiment.jl")
        exit(1)
end

println("\nArchivos de entrada encontrados")

# Crear directorio de plots
mkpath("plots")

# ============================================================
# PARTE 1: VISUALIZACIONES DE RENDIMIENTO
# ============================================================

println("\n" * "="^80)
println("PARTE 1: GENERANDO VISUALIZACIONES DE RENDIMIENTO")
println("="^80)

println("\nEjecutando plot_single_experiment.jl...")
include("plot_single_experiment.jl")

# ============================================================
# PARTE 2: VISUALIZACIONES DEL ÁRBOL
# ============================================================

println("\n" * "="^80)
println("PARTE 2: GENERANDO VISUALIZACIONES DEL ÁRBOL")
println("="^80)

println("\nEjecutando plot_tree_visualization.jl...")
include("plot_tree_visualization.jl")

# ============================================================
# RESUMEN FINAL
# ============================================================

println("\n" * "="^80)
println("TODAS LAS VISUALIZACIONES COMPLETADAS")
println("="^80)

# Contar archivos generados
plot_files = filter(f -> contains(f, exp_id), readdir("plots"))
png_files = filter(f -> endswith(f, ".png"), plot_files)
txt_files = filter(f -> endswith(f, ".txt"), plot_files)

println("\nResumen de archivos generados:")
println("   Gráficas PNG: $(length(png_files))")
println("   Reportes TXT: $(length(txt_files))")
println("   Total: $(length(plot_files)) archivos")

println("\nUbicación: plots/")

println("\nGráficas generadas:")
for (i, file) in enumerate(sort(png_files))
        println("   $i. $file")
end

println("\nReportes generados:")
for (i, file) in enumerate(sort(txt_files))
        println("   $i. $file")
end

println("\nSugerencias:")
println("   1. Visualizar resumen: xdg-open plots/$(exp_id)_08_summary_grid.png")
println("   2. Leer análisis: cat plots/$(exp_id)_REPORTE_VISUALIZACIONES.txt")
println("   3. Ver importancia variables: xdg-open plots/$(exp_id)_10_variable_importance.png")

println("\n" * "="^80)
println("Proceso completado con éxito!")
println("="^80)