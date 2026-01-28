using Plots
using Statistics
using CSV
using DataFrames
using Printf
using Dates
using StatsPlots
using Distributions  # AÑADIDO: Necesario para Normal() en Q-Q plot

"""
Script para generar visualizaciones de un experimento individual de DoME

Uso:
    julia plot_single_experiment.jl [exp_id]
    
Ejemplo:
    julia plot_single_experiment.jl exp_1
"""

println("="^80)
println("GENERANDO VISUALIZACIONES DEL EXPERIMENTO")
println("="^80)

# Determinar el ID del experimento
exp_id = length(ARGS) >= 1 ? ARGS[1] : "exp_1"

println("\nExperimento: $exp_id")

# Crear directorio para plots
mkpath("plots")

# ============================================================
# CARGAR DATOS
# ============================================================

println("\nCargando datos...")

# Cargar predicciones
predictions_file = "results/$(exp_id)_predictions.csv"
if !isfile(predictions_file)
        println("Error: No se encontró $predictions_file")
        println("   Ejecuta primero run_experiment.jl con exp_id='$exp_id'")
        exit(1)
end

df_pred = CSV.read(predictions_file, DataFrame)
println("Predicciones cargadas: $(nrow(df_pred)) muestras")

# Cargar resumen
summary_file = "results/$(exp_id).csv"
if isfile(summary_file)
        df_summary = CSV.read(summary_file, DataFrame)
        println("Resumen cargado")
else
        df_summary = nothing
        println("No se encontró archivo de resumen")
end

# Leer el reporte de texto para extraer información
report_file = "results/$(exp_id).txt"

# Declarar variables globales correctamente
global expression = "No disponible"
global train_mse = 0.0
global test_mse = 0.0
global num_nodes = 0

if isfile(report_file)
        lines = readlines(report_file)
        for (i, line) in enumerate(lines)
                if contains(line, "MSE entrenamiento (final):")
                        global train_mse = parse(Float64, strip(split(line, ":")[2]))
                elseif contains(line, "MSE test:")
                        global test_mse = parse(Float64, strip(split(line, ":")[2]))
                elseif contains(line, "Número de nodos:")
                        global num_nodes = parse(Int, strip(split(line, ":")[2]))
                elseif contains(line, "EXPRESIÓN MATEMÁTICA:")
                        # La expresión está en la siguiente línea (i+2)
                        if i + 2 <= length(lines)
                                global expression = strip(lines[i+2])
                        end
                end
        end
        println("Información del modelo extraída")
end

# Extraer datos
y_true = df_pred.y_true
y_pred = df_pred.y_pred
errors = df_pred.error
squared_errors = df_pred.squared_error

# Calcular métricas adicionales
mse = mean(squared_errors)
rmse = sqrt(mse)
mae = mean(abs.(errors))
r2 = 1 - sum(squared_errors) / sum((y_true .- mean(y_true)).^2)

println("\nMétricas calculadas:")
println("   MSE:  $(round(mse, digits=2))")
println("   RMSE: $(round(rmse, digits=2))")
println("   MAE:  $(round(mae, digits=2))")
println("   R²:   $(round(r2, digits=4))")

# ============================================================
# PLOT 1: PREDICCIONES VS VALORES REALES (Scatter)
# ============================================================

println("\nGenerando Plot 1: Predicciones vs Valores Reales...")

# Calcular límites para la línea diagonal
min_val = min(minimum(y_true), minimum(y_pred))
max_val = max(maximum(y_true), maximum(y_pred))

p1 = scatter(
        y_true, y_pred,
        xlabel = "Valores Reales",
        ylabel = "Predicciones",
        title = "Predicciones vs Valores Reales\n(R² = $(round(r2, digits=4)))",
        label = "Predicciones",
        markersize = 4,
        markercolor = :blue,
        markeralpha = 0.6,
        markerstrokewidth = 0,
        grid = true,
        legend = :topleft,
        size = (800, 700)
)

# Línea diagonal perfecta (y = x)
plot!(p1, [min_val, max_val], [min_val, max_val],
            linestyle = :dash, linecolor = :red, linewidth = 2,
            label = "Predicción Perfecta")

# Añadir texto con métricas
annotate!(p1, min_val + 0.05 * (max_val - min_val),
                    max_val - 0.1 * (max_val - min_val),
                    text("MSE: $(round(mse, sigdigits=4))\nRMSE: $(round(rmse, sigdigits=4))\nMAE: $(round(mae, sigdigits=4))",
                             :left, 10))

savefig(p1, "plots/$(exp_id)_01_predictions_vs_real.png")
println("Guardado: plots/$(exp_id)_01_predictions_vs_real.png")

# ============================================================
# PLOT 2: SERIE TEMPORAL DE PREDICCIONES
# ============================================================

println("Generando Plot 2: Serie Temporal...")

p2 = plot(
        1:length(y_true), y_true,
        xlabel = "Índice de muestra (test)",
        ylabel = "Valor",
        title = "Serie Temporal: Predicciones vs Valores Reales",
        label = "Valores Reales",
        linewidth = 2,
        linecolor = :blue,
        legend = :topright,
        size = (1200, 500),
        grid = true
)

plot!(p2, 1:length(y_pred), y_pred,
            label = "Predicciones",
            linewidth = 2,
            linecolor = :red,
            linestyle = :dash)

savefig(p2, "plots/$(exp_id)_02_time_series.png")
println("Guardado: plots/$(exp_id)_02_time_series.png")

# ============================================================
# PLOT 3: DISTRIBUCIÓN DE ERRORES
# ============================================================

println("Generando Plot 3: Distribución de Errores...")

p3 = histogram(
        errors,
        xlabel = "Error (Real - Predicción)",
        ylabel = "Frecuencia",
        title = "Distribución de Errores\n(Media: $(round(mean(errors), digits=2)), Std: $(round(std(errors), digits=2)))",
        label = "Errores",
        bins = 30,
        fillcolor = :lightblue,
        linecolor = :blue,
        legend = :topright,
        size = (800, 600),
        grid = true
)

# Línea vertical en 0
vline!(p3, [0], linecolor = :red, linewidth = 2, linestyle = :dash, label = "Error = 0")

# Línea vertical en la media
vline!(p3, [mean(errors)], linecolor = :green, linewidth = 2, label = "Media")

savefig(p3, "plots/$(exp_id)_03_error_distribution.png")
println("Guardado: plots/$(exp_id)_03_error_distribution.png")

# ============================================================
# PLOT 4: ERRORES A LO LARGO DEL TIEMPO
# ============================================================

println("Generando Plot 4: Evolución de Errores...")

p4 = plot(
        1:length(errors), errors,
        xlabel = "Índice de muestra",
        ylabel = "Error (Real - Predicción)",
        title = "Evolución de Errores en Test",
        label = "Error",
        linewidth = 1,
        linecolor = :blue,
        legend = :topright,
        size = (1200, 500),
        grid = true
)

# Línea en 0
hline!(p4, [0], linecolor = :red, linewidth = 2, linestyle = :dash, label = "Error = 0")

# Banda de ±1 desviación estándar
std_err = std(errors)
hline!(p4, [std_err], linecolor = :orange, linewidth = 1, linestyle = :dot, label = "±1 Std")
hline!(p4, [-std_err], linecolor = :orange, linewidth = 1, linestyle = :dot, label = "")

savefig(p4, "plots/$(exp_id)_04_error_evolution.png")
println("Guardado: plots/$(exp_id)_04_error_evolution.png")

# ============================================================
# PLOT 5: BOXPLOT DE ERRORES
# ============================================================

println("Generando Plot 5: Boxplot de Errores...")

p5 = StatsPlots.boxplot(
        ["Errores"], errors,
        ylabel = "Error (Real - Predicción)",
        title = "Distribución de Errores (Boxplot)",
        legend = false,
        fillcolor = :lightgreen,
        linecolor = :darkgreen,
        grid = true,
        size = (600, 600),
        outliers = true
)

# Añadir línea en 0
hline!(p5, [0], linecolor = :red, linewidth = 2, linestyle = :dash)

# Añadir estadísticas como texto
q1 = quantile(errors, 0.25)
q2 = median(errors)
q3 = quantile(errors, 0.75)
iqr = q3 - q1

annotate!(p5, 1.35, maximum(errors) * 0.9,
         text("Mediana: $(round(q2, digits=2))\nQ1: $(round(q1, digits=2))\nQ3: $(round(q3, digits=2))\nIQR: $(round(iqr, digits=2))",
              :left, 8))

savefig(p5, "plots/$(exp_id)_05_error_boxplot.png")
println("Guardado: plots/$(exp_id)_05_error_boxplot.png")

# ============================================================
# PLOT 6: Q-Q PLOT PARA NORMALIDAD
# ============================================================

println("Generando Plot 6: Q-Q Plot...")

# Ordenar errores
sorted_errors = sort(errors)
n = length(sorted_errors)

# Calcular cuantiles teóricos de una distribución normal
# CORREGIDO: Ahora que importamos Distributions, esto funciona
theoretical_quantiles = quantile(Normal(mean(errors), std(errors)), 
                                (1:n) ./ (n + 1))

p6 = scatter(
        theoretical_quantiles, sorted_errors,
        xlabel = "Cuantiles Teóricos (Normal)",
        ylabel = "Cuantiles Observados",
        title = "Q-Q Plot (Test de Normalidad)",
        label = "Datos",
        markersize = 3,
        markercolor = :blue,
        markeralpha = 0.6,
        markerstrokewidth = 0,
        grid = true,
        legend = :topleft,
        size = (700, 700)
)

# Línea de referencia (si los datos fueran perfectamente normales)
min_q = minimum([minimum(theoretical_quantiles), minimum(sorted_errors)])
max_q = maximum([maximum(theoretical_quantiles), maximum(sorted_errors)])
plot!(p6, [min_q, max_q], [min_q, max_q],
     linecolor = :red, linewidth = 2, linestyle = :dash,
     label = "Línea de Referencia")

savefig(p6, "plots/$(exp_id)_06_qq_plot.png")
println("Guardado: plots/$(exp_id)_06_qq_plot.png")

# ============================================================
# PLOT 7: PANEL DE MÉTRICAS
# ============================================================

println("Generando Plot 7: Panel de Métricas...")

sp1 = scatter([1], [mse], ylims=(0, mse*1.2), xlims=(0.5, 1.5),
                            title="MSE", legend=false, markersize=20, markercolor=:red,
                            xticks=[], ylabel="", size=(200, 200))
annotate!(sp1, 1, mse*1.1, text(@sprintf("%.2e", mse), :center, 10))

sp2 = scatter([1], [rmse], ylims=(0, rmse*1.2), xlims=(0.5, 1.5),
                            title="RMSE", legend=false, markersize=20, markercolor=:orange,
                            xticks=[], ylabel="", size=(200, 200))
annotate!(sp2, 1, rmse*1.1, text(@sprintf("%.2f", rmse), :center, 10))

sp3 = scatter([1], [mae], ylims=(0, mae*1.2), xlims=(0.5, 1.5),
                            title="MAE", legend=false, markersize=20, markercolor=:green,
                            xticks=[], ylabel="", size=(200, 200))
annotate!(sp3, 1, mae*1.1, text(@sprintf("%.2f", mae), :center, 10))

sp4 = scatter([1], [r2], ylims=(min(0, r2*1.1), max(1, r2*1.1)), xlims=(0.5, 1.5),
                            title="R²", legend=false, markersize=20, markercolor=:blue,
                            xticks=[], ylabel="", size=(200, 200))
annotate!(sp4, 1, r2 > 0 ? r2*1.05 : r2*0.95, text(@sprintf("%.4f", r2), :center, 10))

p7 = plot(sp1, sp2, sp3, sp4, layout=(2,2), size=(800, 600),
                    plot_title="Métricas de Rendimiento")

savefig(p7, "plots/$(exp_id)_07_metrics_panel.png")
println("Guardado: plots/$(exp_id)_07_metrics_panel.png")

# ============================================================
# PLOT 8: GRID RESUMEN
# ============================================================

println("Generando Plot 8: Grid Resumen...")

# Mini scatter plot
gp1 = scatter(y_true, y_pred, xlabel="Real", ylabel="Pred",
                            title="Predicciones", legend=false, markersize=2,
                            markercolor=:blue, markeralpha=0.5, grid=true)
plot!(gp1, [min_val, max_val], [min_val, max_val],
            linecolor=:red, linewidth=1, linestyle=:dash)

# Mini histograma de errores
gp2 = histogram(errors, xlabel="Error", ylabel="Freq",
                                title="Distribución Errores", legend=false,
                                bins=20, fillcolor=:lightblue, grid=true)
vline!(gp2, [0], linecolor=:red, linewidth=1)

# Serie temporal (primeros 100 puntos)
n_show = min(100, length(y_true))
gp3 = plot(1:n_show, y_true[1:n_show], label="Real", linewidth=1,
                     xlabel="Muestra", ylabel="Valor", title="Serie Temporal (primeras $n_show)",
                     linecolor=:blue, grid=true)
plot!(gp3, 1:n_show, y_pred[1:n_show], label="Pred", linewidth=1,
            linecolor=:red, linestyle=:dash)

# Boxplot errores
gp4 = StatsPlots.boxplot(["Errores"], errors, ylabel="Error",
                            title="Boxplot Errores", legend=false,
                            fillcolor=:lightgreen, grid=true)

p8 = plot(gp1, gp2, gp3, gp4, layout=(2,2), size=(1200, 900),
                    plot_title="Resumen del Experimento: $exp_id")

savefig(p8, "plots/$(exp_id)_08_summary_grid.png")
println("Guardado: plots/$(exp_id)_08_summary_grid.png")

# ============================================================
# PLOT 9: RESIDUOS VS PREDICCIONES
# ============================================================

println("Generando Plot 9: Residuos vs Predicciones...")

p9 = scatter(
        y_pred, errors,
        xlabel = "Predicciones",
        ylabel = "Residuos (Real - Predicción)",
        title = "Residuos vs Predicciones\n(Heterocedasticidad)",
        label = "Residuos",
        markersize = 4,
        markercolor = :purple,
        markeralpha = 0.6,
        markerstrokewidth = 0,
        grid = true,
        legend = :topright,
        size = (900, 600)
)

# Línea en 0
hline!(p9, [0], linecolor = :red, linewidth = 2, linestyle = :dash, label = "Residuo = 0")

# Bandas de ±2 std
hline!(p9, [2*std(errors)], linecolor = :orange, linewidth = 1, linestyle = :dot, label = "±2 Std")
hline!(p9, [-2*std(errors)], linecolor = :orange, linewidth = 1, linestyle = :dot, label = "")

savefig(p9, "plots/$(exp_id)_09_residuals_vs_predictions.png")
println("Guardado: plots/$(exp_id)_09_residuals_vs_predictions.png")

# ============================================================
# GENERAR REPORTE DE VISUALIZACIONES
# ============================================================

println("\nGenerando reporte...")

report = """
================================================================================
REPORTE DE VISUALIZACIONES: $exp_id
================================================================================

Fecha: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

INFORMACIÓN DEL MODELO:
------------------------------------------------------------------------------
Número de nodos: $num_nodes
Expresión matemática:
    $expression

MÉTRICAS DE RENDIMIENTO:
------------------------------------------------------------------------------
MSE (Mean Squared Error):      $(round(mse, sigdigits=6))
RMSE (Root MSE):               $(round(rmse, sigdigits=6))
MAE (Mean Absolute Error):     $(round(mae, sigdigits=6))
R² (Coef. Determinación):      $(round(r2, sigdigits=6))

MSE Entrenamiento:             $(round(train_mse, sigdigits=6))
MSE Test:                      $(round(test_mse, sigdigits=6))

ESTADÍSTICAS DE ERRORES:
------------------------------------------------------------------------------
Media de errores:              $(round(mean(errors), digits=4))
Mediana de errores:            $(round(median(errors), digits=4))
Desviación estándar:           $(round(std(errors), digits=4))
Error mínimo:                  $(round(minimum(errors), digits=4))
Error máximo:                  $(round(maximum(errors), digits=4))

INTERPRETACIÓN:
------------------------------------------------------------------------------
R² = $(round(r2, digits=4)):
    $(r2 > 0.9 ? "Excelente ajuste" :
         r2 > 0.7 ? "Buen ajuste" :
         r2 > 0.5 ? "Ajuste moderado" :
         r2 > 0 ? "Ajuste pobre" : "Modelo peor que la media")

MSE Train vs Test:
    Ratio: $(round(test_mse / train_mse, digits=3))
    $(abs(test_mse / train_mse - 1) < 0.3 ? "Buena generalización" :
        test_mse > train_mse * 1.5 ? "Posible overfitting" :
        "Revisar modelo")

Normalidad de errores:
    Sesgo: $(round(mean(errors) / std(errors), digits=3))
    $(abs(mean(errors)) < std(errors) * 0.1 ? "Errores centrados en 0" :
        "Sesgo en las predicciones")

GRÁFICAS GENERADAS:
------------------------------------------------------------------------------
1. $(exp_id)_01_predictions_vs_real.png
     - Scatter plot: predicciones vs valores reales
     - Línea diagonal = predicción perfecta
     - Útil para: evaluar ajuste global

2. $(exp_id)_02_time_series.png
     - Serie temporal de predicciones vs reales
     - Útil para: identificar tendencias y patrones

3. $(exp_id)_03_error_distribution.png
     - Histograma de errores
     - Útil para: verificar normalidad de residuos

4. $(exp_id)_04_error_evolution.png
     - Evolución de errores a lo largo del test
     - Útil para: detectar problemas sistemáticos

5. $(exp_id)_05_error_boxplot.png
     - Boxplot de errores con estadísticas
     - Útil para: identificar outliers

6. $(exp_id)_06_qq_plot.png
     - Q-Q plot para normalidad
     - Útil para: validar supuestos estadísticos

7. $(exp_id)_07_metrics_panel.png
     - Panel con métricas principales
     - Útil para: presentaciones

8. $(exp_id)_08_summary_grid.png
     - Grid con resumen de todas las visualizaciones
     - Útil para: visión general rápida

9. $(exp_id)_09_residuals_vs_predictions.png
     - Residuos vs predicciones
     - Útil para: detectar heterocedasticidad

RECOMENDACIONES:
------------------------------------------------------------------------------
$(r2 < 0.5 ? "Considera aumentar max_nodes o ajustar window" : "")
$(abs(test_mse / train_mse - 1) > 0.5 ? "Posible overfitting: reduce max_nodes o aumenta datos" : "")
$(abs(mean(errors)) > std(errors) * 0.2 ? "Sesgo en predicciones: revisar preprocesamiento" : "")

================================================================================
"""

# Guardar reporte
open("plots/$(exp_id)_REPORTE_VISUALIZACIONES.txt", "w") do f
        write(f, report)
end

println("Guardado: plots/$(exp_id)_REPORTE_VISUALIZACIONES.txt")

# ============================================================
# RESUMEN FINAL
# ============================================================

println("\n" * "="^80)
println("VISUALIZACIONES COMPLETADAS")
println("="^80)
println("\nMétricas del modelo:")
println("   MSE:  $(round(mse, sigdigits=6))")
println("   RMSE: $(round(rmse, sigdigits=6))")
println("   MAE:  $(round(mae, sigdigits=6))")
println("   R²:   $(round(r2, sigdigits=6))")

println("\nArchivos generados (9 gráficas + 1 reporte):")
for i in 1:9
        files = filter(f -> contains(f, "_0$i") && contains(f, exp_id), readdir("plots"))
        if !isempty(files)
                println("   $(files[1])")
        end
end
println("   $(exp_id)_REPORTE_VISUALIZACIONES.txt")

println("\nPara ver las gráficas:")
println("   cd plots && xdg-open $(exp_id)_08_summary_grid.png")
println("\n" * "="^80)