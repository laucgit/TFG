using Plots
using Printf

"""
Script para visualizar el árbol de DoME como diagrama

Lee la expresión matemática y crea una representación visual simplificada
"""

println("="^80)
println("VISUALIZACIÓN DEL ÁRBOL DoME")
println("="^80)

exp_id = length(ARGS) >= 1 ? ARGS[1] : "exp_1"

println("\nExperimento: $exp_id")

# Crear directorio para plots
mkpath("plots")

# ============================================================
# CARGAR EXPRESIÓN
# ============================================================

report_file = "results/$(exp_id).txt"
if !isfile(report_file)
    println("Error: No se encontró $report_file")
    exit(1)
end

# Declarar como global
global expression = ""

lines = readlines(report_file)
for (i, line) in enumerate(lines)
    if contains(line, "EXPRESIÓN MATEMÁTICA:")
        # La expresión está dos líneas después
        if i + 2 <= length(lines)
            global expression = String(strip(lines[i+2]))  # CORREGIDO: Convertir a String
        end
        break
    end
end

if isempty(expression)
    println("No se pudo extraer la expresión matemática")
    println("\nContenido del archivo para debug:")
    for (i, line) in enumerate(lines)
        if contains(line, "EXPRESIÓN")
            println("Línea $i: $line")
            if i+1 <= length(lines)
                println("Línea $(i+1): $(lines[i+1])")
            end
            if i+2 <= length(lines)
                println("Línea $(i+2): $(lines[i+2])")
            end
        end
    end
    exit(1)
end

println("\nExpresión encontrada:")
println("   $expression")

# ============================================================
# ANALIZAR EXPRESIÓN
# ============================================================

# Contar variables
variables_used = Dict{String, Int}()
for i in 1:24
    var = "X$i"
    count = length(collect(eachmatch(Regex(var), expression)))
    if count > 0
        variables_used[var] = count
    end
end

# Contar operadores
operators = Dict(
    "+" => length(collect(eachmatch(r"\+", expression))),
    "-" => length(collect(eachmatch(r"-", expression))),
    "*" => length(collect(eachmatch(r"\*", expression))),
    "/" => length(collect(eachmatch(r"/", expression)))
)

# Contar constantes (números)
constants = length(collect(eachmatch(r"\d+\.\d+", expression)))

println("\nAnálisis de la expresión:")
println("   Variables únicas usadas: $(length(variables_used))")
println("   Operadores:")
for (op, count) in sort(collect(operators), by=x->x[2], rev=true)
    count > 0 && println("      $op: $count veces")
end
println("   Constantes numéricas: $constants")

# ============================================================
# PLOT 1: IMPORTANCIA DE VARIABLES
# ============================================================

println("\nGenerando Plot 1: Importancia de Variables...")

if !isempty(variables_used)
    var_names = collect(keys(variables_used))
    var_counts = collect(values(variables_used))
    
    # Ordenar por importancia
    sorted_indices = sortperm(var_counts, rev=true)
    var_names = var_names[sorted_indices]
    var_counts = var_counts[sorted_indices]
    
    p1 = bar(
        var_names, var_counts,
        xlabel = "Variable",
        ylabel = "Número de apariciones",
        title = "Importancia de Variables en la Expresión DoME",
        legend = false,
        fillcolor = :lightblue,
        linecolor = :blue,
        grid = true,
        size = (800, 600),
        xrotation = 45
    )
    
    # Añadir etiquetas con los valores
    for (i, (name, count)) in enumerate(zip(var_names, var_counts))
        annotate!(p1, i, count + maximum(var_counts) * 0.02,
                 text(string(count), :center, 8))
    end
    
    savefig(p1, "plots/$(exp_id)_10_variable_importance.png")
    println("Guardado: plots/$(exp_id)_10_variable_importance.png")
else
    println("No se encontraron variables en la expresión")
end

# ============================================================
# PLOT 2: DISTRIBUCIÓN DE OPERADORES
# ============================================================

println("Generando Plot 2: Distribución de Operadores...")

op_names = collect(keys(operators))
op_counts = collect(values(operators))

# Filtrar operadores no usados
non_zero = op_counts .> 0
op_names = op_names[non_zero]
op_counts = op_counts[non_zero]

if !isempty(op_counts)
    p2 = pie(
        op_names, op_counts,
        title = "Distribución de Operadores",
        legend = :right,
        size = (700, 600)
    )
    
    savefig(p2, "plots/$(exp_id)_11_operators_distribution.png")
    println("Guardado: plots/$(exp_id)_11_operators_distribution.png")
end

# ============================================================
# PLOT 3: ESTRUCTURA SIMPLIFICADA DEL ÁRBOL
# ============================================================

println("Generando Plot 3: Estructura del Árbol...")

# Crear un diagrama de texto ASCII del árbol
function create_tree_diagram(expr::String)
    # Simplificar la expresión para visualización
    # Esto es una representación muy simplificada
    
    lines = String[]
    push!(lines, "ESTRUCTURA DEL ÁRBOL DoME")
    push!(lines, "="^60)
    push!(lines, "")
    push!(lines, "Expresión completa:")
    push!(lines, expr)
    push!(lines, "")
    push!(lines, "Estructura jerárquica:")
    push!(lines, "")
    
    # Contar niveles de anidamiento (paréntesis)
    max_depth = 0
    current_depth = 0
    for c in expr
        if c == '('
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elseif c == ')'
            current_depth -= 1
        end
    end
    
    push!(lines, "  Profundidad máxima: $max_depth niveles")
    push!(lines, "")
    push!(lines, "Componentes principales:")
    push!(lines, "  ├─ Variables: $(length(variables_used))")
    push!(lines, "  ├─ Constantes: $constants")
    push!(lines, "  └─ Operaciones: $(sum(values(operators)))")
    push!(lines, "")
    
    if !isempty(variables_used)
        push!(lines, "Variables más importantes:")
        sorted_vars = sort(collect(variables_used), by=x->x[2], rev=true)
        for (i, (var, count)) in enumerate(sorted_vars[1:min(5, length(sorted_vars))])
            prefix = i < min(5, length(sorted_vars)) ? "├─" : "└─"
            push!(lines, "  $prefix $var: usado $count $(count == 1 ? "vez" : "veces")")
        end
    end
    
    return join(lines, "\n")
end

# CORREGIDO: Asegurar que expression es String
tree_text = create_tree_diagram(String(expression))

# Crear plot con texto
p3 = plot(
    xlims = (0, 10), ylims = (0, 10),
    legend = false,
    grid = false,
    axis = false,
    size = (800, 600),
    title = "Estructura del Árbol DoME"
)

# Añadir texto
annotate!(p3, 5, 5,
         text(tree_text, :center, 8, :courier))

savefig(p3, "plots/$(exp_id)_12_tree_structure.png")
println("Guardado: plots/$(exp_id)_12_tree_structure.png")

# ============================================================
# PLOT 4: VISUALIZACIÓN DE COMPLEJIDAD
# ============================================================

println("Generando Plot 4: Métricas de Complejidad...")

# Crear un panel con métricas de complejidad
metrics = [
    ("Variables\nÚnicas", length(variables_used)),
    ("Total\nOperaciones", sum(values(operators))),
    ("Constantes", constants),
    ("Profundidad\nMáxima", 0)  # Calcular profundidad
]

# Calcular profundidad real
max_depth = 0
current_depth = 0
for c in expression
    if c == '('
        current_depth += 1
        max_depth = max(max_depth, current_depth)
    elseif c == ')'
        current_depth -= 1
    end
end
metrics[4] = ("Profundidad\nMáxima", max_depth)

p4 = plot(layout = (2, 2), size = (900, 700),
          plot_title = "Métricas de Complejidad del Modelo")

colors = [:blue, :green, :orange, :red]

for (i, (name, value)) in enumerate(metrics)
    subplot = i
    plot!(p4, [1], [value],
          subplot = subplot,
          seriestype = :bar,
          title = name,
          ylabel = "Valor",
          ylims = (0, value * 1.2),
          legend = false,
          fillcolor = colors[i],
          xticks = [],
          grid = true)
    
    annotate!(p4, 1, value * 1.1,
             text(string(value), :center, 12, :bold),
             subplot = subplot)
end

savefig(p4, "plots/$(exp_id)_13_complexity_metrics.png")
println("Guardado: plots/$(exp_id)_13_complexity_metrics.png")

# ============================================================
# GUARDAR ANÁLISIS TEXTUAL
# ============================================================

println("\nGenerando análisis textual...")

analysis = """
================================================================================
ANÁLISIS DEL ÁRBOL DoME: $exp_id
================================================================================

EXPRESIÓN MATEMÁTICA COMPLETA:
------------------------------------------------------------------------------
$expression

ESTADÍSTICAS DE COMPLEJIDAD:
------------------------------------------------------------------------------
Variables únicas utilizadas:    $(length(variables_used))
Total de operaciones:           $(sum(values(operators)))
Constantes numéricas:           $constants
Profundidad máxima del árbol:   $max_depth

OPERADORES UTILIZADOS:
------------------------------------------------------------------------------
$(join(["$op: $count veces" for (op, count) in sort(collect(operators), by=x->x[2], rev=true) if count > 0], "\n"))

VARIABLES MÁS IMPORTANTES:
------------------------------------------------------------------------------
$(if !isempty(variables_used)
    sorted_vars = sort(collect(variables_used), by=x->x[2], rev=true)
    join(["$var: $count apariciones ($(round(count/sum(values(variables_used))*100, digits=1))%)"
          for (var, count) in sorted_vars], "\n")
else
    "No se encontraron variables"
end)

INTERPRETACIÓN:
------------------------------------------------------------------------------
Complejidad del modelo:
  $(length(variables_used) > 15 ? "Alta - usa muchas variables" :
    length(variables_used) > 8 ? "Media - usa cantidad moderada de variables" :
    "Baja - modelo simple con pocas variables")

Diversidad de operaciones:
  $(sum(values(operators)) > 20 ? "Alta - expresión muy elaborada" :
    sum(values(operators)) > 10 ? "Media - expresión moderada" :
    "Baja - expresión simple")

Profundidad:
  $(max_depth > 5 ? "Árbol profundo - operaciones muy anidadas" :
    max_depth > 3 ? "Profundidad media" :
    "Árbol poco profundo")

RECOMENDACIONES:
------------------------------------------------------------------------------
$(length(variables_used) > 15 ? "Advertencia: Modelo complejo: considera reducir max_nodes para simplicidad\n" : "")$(max_depth > 6 ? "Advertencia: Árbol muy profundo: puede ser difícil de interpretar\n" : "")$(length(variables_used) < 5 ? "Modelo simple y interpretable\n" : "")

VISUALIZACIONES GENERADAS:
------------------------------------------------------------------------------
1. $(exp_id)_10_variable_importance.png
   → Importancia de cada variable (frecuencia de uso)

2. $(exp_id)_11_operators_distribution.png
   → Distribución de operadores matemáticos

3. $(exp_id)_12_tree_structure.png
   → Diagrama de estructura del árbol

4. $(exp_id)_13_complexity_metrics.png
   → Panel de métricas de complejidad

================================================================================
"""

open("plots/$(exp_id)_ANALISIS_ARBOL.txt", "w") do f
    write(f, analysis)
end

println("Guardado: plots/$(exp_id)_ANALISIS_ARBOL.txt")

# ============================================================
# RESUMEN
# ============================================================

println("\n" * "="^80)
println("VISUALIZACIÓN DEL ÁRBOL COMPLETADA")
println("="^80)
println("\nEstadísticas del modelo:")
println("   Variables usadas: $(length(variables_used))")
println("   Operaciones totales: $(sum(values(operators)))")
println("   Profundidad: $max_depth niveles")

if !isempty(variables_used)
    println("\nVariables más importantes:")
    sorted_vars = sort(collect(variables_used), by=x->x[2], rev=true)
    for (i, (var, count)) in enumerate(sorted_vars[1:min(5, length(sorted_vars))])
        println("   $i. $var ($(count) apariciones)")
    end
end

println("\nArchivos generados:")
println("   $(exp_id)_10_variable_importance.png")
println("   $(exp_id)_11_operators_distribution.png")
println("   $(exp_id)_12_tree_structure.png")
println("   $(exp_id)_13_complexity_metrics.png")
println("   $(exp_id)_ANALISIS_ARBOL.txt")

println("\n" * "="^80)