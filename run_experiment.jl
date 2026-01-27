using SymDoME
using Statistics
using CSV
using Dates
using DataFrames
using DelimitedFiles

include("load_data.jl")

# Definir la estructura DoMEParams
struct DoMEParams
    max_nodes::Int
    strategy::Symbol
    min_improvement::Float64
end

# Intentar importar las funciones de estrategia desde SymDoME (ver esto, igual hay que quitarlo)
STRATEGY_FUNCTIONS = Dict{Symbol, Any}()

try
    STRATEGY_FUNCTIONS[:selective] = SymDoME.selectiveStrategy
    println("[OK] Estrategia 'selective' cargada desde SymDoME")
catch
    println("[INFO] No se encontró SymDoME.selectiveStrategy")
end

try
    STRATEGY_FUNCTIONS[:exhaustive] = SymDoME.exhaustiveStrategy
    println("[OK] Estrategia 'exhaustive' cargada desde SymDoME")
catch
    println("[INFO] No se encontró SymDoME.exhaustiveStrategy")
end

# Función para obtener la estrategia
function get_strategy_function(strategy_symbol::Symbol)
    return get(STRATEGY_FUNCTIONS, strategy_symbol, nothing)
end

function run_experiment(;
    dataset::String,
    window::Int,
    horizon::Int = 1,
    max_nodes::Int = 15,
    strategy::Symbol = :selective,
    min_improvement::Float64 = 1e-6,
    seed::Int = 1,
    exp_id::String = "exp",
    target_col::Union{String,Int,Nothing} = nothing
)

    # Cargar el dataset
    Xtr, ytr, Xte, yte = load_electricity_dataset(
        dataset, 
        window=window, 
        horizon=horizon,
        target_col=target_col  # Ahora pasa correctamente
    )

    println("\n[DATA] Datos cargados: Xtr=$(size(Xtr)), ytr=$(size(ytr)), Xte=$(size(Xte)), yte=$(size(yte))")
    println("   Tipos: Xtr=$(eltype(Xtr)), ytr=$(eltype(ytr))")

    # Crear los parámetros del modelo DoME
    params = DoMEParams(max_nodes, strategy, min_improvement)

    println("\n[SETUP] Entrenando DoME con parámetros: ", params)

    # Intentar obtener la función de estrategia
    strategy_fn = get_strategy_function(strategy)
    
    # Preparar los kwargs para dome()
    dome_kwargs = Dict{Symbol, Any}(
        :maximumNodes => params.max_nodes,
        :minimumReductionMSE => params.min_improvement
    )
    
    # Solo añadir strategy si encontramos la función
    if strategy_fn !== nothing
        dome_kwargs[:strategy] = strategy_fn
        println("   Usando función de estrategia: $strategy")
    else
        println("   [INFO] Función de estrategia no encontrada, usando configuración por defecto")
    end

    # Ejecutar el algoritmo DoME
    tree = nothing
    history = nothing
    
    try
        tree, history = dome(Xtr, ytr; dome_kwargs...)
        println("\n[SUCCESS] Entrenamiento completado. MSE final: ", history[end])
    catch e
        println("\n[WARNING] Error durante el entrenamiento:")
        println("   Tipo: ", typeof(e))
        println("   Mensaje: ", e)
        
        # Intentar sin el parámetro strategy
        println("\n   Reintentando sin especificar estrategia...")
        tree, history = dome(
            Xtr, 
            ytr;
            maximumNodes=params.max_nodes,
            minimumReductionMSE=params.min_improvement
        )
        println("[SUCCESS] Entrenamiento completado (sin estrategia). MSE final: ", history[end])
    end

    # Evaluación en el conjunto de prueba
    println("\n[TEST] Evaluando en conjunto de test...")
    ŷ_test = zeros(size(Xte, 1))
    
    for i in eachindex(ŷ_test)
        ŷ_test[i] = SymDoME.evaluateTree(tree, Xte[i, :])
    end
    
    mse_test = mean((ŷ_test .- yte).^2)

    println("   MSE en test: ", mse_test)

    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

    # ===== Guardado en CSV (resumen del experimento) =====
    mkpath("results")

    csv_file = "results/$exp_id.csv"

    # Crear DataFrame con los resultados
    df = DataFrame(
        dataset = dataset,
        window = window,
        horizon = horizon,
        max_nodes = max_nodes,
        strategy = String(strategy),
        min_improvement = min_improvement,
        seed = seed,
        train_mse_final = history[end],
        test_mse = mse_test,
        num_nodes = length(history),
        timestamp = timestamp
    )

    # Guardar CSV
    CSV.write(csv_file, df)

    # ===== Guardado en TXT (información detallada) =====
    txt_file = "results/$exp_id.txt"
    
    open(txt_file, "w") do io
        println(io, "="^60)
        println(io, "RESULTADOS DEL EXPERIMENTO: $exp_id")
        println(io, "="^60)
        println(io, "Timestamp: $timestamp")
        println(io, "")
        
        println(io, "CONFIGURACIÓN:")
        println(io, "-"^60)
        println(io, "Dataset: $dataset")
        println(io, "Target column: $target_col")
        println(io, "Window: $window")
        println(io, "Horizon: $horizon")
        println(io, "Max nodes: $max_nodes")
        println(io, "Strategy: $strategy")
        println(io, "Min improvement: $min_improvement")
        println(io, "Seed: $seed")
        println(io, "")
        
        println(io, "DIMENSIONES DE LOS DATOS:")
        println(io, "-"^60)
        println(io, "Train: X=$(size(Xtr)), y=$(size(ytr))")
        println(io, "Test:  X=$(size(Xte)), y=$(size(yte))")
        println(io, "")
        
        println(io, "RESULTADOS:")
        println(io, "-"^60)
        println(io, "MSE entrenamiento (final): $(history[end])")
        println(io, "MSE test: $mse_test")
        println(io, "Número de nodos: $(length(history))")
        println(io, "")
        
        println(io, "HISTORIAL DE ENTRENAMIENTO:")
        println(io, "-"^60)
        println(io, "Iteración\tMSE")
        for (i, mse) in enumerate(history)
            println(io, "$i\t$mse")
        end
        println(io, "")
        
        println(io, "EXPRESIÓN MATEMÁTICA:")
        println(io, "-"^60)
        println(io, string(tree))
        println(io, "")
        
        println(io, "ÁRBOL DEL MODELO (representación interna):")
        println(io, "-"^60)
        println(io, tree)
        println(io, "")
        
        println(io, "="^60)
        println(io, "FIN DEL REPORTE")
        println(io, "="^60)
    end

    # ===== Guardar predicciones en CSV =====
    predictions_file = "results/$(exp_id)_predictions.csv"
    
    predictions_df = DataFrame(
        y_true = yte,
        y_pred = ŷ_test,
        error = yte .- ŷ_test,
        squared_error = (yte .- ŷ_test).^2
    )
    
    CSV.write(predictions_file, predictions_df)

    println("\n[RESULTS] Resultados guardados en:")
    println("  - Resumen CSV: $csv_file")
    println("  - Reporte detallado TXT: $txt_file")
    println("  - Predicciones CSV: $predictions_file")

    # Retornar diccionario con resultados
    result = Dict(
        "dataset" => dataset,
        "window" => window,
        "horizon" => horizon,
        "max_nodes" => max_nodes,
        "strategy" => strategy,
        "min_improvement" => min_improvement,
        "seed" => seed,
        "tree" => tree,
        "history" => history,
        "train_mse_final" => history[end],
        "test_mse" => mse_test,
        "predictions" => ŷ_test,
        "y_true" => yte
    )

    return result
end

# ==========================================
# Ejecución directa del script
# ==========================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^60)
    println("Ejecutando experimento DoME")
    println("="^60)

    run_experiment(
        dataset = "data/ElectricDevices/LD2011_2014_mini.txt", #hay que cambiarlo para no hardcodear
        target_col = "MT_196",  # Especificar la columna objetivo
        window = 24,
        horizon = 1,
        max_nodes = 15,
        strategy = :selective,
        min_improvement = 1e-6,
        seed = 1,
        exp_id = "exp_1" 
    )

    println("\n" * "="^60)
    println("Experimento terminado")
    println("="^60)
end