using SymDoME
using Serialization
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

# Intentar importar las funciones de estrategia desde SymDoME
# Si no existen, simplemente las ignoraremos
STRATEGY_FUNCTIONS = Dict{Symbol, Any}()

try
    # Intentar importar selectiveStrategy desde SymDoME
    STRATEGY_FUNCTIONS[:selective] = SymDoME.selectiveStrategy
    println("✓ Estrategia 'selective' cargada desde SymDoME")
catch
    println("⚠ No se encontró SymDoME.selectiveStrategy")
end

try
    # Intentar importar exhaustiveStrategy desde SymDoME
    STRATEGY_FUNCTIONS[:exhaustive] = SymDoME.exhaustiveStrategy
    println("✓ Estrategia 'exhaustive' cargada desde SymDoME")
catch
    println("⚠ No se encontró SymDoME.exhaustiveStrategy")
end

# Función para obtener la estrategia
function get_strategy_function(strategy_symbol::Symbol)
    if haskey(STRATEGY_FUNCTIONS, strategy_symbol)
        return STRATEGY_FUNCTIONS[strategy_symbol]
    else
        # Si no se encontró la función, retornar nothing
        # El código que llama manejará esto
        return nothing
    end
end

function run_experiment(;
    dataset::String,
    window::Int,
    horizon::Int = 1,
    max_nodes::Int = 15,
    strategy::Symbol = :selective,
    min_improvement::Float64 = 1e-6,
    seed::Int = 1,
    exp_id::String = "exp"
)

    # Cargar el dataset
    Xtr, ytr, Xte, yte = load_electricity_dataset(
        dataset, 
        window=window, 
        horizon=horizon
    )

    println("Datos cargados: ",
        "Xtr=$(size(Xtr)), ytr=$(size(ytr)), ",
        "Xte=$(size(Xte)), yte=$(size(yte))"
    )
    
    println("Tipos: Xtr=$(eltype(Xtr)), ytr=$(eltype(ytr))")

    # Crear los parámetros del modelo DoME
    params = DoMEParams(max_nodes, strategy, min_improvement)

    println("Entrenando DoME con parámetros: ", params)

    # Intentar obtener la función de estrategia
    strategy_fn = get_strategy_function(strategy)
    
    # Preparar los kwargs para dome()
    dome_kwargs = Dict{Symbol, Any}(
        :maximumNodes => params.max_nodes,
        :minimumReductionMSE => params.min_improvement
    )
    
    # Solo agregar strategy si encontramos la función
    if strategy_fn !== nothing
        dome_kwargs[:strategy] = strategy_fn
        println("Usando función de estrategia: $strategy_fn")
    else
        println("⚠ Función de estrategia no encontrada, usando configuración por defecto de SymDoME")
    end

    # Ejecutar el algoritmo DoME
    tree = nothing
    history = nothing
    
    try
        tree, history = dome(Xtr, ytr; dome_kwargs...)
        
        println("✓ Entrenamiento completado. MSE final: ", history[end])
    catch e
        println("❌ Error durante el entrenamiento:")
        println("   Tipo de error: ", typeof(e))
        println("   Mensaje: ", e)
        
        # Intentar sin el parámetro strategy
        println("\n🔄 Reintentando sin especificar estrategia...")
        tree, history = dome(
            Xtr, 
            ytr;
            maximumNodes=params.max_nodes,
            minimumReductionMSE=params.min_improvement
        )
        println("✓ Entrenamiento completado (sin estrategia). MSE final: ", history[end])
    end

    # Evaluación en el conjunto de prueba
    # Declarar ŷ_test FUERA del try-catch para que esté disponible después
    ŷ_test = nothing
    
    # La función evaluate debe venir de SymDoME, si no existe, usar una implementación manual
    try
        # Intentar usar SymDoME.evaluate
        ŷ_test = SymDoME.evaluate(tree, Xte)
        println("✓ Usando SymDoME.evaluate")
    catch e1
        println("⚠ SymDoME.evaluate no disponible")
        try
            # Si evaluate no existe, intentar con predict
            ŷ_test = SymDoME.predict(tree, Xte)
            println("✓ Usando SymDoME.predict")
        catch e2
            # Última opción: implementación manual básica
            println("⚠ Usando implementación de predicción manual...")
            try
                ŷ_test = zeros(size(Xte, 1))
                for i in 1:size(Xte, 1)
                    ŷ_test[i] = SymDoME.evaluateTree(tree, Xte[i, :])
                end
                println("✓ Predicción manual completada")
            catch e3
                println("❌ Error en predicción manual: $e3")
                # Último recurso: intentar aplicar el árbol directamente
                try
                    ŷ_test = [tree(Xte[i, :]) for i in 1:size(Xte, 1)]
                    println("✓ Usando aplicación directa del árbol")
                catch e4
                    error("No se pudo evaluar el modelo. Errores: evaluate=$e1, predict=$e2, evaluateTree=$e3, direct=$e4")
                end
            end
        end
    end
    
    mse_test = mean((ŷ_test .- yte).^2)

    println("MSE en test: ", mse_test)

    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

    # ===== Guardado binario (completo) =====
    result = Dict(
        "dataset" => dataset,
        "window" => window,
        "horizon" => horizon,
        "max_nodes" => max_nodes,
        "strategy" => strategy,
        "min_improvement" => min_improvement,
        "seed" => seed,
        "tree" => tree,
        "train_mse_final" => history[end],
        "test_mse" => mse_test
    )

    mkpath("results/raw")
    serialize("results/raw/$exp_id.jls", result)

    # ===== Guardado CSV (resumen) =====
    mkpath("resultados_experimentos")

    csv_file = "resultados_experimentos/$exp_id.csv"

    header = [
        "dataset", "window", "horizon", "max_nodes",
        "strategy", "min_improvement", "seed",
        "train_mse_final", "test_mse", "timestamp"
    ]

    row = [
        dataset, window, horizon, max_nodes,
        String(strategy), min_improvement, seed,
        history[end], mse_test, timestamp
    ]

    writedlm(csv_file, [header; row], ',')

    println("\n✓ Resultados guardados en:")
    println("  - Binario: results/raw/$exp_id.jls")
    println("  - CSV: $csv_file")

    return result
end

# ==========================================
# Ejecución directa del script
# ==========================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("=".^60)
    println("Ejecutando experimento DoME…")
    println("=".^60)

    run_experiment(
        dataset = "data/ElectricDevices/LD2011_2014_mini.txt",
        window = 24,
        horizon = 1,
        max_nodes = 15,
        strategy = :selective,
        min_improvement = 1e-6,
        seed = 1,
        exp_id = "exp_1"
    )

    println("\n" * "=".^60)
    println("Experimento terminado")
    println("=".^60)
end