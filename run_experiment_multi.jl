using SymDoME
using Statistics
using CSV
using Dates
using DataFrames
using DelimitedFiles

include("load_data_multi.jl")

struct DoMEParams
    max_nodes::Int
    min_improvement::Float64
    use_division::Bool
    strategy::Function
end

"""
Configuraciones de modelos según especificación de los profesores.

minimumReductionMSE es un MULTIPLICADOR del MSE actual, NO un valor absoluto.
Línea 775 de DoME.jl: maximumReduction = obj.minimumReductionMSE * obj.mse
Con MSE normalizado ~0.01-0.05 y minimumReductionMSE=0.001:
  → maximumReduction = 0.001 * 0.01 = 0.00001 (demasiado pequeño)
SOLUCIÓN: Usar valores MUCHO más pequeños: 1e-5 a 1e-9
"""
function modelConfigurations(model::Symbol)
    if model == :DoME
        MinimumReductionsMSE = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

        # CORRECCIÓN: maxNodes=1 eliminado. Un árbol de 1 nodo es solo mean(y),
        # Step! retorna false inmediatamente → experimento inútil.
        MaxNumNodes = [5, 10, 20, 30, 50, 75, 100, 150, 200]

        Strategies = [
            SymDoME.Strategy4,  # Más rápida
            SymDoME.Strategy3   # Default
        ]

        configurations = [
            Dict{String,Any}(
                "minimumReductionMSE" => minimumReductionMSE,
                "maxNumNodes" => maxNumNodes,
                "useDivisionOperator" => useDivisionOperator,
                "strategy" => strategy,
                "strategyName" => string(strategy)
            )
            for maxNumNodes in MaxNumNodes,
                minimumReductionMSE in MinimumReductionsMSE,
                useDivisionOperator in [false, true],
                strategy in Strategies
        ][:]
    else
        error("Modelo no soportado: $model")
    end
    return configurations
end

# =============================================================================
# CORRECCIÓN: Imputación de NaN/Inf
# Los scripts de test (test_go_no_go, test_datasets) hacían esto antes de
# normalizar, pero run_experiment() no lo hacía. DoME hace @assert(!any(isnan))
# en el constructor → cualquier dato sucio crasheaba en CESGA.
# =============================================================================

function impute_nonfinite!(X::Matrix{Float64})
    n, d = size(X)
    for j in 1:d
        s = 0.0; c = 0
        @inbounds for i in 1:n
            v = X[i, j]
            if isfinite(v)
                s += v; c += 1
            end
        end
        μ = (c > 0) ? (s / c) : 0.0
        @inbounds for i in 1:n
            if !isfinite(X[i, j])
                X[i, j] = μ
            end
        end
    end
    return X
end

function impute_nonfinite!(y::Vector{Float64})
    s = 0.0; c = 0
    @inbounds for v in y
        if isfinite(v)
            s += v; c += 1
        end
    end
    μ = (c > 0) ? (s / c) : 0.0
    @inbounds for i in eachindex(y)
        if !isfinite(y[i])
            y[i] = μ
        end
    end
    return y
end

"""
FUNCIÓN DE ENTRENAMIENTO - API con Step! manual para capturar history.

dome() no devuelve history. El struct DoME existe pero no se exporta.
Solución: Usar SymDoME.DoME (struct interno) y Step! para iterar manualmente.
"""
function train_dome(
    Xtr::Matrix{Float64},
    ytr::Vector{Float64},
    params::DoMEParams
)
    println("\n[TRAIN] Entrenando DoME...")
    println("  Parámetros:")
    println("    - max_nodes: $(params.max_nodes)")
    println("    - min_improvement: $(params.min_improvement)")
    println("    - use_division: $(params.use_division)")
    println("    - strategy: $(params.strategy)")
    println("  Datos:")
    println("    - Xtr: $(size(Xtr))")
    println("    - ytr: $(size(ytr))")

    dome_obj = SymDoME.DoME(
        Xtr, ytr;
        maximumNodes=params.max_nodes,
        minimumReductionMSE=params.min_improvement,
        useDivisionOperator=params.use_division,
        strategy=params.strategy
    )

    history = Float64[dome_obj.mse]

    max_iterations = 1000  # Límite de seguridad

    for iteration in 1:max_iterations
        improved = SymDoME.Step!(dome_obj)
        push!(history, dome_obj.mse)

        if !improved
            println("  Convergencia alcanzada en iteración $iteration")
            break
        end
    end

    tree = dome_obj.tree

    println("  Entrenamiento completado")
    println("    - MSE inicial: $(history[1])")
    println("    - MSE final: $(history[end])")
    println("    - Iteraciones: $(length(history))")
    println("    - Mejora: $(round((1 - history[end]/history[1])*100, digits=2))%")

    return tree, history
end

"""
FUNCIÓN DE EVALUACIÓN - vectorizada.

CORRECCIÓN: La versión anterior hacía un bucle fila por fila:
    for i in eachindex(ŷ_test)
        ŷ_test[i] = SymDoME.evaluateTree(tree, Xte[i, :])
    end
Cada iteración hacía reshape(vector, 1, :) → asignación extra.
Con 4343 muestras × 2400 experimentos eso es mucho.

evaluateTree(tree::Tree, dataset::Matrix) existe (Tree.jl línea 476) y
recompute desde los hijos usando el dataset pasado, sin usar semantics
de entrenamiento. Es seguro y vectorizado.
"""
function evaluate_dome(
    tree,
    Xte::Matrix{Float64},
    yte::Vector{Float64}
)
    println("\n[EVAL] Evaluando en conjunto de test...")
    println("  Datos test: $(size(Xte))")

    # Evaluación vectorizada: pasa toda la matriz de una vez
    ŷ_test = collect(SymDoME.evaluateTree(tree, Xte))

    mse_test = mean((ŷ_test .- yte).^2)

    println("  MSE test: $mse_test")

    return ŷ_test, mse_test
end

"""
FUNCIÓN PRINCIPAL DE EXPERIMENTO
"""
function run_experiment(;
    dataset::String,
    window::Int,
    normalization::String = "MaxMin",
    model::Symbol = :DoME,
    config_id::Int,
    seed::Int = 1,
    data_dir::String = "data",
    horizon::Int = 1,
    target_col::Union{String,Int,Nothing} = nothing,
    train_ratio::Float64 = 0.75
)
    # 1. Obtener configuración del modelo
    configurations = modelConfigurations(model)

    if config_id < 1 || config_id > length(configurations)
        error("config_id fuera de rango: $config_id (total: $(length(configurations)))")
    end

    config = configurations[config_id]
    max_nodes = config["maxNumNodes"]
    min_improvement = config["minimumReductionMSE"]
    use_division = config["useDivisionOperator"]
    strategy = config["strategy"]

    println("="^70)
    println("EJECUTANDO EXPERIMENTO")
    println("="^70)
    println("Dataset: $dataset")
    println("Window: $window")
    println("Normalización: $normalization")
    println("Modelo: $model")
    println("Config ID: $config_id")
    println("Seed: $seed")
    println("\nConfiguración del modelo:")
    println("  - maxNumNodes: $max_nodes")
    println("  - minimumReductionMSE: $min_improvement")
    println("  - useDivisionOperator: $use_division")
    println("  - strategy: $strategy")
    println("="^70)

    # 2. Cargar datos
    Xtr, ytr, Xte, yte = load_dataset(
        dataset;
        data_dir=data_dir,
        window=window,
        horizon=horizon,
        target_col=target_col,
        train_ratio=train_ratio
    )

    println("\nDatos cargados:")
    println("  Xtr: $(size(Xtr))")
    println("  ytr: $(size(ytr))")
    println("  Xte: $(size(Xte))")
    println("  yte: $(size(yte))")

    # 3. CORRECCIÓN: Imputar NaN/Inf ANTES de normalizar
    println("\nImputando valores no finitos...")
    impute_nonfinite!(Xtr)
    impute_nonfinite!(ytr)
    impute_nonfinite!(Xte)
    impute_nonfinite!(yte)
    println("  Imputación completada")

    # 4. Normalizar
    println("\nNormalizando datos...")
    if normalization == "MaxMin"
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
    else
        error("Normalización no soportada: $normalization")
    end

    println("  Datos normalizados")

    # 5. Crear parámetros del modelo
    params = DoMEParams(max_nodes, min_improvement, use_division, strategy)

    # 6. ENTRENAR
    tree, history = train_dome(Xtr_norm, ytr_norm, params)

    # 7. EVALUAR
    ŷ_test, mse_test = evaluate_dome(tree, Xte_norm, yte_norm)

    # 8. Guardar resultados
    results_dir = "results"
    isdir(results_dir) || mkdir(results_dir)

    dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")
    base_name = "$(dataset_name)_$(normalization)_$(model)_w$(window)_config$(config_id)_seed$(seed)"

    # CSV resumen
    csv_file = joinpath(results_dir, "$(base_name).csv")
    results_df = DataFrame(
        dataset = [dataset],
        window = [window],
        normalization = [normalization],
        model = [string(model)],
        config_id = [config_id],
        seed = [seed],
        max_nodes = [max_nodes],
        min_improvement = [min_improvement],
        use_division = [use_division],
        strategy = [string(strategy)],
        mse_initial = [history[1]],
        mse_final_train = [history[end]],
        mse_test = [mse_test],
        improvement_pct = [(1 - history[end]/history[1]) * 100],
        iterations = [length(history)],
        converged = [length(history) > 1],
        error = [""]  # CORRECCIÓN: columna error presente siempre (vacía si no hay error)
    )
    CSV.write(csv_file, results_df)
    println("\n[RESULTADOS] Guardado resumen en: $csv_file")

    # TXT reporte
    txt_file = joinpath(results_dir, "$(base_name).txt")
    open(txt_file, "w") do io
        println(io, "="^70)
        println(io, "REPORTE DE EXPERIMENTO")
        println(io, "="^70)
        println(io, "Fecha: $(Dates.now())")
        println(io, "\nDATASET:")
        println(io, "  - Nombre: $dataset")
        println(io, "  - Window: $window")
        println(io, "  - Normalización: $normalization")
        println(io, "  - Train ratio: $train_ratio")
        println(io, "\nMODELO:")
        println(io, "  - Tipo: $model")
        println(io, "  - Config ID: $config_id")
        println(io, "  - Seed: $seed")
        println(io, "\nHIPERPARÁMETROS:")
        println(io, "  - maxNumNodes: $max_nodes")
        println(io, "  - minimumReductionMSE: $min_improvement")
        println(io, "  - useDivisionOperator: $use_division")
        println(io, "  - strategy: $strategy")
        println(io, "\nDIMENSIONES:")
        println(io, "  - Xtr: $(size(Xtr))")
        println(io, "  - ytr: $(size(ytr))")
        println(io, "  - Xte: $(size(Xte))")
        println(io, "  - yte: $(size(yte))")
        println(io, "\nRESULTADOS:")
        println(io, "  - MSE inicial: $(history[1])")
        println(io, "  - MSE final (train): $(history[end])")
        println(io, "  - MSE test: $mse_test")
        println(io, "  - Mejora: $((1 - history[end]/history[1]) * 100)%")
        println(io, "  - Iteraciones: $(length(history))")
        println(io, "  - Convergencia: $(length(history) > 1 ? "SÍ" : "NO")")
        println(io, "\nHISTORIAL DE ENTRENAMIENTO:")
        for (i, mse) in enumerate(history)
            println(io, "  Iter $i: MSE = $mse")
        end
        println(io, "\nEXPRESIÓN MATEMÁTICA:")
        try
            expr = SymDoME.vectorString(tree)
            println(io, "  $expr")
        catch e
            println(io, "  [Error al obtener expresión: $e]")
        end
        println(io, "="^70)
    end
    println("[RESULTADOS] Guardado reporte en: $txt_file")

    # CSV predicciones
    pred_file = joinpath(results_dir, "$(base_name)_predictions.csv")
    pred_df = DataFrame(
        y_true = yte_norm,
        y_pred = ŷ_test,
        error = ŷ_test .- yte_norm,
        abs_error = abs.(ŷ_test .- yte_norm),
        squared_error = (ŷ_test .- yte_norm).^2
    )
    CSV.write(pred_file, pred_df)
    println("[RESULTADOS] Guardado predicciones en: $pred_file")

    println("\n" * "="^70)
    println("EXPERIMENTO COMPLETADO")
    println("="^70)

    return results_df, tree, history
end

# Script ejecutable
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 6
        println("Uso: julia run_experiment_multi.jl <dataset> <window> <norm> <model> <config_id> <seed>")
        println("Ejemplo: julia run_experiment_multi.jl ETT/ETTh2.csv 12 MaxMin DoME 1 1")
        exit(1)
    end

    dataset = ARGS[1]
    window = parse(Int, ARGS[2])
    normalization = ARGS[3]
    model = Symbol(ARGS[4])
    config_id = parse(Int, ARGS[5])
    seed = parse(Int, ARGS[6])

    run_experiment(
        dataset=dataset,
        window=window,
        normalization=normalization,
        model=model,
        config_id=config_id,
        seed=seed
    )
end