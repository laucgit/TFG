using CSV
using DataFrames
using Statistics

"""
    load_electricity_dataset_multivariate(
        path::String;
        window::Int,
        horizon::Int = 1,
        train_ratio::Float64 = 0.75,
        target_column::Union{Int,Nothing} = nothing,
        feature_columns::Union{Vector{Int},Nothing} = nothing
    )

Carga el dataset de electricidad y crea ventanas temporales MULTIVARIANTES.
Esta versión permite usar múltiples series temporales como features.

Parámetros:
- path: Ruta al archivo CSV
- window: Tamaño de la ventana temporal
- horizon: Pasos adelante para predecir
- train_ratio: Proporción de datos para entrenamiento
- target_column: Índice de la columna a predecir (por defecto: última columna)
- feature_columns: Índices de columnas a usar como features (por defecto: todas excepto target)

Ejemplo:
    # Usar solo la columna 5 para predecir la columna 10
    Xtr, ytr, Xte, yte = load_electricity_dataset_multivariate(
        "data.txt", window=24, target_column=10, feature_columns=[5]
    )
    # Resultado: Xtr = (N, 24) - ventanas de 24 valores de la columna 5
    
    # Usar columnas 5, 10, 15 para predecir columna 20
    Xtr, ytr, Xte, yte = load_electricity_dataset_multivariate(
        "data.txt", window=24, target_column=20, feature_columns=[5, 10, 15]
    )
    # Resultado: Xtr = (N, 72) - ventanas de 24×3 valores intercalados
"""
function load_electricity_dataset_multivariate(
    path::String; 
    window::Int, 
    horizon::Int = 1, 
    train_ratio::Float64 = 0.75,
    target_column::Union{Int,Nothing} = nothing,
    feature_columns::Union{Vector{Int},Nothing} = nothing
)
    # Leer el archivo
    println("Cargando dataset desde: $path")
    data = CSV.File(path, delim=";", quotechar='"', header=1)
    df = DataFrame(data)
    
    println("Dimensiones originales: $(size(df))")
    
    # Eliminar timestamps (primera columna)
    df = df[:, 2:end]
    println("Dimensiones después de eliminar timestamps: $(size(df))")
    
    # Convertir a Float64
    for col in names(df)
        if eltype(df[!, col]) != Float64
            try
                df[!, col] = Float64.(df[!, col])
            catch
                try
                    df[!, col] = parse.(Float64, replace.(string.(df[!, col]), "," => "."))
                catch e
                    println("Warning: Error convirtiendo columna $col: $e")
                    df[!, col] = zeros(Float64, nrow(df))
                end
            end
        end
    end
    
    # Determinar columnas
    if target_column === nothing
        target_column = ncol(df)
    end
    
    if feature_columns === nothing
        # Por defecto: usar solo la columna objetivo como feature
        feature_columns = [target_column]
    end
    
    n_features = length(feature_columns)
    
    println("\nConfiguración:")
    println("  Target: Columna $target_column - $(names(df)[target_column])")
    println("  Features: $n_features columnas - $(names(df)[feature_columns])")
    println("  Window size: $window")
    println("  Total features por ventana: $(window * n_features)")
    
    # División train/test
    total_rows = nrow(df)
    train_size = Int(round(total_rows * train_ratio))
    
    train_data = df[1:train_size, :]
    test_data = df[(train_size+1):end, :]
    
    # Crear ventanas
    n_train_samples = train_size - window - horizon + 1
    n_test_samples = nrow(test_data) - window - horizon + 1
    
    println("\nVentanas de entrenamiento: $n_train_samples")
    println("Ventanas de test: $n_test_samples")
    
    if n_train_samples <= 0 || n_test_samples <= 0
        error("No hay suficientes datos para crear ventanas.")
    end
    
    # Crear matrices
    # OPCIÓN 1: Intercalar features (X1_t1, X2_t1, X3_t1, X1_t2, X2_t2, X3_t2, ...)
    # OPCIÓN 2: Concatenar features (X1_t1...X1_t24, X2_t1...X2_t24, X3_t1...X3_t24)
    # Usaremos OPCIÓN 2 (más intuitiva para DoME)
    
    Xtr = zeros(Float64, n_train_samples, window * n_features)
    ytr = zeros(Float64, n_train_samples)
    
    for i in 1:n_train_samples
        # Para cada feature, extraer su ventana y concatenar
        for (feat_idx, col_idx) in enumerate(feature_columns)
            start_col = (feat_idx - 1) * window + 1
            end_col = feat_idx * window
            Xtr[i, start_col:end_col] = train_data[i:(i+window-1), col_idx]
        end
        ytr[i] = train_data[i+window+horizon-1, target_column]
    end
    
    Xte = zeros(Float64, n_test_samples, window * n_features)
    yte = zeros(Float64, n_test_samples)
    
    for i in 1:n_test_samples
        for (feat_idx, col_idx) in enumerate(feature_columns)
            start_col = (feat_idx - 1) * window + 1
            end_col = feat_idx * window
            Xte[i, start_col:end_col] = test_data[i:(i+window-1), col_idx]
        end
        yte[i] = test_data[i+window+horizon-1, target_column]
    end
    
    println("\nDatos preparados:")
    println("  Xtr: $(size(Xtr))")
    println("  ytr: $(size(ytr))")
    println("  Xte: $(size(Xte))")
    println("  yte: $(size(yte))")
    
    println("\nEstadísticas:")
    println("  ytr - Media: $(round(mean(ytr), digits=4)), Std: $(round(std(ytr), digits=4))")
    println("  yte - Media: $(round(mean(yte), digits=4)), Std: $(round(std(yte), digits=4))")
    
    if n_features > 10
        @warn "Warning: Usando $n_features features × $window timesteps = $(window*n_features) variables totales. Esto puede ser demasiado para DoME."
    end
    
    return Xtr, ytr, Xte, yte
end