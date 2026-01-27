using CSV
using DataFrames
using Statistics

"""
    load_electricity_dataset(
        path::String;
        window::Int,
        horizon::Int = 1,
        train_ratio::Float64 = 0.75,
        target_col::Union{String,Int,Nothing} = nothing,
        target_column::Union{Int,Nothing} = nothing
    )

Carga el dataset y crea ventanas temporales.

Parámetros:
- path: Ruta al archivo CSV
- window: Tamaño de la ventana temporal
- horizon: Pasos adelante para predecir
- train_ratio: Proporción de datos para entrenamiento
- target_col: Nombre de la columna a predecir (string) o índice (int)
- target_column: Índice de la columna a predecir (alternativa a target_col)

Nota: target_col y target_column son sinónimos. Si ninguno se especifica, usa la última columna.
"""
function load_electricity_dataset(
    path::String; 
    window::Int, 
    horizon::Int = 1, 
    train_ratio::Float64 = 0.75,
    target_col::Union{String,Int,Nothing} = nothing,
    target_column::Union{Int,Nothing} = nothing
)
    # Lee el archivo usando CSV.jl
    println("Cargando dataset desde: $path")
    data = CSV.File(path, delim=";", quotechar='"', header=1)
    
    # Convierte el CSV a DataFrame
    df = DataFrame(data)
    
    println("Dimensiones originales: $(size(df))")
    println("Nombres de columnas (primeras 5): $(names(df)[1:min(5, length(names(df)))])")
    
    # IMPORTANTE: Elimina la primera columna que contiene los timestamps
    df = df[:, 2:end]
    
    println("Dimensiones después de eliminar timestamps: $(size(df))")
    
    # Convierte todas las columnas a Float64
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
    
    println("Tipos de datos después de conversión: $(eltype.(eachcol(df))[1:min(5, ncol(df))])")
    
    # Determinar la columna objetivo
    # Prioridad: target_col > target_column > última columna
    target_col_idx = nothing
    
    if target_col !== nothing
        if target_col isa String
            # Buscar el índice por nombre de columna
            col_names = names(df)
            idx = findfirst(==(target_col), col_names)
            if idx === nothing
                error("Columna '$target_col' no encontrada. Columnas disponibles: $col_names")
            end
            target_col_idx = idx
            println("Usando columna '$target_col' (índice $target_col_idx)")
        elseif target_col isa Int
            target_col_idx = target_col
            println("Usando columna $target_col_idx: $(names(df)[target_col_idx])")
        end
    elseif target_column !== nothing
        target_col_idx = target_column
        println("Usando columna $target_column: $(names(df)[target_column])")
    else
        target_col_idx = ncol(df)  # Última columna por defecto
        println("Usando última columna (índice $target_col_idx): $(names(df)[target_col_idx])")
    end
    
    # Verificar que el índice es válido
    if target_col_idx < 1 || target_col_idx > ncol(df)
        error("Índice de columna $target_col_idx fuera de rango. Total columnas: $(ncol(df))")
    end
    
    # Obtener el total de filas
    total_rows = nrow(df)
    train_size = Int(round(total_rows * train_ratio))
    
    println("Total de filas: $total_rows")
    println("Filas de entrenamiento: $train_size")
    println("Filas de test: $(total_rows - train_size)")
    
    # Dividir en entrenamiento y test
    train_data = df[1:train_size, :]
    test_data = df[(train_size+1):end, :]
    
    # Crear las ventanas temporales
    n_train_samples = train_size - window - horizon + 1
    n_test_samples = nrow(test_data) - window - horizon + 1
    
    println("Ventanas de entrenamiento: $n_train_samples")
    println("Ventanas de test: $n_test_samples")
    
    if n_train_samples <= 0 || n_test_samples <= 0
        error("No hay suficientes datos para crear ventanas. Necesitas al menos $(window + horizon) filas.")
    end
    
    # Solo usar la columna objetivo para las features
    println("\nFORMATO CORRECTO PARA DOME:")
    println("   - Cada ventana tiene $window valores de la variable objetivo")
    println("   - Cada fila = 1 ventana con $window features")
    
    # Extraer la serie temporal de la variable objetivo
    target_series_train = train_data[!, target_col_idx]
    target_series_test = test_data[!, target_col_idx]
    
    # Crear matrices para los datos de entrenamiento
    Xtr = zeros(Float64, n_train_samples, window)
    ytr = zeros(Float64, n_train_samples)
    
    for i in 1:n_train_samples
        # Extraer ventana: valores consecutivos de la serie temporal
        Xtr[i, :] = target_series_train[i:(i+window-1)]
        # Target: valor "horizon" pasos adelante
        ytr[i] = target_series_train[i+window+horizon-1]
    end
    
    # Crear matrices para los datos de test
    Xte = zeros(Float64, n_test_samples, window)
    yte = zeros(Float64, n_test_samples)
    
    for i in 1:n_test_samples
        Xte[i, :] = target_series_test[i:(i+window-1)]
        yte[i] = target_series_test[i+window+horizon-1]
    end
    
    println("\nDatos preparados exitosamente:")
    println("  Xtr: $(size(Xtr)), tipo: $(eltype(Xtr))")
    println("  ytr: $(size(ytr)), tipo: $(eltype(ytr))")
    println("  Xte: $(size(Xte)), tipo: $(eltype(Xte))")
    println("  yte: $(size(yte)), tipo: $(eltype(yte))")
    
    # Mostrar estadísticas
    println("\nEstadísticas de los targets:")
    println("  ytr - Media: $(round(mean(ytr), digits=4)), Std: $(round(std(ytr), digits=4))")
    println("  yte - Media: $(round(mean(yte), digits=4)), Std: $(round(std(yte), digits=4))")
    
    if std(ytr) < 1e-10
        @warn "La varianza de ytr es muy baja. Puede que todos los valores sean iguales."
    end
    
    return Xtr, ytr, Xte, yte
end