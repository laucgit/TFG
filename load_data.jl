using CSV
using DataFrames

"""
    load_electricity_dataset(
        path::String;
        window::Int,
        horizon::Int = 1,
        train_ratio::Float64 = 0.75
    )

Carga el dataset de electricidad (único archivo), crea ventanas temporales
y divide en entrenamiento y test.
"""
function load_electricity_dataset(path::String; window::Int, horizon::Int = 1, train_ratio::Float64 = 0.75)
    # Leer el archivo usando CSV.jl
    println("Cargando dataset desde: $path")
    data = CSV.File(path, delim=";", quotechar='"', header=1)
    
    # Convertir el CSV a DataFrame
    df = DataFrame(data)
    
    println("Dimensiones originales: $(size(df))")
    println("Nombres de columnas (primeras 5): $(names(df)[1:min(5, length(names(df)))])")
    
    # IMPORTANTE: Eliminar la primera columna que contiene los timestamps
    # La primera columna suele llamarse "" o contener las fechas
    df = df[:, 2:end]  # Seleccionar todas las columnas excepto la primera
    
    println("Dimensiones después de eliminar timestamps: $(size(df))")
    
    # Convertir todas las columnas a Float64
    # Reemplazar comas por puntos si es necesario (formato europeo de decimales)
    for col in names(df)
        if eltype(df[!, col]) != Float64
            try
                # Intentar conversión directa
                df[!, col] = Float64.(df[!, col])
            catch
                try
                    # Si falla, intentar reemplazar comas por puntos y parsear
                    df[!, col] = parse.(Float64, replace.(string.(df[!, col]), "," => "."))
                catch e
                    println("Error convirtiendo columna $col: $e")
                    # Si todo falla, llenar con ceros
                    df[!, col] = zeros(Float64, nrow(df))
                end
            end
        end
    end
    
    println("Tipos de datos después de conversión: $(eltype.(eachcol(df))[1:min(5, ncol(df))])")
    
    # Obtener el total de filas
    total_rows = nrow(df)
    
    # Calcular el tamaño de entrenamiento
    train_size = Int(round(total_rows * train_ratio))
    
    println("Total de filas: $total_rows")
    println("Filas de entrenamiento: $train_size")
    println("Filas de test: $(total_rows - train_size)")
    
    # Dividir en entrenamiento y test
    train_data = df[1:train_size, :]
    test_data = df[(train_size+1):end, :]
    
    # Crear las ventanas temporales
    # Número de muestras que podemos crear
    n_train_samples = train_size - window - horizon + 1
    n_test_samples = nrow(test_data) - window - horizon + 1
    
    println("Ventanas de entrenamiento: $n_train_samples")
    println("Ventanas de test: $n_test_samples")
    
    if n_train_samples <= 0 || n_test_samples <= 0
        error("No hay suficientes datos para crear ventanas. Necesitas al menos $(window + horizon) filas.")
    end
    
    # Determinar número de features (todas las columnas menos la última que es el target)
    n_features = ncol(df) - 1
    
    println("Número de features: $n_features")
    println("Tamaño de cada ventana aplanada: $(window * n_features)")
    
    # Crear matrices para los datos de entrenamiento
    # Cada fila será una muestra aplanada de la ventana temporal
    Xtr = zeros(Float64, n_train_samples, window * n_features)
    ytr = zeros(Float64, n_train_samples)
    
    for i in 1:n_train_samples
        # Extraer la ventana (todas las columnas excepto la última)
        window_data = Matrix(train_data[i:(i+window-1), 1:n_features])
        # Aplanar la ventana en un vector fila
        Xtr[i, :] = vec(window_data')  # Transponer para mantener el orden temporal
        # El target está 'horizon' pasos adelante (última columna)
        ytr[i] = train_data[i+window+horizon-1, end]
    end
    
    # Crear matrices para los datos de test
    Xte = zeros(Float64, n_test_samples, window * n_features)
    yte = zeros(Float64, n_test_samples)
    
    for i in 1:n_test_samples
        window_data = Matrix(test_data[i:(i+window-1), 1:n_features])
        Xte[i, :] = vec(window_data')
        yte[i] = test_data[i+window+horizon-1, end]
    end
    
    println("Datos preparados exitosamente:")
    println("  Xtr: $(size(Xtr)), tipo: $(eltype(Xtr))")
    println("  ytr: $(size(ytr)), tipo: $(eltype(ytr))")
    println("  Xte: $(size(Xte)), tipo: $(eltype(Xte))")
    println("  yte: $(size(yte)), tipo: $(eltype(yte))")
    
    return Xtr, ytr, Xte, yte
end