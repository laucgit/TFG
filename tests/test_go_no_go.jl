include("run_experiment_multi.jl")
using Random
using Statistics
using SymDoME

println("="^70)
println("VALIDACIÓN GO/NO-GO ANTES DE CESGA")
println("="^70)
println("\nPLAN (según feedback de ChatGPT):")
println("1. Test sintético: ¿DoME funciona en vacío?")
println("2. Test real: ¿Los datos tienen señal?")
println("3. Criterio de pasa: length(history) > 1 y MSE mejora")
println("="^70)

# =============================================================================
# PASO 1: TEST SINTÉTICO - ¿DoME funciona en vacío?
# =============================================================================

println("\n" * "="^70)
println("PASO 1: TEST SINTÉTICO")
println("="^70)

Random.seed!(1)
N = 2000
P = 5
X = rand(N, P)
y = 0.7 .* X[:,1] .- 0.2 .* X[:,3] .+ 0.05 .* randn(N)

println("\n[DATOS SINTÉTICOS]")
println("  Muestras: $N")
println("  Features: $P")
println("  Relación: y = 0.7*X₁ - 0.2*X₃ + ruido")

println("\n[ENTRENANDO DoME]")
# CORRECCIÓN DEFINITIVA: Usar SymDoME.DoME (struct interno)
dome_obj = SymDoME.DoME(
    X, y;
    maximumNodes=30,
    minimumReductionMSE=1e-6,
    useDivisionOperator=false,
    strategy=SymDoME.Strategy3
)

hist = Float64[dome_obj.mse]
max_iterations = 1000

for iteration in 1:max_iterations
    improved = SymDoME.Step!(dome_obj)
    push!(hist, dome_obj.mse)
    
    if !improved
        break
    end
end

tree = dome_obj.tree

println("\n[RESULTADO SINTÉTICO]")
println("  Iteraciones: $(length(hist))")
println("  MSE inicial: $(round(hist[1], digits=6))")
println("  MSE final: $(round(hist[end], digits=6))")
println("  Mejora: $(round((1 - hist[end]/hist[1])*100, digits=2))%")

# Obtener expresión del árbol
print("  Expresión: ")
try
    println(SymDoME.vectorString(tree))
catch e
    println("[error: $e]")
end

synthetic_pass = length(hist) > 1 && hist[end] < hist[1]

if synthetic_pass
    println("\n[PASO 1 EXITOSO]: DoME funciona correctamente")
else
    println("\n[PASO 1 FALLIDO]: Problema con DoME o su instalación")
    println("\nRECOMENDACIÓN: Verificar instalación de SymDoME")
    println("NO continuar a CESGA hasta resolver esto")
    exit(1)
end

# =============================================================================
# PASO 2: TEST REAL - ¿Los datos tienen señal?
# =============================================================================

println("\n" * "="^70)
println("PASO 2: TEST CON DATOS REALES")
println("="^70)

count_nonfinite(A::AbstractArray{Float64}) = count(v -> !isfinite(v), A)

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

# Usar ETTh2 - dataset conocido
dataset_name = "ETT/ETTh2.csv"
target_col = "OT"
window = 12

println("\n[CONFIGURACIÓN]")
println("  Dataset: $dataset_name")
println("  Target: $target_col")
println("  Window: $window")
println("  Config: maxNodes=30, minReduction=1e-6, Strategy3")

try
    # IMPORTANTE: Asegurarse de que load_dataset INCLUYE lags del objetivo
    Xtr, ytr, Xte, yte = load_dataset(
        dataset_name,
        data_dir="data",
        window=window,
        horizon=1,
        target_col=target_col,
        train_ratio=0.75
    )
    
    println("\n[DATOS CARGADOS]")
    println("  Xtr: $(size(Xtr))")
    println("  ytr: $(size(ytr))")
    println("  Features/samples: $(round(size(Xtr,2)/size(Xtr,1), digits=3))")
    
    # Verificar si incluye lags del objetivo
    if size(Xtr, 2) % window != 0
        println("\n[ADVERTENCIA]: Número de features no es múltiplo de window")
        println("    Esto puede indicar que los lags del objetivo fueron eliminados")
    end
    
    # Imputar
    impute_nonfinite!(Xtr)
    impute_nonfinite!(ytr)
    impute_nonfinite!(Xte)
    impute_nonfinite!(yte)
    
    # Normalizar
    println("\n[NORMALIZANDO]")
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
    
    println("  [Normalizado al rango [0, 1]]")
    
    # Entrenar con una configuración simple pero efectiva
    println("\n[ENTRENANDO DoME CON DATOS REALES]")
    
    params = DoMEParams(
        30,      # maxNodes
        1e-6,    # minReduction
        false,   # useDiv
        SymDoME.Strategy3
    )
    
    tree, history = train_dome(Xtr_norm, ytr_norm, params)
    ŷ_test, mse_test = evaluate_dome(tree, Xte_norm, yte_norm)
    
    converged = length(history) > 1
    improvement = (1 - history[end]/history[1]) * 100
    
    println("\n[RESULTADO REAL]")
    println("  Iteraciones: $(length(history))")
    println("  MSE inicial: $(round(history[1], digits=6))")
    println("  MSE final: $(round(history[end], digits=6))")
    println("  MSE test: $(round(mse_test, digits=6))")
    println("  Mejora: $(round(improvement, digits=2))%")
    
    real_pass = length(history) > 1 && history[end] < history[1]
    
    println("\n" * "="^70)
    println("RESULTADO FINAL GO/NO-GO")
    println("="^70)
    
    if real_pass
        println("\n[VALIDACIÓN EXITOSA]")
        println("\n1. [OK] DoME funciona (test sintético pasado)")
        println("2. [OK] DoME aprende de tus datos (test real pasado)")
        println("3. [OK] Iteraciones > 1: $(length(history))")
        println("4. [OK] MSE mejora: $(round(improvement, digits=2))%")
        println("\n>>> LISTO PARA CESGA <<<")
        println("\nPróximos pasos:")
        println("  1. Subir scripts corregidos a CESGA")
        println("  2. Ejecutar: julia sweep_experiments_multi.jl")
        println("\nTiempo estimado: ~8-12 horas para 2,400 experimentos")
    else
        println("\n[VALIDACIÓN PROBLEMÁTICA]")
        println("\n1. [OK] DoME funciona (test sintético pasado)")
        println("2. [FALLO] DoME NO aprende de tus datos (test real fallido)")
        println("3. [FALLO] Iteraciones: $(length(history))")
        println("4. [FALLO] Mejora: $(round(improvement, digits=2))%")
        println("\nPOSIBLES CAUSAS:")
        println("  A) Los lags del objetivo AÚN se están eliminando")
        println("     -> Verificar load_data_multi.jl línea ~466")
        println("  B) Los datos no tienen señal simbólica clara")
        println("     -> Probar con window más pequeño (6)")
        println("  C) DoME necesita más nodos (probar 50-100)")
        println("\nOPCIONES:")
        println("  1. Ejecutar en CESGA de todas formas (documentar fallo)")
        println("  2. Ajustar hiperparámetros más agresivos")
        println("  3. Consultar con profesor")
    end
    println("="^70)
    
catch e
    println("\n[ERROR] en test real:")
    println("  $e")
    showerror(stdout, e, catch_backtrace())
    println()
end