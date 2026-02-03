using Random
using Statistics
using SymDoME

println("="^70)
println("TEST SINTÉTICO - Validación de DoME en Vacío")
println("="^70)
println("\nOBJETIVO: Verificar que DoME funciona con datos sintéticos")
println("DATOS: y = 0.7*X₁ - 0.2*X₃ + ruido")
println("="^70)

Random.seed!(1)
N = 2000
P = 5
X = rand(N, P)
y = 0.7 .* X[:,1] .- 0.2 .* X[:,3] .+ 0.05 .* randn(N)

println("\n[DATOS]")
println("  N muestras: $N")
println("  P features: $P")
println("  Relación verdadera: y = 0.7*X₁ - 0.2*X₃ + ruido")

println("\n[ENTRENANDO DoME - API DE BAJO NIVEL]")
println("  Parámetros:")
println("    - maximumNodes: 30")
println("    - minimumReductionMSE: 1e-6")
println("    - useDivisionOperator: false")
println("    - strategy: Strategy3")

# CORRECCIÓN DEFINITIVA: Usar SymDoME.DoME (struct interno no exportado)
dome_obj = SymDoME.DoME(
    X, y;
    maximumNodes=30,
    minimumReductionMSE=1e-6,
    useDivisionOperator=false,
    strategy=SymDoME.Strategy3
)

# Registrar MSE inicial
hist = Float64[dome_obj.mse]

# Iterar hasta convergencia
max_iterations = 1000

for iteration in 1:max_iterations
    improved = SymDoME.Step!(dome_obj)
    push!(hist, dome_obj.mse)
    
    if !improved
        println("  Convergió en iteración $iteration")
        break
    end
end

tree = dome_obj.tree

println("\n[RESULTADO]")
println("  Iteraciones: $(length(hist))")
println("  MSE inicial: $(round(hist[1], digits=6))")
println("  MSE final: $(round(hist[end], digits=6))")
println("  Mejora: $(round((1 - hist[end]/hist[1])*100, digits=2))%")
println("\n  Expresión encontrada:")
# vectorString funciona (confirmado en diagnóstico)
try
    println("    $(SymDoME.vectorString(tree))")
catch e
    println("    [Error: $e]")
end

println("\n" * "="^70)
println("DIAGNÓSTICO")
println("="^70)

if length(hist) > 1
    println("\n[OK] ÉXITO: DoME iteró y mejoró")
    println("\nDoME funciona correctamente.")
    println("Si tus datos reales fallan, el problema está en:")
    println("  1. Las ventanas temporales (setup de features)")
    println("  2. La columna objetivo está siendo eliminada de los lags")
    println("  3. Los datos no tienen señal simbólica clara")
else
    println("\n[ERROR] FALLO: DoME se quedó en 1 iteración")
    println("\nProblemas posibles:")
    println("  1. Versión incorrecta de SymDoME")
    println("  2. Bug en cómo se recoge history")
    println("  3. Problema con la instalación")
    println("\nRECOMENDACIÓN: Verificar instalación de SymDoME")
end

if hist[end] < hist[1]
    println("\n[OK] MSE mejoró: $(round(hist[1], digits=6)) → $(round(hist[end], digits=6))")
else
    println("\n[WARNING] MSE NO mejoró")
end

println("="^70)