using SymDoME
using Random

println("="^70)
println("DIAGNÓSTICO AVANZADO DE SymDoME")
println("="^70)

# 1. Ver TODOS los nombres (incluyendo no exportados)
println("\n[1] TODOS los símbolos en SymDoME (incluyendo internos):")
all_names = names(SymDoME, all=true)
dome_related = filter(n -> occursin(r"dome|DOME|DoME"i, string(n)), all_names)
println("\nRelacionados con 'dome':")
for name in dome_related
    println("  - $name")
end

# 2. Verificar si DOME es un tipo o función
println("\n[2] Tipo de DOME:")
if isdefined(SymDoME, :DOME)
    println("  ✓ DOME existe en SymDoME")
    println("  Tipo: $(typeof(SymDoME.DOME))")
    if SymDoME.DOME isa Type
        println("  Es un TIPO (struct)")
        println("  Campos: $(fieldnames(SymDoME.DOME))")
    else
        println("  Es una FUNCIÓN")
    end
else
    println("  ✗ DOME NO existe en SymDoME")
    println("  Probando con minúsculas...")
    if isdefined(SymDoME, :dome)
        println("  ✓ dome (minúscula) existe")
    end
end

# 3. Probar llamada directa con namespace
println("\n[3] Probando SymDoME.DOME directamente:")
Random.seed!(1)
X = rand(50, 3)
y = 0.5 .* X[:,1] .+ 0.3 .* X[:,2]

try
    result = SymDoME.DOME(X, y; maximumNodes=10)
    println("  ✓ SymDoME.DOME funcionó!")
    println("  Tipo resultado: $(typeof(result))")
catch e
    println("  ✗ SymDoME.DOME falló: $e")
end

# 4. Ver qué devuelve dome() exactamente
println("\n[4] Investigando qué devuelve dome():")
try
    result = dome(X, y; maximumNodes=10, minimumReductionMSE=1e-6)
    println("  Tipo: $(typeof(result))")
    
    if result isa Tuple
        println("  Longitud tupla: $(length(result))")
        for (i, elem) in enumerate(result)
            println("  Elemento $i: tipo=$(typeof(elem))")
            if elem isa AbstractFloat
                println("    Valor: $elem")
            elseif elem isa AbstractVector && length(elem) < 20
                println("    Vector longitud $(length(elem)): $elem")
            elseif elem isa AbstractVector
                println("    Vector longitud $(length(elem))")
                println("    Primeros 5: $(elem[1:min(5,end)])")
                println("    Últimos 5: $(elem[max(1,end-4):end])")
            end
        end
    end
    
    # Intentar extraer árbol
    println("\n[5] Intentando obtener árbol del resultado:")
    if result isa Tuple && length(result) >= 1
        tree = result[1]
        println("  Árbol obtenido, tipo: $(typeof(tree))")
        
        # Probar métodos de visualización
        println("\n[6] Probando métodos de visualización del árbol:")
        
        methods_to_try = [
            :writeAsTree,
            :latexString,
            :vectorString,
            :string,
            :print,
            :show
        ]
        
        for method in methods_to_try
            if isdefined(SymDoME, method)
                try
                    expr = getfield(SymDoME, method)(tree)
                    println("  ✓ $method funcionó: $expr")
                catch e
                    println("  ✗ $method falló: $e")
                end
            else
                println("  - $method no existe")
            end
        end
    end
    
catch e
    println("  ✗ Error: $e")
    showerror(stdout, e, catch_backtrace())
end

# 5. Buscar funciones Step o iterate
println("\n\n[7] Buscando funciones de iteración:")
step_related = filter(n -> occursin(r"step|Step|iterate"i, string(n)), all_names)
for name in step_related
    println("  - $name")
end

println("\n" * "="^70)
println("CONCLUSIÓN:")
println("="^70)
println("Con esta información sabremos:")
println("1. Si DOME existe y cómo usarlo")
println("2. Qué devuelve dome() realmente")
println("3. Cómo obtener la expresión del árbol")
println("4. Si hay funciones para iterar manualmente")
println("="^70)