10/12/2025

- CRISP-DM (metodología)

- mecanismo olvido: ventana temporal

- cuenta cesga: cesga.es
- firma digital
- datasets de papers: verlos y mirar si hay código, respos o ŕomulas
- hacerse con librería de dome
- SLUM
- experimentar con distintos tamaños de ventana, coger dataset y ver los tamaños y atributos, experimentar con número de nodos (esto para los experimentos)

- documentar el proceso

- ir empezando la memoria (subcapítulo por iteración)

- convocatorias de becas predoctorales



11/12/2025

Estoy haciendo la cuenta del cesga. Problemas: 

- Hay un apartado de Puesto, Responsable, Dirección, Área, ORCID (no obligatioria), Researcher ID (no ibligatoria), Scopus ID (no obligatoria), Código. No sé exactamente qué meter ahí. **Preguntar**


Comprobación de la firma digital. Problemas:

- Por un momento la confundí con la electrónica, pero firma digital personal no tengo. Conseguirla o pedir "prestada?".


Datasets de papers: hecho en latex

Librería de DOME: https://github.com/danielriveroc/SymDoME.jl

SLUM: colas para el cesga creo, esto cuando tenga el cesga


17/12/2025

Para el cesga ya he pedido la cuenta, ahora toca hacer lo que dijo dani de los primeros experimentos


27/01/2026

Los scripts que he trabajado hoy incluyen:

1. **`run_experiment.jl`**: Este script ejecuta el experimento principal, donde se lleva a cabo la generación y evaluación de las expresiones matemáticas. He corregido y ajustado la estructura para asegurar que los modelos generados se ajusten correctamente a los datos de entrada y salida.

2. **`sweep_experiments.jl`**: Este script permite realizar barridos de parámetros, evaluando diferentes configuraciones de hiperparámetros para el modelo. Esto me ha permitido experimentar con varias combinaciones para encontrar la configuración más óptima.

3. **`load_data.jl`**: Este archivo se encarga de cargar los datos necesarios para los experimentos, asegurando que los conjuntos de entrenamiento y prueba estén listos para ser utilizados en el proceso de ajuste de los modelos.

4. **`crear_minidataset.jl`**: He utilizado este script para crear un conjunto de datos reducido que facilita las pruebas rápidas y la depuración de los experimentos sin necesidad de cargar grandes volúmenes de datos.

falta pulir los scripts de plot


28/01/2026

Hechos los scripts multis
Arreglar (???) el árbol de los resultados
Elegir el tercer dataset
De momento, los datasets que tengo son Electricity Load Diagrams 2011–2014 (paper: A. Halstead et al. – “Recurrent Concept Drifts on Data Streams” (IJCAI, 2024)) y CinCECGTorso (paper: S. Hinder et al. – “Data streams classification using
deep learning under different concept drifts” (Journal of Logic and Computation, 2023)) (descartado)
Con un dataset de 678,1 MiB, tarda 5 minutos 19 segundos en local con CPU
Hecho sweep_experiments_multi para probar con distintos hiperparámetros, funciona correctamente con ambos datasets. Se guardan los resultados correctamente.
Necesito más datasets, esto es algo realmente importante

Dudas:

- Para la fase de stream learning necesito los mismos datasets? yep
- Algunos de estos datasets no tienen métricas en el paper, q hago? no passa na

Debería buscar más datasets y papers


03/02/2026

Vale, tengo que hacer:

- Hold out (done (comprueba))
- Aislar Train (debería estar más aislado)
- Entrenar hasta que el mse no mejore (done)
- Implementar la función de Dani (done)
- Graficar el drift

La cuenta del CESGA ya está activa, pero aún tengo problemas para hacer lo de la validación del MSE. Hoy o mañana tengo que mandar los experimentos, entonces también tienes que mirar bien si los datasets sirven bien con sus respectivos papers, y si son suficientes.

Tengo que:

- hacer la función de graficar

- cambiar cómo guardo los resultados

- aislar más la función de train


04/02/2026

- Debería guardar el árbol??

Tengo que:

- hacer la función de graficar (done)

- cambiar cómo guardo los resultados (done)

- aislar más la función de train (done)

- quitar el parámetro de _model_ y ver si hay que quitar también el de _norm_: creo que se queda para reproducibilidad (done)

- ver cómo mando todo el entorno al CESGA para evitar tener problemas

- debería guardar las predicciones? pa graficar sep


05/02/2026

Tengo que :

- comprobar que todo funciona correctamente y está listo para mandar al CESGA (done)

- mandar al cesga los experimentos ejecutando con un solo dataset (done)

Preguntar qué debo tener exactamente en el script del CESGA, es decir, qué parámetros (done)

- yo añadiría lo de la notificación de end/fail del job en el CESGA

- escribir todo lo de hasta ahora en la memoria


10/02/2026

El primer dataset ya se está ejecutando en el CESGA.
La memoria está avanzada en el capítulo de la parte de DoME sin SL.

12/02/2026

Tuve problemas con el script y el guardado de los .jd2 pero ya está resuelto y ahora faltan las últimas configuraciones con ese dataset, pero faltan los demás, me preocupa que no lleguen, hay que ir apresurando esa parte.
La memoria está avanzada en el capítulo de la parte de DoME sin SL, corregidos los detalles grandes de la primera parte, falta corregir los detalles pequeños, y luego reafinar la parte de después de los resultados.
Debería cambiar el script de analizar para que me vaya ya dando resultados para graficar.

23/02/2026

LISTA DE COSAS QUE HAY QUE HACER pt 2

GENERAL / PREVIO
- [ ] Documentar en la memoria lo ya hecho (mientras se conceptualiza la siguiente iteración).
- [ ] Confirmar que cada dataset está bien tipado:
      - Si fuese CLASIFICACIÓN: los targets deben ser un vector de booleanos.
      - Si los targets son reales: DoME lo tratará como REGRESIÓN (que es lo que quieres en forecasting).
- [ ] Meter el resto de datasets en el pipeline (y comprobar el punto anterior para cada uno).
- [ ] No usar en la parte de stream learning el dataset que “no es de stream” (se queda fuera de esa iteración).

ITERACIÓN 1 (ventanas / batch): hacerlo más exhaustivo y dejarlo cerrado
- [ ] Probar más tamaños de ventana, no solo 12/24/48:
      - Barrer alrededor del que mejor salga (p. ej. 10, 15, 20, 25, 30, 35...) y ver dónde empeora el MSE.
- [ ] Sacar la “configuración ganadora” por dataset:
      - Mejor configuración y su MSE para el mejor tamaño de ventana (conclusión clave de la iteración 1).
- [ ] Hacer una tabla comparativa final por dataset, con al menos:
      - Tamaños de ventana probados
      - MSE obtenido
      - Valor de referencia (baseline)
      - Configuración (y si puedes, nº de nodos)
      - (Opcional) tiempo computacional (aquí menos relevante, pero si lo metes, que sea consistente)
- [ ] Decidir qué gráficas merece la pena:
      - Si haces evolución vs max nodos: implicaría replicarlo por dataset y por tamaño de ventana (y además por estrategia);
        es decir, muchas gráficas. Dejarlo para decidir al escribir.

ITERACIÓN 2 (stream learning): programar el bucle y experimentar “cuánto entrenas” por muestra
- [ ] Hacer stream learning con warm start (si es lo que hacen las referencias):
      1) Elegir el mejor tamaño de ventana (p. ej. 24)
      2) Entrenar DoME con esa ventana -> modelo inicial
      3) Bucle streaming: por cada nueva muestra:
         - predecir
         - comparar con el valor real
         - actualizar ventana (entra nueva, sale la más vieja)
         - re-entrenar con la ventana actual
- [ ] Experimentar con el número de iteraciones/ciclos de DoME por cada muestra que llega (p. ej. 1, 5, ...).
      -> Este es el parámetro clave de la iteración 2.
- [ ] Comparar estrategias en stream:
      - una optimiza constantes y la otra hace cambios pequeños -> puede afectar a la reactividad.
- [ ] Comprobar que el coste es práctico:
      - El tiempo alto medido era con todo el dataset “enventanado”; con ventana 24 debería bajar,
        pero hay que verificarlo, sobre todo si los datos llegan muy frecuentemente (minuto vs diario).
- [ ] Reutilizar la implementación que ya tienes:
      - usar el parámetro opcional initialTree para continuar desde el modelo previo
      - reutilizar la función para “añadir muestra y actualizar ventana”
- [ ] Sacar una gráfica de serie real vs predicción en streaming (real vs DoME) y usarla en discusión:
      - señalar tramos donde deja de seguir y luego se readapta.

ITERACIÓN 3 (si procede): drift
- [ ] No meter drift en la iteración 2.
- [ ] Si en las gráficas de stream se aprecia drift: iteración 3 = combinar DoME + detector de drift,
      justificándolo con los tramos donde DoME no sigue bien.

MÁS ADELANTE (ajuste fino)
- [ ] Asumir que la mejor configuración en batch puede NO ser la misma en stream;
      más adelante se puede retocar nº de nodos/config ya orientado a streaming.


3/3/2026

-Ir haciendo las gráficas ya de ya
-Ir programando la fase 3
-Ir haciendo la memoria (cambiar los datasets e ir haciendo las tablas formato Dani)

ejemplo tabla: 

dataset|MSE ref|valor de referencia|window|dome (no sé muy bien cómo llamar a esta columna)|configuración

ETTH2 | 0.6 | [7] | 10 15 20 ... | 0.5 0.4 0.6 | .....


ett: https://github.com/zhouhaoyi/ETDataset


MEJOR CONFIG POR VENTANA                                                                                                                                                                                                                                    
==========================================================================================                                                                                                                                                                  
w10 | c80 | accuracy=0.992226 | max_nodes=50 | min_improvement=1.0e-7 | strategy=StrategySelective | file=results_unsw_fr/unsw_bin_w10__20260329_160045.jld2                                                                                                
w15 | c3 | accuracy=0.950988 | max_nodes=15 | min_improvement=0.01 | strategy=StrategySelective | file=results_unsw_fr/unsw_bin_w15__20260329_210720.jld2                                                                                                   
w20 | c109 | accuracy=0.988789 | max_nodes=60 | min_improvement=0.001 | strategy=StrategySelectiveWithConstantOptimization | file=results_unsw/unsw_bin_w20__20260329_165145.jld2                                                                           
w25 | c19 | accuracy=0.997507 | max_nodes=25 | min_improvement=0.001 | strategy=StrategySelective | file=results_unsw/unsw_bin_w25__20260329_165145.jld2                                                                                                    
w30 | c108 | accuracy=0.987679 | max_nodes=50 | min_improvement=0.001 | strategy=StrategySelectiveWithConstantOptimization | file=results_unsw/unsw_bin_w30__20260329_165145.jld2                                                                           
w35 | c107 | accuracy=0.989543 | max_nodes=45 | min_improvement=0.001 | strategy=StrategySelectiveWithConstantOptimization | file=results_unsw/unsw_bin_w35__20260329_165145.jld2                                                                           
w40 | c19 | accuracy=0.980319 | max_nodes=25 | min_improvement=0.001 | strategy=StrategySelective | file=results_unsw/unsw_bin_w40__20260329_165145.jld2                                                                                                    
w45 | c3 | accuracy=0.953259 | max_nodes=15 | min_improvement=0.01 | strategy=StrategySelective | file=results_unsw/unsw_bin_w45__20260329_165145.jld2       


16/04/2026

Después de muchas cosas hechas y desechas, ya tenemos el pipeline funcional.

- Terminar los scripts bien hechos
- Corregir la memoria
- Terminar el procedimiento de stream
- Hacer la parte de la deriva