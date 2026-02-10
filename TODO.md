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


05/02/2025

Tengo que :

- comprobar que todo funciona correctamente y está listo para mandar al CESGA (done)

- mandar al cesga los experimentos ejecutando con un solo dataset (done)

Preguntar qué debo tener exactamente en el script del CESGA, es decir, qué parámetros

- yo añadiría lo de la notificación de end/fail del job en el CESGA