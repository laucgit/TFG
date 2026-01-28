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

