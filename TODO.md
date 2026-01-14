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


14/01/2026

\chapter{Metodología y validación del pipeline}
\label{chap:metodoloxia}

En este capítulo se describe el pipeline de procesamiento y entrenamiento desarrollado, así como el procedimiento seguido para su validación preliminar. El objetivo principal de esta fase es garantizar el correcto funcionamiento del sistema antes de su ejecución a gran escala en un entorno de computación de alto rendimiento.

\section{Pipeline de procesamiento de series temporales}

Se implementó un pipeline base para el procesamiento de series temporales y el entrenamiento de modelos predictivos. Este pipeline constituye la base sobre la que se apoyarán posteriormente los experimentos masivos y la comparación con métodos más avanzados.

El pipeline se estructura en las siguientes etapas:

\subsection{Carga del dataset}

Se empleó el conjunto de datos \textit{ElectricDevices}, perteneciente al repositorio UCR/UEA de series temporales. El sistema de carga se diseñó de forma robusta para soportar automáticamente los formatos \texttt{.ts} y \texttt{.tsv}, detectando el formato disponible y parseando correctamente tanto las series temporales como sus etiquetas.

Este diseño permite reutilizar el pipeline con otros datasets del repositorio sin necesidad de modificaciones adicionales en el código.

\subsection{Normalización}

Las series temporales se normalizaron mediante estandarización \textit{z-score}, calculando la media y la desviación típica por característica. Este paso tiene como objetivo evitar diferencias de escala entre variables y mejorar la estabilidad del entrenamiento de los modelos.

\subsection{Construcción de ventanas temporales}

A partir de las series originales se generaron ventanas temporales deslizantes de tamaño fijo. Este proceso transforma el problema original en un formato supervisado, donde cada ventana se asocia a una salida desplazada en el tiempo según un horizonte de predicción configurable.

El tamaño de ventana y el horizonte de predicción se definen como hiperparámetros del sistema y se utilizan posteriormente en el diseño experimental.

\subsection{División temporal de los datos}

Los datos se dividieron en conjuntos de entrenamiento y test respetando el orden temporal de las muestras. El conjunto de test se corresponde con el tramo final de la serie temporal, evitando así fugas de información temporal y reproduciendo un escenario de predicción realista.

\section{Modelo baseline}

Como modelo de referencia se empleó un modelo lineal entrenado mediante mínimos cuadrados. Este modelo se utiliza con un propósito exclusivamente metodológico, sirviendo como baseline para:

\begin{itemize}
    \item validar el correcto funcionamiento del pipeline,
    \item comprobar la coherencia de los datos generados,
    \item establecer una línea base para comparaciones posteriores.
\end{itemize}

La evaluación del modelo se realiza mediante el error cuadrático medio (MSE), métrica estándar en problemas de predicción de series temporales.

\section{Definición del experimento}

Se definió un experimento como una combinación concreta de los siguientes parámetros:

\begin{itemize}
    \item tamaño de ventana temporal,
    \item horizonte de predicción,
    \item proporción de datos destinada a test.
\end{itemize}

Para cada configuración, el pipeline completo se ejecuta de forma automática, entrenando el modelo y evaluándolo sobre el conjunto de test.

Con el objetivo de garantizar la reproducibilidad y facilitar el análisis posterior, los resultados de cada experimento se almacenan automáticamente en ficheros CSV, incluyendo tanto los valores de los parámetros como la métrica de evaluación obtenida.

\section{Validación preliminar del sistema}

Antes de ejecutar experimentos a gran escala en un entorno de computación distribuida, se llevó a cabo una validación preliminar en local con un conjunto reducido de configuraciones. Esta fase tiene como finalidad:

\begin{itemize}
    \item verificar la estabilidad del pipeline,
    \item comprobar la correcta ejecución de los experimentos,
    \item asegurar el almacenamiento correcto de los resultados,
    \item confirmar que el sistema es reproducible y escalable.
\end{itemize}

Esta validación se limita a comprobar el correcto funcionamiento del sistema y no pretende extraer conclusiones sobre el rendimiento del modelo ni sobre configuraciones óptimas, aspectos que se abordarán en fases posteriores del trabajo.

\section{Preparación para ejecución a gran escala}

Una vez validado el pipeline y el sistema de experimentos en local, el código quedó preparado para su ejecución en un entorno de computación de alto rendimiento. El uso de scripts autónomos, la gestión explícita de dependencias y el almacenamiento automático de resultados permiten escalar el número de experimentos sin modificar la lógica del sistema.

En etapas posteriores, este pipeline se utilizará para realizar barridos completos de hiperparámetros y para comparar el modelo baseline con métodos más avanzados.

