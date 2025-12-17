
- G. I. Webb et al. – “Supervised Learning from Data Streams: An Overview and Update” (ACM Computing Surveys, 2024)Supervised Learning from Data Streams: An Overview and Update : 

    Se mencionan datasets populares que se han utilizado en la investigación sobre el aprendizaje de flujos de datos, pero se hacen algunas aclaraciones importantes sobre su naturaleza:

        Forest Covertype: dataset clásico utilizado en investigaciones de minería de datos. Sin embargo, el artículo señala que este dataset **no está diseñado como un flujo de datos**. Dataset estático que se ha **transformado artificialmente** en un flujo de datos para ciertos experimentos.

        Poker Hand: dataset que no es un flujo de datos real, convertido en un flujo artificialmente. Usado para probar algoritmos de clasificación, y fuera del contexto de flujos de datos.

        Electricity Dataset: utilizado como un benchmark en flujos de datos, particularmente para el análisis de dependencia temporal en los datos. Aunque se menciona en el artículo como un dataset de referencia, también se critica por no ser un flujo de datos real, ya que no está diseñado específicamente para estudiar el concepto de "drift" en flujos de datos.

        "Many of these studies propose methods designed for or tested in sandbox settings that are sometimes only loosely connected to everyday real-world challenges. These papers commonly cite the ubiquity of streams in the real world, yet typically only test their methods on synthetic or static datasets which are at best small snapshots of past streams or are artificially turned into streams by manual sorting. Many of the popular benchmark sets are decades old and have relatively small dimensions"


- A. Halstead et al. – “Recurrent Concept Drifts on Data Streams” (IJCAI, 2024)


    Repositorios de Código y Software:

        MOA (Massive Online Analysis): El artículo menciona que muchos de los métodos discutidos tienen implementaciones disponibles en MOA, como LEARN++.NSE, PCCF y ProChange. Las implementaciones de CPF, ECPF, MDP y PEARL también están disponibles como repositorios separados en MOA.

        scikit-multiflow: Repositorios basados en scikit-multiflow, como scikit-ika, que contiene implementaciones de PEARL y Nacre.

        River: El artículo menciona FALL (una plataforma modular de aprendizaje basada en River) que permite el uso de clasificadores y detectores de drift de River. También implementa los mecanismos de selección y evaluación de contexto C-F1 utilizados en métodos como FiCSUM.

        Repositores en GitHub: Se mencionan forks en GitHub de frameworks populares de aprendizaje automático para flujos de datos como MOA, scikit-multiflow y River, con implementación de los métodos revisados en el artículo

    Datasets:

        Datasets sintéticos: Mencionan generadores de datos sintéticos como SEA, Hyperplane, Agrawal, Random Tree, y LED, los cuales están disponibles en MOA (Massive Online Analysis).

        Datasets reales:

            Electricity: Un dataset utilizado en la literatura de SL para flujos de datos con drifts recurrentes.

            Sensor: Otro dataset real que también se utiliza en investigaciones de flujos de datos con drifts recurrentes. Aunque se utiliza ampliamente, su recurrencia exacta no está completamente definida.

            Aedes-Culex y Aedes-Sex: Datos de comportamiento de insectos influenciados por condiciones ambientales (por ejemplo, temperatura) y su actividad recurrente.

        SODA10M: En el contexto de aprendizaje continuo, SODA10M es un dataset de imágenes sin etiquetar con 10M de imágenes y 20k imágenes etiquetadas, utilizado en benchmark para la conducción autónoma. Incluye datos de cámaras de vehículos que circulan por cuatro ciudades chinas, con anotaciones de 6 clases de objetos

        "Some of the commonly used synthetic data generators such as SEA, Hyperplane, Agrawal, Random Tree, LED and different concept drift simulators are available in MOA. Although real-world datasets Electricity and Sensor are widely used in SL literature for data streams with recurrent concept drifts, their exact concept recurrences are unclear."

    Fórmulas:

        El artículo aborda métodos estadísticos utilizados para detectar drifts recurrentes, y cómo se manejan los conceptos recurrentes:

        Jensen-Shannon Divergence: Se utiliza en métodos como ESCR para medir la divergencia entre distribuciones de puntuaciones de confianza de clasificadores. Esto ayuda a detectar drifts recurrentes.

        Cumulative Accuracy Gain (CAG): Se menciona como una métrica para evaluar métodos en flujos de datos con drifts recurrentes. La fórmula para CAG es:

                      \[
                        CAG = \sum \left( \text{accuracy}(A) - \text{accuracy}(B) \right)
                      \]

        donde accuracy(A) y accuracy(B) son las precisiones de los clasificadores en comparación con una línea base.

        
        Además, se emplean técnicas de agrupación no supervisada (como el algoritmo k-Means) para crear clusters de conceptos y detectar drifts recurrentes



- S. Hinder et al. – “Data streams classification using deep learning under different concept drifts” (Journal of Logic and Computation, 2023)

    Datasets:

        UCR Repository: Se utilizan 29 datasets de time series provenientes del UCR repository, los cuales se simulan como flujos de datos. Estos datasets cubren diferentes dominios y características, y se usan para evaluar el rendimiento de los modelos de aprendizaje profundo. Ejemplos de estos datasets incluyen TwoPatterns, CinCECGtorso, Pendigits, y ElectricDevices. Se mencionan con detalles como el número de instancias, longitud de la serie temporal y el número de clases

        Concept Drift Datasets: Se generan 15 datasets sintéticos diseñados para simular distintos tipos de concept drift. Estos incluyen cambios abruptos, graduales e incrementales en las distribuciones de datos, utilizando generadores como RBF, LED, RandomTree, Agrawal, y SEA. Estos se implementan usando la libería River

    Repositorios:

        ADLStream Framework: El código del ADLStream Python library (framework utilizado en el estudio para la clasificación de flujos de datos) está disponible en este repositorio de GitHub: https://github.com/pedrolarben/ADLStream

        Deep Learning Online Classification: La implementación de los experimentos de clasificación de datos en tiempo real, utilizando el framework de ADLStream, también está disponible en este otro repositorio: https://github.com/pedrolarben/deep-learning-online-classification

    Fórmulas:

        El artículo utiliza la precisión prequential y el Kappa Statistic como métricas para evaluar el rendimiento de los modelos en flujos de datos. Las fórmulas para estas métricas son las siguientes:

            Prequential Accuracy:

            \[
            P_{\alpha}(i) = \frac{\sum_{k=1}^{i} \alpha^{i-k} L(y_k, o_k)}{\sum_{k=1}^{i} \alpha^{i-k}} = \frac{S_{\alpha}(i)}{B_{\alpha}(i)}
            \]
            Donde \( L(y_k, o_k) \) es la pérdida y \( \alpha \) es un factor de decaimiento para dar más peso a los datos más recientes.

            Kappa Statistic:

            \[
            \kappa = \frac{p_0 - p_c}{1 - p_c}
            \]
            Donde \( p_0 \) es la precisión prequential y \( p_c \) es la probabilidad hipotética de un acuerdo aleatorio.

    

- M. Mena-Torres et al. – “A survey on machine learning for recurring concept drifting data streams” (2023)

    No se mencionan datasets, fórmulas o código específico. Se incluye una descripción de métodos de Drift Detection:

        ADWIN y ADWIN2: Se utilizan para detectar concept drift en el flujo de datos, basándose en la diferencia de medias entre dos ventanas de datos.

        DDM (Drift Detection Method): Detecta el drift observando el tasa de error en el clasificador.

        RDDM: Un método reactivo para detectar drift, ajustando el umbral cuando se detectan cambios.

        HDDM: Utiliza desigualdad de Hoeffding para detectar cambios abruptos y graduales en los flujos de datos.

- N. Lu et al. – “One or two things we know about concept drift—a survey on unsupervised drift detection” (Frontiers in Artificial Intelligence, 2024)

    Datasets:

        El artículo menciona varios datasets sintéticos utilizados para evaluar los métodos de detección de concept drift. Estos datasets están diseñados para estudiar diferentes tipos de drift (abrupto, gradual, etc.):

        Uniform Dataset: Este dataset se genera de manera uniforme dentro de un cuadrado unitario, con el drift introducido por un desplazamiento a lo largo de la diagonal.

        Gaussian Dataset: Los datos son generados a partir de una distribución normal con características correlacionadas, y el drift cambia la señal de correlación entre las características.

        Two Overlapping Dataset: Los datos provienen de dos cuadrados uniformes superpuestos, y el drift se genera mediante una rotación de 90 grados.


    Repositorios:

        El artículo menciona repositorios de código en línea donde se pueden encontrar implementaciones de los métodos y experimentos descritos:

            El código experimental de este estudio está disponible en GitHub en el repositorio: https://github.com/FabianHinder/One-or-Two-Things-We-Know-about-Concept-Drift

        
    
    Fŕomulas:

        Kolmogorov-Smirnov Test:

            Se utiliza para medir la discrepancia máxima entre dos muestras. La fórmula para la estadística del test es:

            \hat{d}(S^-(t), S^+(t)) = \sup_x \left| F_{S^+(t)}(x) - F_{S^-(t)}(x) \right|


            donde F es la función de distribución empírica acumulada (CDF) de cada muestra.


        MMD (Maximum Mean Discrepancy):

            Utiliza kernels para medir la discrepancia entre dos distribuciones de muestras P y Q, y se expresa como:

            \text{MMD}(P, Q) = \max_{f \in \mathcal{H}} \left| \mathbb{E}_{X \sim P}[f(X)] - \mathbb{E}_{X \sim Q}[f(X)] \right|

            donde H es el espacio de funciones de un kernel elegido, y f es una función de ese espacio.




