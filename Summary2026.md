# Resumen del trabajo realizado en Intensive-Care-Hidden-Mortality

Fecha de elaboración: 9 de noviembre de 2026.

Este documento reúne el trabajo que consta en los archivos y en el historial de Git del repositorio. El desarrollo registrado comienza el 11 de enero de 2024 y el último commit disponible es `2cf8140`, del 18 de diciembre de 2025. El nombre del documento corresponde al año de elaboración del resumen; no implica que se hayan registrado nuevas implementaciones durante 2026.

La revisión se basa en el código, las celdas y salidas guardadas de los cuadernos, los archivos de resultados y configuración, los artefactos versionados y los mensajes de los commits. No se reentrenaron modelos ni se ejecutaron consultas a servicios de IA para redactarlo. Las métricas citadas son resultados históricos conservados, no mediciones nuevas.

## Continuidad del trabajo e integración en SADUCI

La capacidad predictiva de este proyecto ya tiene una integración en el repositorio público [coslatte/saduci](https://github.com/coslatte/saduci). Allí se utiliza una copia binariamente idéntica de `new_workflow.joblib` dentro del servicio de predicción, con contratos HTTP, persistencia y generación narrativa. Esto identifica un destino concreto para la ejecución integrada del modelo, sin implicar que todos los cuadernos o explicadores se hayan trasladado.

El [informe de integración y traslado](IntegracionSADUCI2026.md) describe qué elementos permanecen como investigación en este repositorio, cuáles se incorporaron o adaptaron en SADUCI y qué diferencias requieren actualización. También distingue ese movimiento funcional de la contribución de ambos informes al repositorio original mediante un fork. La documentación conserva el origen de los experimentos y enlaza su continuación en el sistema integrado.

## 1. Propósito y alcance del proyecto

El proyecto estudia la predicción de mortalidad en pacientes de una unidad de cuidados intensivos mediante aprendizaje automático. Su evolución combina experimentación con datos clínicos, comparación de clasificadores, tratamiento del desequilibrio entre clases y explicación de predicciones individuales. Posteriormente se incorporó una interfaz web para introducir características de un paciente y consultar una probabilidad de no supervivencia al egreso de la UCI.

La expresión «mortalidad oculta» aparece en el contexto y en las instrucciones de los modelos de lenguaje. En el entrenamiento revisado, la variable objetivo es `Fallece`. No se encontró en esos flujos una segunda etiqueta que identifique de forma independiente la condición de mortalidad oculta. Por tanto, el resultado computacional documentado es una predicción de fallecimiento, interpretada dentro de ese contexto de investigación.

El repositorio conserva distintas generaciones de experimentos. Algunos cuadernos representan comparaciones iniciales, otros exploran explicabilidad y otros construyen flujos de preprocesamiento y clasificación reutilizables. La aplicación integra parte de ese trabajo en un prototipo interactivo.

## 2. Evolución histórica

### Enero de 2024: preparación de datos y primeros modelos

El primer commit, `bdba1c0`, incorporó cuadernos de clasificación y regresión. Durante los días siguientes se trabajó en la inspección de datos, la comprobación de valores nulos y la construcción del conjunto de variables que recibirían los algoritmos.

El historial de `My_regression.ipynb`, posteriormente reorganizado, registra la separación de `Fallece` respecto a los predictores, la selección de observaciones individuales y la exploración de la frecuencia de las clases. También se añadieron gráficas de barras para representar esa distribución.

Entre el 15 y el 17 de enero se incorporaron SMOTE, la división entre entrenamiento y prueba, la regresión logística, las predicciones, sus métricas y la validación cruzada. Esa secuencia estableció el núcleo experimental del proyecto.

### Marzo de 2024: particiones y comparación sistemática

En marzo se reorganizaron los cuadernos y se introdujeron conjuntos de entrenamiento, validación y prueba. El commit `632585f`, del 19 de marzo, documenta una proporción de 80:10:10.

Los commits del 29 y 30 de marzo registran varias versiones para un foro. Una de ellas, `cff8265`, describe la eliminación de variables dummy en esa versión y el uso de F1 como criterio de búsqueda de hiperparámetros. Estas decisiones corresponden a una etapa concreta, ya que los flujos posteriores vuelven a incorporar codificación categórica explícita.

### Abril de 2025: explicabilidad, interfaz y nuevo flujo

La actividad de abril de 2025 amplió la comparación sin sobremuestreo y consolidó el trabajo visible en la aplicación. También se retiraron del seguimiento varios archivos de datos y exportaciones de Excel, según `9fa32a3`, del 4 de abril.

El 9 de abril se añadió un contenedor de desarrollo. El 15 de abril se mejoró la selección de diagnósticos mediante descripciones y se modificó el flujo de explicación de la interfaz. Entre el 18 y el 21 de abril se introdujo un nuevo cuaderno de trabajo, se guardó `new_workflow.joblib` y se actualizó la aplicación para utilizarlo.

### Mayo de 2025: inferencia y aumento de datos

El 15 de mayo se incorporó `inference.ipynb` y se amplió la comparación entre técnicas de aumento de datos en `test-workflow.ipynb`. El trabajo pasó a incluir inspección de probabilidades, revisión de errores y evaluación con distintas variantes de sobremuestreo.

### Junio de 2025: regresión logística y explicaciones narrativas

El 18 de junio se añadió un flujo específico de regresión logística y se separaron los cuadernos de MLP y regresión logística. Se guardó además `new_workflow_lr.joblib`.

El 21 de junio se cambió el idioma de la interfaz y se ajustó la presentación del DataFrame. Los días 28 y 29 se añadió un experimento con modelos de lenguaje en `llms/model-explain-text.ipynb`, se llevó la generación de narrativas a la aplicación y se corrigió la configuración de la clave de API. Varias revisiones del mismo período ajustaron dependencias y resolvieron incompatibilidades del entorno.

### Diciembre de 2025: integración de Gemini

El commit `2cf8140`, del 18 de diciembre, actualizó `web_app.py` y `requirements.txt` para la integración con Gemini. La versión revisada solicita narrativas a `gemini-2.5-flash-lite` mediante `google.generativeai`. Esta es una descripción del código conservado, no una comprobación de disponibilidad actual del servicio.

## 3. Datos clínicos y preparación

Los cuadernos cargan principalmente `Datos_regresion.csv`. Las pruebas de campo utilizan `prueba_de_campo.xls` y, en flujos posteriores, `pc.csv`. La configuración de Git excluye archivos CSV, XLS y XLSX, por lo que esos conjuntos no forman parte del material de datos versionado revisado.

Las variables de entrada del flujo usado por la aplicación son las siguientes:

| Variable | Significado en el proyecto |
| --- | --- |
| `Edad` | Edad del paciente. |
| `Diag.Ing1` | Primer diagnóstico de ingreso. |
| `Diag.Ing2` | Segundo diagnóstico de ingreso. |
| `Diag.Egr2` | Segundo diagnóstico de egreso. |
| `APACHE` | Valor presentado en la interfaz como APACHE II. |
| `TiempoVAM` | Tiempo de ventilación artificial mecánica. |

En los análisis iniciales también se utiliza `Diag.Egr1`. Se compararon variantes con todos los rasgos, sin el primer diagnóstico de egreso y sin ambos diagnósticos de egreso. La aplicación actual conserva `Diag.Egr2` y excluye `Diag.Egr1`.

El análisis exploratorio incluye estadísticas descriptivas, tipos de datos, frecuencias por diagnóstico y tablas que cruzan diagnósticos con `Fallece`. En `Classification.ipynb` se inspeccionan valores de `TiempoVAM` superiores a 120 y se elimina la fila con índice 207. Esa modificación ayuda a explicar que algunas salidas muestren 207 observaciones y otras 208.

Una salida de clasificación registra 176 observaciones de clase 0 y 31 de clase 1 antes del remuestreo. En `inference.ipynb` aparecen 176 de clase 0 y 32 de clase 1. Estas cifras pertenecen a estados distintos del conjunto y no deben combinarse como si procedieran de una única ejecución.

Para la prueba de campo se eliminan columnas como posición, sexo, piel, fechas de ingreso y egreso, traslado, procedencia y modos de ventilación. En los flujos nuevos también se divide `TiempoVAM` entre 24 al preparar `pc.csv`. Esa conversión debe conservarse al interpretar las unidades del modelo.

## 4. Tratamiento del desequilibrio de clases

El desequilibrio entre supervivencia y fallecimiento motivó una parte central de los experimentos. Se construyeron variantes sin sobremuestreo y variantes con Random Over Sampling y SMOTE. En las etapas posteriores se añadieron ADASYN y BorderlineSMOTE.

Random Over Sampling replica observaciones de la clase minoritaria. SMOTE y sus variantes generan observaciones sintéticas. Los cuadernos comparan cómo esos cambios afectan a la detección de fallecimientos y a los errores sobre supervivientes.

La organización conserva estas líneas experimentales en archivos separados:

| Cuaderno | Trabajo principal |
| --- | --- |
| `Classification_RandomOverSampling.ipynb` | Experimentos con ROS y comparaciones adicionales de preprocesamiento y remuestreo. |
| `Classification_SMOTE.ipynb` | Experimentos con generación sintética mediante SMOTE. |
| `Classification_without_oversampling.ipynb` | Comparación sin sobremuestreo con búsqueda de hiperparámetros. |
| `Classification_without_oversampling_without_gridsearch.ipynb` | Referencia sin sobremuestreo ni búsqueda exhaustiva. |

En algunos flujos iniciales el sobremuestreo se aplica al conjunto completo antes de dividirlo. Eso puede introducir dependencia entre entrenamiento y evaluación, por ejemplo si copias de una observación terminan en particiones diferentes. Los flujos posteriores separan primero entrenamiento y prueba y generan los conjuntos aumentados a partir del entrenamiento.

También se conservan experimentos que aplican métodos sintéticos sobre columnas de diagnósticos codificadas numéricamente antes de la codificación categórica. La validez clínica de valores interpolados en esas columnas necesita una revisión específica. El resumen registra el experimento sin asumir que todas sus observaciones sintéticas sean categorías diagnósticas válidas.

## 5. Modelos y búsqueda de hiperparámetros

Las principales familias comparadas son K vecinos más próximos, regresión logística y perceptrón multicapa.

Para KNN se exploraron número de vecinos, ponderación uniforme o por distancia, algoritmo de búsqueda, tamaño de hoja y distancia de Minkowski. Para regresión logística se variaron `C`, penalización y solucionador. Para MLP se ensayaron tamaños de capas, activaciones, optimizadores, regularización y número máximo de iteraciones.

Los cuadernos iniciales incorporan `GridSearchCV` con cinco particiones y F1 como criterio de selección. `Classification.ipynb` incluye además validación cruzada estratificada de diez particiones y métricas tanto de entrenamiento como de prueba.

Las métricas utilizadas incluyen sensibilidad o recall, F1, precisión, matrices de confusión, exactitud balanceada, kappa de Cohen y valores rotulados como AUC-ROC. La lectura conjunta de esas métricas permite observar problemas que la exactitud global puede ocultar en un conjunto desequilibrado.

Los archivos `configuracion_ROS.txt` y `configuracion_SMOTE.txt` guardan hiperparámetros para distintas selecciones de rasgos. Por ejemplo, el primer bloque de ROS registra un MLP de una capa de 50 neuronas con `tanh`, mientras que el primer bloque de SMOTE registra una capa de 100 neuronas con `relu`. Son configuraciones históricas y no describen necesariamente el modelo que carga hoy la interfaz.

## 6. Resultados históricos de ROS y SMOTE

Los archivos de texto contienen varios bloques, algunos repetidos. La tabla siguiente reproduce exclusivamente el primer bloque titulado «Con todos los rasgos» de cada archivo. Los vectores se presentan en el orden conservado en la fuente.

| Método | Modelo | Recall por clase | F1 por clase | Valor rotulado AUC-ROC |
| --- | --- | --- | --- | --- |
| ROS | KNN | 1,00 / 0,85 | 0,9091 / 0,9189 | 0,9250 |
| ROS | Regresión logística | 0,80 / 0,70 | 0,7273 / 0,7568 | 0,7500 |
| ROS | MLP | 1,00 / 0,90 | 0,9375 / 0,9474 | 0,9500 |
| SMOTE | KNN | 0,8667 / 0,75 | 0,7879 / 0,8108 | 0,8083 |
| SMOTE | Regresión logística | 1,00 / 0,75 | 0,8571 / 0,8571 | 0,8750 |
| SMOTE | MLP | 0,80 / 0,90 | 0,8276 / 0,8780 | 0,8500 |

Las variantes que eliminan diagnósticos modifican esos resultados. Por ejemplo, el bloque de SMOTE sin `Diag.Egr1` registra para MLP recall de 0,9333 y 0,90, F1 de 0,9032 y 0,9231, y un valor rotulado AUC-ROC de 0,9167.

Estos archivos documentan ejecuciones y permiten reconstruir las comparaciones realizadas. Sus repeticiones y la falta de identificadores únicos por ejecución impiden tratarlos como un registro experimental normalizado o atribuir cada bloque con certeza a un artefacto serializado.

Además, distintas celdas calculan `roc_auc_score` usando `y_pred`, es decir, etiquetas discretas. Esos valores no equivalen a evaluar la capacidad de ordenación de las probabilidades mediante una curva ROC completa. Por esa razón se conserva la denominación original con una aclaración y no se presenta el mayor valor como prueba de superioridad clínica definitiva.

## 7. Nuevos flujos de preprocesamiento y entrenamiento

`test-workflow.ipynb`, `test-mlp-workflow.ipynb` y `test-lr-workflow.ipynb` organizan una generación posterior de experimentos. En ellos se separan los diagnósticos como variables categóricas y edad, APACHE y tiempo de ventilación como variables numéricas.

El preprocesamiento utiliza `ColumnTransformer`, `OneHotEncoder` con `handle_unknown="infrequent_if_exist"` y `RobustScaler`. De esta manera, el flujo conserva juntos la transformación de columnas y el clasificador mediante un pipeline.

La partición principal reserva un 10 % para prueba, con estratificación por `Fallece` y semilla 42. El MLP de esa comparación tiene capas ocultas de 200, 150, 100 y 50 neuronas, activación ReLU, optimizador Adam, parada temprana y hasta 1000 iteraciones.

Los cuadernos generan conjuntos de entrenamiento con ADASYN, SMOTE, BorderlineSMOTE y ROS y reajustan el pipeline con cada variante. La regresión logística utiliza una estructura semejante para facilitar la comparación. En esa variante, el paso del clasificador mantiene el nombre interno `mlp`, aunque su objeto sea `LogisticRegression`.

`test-workflow.ipynb` contiene además una búsqueda posterior con tres particiones y un pipeline que integra SMOTE después del preprocesamiento. Esa sección explora un MLP de dos capas de 200 y 150 neuronas. Convive con la comparación anterior y no debe confundirse automáticamente con el artefacto guardado antes en el cuaderno.

Los flujos se serializan como `new_workflow.joblib` y `new_workflow_lr.joblib`. La aplicación carga el primero. La correspondencia exacta entre sus parámetros internos y una ejecución concreta no se comprobó deserializando los archivos durante esta revisión.

## 8. Inferencia y pruebas de campo

`inference.ipynb` carga el flujo guardado, calcula clases y probabilidades y añade al conjunto columnas de predicción. También identifica observaciones donde la predicción difiere de `Fallece`.

La salida guardada muestra una matriz de confusión `[[158, 18], [6, 26]]` para 208 observaciones. El informe registra exactitud aproximada de 0,88, recall de 0,90 para la clase 0 y de 0,81 para la clase 1, con F1 de 0,93 y 0,68. El valor mostrado como AUC-ROC es 0,86 y se calcula sobre clases predichas. Como el cuaderno aplica el modelo al conjunto cargado, estas cifras no establecen por sí solas rendimiento en pacientes independientes del entrenamiento.

`Prueba de campo.ipynb` incorpora una evaluación sobre un archivo Excel. Su salida conserva exactitud balanceada aproximada de 0,5982 y kappa de Cohen de 0,0862. El recall por clase aparece como 0,3333 y 0,8630, con F1 de 0,1429 y 0,9130. La codificación de clases debe revisarse antes de compararla directamente con los flujos posteriores.

En los flujos nuevos, las salidas guardadas de prueba interna utilizan 21 observaciones, de las cuales tres pertenecen a la clase 1. El bloque inicial tanto de MLP como de regresión logística muestra recall de 1,00 para esa clase y precisión de 0,38. Esto representa tres casos detectados, con varios falsos positivos, y no una estimación precisa de sensibilidad poblacional.

Las pruebas de campo de esos cuadernos muestran 76 observaciones, 73 de clase 0 y tres de clase 1:

| Flujo | Exactitud | Precisión de clase 1 | Recall de clase 1 | F1 de clase 1 | Valor rotulado AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| MLP | 0,82 | 0,18 | 1,00 | 0,30 | 0,90 |
| Regresión logística | 0,68 | 0,08 | 0,67 | 0,14 | 0,68 |

Son salidas conservadas, no resultados reproducidos para este documento. La proporción de fallecimientos y el reducido número de casos positivos limitan las conclusiones. La precisión baja también muestra que detectar esos casos puede acompañarse de muchas alertas sobre pacientes supervivientes.

## 9. Explicabilidad de las predicciones

`Classification.ipynb` reúne experimentos con LIME, SHAP, Integrated Gradients y Saliency. Se utilizaron tanto las implementaciones de `interpret.blackbox` como bibliotecas específicas de LIME y SHAP. Para los métodos de gradiente se empleó Captum sobre PyTorch.

Una parte del trabajo copia coeficientes e interceptos de un MLP de scikit-learn a una red de PyTorch. La arquitectura declarada en esa celda tiene seis entradas, dos capas de 50 neuronas con activación `tanh` y una salida sigmoide. La transferencia permite aplicar técnicas basadas en gradientes a una representación del clasificador.

El repositorio conserva explicadores serializados en `Explainers/`, incluidos archivos para Integrated Gradients, Saliency, LIME y SHAP. Se generaron visualizaciones locales y se exploró la relevancia positiva o negativa de las variables para observaciones concretas.

También se implementó una evaluación por perturbaciones. `modify_instance` modifica características seleccionadas por su relevancia, y `calculate_fidelity` calcula la proporción de repeticiones en que se mantiene la clase predicha. Ese procedimiento registra estabilidad bajo las perturbaciones definidas en el cuaderno. No demuestra por sí solo causalidad clínica ni una validación universal de fidelidad de la explicación.

La aplicación actual utiliza `Explainers/ig_explainer.pkl`, mientras que la predicción procede de `new_workflow.joblib`. La revisión no verificó que ese explicador represente exactamente el mismo modelo y preprocesamiento del flujo actualizado. La relación entre ambos artefactos es un punto pendiente para interpretar sus atribuciones con confianza.

## 10. Aplicación web con Streamlit

`web_app.py` implementa una interfaz para consultar la probabilidad de no supervivencia al egreso de UCI. La barra lateral recoge edad, dos diagnósticos de ingreso, segundo diagnóstico de egreso, APACHE II y tiempo de ventilación.

La aplicación incorpora un catálogo de 49 códigos, del 0 al 48. El código 0 corresponde a una opción vacía, mientras que los restantes incluyen diagnósticos y condiciones como sepsis, shock, traumatismo craneoencefálico, insuficiencia renal y otros. Los diccionarios permiten mostrar descripciones y enviar al modelo los códigos esperados.

El recorrido implementado es el siguiente:

1. Introducir las características del paciente.
2. Ver una tabla de las entradas con etiquetas en inglés.
3. Pulsar `Predict` para obtener `predict_proba(input_df)[:, 1]`.
4. Ver la probabilidad en formato porcentual, bajo un encabezado que menciona un corte del 50 %.
5. Pulsar `Explain` para calcular atribuciones y solicitar una narrativa.

La probabilidad se conserva en `st.session_state`. Si se intenta explicar antes de predecir, la interfaz pide realizar primero una predicción. La gráfica de relevancia ordena las variables por valor absoluto, con naranja para contribuciones positivas y azul para negativas.

El flujo depende de que los archivos serializados estén disponibles y sean compatibles con las bibliotecas instaladas. El resumen no incluye una prueba de arranque de Streamlit ni una comprobación interactiva de su funcionamiento.

## 11. Narrativas mediante modelos de lenguaje

`llms/model-explain-text.ipynb` explora la transformación de explicaciones numéricas en texto. Carga el pipeline, prepara explicaciones SHAP y construye un prompt con nombres de variables, valores observados y contribuciones. El historial registra su incorporación antes de trasladar esa capacidad a la interfaz.

En la versión actual de `web_app.py`, el texto se construye a partir de atribuciones de Integrated Gradients. Los códigos diagnósticos se convierten en descripciones antes de enviarse al generador. El prompt solicita una síntesis clínica de los factores más influyentes en el contexto de mortalidad oculta en UCI.

La clave se obtiene de `.env` mediante `GEMINI_API_KEY`. El código configura Gemini, utiliza el identificador `gemini-2.5-flash-lite` y establece temperatura 0,7. La respuesta debe comenzar por `Narrative:`; la interfaz elimina ese prefijo y muestra el texto en un cuadro informativo.

Esta integración añade una capa de presentación textual de las atribuciones. El cálculo de la probabilidad sigue correspondiendo al pipeline local. El código transmite al proveedor los valores clínicos y las atribuciones incluidos en el prompt; esta revisión no realizó esa transmisión ni evaluó la corrección clínica de una narrativa generada.

## 12. Artefactos y organización del repositorio

Los resultados del trabajo se distribuyen entre cuadernos, modelos, explicadores y archivos de apoyo:

| Archivo o grupo | Aporte conservado |
| --- | --- |
| `Classification.ipynb` | Exploración, clasificación, explicabilidad, transferencia de pesos y perturbaciones. |
| Cuadernos `Classification_*` | Variantes de sobremuestreo y selección de modelos. |
| `All.ipynb` | Cuaderno auxiliar con importaciones de herramientas de explicabilidad. |
| `Prueba de campo.ipynb` | Evaluación con datos de un archivo de campo en Excel. |
| `inference.ipynb` | Predicciones, probabilidades y revisión de errores. |
| Cuadernos `test-*workflow.ipynb` | Preprocesamiento categórico y numérico, comparación de aumentos y serialización. |
| `llms/model-explain-text.ipynb` | Experimento de narrativas a partir de explicaciones. |
| `Modelos/*.pkl` | Clasificadores históricos de KNN, regresión logística y MLP. |
| `Explainers/*.pkl` | Explicadores guardados con distintas bibliotecas. |
| `modelo_completo.pth`, `modelo_mlp_ROS_pt.pth` | Artefactos de PyTorch. |
| `new_workflow.joblib`, `new_workflow_lr.joblib` | Flujos serializados de generaciones posteriores. |
| `Resultados_*.txt`, `configuracion_*.txt` | Métricas e hiperparámetros de experimentos. |
| `web_app.py` | Interfaz Streamlit e integración narrativa con Gemini. |
| `requirements.txt` | Dependencias del entorno. |
| `.devcontainer/devcontainer.json` | Configuración del entorno de desarrollo y arranque de Streamlit. |

También hay archivos de configuración del IDE, checkpoints de Jupyter y restos en `.tmp.driveupload/`. Esos elementos forman parte del material versionado, pero no representan nuevas capacidades del modelo.

## 13. Entorno de desarrollo y dependencias

El entorno reúne herramientas de análisis de datos, aprendizaje automático, visualización, explicabilidad y generación de texto. Entre las dependencias declaradas figuran pandas, NumPy, scikit-learn, imbalanced-learn, PyTorch, Captum, LIME, SHAP, InterpretML, Matplotlib, Streamlit, joblib, cloudpickle y clientes de servicios de lenguaje.

El historial muestra varias ampliaciones y reducciones de `requirements.txt`, además de ajustes para resolver conflictos y retirar dependencias específicas de Windows. El archivo revisado contiene numerosas versiones fijadas y se guarda en UTF-16. Los archivos de configuración de experimentos utilizan otra codificación, por lo que no todo el material textual se lee directamente como UTF-8.

El contenedor de desarrollo utiliza una imagen de Python 3.11 sobre Debian Bullseye. Instala dependencias, prepara Streamlit y configura la ejecución de `web_app.py`, con reenvío del puerto 8501. Esa configuración documenta un camino de ejecución para desarrollo; no demuestra un despliegue público operativo.

La exclusión de `.env` evita que nuevas claves locales entren en Git por la vía habitual. También se excluyen datos tabulares y un directorio de entorno virtual. La reproducción completa requiere disponer de los conjuntos externos que los cuadernos referencian.

## 14. Qué queda establecido y qué requiere comprobación

El trabajo conservado establece una cadena experimental amplia: inspección de datos clínicos, comparación de tres familias de clasificadores, distintas formas de sobremuestreo, selección de hiperparámetros, evaluación interna y de campo, explicaciones locales y una interfaz con generación narrativa.

La evidencia revisada deja varias comprobaciones abiertas:

- Reproducir los experimentos con los datos originales y un registro inequívoco de las particiones.
- Separar rigurosamente ajuste, sobremuestreo y evaluación, especialmente al interpretar resultados antiguos.
- Verificar el significado y la consistencia de la clase positiva entre versiones de los conjuntos.
- Calcular AUC sobre probabilidades cuando se quiera medir discriminación y distinguirla de la evaluación sobre etiquetas.
- Confirmar las unidades de `TiempoVAM` y la disponibilidad temporal de los diagnósticos de egreso para el uso previsto.
- Verificar que el explicador cargado por la aplicación corresponde al pipeline que produce la probabilidad.
- Asociar cada modelo serializado con sus datos, parámetros, versiones y métricas de origen.
- Evaluar con más casos positivos la sensibilidad, la precisión y el volumen de falsas alertas.
- Comprobar calibración de probabilidades y utilidad del umbral si se pretende utilizar el resultado en decisiones clínicas.

Estos puntos son límites y posibles pasos de continuación, no tareas completadas durante la redacción. La presencia de un prototipo y de métricas guardadas no permite afirmar que exista validación clínica prospectiva, integración hospitalaria o uso asistencial demostrado.

## 15. Balance del trabajo realizado

El repositorio muestra el paso de pruebas iniciales de clasificación a un prototipo que recibe datos clínicos, produce una probabilidad y ofrece una explicación visual acompañada de texto. El desarrollo conserva alternativas de modelado y varios intentos de mejorar el tratamiento de los diagnósticos, las escalas numéricas y el desequilibrio de clases.

Su aportación documentada incluye los experimentos de KNN, regresión logística y MLP, la comparación de ROS y técnicas sintéticas, las pruebas de campo, el uso de cuatro familias de explicadores y la incorporación de un modelo de lenguaje para redactar narrativas. Los archivos permiten estudiar tanto los avances como las diferencias entre generaciones del flujo.

El estado descrito corresponde a un proyecto de investigación y prototipado con resultados históricos conservados. La siguiente etapa de evaluación deberá vincular esos resultados a ejecuciones reproducibles y comprobar que predicción, preprocesamiento y explicación describen el mismo sistema.

## 16. Fuentes internas de este resumen

La reconstrucción histórica procede del historial Git disponible, entre `bdba1c0` y `2cf8140`. La descripción funcional se apoya en `web_app.py`, los cuadernos enumerados en este documento, los archivos de resultados y configuración, `requirements.txt`, `.gitignore` y `.devcontainer/devcontainer.json`.

Se comprobó la existencia de los modelos y explicadores versionados, sin ejecutar su deserialización. Los cuadernos se inspeccionaron como documentos JSON, incluyendo código y salidas guardadas, sin ejecutar sus celdas. No se consultaron fuentes externas ni se incorporaron datos identificativos de pacientes al resumen.
