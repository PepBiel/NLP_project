# Fine-tuning de mT5 para generar resúmenes de noticias deportivas en español

## Resumen ejecutivo

Este proyecto desarrolla un **sistema de resumen automático de noticias deportivas en español** mediante el ajuste fino de un modelo **Transformer encoder-decoder**.  
El objetivo es convertir artículos largos en resúmenes breves, legibles y útiles para contextos como:

- medios digitales y redacciones,
- monitorización de prensa,
- curación de contenidos,
- alertas informativas,
- asistentes internos para equipos de comunicación o análisis.

La solución se ha construido sobre **mT5-small** y el dataset **MLSUM en español**, filtrado por temática de **deportes**.  
El resultado es una **prueba de concepto funcional** capaz de captar la idea principal de una noticia y generar resúmenes con estilo periodístico, aunque todavía presenta limitaciones típicas de los modelos generativos, como **repeticiones** y **alucinaciones fácticas**.

---

## Problema que resuelve

Los equipos de contenido y análisis trabajan cada día con grandes volúmenes de texto: crónicas, previas, notas de prensa y noticias deportivas. Leer, sintetizar y redistribuir esa información manualmente consume tiempo y escala mal.

Este proyecto aborda ese problema con un pipeline de NLP que:

1. recibe una noticia completa en español,
2. procesa el texto con un modelo seq2seq,
3. genera un resumen abstractivo de forma automática.

A diferencia de un sistema extractivo, aquí el modelo **reescribe** la información en lugar de limitarse a copiar frases del texto original.

---

## Qué hace el sistema

- Resume noticias deportivas en español.
- Utiliza un modelo **multilingüe** preparado para tareas de tipo texto-a-texto.
- Está especializado en el dominio de **deportes** gracias al filtrado temático del dataset.
- Evalúa el rendimiento con métricas estándar de summarization: **ROUGE-1, ROUGE-2, ROUGE-L y ROUGE-Lsum**.
- Permite analizar de forma cualitativa ejemplos reales de entrada, referencia humana y resumen generado.

---

## Enfoque técnico

### Modelo
- **Modelo base:** `google/mt5-small`
- **Tipo de arquitectura:** Encoder-Decoder (Seq2Seq)
- **Motivo de elección:** T5 es una arquitectura adecuada para generación condicional, y **mT5** permite trabajar correctamente con texto en español.

### Dataset
- **Fuente:** `MLSUM` (subconjunto en español)
- **Tarea:** resumen abstractivo
- **Filtrado temático:** solo noticias de **deportes**

### Tamaño de datos
- Dataset completo: **290.645** ejemplos
- Dataset filtrado por deportes: **20.239** ejemplos
- Subconjunto utilizado para entrenar por limitaciones de cómputo:
  - **Train:** 1.000
  - **Validation:** 200
  - **Test:** 200

### Preprocesamiento
- Se añade el prefijo `summarize:` a cada entrada, siguiendo el formato esperado por T5.
- Longitud máxima de entrada: **1024 tokens**
- Longitud máxima de salida: **128 tokens**
- Tokenización con `AutoTokenizer`
- Padding configurado a la izquierda para facilitar la generación por lotes

---

## Pipeline del proyecto

```text
Noticia en español
   ↓
Filtrado / carga de datos MLSUM
   ↓
Tokenización + prefijo "summarize:"
   ↓
Fine-tuning de mT5-small
   ↓
Evaluación con ROUGE
   ↓
Generación de resúmenes sobre noticias nuevas
```

---

## Stack tecnológico

- **Python**
- **Jupyter Notebook / Google Colab**
- **PyTorch**
- **Hugging Face Transformers**
- **Hugging Face Datasets**
- **Evaluate**
- **rouge_score**
- **Accelerate**

Entorno de ejecución utilizado en el experimento:
- **Google Colab**
- **GPU Tesla T4**

---

## Configuración de entrenamiento

Se ha usado una configuración conservadora para adaptar el modelo sin disparar el consumo de memoria:

- **Optimizador:** `Adafactor`
- **Learning rate:** `5e-4`
- **Batch size por dispositivo:** `4`
- **Gradient accumulation:** `8`
- **Batch size efectivo:** `32`
- **Warmup steps:** `500`
- **Weight decay:** `0.01`
- **Epochs:** `3`
- **Evaluación:** al final de cada época
- **Selección automática del mejor modelo:** sí, usando `rouge1`

---

## Resultados principales

### Métricas de validación por época

| Época | Validation Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|------:|----------------:|--------:|--------:|--------:|-----------:|
| 1 | 3.2696 | 0.2244 | 0.0521 | 0.1823 | 0.1822 |
| 2 | 3.1215 | **0.2282** | **0.0571** | **0.1832** | **0.1832** |
| 3 | 3.0213 | 0.2201 | 0.0566 | 0.1783 | 0.1789 |

### Lectura de resultados
- El modelo **mejora entre la época 1 y la 2**.
- En la **época 3**, la pérdida sigue bajando, pero **ROUGE empeora**.
- Esto sugiere un inicio de **overfitting**.
- El mejor checkpoint es el de la **época 2**, que se recupera automáticamente con `load_best_model_at_end=True`.

---

## Qué demuestra este proyecto

### Fortalezas
- Capacidad de **captar la idea central** de noticias largas.
- Generación de resúmenes con un **tono cercano al estilo periodístico deportivo**.
- Diseño técnico coherente y reproducible para una tarea real de NLP.
- Uso de herramientas estándar de la industria en IA aplicada al lenguaje.

### Limitaciones actuales
- **Alucinaciones fácticas:** en algunos casos inventa o mezcla nombres, resultados o detalles.
- **Repetición de expresiones:** especialmente en inferencia libre.
- **Subconjunto reducido de entrenamiento:** solo 1.000 ejemplos, lo que limita la generalización.
- La solución actual está validada como **prototipo técnico**, no como sistema listo para producción.

---

## Cómo ejecutar el proyecto

### Requisitos
Instala las dependencias principales:

```bash
pip install transformers accelerate evaluate rouge_score
pip install datasets==2.19.0
```

### Ejecución
1. Abre el notebook del proyecto.
2. Ejecuta las celdas de instalación.
3. Carga el dataset MLSUM en español.
4. Aplica el filtrado por tema `deportes`.
5. Lanza el proceso de tokenización y entrenamiento.
6. Evalúa el modelo y genera ejemplos de resumen.

### Archivo principal
- `Fornes_Reynes_JosepGabriel.ipynb`

### Modelo guardado
Durante el notebook se guarda el modelo ajustado como:

```text
modelo_resumen_t5_final
```

---

## Estructura del trabajo

```text
.
├── Fornes_Reynes_JosepGabriel.ipynb
├── Fornes_Reynes_JosepGabriel.pdf
├── proyecto_práctico_DL4NLP_24-25.pdf
├── Fornes_Reynes_JosepGabriel_LIMPIO.ipynb
└── README.md
```

---

## Ejemplo de funcionamiento

**Entrada:** una noticia deportiva completa en español.  
**Salida:** un resumen breve generado por el modelo, intentando condensar el hecho principal, los protagonistas y el resultado relevante del artículo.

Este enfoque permite convertir contenido largo en una versión más rápida de consumir, útil para flujos de trabajo con alto volumen documental.

---

## Próximos pasos

Para acercar esta prueba de concepto a un entorno profesional, los siguientes pasos serían razonables:

1. **Aumentar el volumen de entrenamiento** para mejorar robustez y generalización.
2. Aplicar **Beam Search** en inferencia para mejorar calidad lingüística.
3. Añadir `repetition_penalty` para reducir repeticiones.
4. Incorporar validaciones adicionales orientadas a **factualidad**.
5. Construir una pequeña **API o demo web** para que usuarios no técnicos puedan probar el sistema.
6. Comparar el rendimiento con modelos más recientes o más grandes.

---

## Conclusión

Este proyecto demuestra una implementación completa de **IA generativa aplicada al resumen automático en español**, desde la selección del modelo y el dataset hasta el entrenamiento, la evaluación y el análisis crítico de resultados.

Desde una perspectiva empresarial, no solo enseña que el modelo funciona, sino también que el trabajo se ha planteado con criterios útiles para producto: **decisiones técnicas justificadas, evaluación rigurosa, identificación de riesgos y propuesta clara de mejora**.

## Autor

**Josep Gabriel Fornes Reynes**

Proyecto académico de **Deep Learning para el Procesamiento del Lenguaje Natural**, orientado aquí como portfolio técnico para mostrar capacidades de NLP aplicado.
