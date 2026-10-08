# Taller 7 — Inteligencia Artificial Generativa

Solución de los **5 ejercicios** de la sección 7.11 del Taller 7 (material del Prof. José J. Martínez P.), sobre IA Generativa, Transformers, RAG y agentes.

El proyecto está pensado para ejecutarse en **Google Colab** sin necesidad de API keys ni GPU potente: usa **embeddings locales** (`sentence-transformers`), **FAISS** como base de datos vectorial y un **LLM pequeño de Hugging Face** (`flan-t5`) como generador. Todo es **gratuito**.

---

## 📁 Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `colab_taller7.py` | Código completo de los 5 ejercicios, organizado en celdas listas para copiar a Colab. |
| `README.md` | Este documento. |

---

## 🚀 Cómo ejecutar en Google Colab

1. Abre [Google Colab](https://colab.research.google.com/) y crea un notebook nuevo.
2. Copia el contenido de `colab_taller7.py`. Cada bloque marcado con `# %% [N]` es **una celda**: pégalos en orden.
3. Ejecuta la celda `[1]` para instalar dependencias.
4. (Ejercicio 3, opcional) Sube el PDF del taller a Colab con el nombre **`taller7.pdf`** para que el chatbot use el documento real. Si no lo subes, se usa un resumen embebido.
5. Ejecuta el resto de celdas en orden.

> La primera ejecución descarga los modelos de Hugging Face (unos cientos de MB); puede tardar un par de minutos.

---

## 📚 Resumen de los ejercicios

### Ejercicio 1 — Definición de 4 *Skills*
Se generan por código 4 carpetas de *skills* (estilo `SKILL.md` de [aihero.dev](https://www.aihero.dev/)):

1. **documentar-ticket** — Convierte una solicitud informal en un ticket de trabajo completo y accionable.
2. **implementacion** — Genera código **junto con sus pruebas** y lo verifica.
3. **explicacion-html** — Produce un HTML autocontenido que explica el trabajo realizado.
4. **captura-requerimientos** — Lee documentos y extrae requerimientos funcionales y no funcionales trazables.

Cada skill queda escrito en `skills/<nombre>/SKILL.md` con front-matter (`name`, `description`), cuándo usarlo e instrucciones paso a paso.

### Ejercicio 2 — RAG industrial (chatbot sobre manuales técnicos)
Chatbot RAG que responde preguntas sobre manuales técnicos de máquinas, siguiendo el pipeline de la sección 7.7 del taller:
**cargar → limpiar → trocear (chunks) → embeddings → FAISS → recuperar → generar.**

Incluye un **manual de ejemplo embebido** (bomba centrífuga BC-200) para que funcione de inmediato, y una celda opcional para **cargar PDFs reales**.

> El taller sugiere **Ollama**. Como Ollama no es práctico en Colab, la solución principal usa modelos locales de Hugging Face y se documenta la variante Ollama para ejecución local (ver sección más abajo).

### Ejercicio 3 — Chatbot sobre los documentos del curso
Mismo núcleo RAG, pero la fuente de conocimiento es el **PDF del propio Taller 7**. Sube `taller7.pdf` a Colab; si no está disponible, se usa un resumen embebido del contenido. Incluye un **modo chat interactivo** opcional.

### Ejercicio 4 — Agente de mantenimiento
Agente con 4 herramientas (tools) que persisten su estado en archivos JSON:

- `buscar_repuesto(nombre)` → devuelve el **código** del repuesto.
- `consultar_almacen(codigo)` → **existencias**, ubicación, precio y disponibilidad.
- `generar_orden(falla, equipo)` → crea una **orden de trabajo** (`OT-XXXX`).
- `actualizar_historial(...)` → registra la falla en el **historial del equipo**.

El orquestador `agente_mantenimiento(...)` encadena los 4 pasos y **alerta cuando un repuesto no tiene stock**. Se incluye un esquema opcional de *function calling* con LLM.

### Ejercicio 5 — ¿Puede un ESP32 usar un LLM?
Análisis conceptual: un **ESP32 no puede ejecutar un LLM localmente** (RAM ~520 KB, flash 4–16 MB, CPU ~240 MHz sin GPU), pero **sí puede usarlo como cliente** enviando prompts por WiFi/HTTPS a un LLM en la nube o a un servidor local con Ollama. Se detallan las restricciones (conectividad, latencia, seguridad de la API key, costo por tokens, RAM para la respuesta) y la alternativa **TinyML** para inferencia local de modelos diminutos. Incluye pseudocódigo Arduino.

---

## 🔁 Variante con Ollama (ejecución local, fuera de Colab)

Para cumplir la sugerencia del taller, en una máquina local:

```bash
# 1. Instalar Ollama desde https://ollama.com
ollama pull llama3.2          # o el modelo que prefieras
pip install ollama sentence-transformers faiss-cpu pypdf
```

```python
import ollama
# Sustituye el generador flan-t5 por:
def generar(prompt):
    r = ollama.chat(model="llama3.2",
                    messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"]
```

El resto del pipeline RAG (chunking, embeddings, FAISS, recuperación) es idéntico.

---

## 🔑 Variante con API key (opcional, mayor calidad)

Si quieres respuestas de mayor calidad puedes usar una API con capa gratuita (p. ej. **Google Gemini** o **Groq**). **No publiques tu clave en GitHub**; en Colab usa los *Secrets* (`from google.colab import userdata`). El pipeline no cambia: solo se reemplaza la función generadora.

---

## 🧠 Conceptos del taller aplicados

- **Embeddings y similitud de coseno**: FAISS con vectores normalizados (`IndexFlatIP`) implementa la similitud de coseno descrita en el taller.
- **Chunking y base de datos vectorial**: pasos 2–3 del pipeline RAG (sección 7.7).
- **Recuperación semántica + generación**: pasos 4–6 del pipeline RAG.
- **Agentes**: un agente necesita modelo, memoria (JSON persistente) y herramientas (sección 7.8).

---

## ⚙️ Dependencias

```
sentence-transformers
faiss-cpu
transformers
pypdf
accelerate
```

## 📄 Licencia / Créditos

Material base del Taller 7 por el **Prof. José J. Martínez P.** Esta solución es con fines educativos.
