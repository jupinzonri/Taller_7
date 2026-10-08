# ==============================================================================
#  TALLER 7 — INTELIGENCIA ARTIFICIAL GENERATIVA
#  Autor del material: Prof. José J. Martínez P.
#  Solución de los 5 ejercicios (Sección 7.11)
# ==============================================================================
#  CÓMO USAR ESTE ARCHIVO EN GOOGLE COLAB
#  --------------------------------------
#  Cada bloque separado por una línea "# %% [N] ..." corresponde a UNA CELDA.
#  Copia el contenido de cada bloque en una celda nueva de Colab, en orden.
#  (También puedes pegar todo en una sola celda, pero se recomienda separarlo
#   para ejecutar paso a paso.)
#
#  ENFOQUE: 100% gratuito, SIN API key. Todo corre dentro de Colab.
#  Se usan embeddings locales (sentence-transformers) + FAISS como base de
#  datos vectorial + un LLM pequeño de Hugging Face (flan-t5) como generador.
#  Al final de cada ejercicio RAG se documenta la variante con Ollama / API.
# ==============================================================================


# %% [1]  INSTALACIÓN DE DEPENDENCIAS  --------------------------------------
# En Colab, el "!" ejecuta comandos de shell. Fuera de Colab, instala con pip.
!pip -q install sentence-transformers faiss-cpu transformers pypdf accelerate


# %% [2]  IMPORTS Y CONFIGURACIÓN GLOBAL  -----------------------------------
import os
import re
import json
import textwrap
from datetime import datetime

import numpy as np

# Semilla para reproducibilidad
np.random.seed(42)

print("Entorno listo. Ejecuta las celdas en orden.")


# ==============================================================================
#  EJERCICIO 1 — DEFINICIÓN DE 4 "SKILLS" (estilo SKILL.md de aihero.dev)
# ==============================================================================
#  Un "Skill" es un paquete (carpeta) con instrucciones, scripts y recursos,
#  anclado por un archivo Markdown llamado SKILL.md. Aquí generamos por código
#  las 4 carpetas de skills con su respectivo SKILL.md.
# ==============================================================================

# %% [3]  EJERCICIO 1 — Generación de los 4 Skills  -------------------------

SKILLS = {
    "documentar-ticket": {
        "name": "Documentar Ticket de Trabajo",
        "description": "Captura y estructura toda la descripción de un trabajo por hacer en un ticket estandarizado.",
        "when_to_use": "Cuando se recibe una solicitud de trabajo y hay que convertirla en un ticket claro, completo y accionable antes de implementar nada.",
        "instructions": """\
## Objetivo
Transformar una solicitud informal en un ticket de trabajo completo y accionable.

## Pasos
1. Identificar el **título** corto y descriptivo del trabajo.
2. Redactar el **contexto / problema**: por qué se necesita este trabajo.
3. Definir el **alcance**: qué SÍ incluye y qué NO incluye.
4. Enumerar los **criterios de aceptación** (checklist verificable).
5. Listar **dependencias y riesgos** conocidos.
6. Estimar **esfuerzo** (S/M/L) y asignar **prioridad** (Alta/Media/Baja).

## Formato de salida (Markdown)
```
# [TICKET-XXX] <título>
**Prioridad:** ...  | **Esfuerzo:** ...  | **Estado:** Abierto
## Contexto
## Alcance
## Criterios de aceptación
- [ ] ...
## Dependencias y riesgos
```

## Reglas
- No implementar código en este skill; sólo documentar.
- Si falta información, listar explícitamente las **preguntas abiertas**.
""",
    },
    "implementacion": {
        "name": "Implementación (código + pruebas)",
        "description": "Genera el código solicitado junto con sus pruebas unitarias y verificación.",
        "when_to_use": "Cuando ya existe un ticket documentado y aprobado, y hay que escribir y validar el código.",
        "instructions": """\
## Objetivo
Implementar la funcionalidad descrita en un ticket, con pruebas automatizadas.

## Pasos
1. Leer el ticket y confirmar los criterios de aceptación.
2. Diseñar la solución (estructura de archivos, funciones, dependencias).
3. Escribir el código siguiendo buenas prácticas (nombres claros, docstrings).
4. Escribir **pruebas unitarias** que cubran los criterios de aceptación.
5. Ejecutar las pruebas y corregir hasta que todas pasen.
6. Reportar cobertura y casos límite considerados.

## Reglas
- Todo código entregado debe venir acompañado de al menos una prueba.
- No marcar como terminado si alguna prueba falla.
- Preferir funciones puras y pequeñas, fáciles de testear.
""",
    },
    "explicacion-html": {
        "name": "Explicación en HTML",
        "description": "Genera un documento HTML autocontenido que explica en detalle el trabajo realizado.",
        "when_to_use": "Al finalizar una implementación, para entregar documentación visual y legible a stakeholders no técnicos.",
        "instructions": """\
## Objetivo
Producir un archivo HTML autocontenido (sin dependencias externas) que explique
qué se hizo, cómo y por qué.

## Pasos
1. Resumir el objetivo del trabajo en 2-3 frases.
2. Explicar la solución con secciones: Contexto, Diseño, Implementación, Resultados.
3. Incluir fragmentos de código resaltados dentro de <pre><code>...</code></pre>.
4. Añadir una sección de "Cómo ejecutar / reproducir".
5. Usar CSS embebido para que el HTML se vea bien sin internet.

## Reglas
- El HTML debe abrir correctamente en un navegador sin archivos externos.
- Lenguaje claro, apto para audiencia mixta (técnica y no técnica).
""",
    },
    "captura-requerimientos": {
        "name": "Captura de Requerimientos",
        "description": "Lee documentos provistos y extrae requerimientos funcionales y no funcionales estructurados.",
        "when_to_use": "Al inicio de un proyecto, cuando el cliente entrega documentos (PDF, specs, correos) y hay que destilar los requerimientos.",
        "instructions": """\
## Objetivo
Leer uno o varios documentos y producir una lista estructurada de requerimientos.

## Pasos
1. Cargar y limpiar el texto de los documentos de entrada.
2. Clasificar los requerimientos en:
   - **Funcionales** (qué debe hacer el sistema).
   - **No funcionales** (rendimiento, seguridad, usabilidad, etc.).
3. Asignar a cada requerimiento un ID único (REQ-F-01, REQ-NF-01...).
4. Marcar los requerimientos ambiguos y generar preguntas de aclaración.
5. Entregar una tabla trazable requerimiento -> fuente (documento/página).

## Reglas
- No inventar requerimientos: todo debe estar respaldado por el documento fuente.
- Separar claramente "requerimiento explícito" de "supuesto / inferido".
""",
    },
}


def crear_skills(base_dir="skills"):
    """Crea la estructura de carpetas y archivos SKILL.md para cada skill."""
    os.makedirs(base_dir, exist_ok=True)
    rutas = []
    for folder, meta in SKILLS.items():
        skill_path = os.path.join(base_dir, folder)
        os.makedirs(skill_path, exist_ok=True)
        md = (
            f"---\n"
            f"name: {meta['name']}\n"
            f"description: {meta['description']}\n"
            f"---\n\n"
            f"# {meta['name']}\n\n"
            f"**Cuándo usar este skill:** {meta['when_to_use']}\n\n"
            f"{meta['instructions']}"
        )
        md_file = os.path.join(skill_path, "SKILL.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md)
        rutas.append(md_file)
    return rutas


rutas_skills = crear_skills()
print("Skills creados:")
for r in rutas_skills:
    print("  -", r)

# Mostrar el contenido de un skill como ejemplo
print("\n----- Ejemplo: skills/documentar-ticket/SKILL.md -----\n")
with open("skills/documentar-ticket/SKILL.md", encoding="utf-8") as f:
    print(f.read())


# ==============================================================================
#  NÚCLEO RAG REUTILIZABLE (lo usan los ejercicios 2 y 3)
# ==============================================================================
#  Pipeline RAG según el documento (sección 7.7):
#   1. Cargar documentos  2. Limpiar  3. Trocear (chunks)
#   4. Embeddings -> base vectorial (FAISS)  5. Consulta -> embedding
#   6. Recuperar chunks cercanos  7. Generar respuesta con el LLM
# ==============================================================================

# %% [4]  NÚCLEO RAG — Carga de modelos  ------------------------------------
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Modelo de embeddings multilingüe (soporta español). Pequeño y gratuito.
EMBED_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# LLM generador pequeño (gratuito, sin API key). flan-t5 responde en es/en.
# Para mejores respuestas en español puedes cambiar a "google/flan-t5-large"
# (más lento) o usar la variante con API documentada más abajo.
GENERADOR = pipeline("text2text-generation", model="google/flan-t5-base")

print("Modelos de embeddings y generación cargados.")


# %% [5]  NÚCLEO RAG — Funciones de limpieza y chunking  --------------------
def limpiar_texto(texto: str) -> str:
    """Elimina ruido básico: espacios múltiples, saltos de línea sobrantes."""
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def trocear(texto: str, tam_chunk: int = 500, solapamiento: int = 80):
    """Divide el texto en chunks por palabras, con solapamiento para no
    perder contexto en los bordes (paso 2 del pipeline RAG)."""
    palabras = texto.split()
    chunks, i = [], 0
    while i < len(palabras):
        chunk = " ".join(palabras[i:i + tam_chunk])
        chunks.append(chunk)
        i += tam_chunk - solapamiento
    return chunks


class BaseVectorial:
    """Base de datos vectorial simple sobre FAISS (búsqueda semántica)."""

    def __init__(self, modelo_embed):
        self.modelo = modelo_embed
        self.index = None
        self.chunks = []

    def indexar(self, chunks):
        self.chunks = chunks
        emb = self.modelo.encode(chunks, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=True)
        dim = emb.shape[1]
        # IndexFlatIP = producto interno; con vectores normalizados equivale a
        # la SIMILITUD DE COSENO descrita en la sección de embeddings del taller.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb.astype("float32"))

    def buscar(self, consulta, k=3):
        q = self.modelo.encode([consulta], convert_to_numpy=True,
                               normalize_embeddings=True)
        sims, idxs = self.index.search(q.astype("float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(idxs[0], sims[0])]


def responder_rag(base: "BaseVectorial", pregunta: str, k: int = 3) -> str:
    """Recupera contexto y genera la respuesta (pasos 5 y 6 del pipeline)."""
    recuperados = base.buscar(pregunta, k=k)
    contexto = "\n\n".join(c for c, _ in recuperados)
    prompt = (
        "Responde la pregunta usando ÚNICAMENTE el siguiente contexto. "
        "Si la respuesta no está en el contexto, di que no tienes esa información.\n\n"
        f"Contexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:"
    )
    salida = GENERADOR(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    return salida.strip(), recuperados


print("Núcleo RAG definido (limpiar_texto, trocear, BaseVectorial, responder_rag).")


# ==============================================================================
#  EJERCICIO 2 — RAG INDUSTRIAL (chatbot sobre manuales técnicos)
# ==============================================================================
#  El taller sugiere descargar manuales reales. Para que el notebook corra en
#  cualquier parte, incluimos un manual técnico de EJEMPLO embebido. Para usar
#  manuales reales, ver la celda [7] (carga de PDFs).
# ==============================================================================

# %% [6]  EJERCICIO 2 — Chatbot RAG con manual de ejemplo  ------------------
MANUAL_EJEMPLO = """\
MANUAL TÉCNICO - BOMBA CENTRÍFUGA MODELO BC-200.
Especificaciones: caudal máximo 200 litros por minuto, presión máxima 6 bar,
potencia del motor 2.2 kW, voltaje 220V trifásico, temperatura máxima del
fluido 80 grados centígrados.
Mantenimiento preventivo: lubricar los rodamientos cada 2000 horas de
operación. Revisar el sello mecánico cada 1000 horas. Cambiar el aceite del
motor cada 4000 horas.
Fallas comunes: si la bomba vibra en exceso, revisar el alineamiento del eje
y el estado de los rodamientos. Si no genera presión, verificar que el
impulsor no esté obstruido y que no haya aire en la succión (cebado).
Si el motor se sobrecalienta, revisar la ventilación y el amperaje; un
amperaje alto puede indicar sobrecarga o rodamientos dañados.
Repuestos principales: sello mecánico código SM-200, rodamiento código RD-6205,
impulsor código IMP-200, empaque de carcasa código EMP-200.
Seguridad: desconectar la energía antes de cualquier intervención. Usar guantes
y gafas de protección. No operar la bomba en seco.
"""

base_manual = BaseVectorial(EMBED_MODEL)
chunks_manual = trocear(limpiar_texto(MANUAL_EJEMPLO), tam_chunk=60, solapamiento=15)
base_manual.indexar(chunks_manual)

# Prueba del chatbot industrial
for pregunta in [
    "¿Cada cuánto se deben lubricar los rodamientos?",
    "¿Qué hago si la bomba no genera presión?",
    "¿Cuál es el código del sello mecánico?",
]:
    resp, fuentes = responder_rag(base_manual, pregunta, k=2)
    print(f"\nP: {pregunta}\nR: {resp}")


# %% [7]  EJERCICIO 2 (opcional) — Cargar manuales PDF reales  --------------
# Sube tus PDFs a Colab (icono de carpeta -> subir) y descomenta:
#
# from pypdf import PdfReader
# def leer_pdf(ruta):
#     texto = ""
#     for pagina in PdfReader(ruta).pages:
#         texto += (pagina.extract_text() or "") + "\n"
#     return texto
#
# textos = [leer_pdf("manual1.pdf"), leer_pdf("manual2.pdf")]
# chunks = []
# for t in textos:
#     chunks += trocear(limpiar_texto(t), tam_chunk=200, solapamiento=40)
# base_manual = BaseVectorial(EMBED_MODEL); base_manual.indexar(chunks)
# print(responder_rag(base_manual, "¿Cuál es la presión máxima?")[0])


# ==============================================================================
#  EJERCICIO 3 — CHATBOT SOBRE LOS DOCUMENTOS DEL CURSO (este PDF)
# ==============================================================================
#  Sube a Colab el PDF del Taller 7 con el nombre "taller7.pdf".
#  Si no lo subes, el código usa un resumen embebido del contenido del taller
#  para que el chatbot funcione igualmente.
# ==============================================================================

# %% [8]  EJERCICIO 3 — Chatbot del curso  ----------------------------------
RESUMEN_CURSO = """\
La IA Generativa produce texto, imágenes, audio, video y datos sintéticos a
partir de patrones aprendidos de datos existentes. El concepto de Transformer
se desarrolló en 2018 y marcó un antes y un después en la IA.
Los modelos fundamentales son algoritmos de Deep Learning pre-entrenados con
data sets extremadamente grandes de internet; se entrenan una vez y luego se
afinan para muchas tareas. Ejemplos: Claude, GPT-4, DeepSeek, DALL-E 2.
Un token es una unidad de datos que viene de descomponer información (palabras,
caracteres o frases). El costo de uso de los LLMs se basa en el número de tokens.
Un embedding es una representación vectorial numérica de datos complejos que
captura significado semántico. La similitud de coseno mide el ángulo entre dos
vectores: 1 es similitud total, 0 es sin relación.
La arquitectura Transformer se basa en mecanismos de atención (artículo
Attention Is All You Need). El encoder comprende la secuencia de entrada; el
decoder genera la secuencia de salida. Usa vectores Q (consulta), K (clave) y
V (valor). El masked multi-head attention evita que el modelo vea palabras
futuras durante el entrenamiento.
RAG (Retrieval-Augmented Generation) mejora la precisión de los LLMs con datos
de fuentes externas, usando una base de datos vectorial y búsqueda semántica.
Un agente de IA necesita tres cosas: un modelo de IA, memoria para contexto, y
herramientas y conocimientos. MCP es un protocolo abierto de Anthropic para
estandarizar cómo las aplicaciones dan contexto y herramientas a los LLM.
"""

# Intentar leer el PDF del taller; si no está, usar el resumen embebido.
texto_curso = None
try:
    from pypdf import PdfReader
    if os.path.exists("taller7.pdf"):
        texto_curso = ""
        for pagina in PdfReader("taller7.pdf").pages:
            texto_curso += (pagina.extract_text() or "") + "\n"
        print("PDF del taller cargado correctamente.")
except Exception as e:
    print("No se pudo leer el PDF:", e)

if not texto_curso:
    print("Usando resumen embebido del curso (sube 'taller7.pdf' para usar el documento real).")
    texto_curso = RESUMEN_CURSO

base_curso = BaseVectorial(EMBED_MODEL)
chunks_curso = trocear(limpiar_texto(texto_curso), tam_chunk=120, solapamiento=25)
base_curso.indexar(chunks_curso)

for pregunta in [
    "¿Qué es un token?",
    "¿Para qué sirve RAG?",
    "¿Qué tres cosas necesita un agente de IA?",
    "¿Qué miden los vectores Q, K y V?",
]:
    resp, _ = responder_rag(base_curso, pregunta, k=3)
    print(f"\nP: {pregunta}\nR: {resp}")


# %% [9]  EJERCICIO 3 — Modo chat interactivo (opcional)  -------------------
# Descomenta para chatear en vivo dentro de Colab:
#
# print("Chatbot del curso. Escribe 'salir' para terminar.")
# while True:
#     q = input("\nTú: ")
#     if q.strip().lower() in ("salir", "exit", "quit"):
#         break
#     r, _ = responder_rag(base_curso, q, k=3)
#     print("Bot:", r)


# ==============================================================================
#  EJERCICIO 4 — AGENTE DE MANTENIMIENTO
# ==============================================================================
#  Herramientas (tools) que el agente puede invocar:
#    - buscar_repuesto(nombre)      -> busca un repuesto y devuelve su código
#    - consultar_almacen(codigo)    -> consulta existencias en el almacén
#    - generar_orden(falla, equipo) -> crea una orden de trabajo
#    - actualizar_historial(...)    -> registra la falla en el historial
#  El estado se persiste en archivos JSON (almacén e historial).
# ==============================================================================

# %% [10]  EJERCICIO 4 — Datos iniciales (almacén, repuestos, historial)  ---
CATALOGO_REPUESTOS = {
    "sello mecanico": "SM-200",
    "rodamiento": "RD-6205",
    "impulsor": "IMP-200",
    "empaque": "EMP-200",
    "correa": "COR-A42",
}

ALMACEN = {  # código -> {stock, ubicacion, precio}
    "SM-200": {"stock": 4, "ubicacion": "A-12", "precio": 85000},
    "RD-6205": {"stock": 10, "ubicacion": "B-03", "precio": 32000},
    "IMP-200": {"stock": 0, "ubicacion": "C-07", "precio": 450000},
    "EMP-200": {"stock": 25, "ubicacion": "A-15", "precio": 12000},
    "COR-A42": {"stock": 6, "ubicacion": "D-01", "precio": 28000},
}

ARCHIVO_ALMACEN = "almacen.json"
ARCHIVO_HISTORIAL = "historial_fallas.json"
ARCHIVO_ORDENES = "ordenes_trabajo.json"

# Persistir estado inicial
with open(ARCHIVO_ALMACEN, "w", encoding="utf-8") as f:
    json.dump(ALMACEN, f, ensure_ascii=False, indent=2)
for arch in (ARCHIVO_HISTORIAL, ARCHIVO_ORDENES):
    if not os.path.exists(arch):
        with open(arch, "w", encoding="utf-8") as f:
            json.dump([], f)

print("Estado inicial del agente persistido en JSON.")


# %% [11]  EJERCICIO 4 — Herramientas del agente (Código)  ------------------
def _cargar(arch):
    with open(arch, encoding="utf-8") as f:
        return json.load(f)


def _guardar(arch, data):
    with open(arch, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def buscar_repuesto(nombre: str) -> dict:
    """Busca un repuesto por nombre y devuelve su código."""
    clave = nombre.strip().lower()
    for nombre_cat, codigo in CATALOGO_REPUESTOS.items():
        if clave in nombre_cat or nombre_cat in clave:
            return {"encontrado": True, "nombre": nombre_cat, "codigo": codigo}
    return {"encontrado": False, "nombre": nombre, "codigo": None}


def consultar_almacen(codigo: str) -> dict:
    """Consulta existencias de un código en el almacén."""
    almacen = _cargar(ARCHIVO_ALMACEN)
    item = almacen.get(codigo)
    if not item:
        return {"codigo": codigo, "existe": False}
    return {"codigo": codigo, "existe": True, **item,
            "disponible": item["stock"] > 0}


def generar_orden(falla: str, equipo: str, repuesto_codigo: str = None) -> dict:
    """Crea una orden de trabajo y la persiste."""
    ordenes = _cargar(ARCHIVO_ORDENES)
    orden = {
        "id": f"OT-{len(ordenes) + 1:04d}",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "equipo": equipo,
        "falla": falla,
        "repuesto": repuesto_codigo,
        "estado": "ABIERTA",
    }
    ordenes.append(orden)
    _guardar(ARCHIVO_ORDENES, ordenes)
    return orden


def actualizar_historial(equipo: str, falla: str, orden_id: str) -> dict:
    """Registra la falla en el historial del equipo."""
    historial = _cargar(ARCHIVO_HISTORIAL)
    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "equipo": equipo,
        "falla": falla,
        "orden": orden_id,
    }
    historial.append(registro)
    _guardar(ARCHIVO_HISTORIAL, historial)
    return registro


print("Herramientas del agente definidas.")


# %% [12]  EJERCICIO 4 — Orquestador del agente  ----------------------------
def agente_mantenimiento(equipo: str, falla: str, repuesto_necesario: str) -> dict:
    """Flujo del agente: busca repuesto -> consulta almacén -> genera orden
    -> actualiza historial. Devuelve un reporte de la intervención."""
    reporte = {"equipo": equipo, "falla": falla, "pasos": []}

    # 1) Buscar el repuesto
    rep = buscar_repuesto(repuesto_necesario)
    reporte["pasos"].append(("buscar_repuesto", rep))

    codigo = rep["codigo"]
    disponibilidad = None
    if codigo:
        # 2) Consultar el almacén
        disponibilidad = consultar_almacen(codigo)
        reporte["pasos"].append(("consultar_almacen", disponibilidad))

    # 3) Generar la orden de trabajo
    orden = generar_orden(falla, equipo, codigo)
    reporte["pasos"].append(("generar_orden", orden))

    # 4) Actualizar el historial de fallas
    hist = actualizar_historial(equipo, falla, orden["id"])
    reporte["pasos"].append(("actualizar_historial", hist))

    # Recomendación según stock
    if disponibilidad and not disponibilidad.get("disponible", False):
        reporte["alerta"] = (f"El repuesto {codigo} NO tiene stock. "
                             f"Se requiere compra antes de ejecutar la orden.")
    return reporte


# Demostración
reporte = agente_mantenimiento(
    equipo="Bomba BC-200 (línea 3)",
    falla="Vibración excesiva y ruido en el eje",
    repuesto_necesario="rodamiento",
)
print(json.dumps(reporte, ensure_ascii=False, indent=2))

# Caso con repuesto sin stock
print("\n--- Caso sin stock ---")
reporte2 = agente_mantenimiento(
    equipo="Bomba BC-200 (línea 1)",
    falla="No genera presión, impulsor obstruido",
    repuesto_necesario="impulsor",
)
print(json.dumps(reporte2, ensure_ascii=False, indent=2))


# %% [13]  EJERCICIO 4 (opcional) — Agente con "function calling" por LLM ----
# El flujo anterior es determinista. Para un agente que DECIDA qué herramienta
# usar con lenguaje natural, se puede usar un LLM con tool-calling (p. ej.
# Gemini/Groq gratis, u Ollama local). Esquema conceptual:
#
#   TOOLS = {"buscar_repuesto": buscar_repuesto, "consultar_almacen": ...}
#   El LLM recibe la lista de tools + la petición del usuario y responde con
#   JSON {"tool": "...", "args": {...}}. El código ejecuta la tool y le
#   devuelve el resultado al LLM hasta resolver la petición.


# ==============================================================================
#  EJERCICIO 5 — ¿PUEDE UN ESP32 USAR UN LLM? RESTRICCIONES
# ==============================================================================
# %% [14]  EJERCICIO 5 — Análisis (texto)  ----------------------------------
ANALISIS_ESP32 = """
EJERCICIO 5 — ¿Puede una aplicación con ESP32 usar un LLM?

RESPUESTA CORTA:
No puede EJECUTAR un LLM grande localmente, pero SÍ puede USAR un LLM como
CLIENTE: el ESP32 envía el prompt por WiFi a un LLM en la nube (o en un
servidor local / PC con Ollama) y recibe la respuesta. También puede ejecutar
localmente modelos diminutos de ML (TinyML), pero no un LLM de miles de millones
de parámetros.

POR QUÉ NO LOCALMENTE — RESTRICCIONES DE HARDWARE:
- Memoria RAM: el ESP32 tiene ~520 KB de SRAM (y hasta pocos MB de PSRAM). Un
  LLM pequeño cuantizado necesita cientos de MB a varios GB. Es inviable.
- Almacenamiento: la flash típica es de 4-16 MB; los pesos de un LLM pesan
  cientos de MB o más.
- CPU: 2 núcleos a ~240 MHz, sin GPU/NPU potente. La inferencia de un
  Transformer sería extremadamente lenta.
- Energía: en aplicaciones a batería, la comunicación por red y el cómputo
  intensivo agotan la batería.

ARQUITECTURA RECOMENDADA (ESP32 como cliente de un LLM):
  [Sensores] -> [ESP32] --WiFi/HTTPS--> [API del LLM en la nube o servidor]
                               <-- respuesta (texto/JSON) --
- El ESP32 arma el prompt (p. ej. con lecturas de sensores) y hace una petición
  HTTPS a la API (OpenAI, Gemini, Groq, o un Ollama en la red local).
- El LLM procesa y responde; el ESP32 actúa sobre la respuesta (muestra texto,
  activa un relé, etc.).

RESTRICCIONES AL USARLO COMO CLIENTE:
- Conectividad: requiere WiFi/red estable; sin internet no hay LLM en la nube.
- Latencia: la respuesta depende de la red y del servidor (segundos).
- Seguridad: manejar HTTPS/TLS y proteger la API key (no quemarla en el firmware
  en texto plano; usar un backend intermedio si es posible).
- Costo y límites: las APIs de pago cobran por token; hay límites de tasa.
- Tamaño de respuesta: la RAM limitada obliga a procesar la respuesta por partes
  (streaming) o pedir respuestas cortas (p. ej., formato JSON compacto).
- Fiabilidad: conviene manejar timeouts, reintentos y un modo de fallo seguro.

ALTERNATIVA LOCAL (TinyML, NO es un LLM):
- Para inferencia 100% local en el ESP32 se usan modelos diminutos con
  TensorFlow Lite Micro (detección de palabras clave, clasificación de gestos,
  anomalías en vibración). Son modelos de KB, no LLMs.

CONCLUSIÓN:
El ESP32 es ideal como PUENTE entre el mundo físico (sensores/actuadores) y un
LLM alojado en otro lugar. Ejecutar el LLM dentro del ESP32 no es viable con la
tecnología actual por memoria, cómputo y energía.
"""
print(ANALISIS_ESP32)


# %% [15]  EJERCICIO 5 (opcional) — Pseudocódigo del cliente ESP32 ----------
PSEUDO_ESP32 = r'''
// Pseudocódigo Arduino/ESP32 (cliente de un LLM en la nube)
#include <WiFi.h>
#include <HTTPClient.h>

void preguntarLLM(String prompt) {
  HTTPClient http;
  http.begin("https://api.proveedor-llm.com/v1/chat");   // endpoint del LLM
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", "Bearer <API_KEY>");   // mejor vía backend
  String body = "{\"prompt\":\"" + prompt + "\",\"max_tokens\":80}";
  int code = http.POST(body);
  if (code == 200) {
    String respuesta = http.getString();   // procesar por partes si es larga
    Serial.println(respuesta);
  } else {
    Serial.println("Error de red / timeout");  // modo de fallo seguro
  }
  http.end();
}
'''
print(PSEUDO_ESP32)

print("\n==== FIN DEL TALLER 7 ====")
