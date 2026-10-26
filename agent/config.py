"""Configuración central del Agente Telios.

Todos los parámetros que se citan en el informe están aquí, para que el
documento y el código digan siempre lo mismo.
"""
from pathlib import Path

# Rutas (relativas a la carpeta del proyecto, no al directorio desde donde se ejecuta)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
TICKETS_DIR = BASE_DIR / "tickets_output"
SESIONES_DIR = BASE_DIR / "sesiones"

# Modelos locales (Ollama)
MODELO_LLM = "llama3.2:3b"          # modelo con soporte de tool calling en Ollama
EMBED_MODEL = "nomic-embed-text"    # modelo de embeddings
TEMPERATURA = 0.0
NUM_PREDICT = 800                   # máximo de tokens que puede generar el modelo por respuesta

# Recuperación semántica (base de conocimiento)
COLECCION_KB = "telios_kb_v2"
CHUNK_PALABRAS = 150                # tamaño de fragmento
CHUNK_SOLAPE = 30                   # solapamiento entre fragmentos
N_RESULTADOS = 2                    # fragmentos que se recuperan por consulta
UMBRAL_DISTANCIA = 0.30             # distancia coseno máxima aceptada (= similitud >= 0.70)
MARGEN = 0.08                       # solo se aceptan fragmentos cercanos al mejor resultado

# Memoria de largo plazo (casos atendidos anteriormente)
COLECCION_MEMORIA = "telios_memoria"
UMBRAL_DISTANCIA_MEMORIA = 0.35

# Memoria de corto plazo
VENTANA_K = 5                       # turnos (pregunta + respuesta) que se reinyectan al modelo

# Control del bucle ReAct
MAX_ITERACIONES = 6                 # evita bucles infinitos y limita costo/latencia
