"""Sistemas de memoria del Agente Telios.

- Memoria de corto plazo: ventana deslizante de los últimos K turnos de la
  conversación con el técnico (equivalente a ConversationBufferWindowMemory),
  guardada además en disco para poder retomar una sesión.
- Memoria de largo plazo: dos colecciones en ChromaDB.
    * telios_kb_v2: base de conocimiento (manuales y catálogo de errores),
      fragmentada en bloques de 150 palabras con 30 de solapamiento.
    * telios_memoria: casos que el agente ya atendió (tickets guardados), para
      que recuerde incidencias anteriores entre sesiones.
"""
import json
from datetime import datetime

import chromadb
import ollama
from langchain_core.messages import AIMessage, HumanMessage

from agent import config


# ---------------------------------------------------------------------------
# Embeddings y cliente vectorial
# ---------------------------------------------------------------------------
def embed(textos):
    """Convierte textos en vectores con el modelo local nomic-embed-text."""
    return ollama.embed(model=config.EMBED_MODEL, input=textos).embeddings


_cliente = None


def cliente():
    global _cliente
    if _cliente is None:
        _cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _cliente


def coleccion(nombre):
    return cliente().get_or_create_collection(nombre, metadata={"hnsw:space": "cosine"})


# ---------------------------------------------------------------------------
# Base de conocimiento (memoria semántica)
# ---------------------------------------------------------------------------
def fragmentar(texto, tam=config.CHUNK_PALABRAS, solape=config.CHUNK_SOLAPE):
    """Divide un texto en bloques de `tam` palabras que se solapan `solape` palabras."""
    palabras = texto.split()
    if len(palabras) <= tam:
        return [" ".join(palabras)] if palabras else []
    paso = tam - solape
    return [" ".join(palabras[i:i + tam]) for i in range(0, len(palabras) - solape, paso)]


def sincronizar_base_conocimiento():
    """Indexa en ChromaDB los fragmentos de data/*.txt que todavía no estén indexados."""
    col = coleccion(config.COLECCION_KB)
    existentes = set(col.get()["ids"]) if col.count() else set()
    ids, docs, metas = [], [], []
    for ruta in sorted(config.DATA_DIR.glob("*.txt")):
        texto = ruta.read_text(encoding="utf-8-sig").strip()
        for i, frag in enumerate(fragmentar(texto)):
            frag_id = f"{ruta.name}#{i}"
            if frag_id not in existentes:
                ids.append(frag_id)
                docs.append(frag)
                metas.append({"fuente": ruta.name, "fragmento": i})
    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
    return col.count()


def buscar_conocimiento(consulta):
    """Devuelve los fragmentos relevantes (con su fuente y distancia) o [] si ninguno supera el umbral."""
    col = coleccion(config.COLECCION_KB)
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=embed([consulta]), n_results=min(config.N_RESULTADOS, col.count()))
    distancias = res["distances"][0]
    mejor = distancias[0]
    encontrados = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], distancias):
        if dist <= config.UMBRAL_DISTANCIA and dist <= mejor + config.MARGEN:
            encontrados.append({"fuente": meta["fuente"], "texto": doc, "distancia": dist})
    return encontrados


# ---------------------------------------------------------------------------
# Memoria de largo plazo: casos atendidos
# ---------------------------------------------------------------------------
def recordar_caso(ticket_id, consulta, resumen, prioridad, estado):
    """Guarda un caso atendido para que el agente pueda recordarlo en sesiones futuras."""
    col = coleccion(config.COLECCION_MEMORIA)
    documento = f"Consulta: {consulta}\nResolución: {resumen}"
    col.upsert(
        ids=[ticket_id],
        documents=[documento],
        metadatas=[{"prioridad": prioridad, "estado": estado,
                    "fecha": datetime.now().isoformat(timespec="seconds")}],
        embeddings=embed([documento]),
    )


def sincronizar_tickets_previos():
    """Carga en la memoria de largo plazo los tickets JSON que existan en tickets_output/."""
    col = coleccion(config.COLECCION_MEMORIA)
    existentes = set(col.get()["ids"]) if col.count() else set()
    for ruta in sorted(config.TICKETS_DIR.glob("*.json")):
        ticket_id = ruta.stem
        if ticket_id in existentes:
            continue
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        consulta = datos.get("consulta", "")
        resumen = datos.get("resumen") or datos.get("borrador") or datos.get("resultado", "")
        recordar_caso(ticket_id, consulta, resumen[:1500],
                      datos.get("prioridad", "SIN_CLASIFICAR"), datos.get("estado", "DESCONOCIDO"))
    return col.count()


def buscar_casos(consulta):
    col = coleccion(config.COLECCION_MEMORIA)
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=embed([consulta]), n_results=min(2, col.count()))
    casos = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= config.UMBRAL_DISTANCIA_MEMORIA:
            casos.append({"ticket_id": cid, "texto": doc, "distancia": dist, **meta})
    return casos


# ---------------------------------------------------------------------------
# Memoria de corto plazo: ventana de conversación
# ---------------------------------------------------------------------------
class MemoriaCortoPlazo:
    """Ventana deslizante de los últimos K turnos, persistida en sesiones/<id>.json."""

    def __init__(self, sesion_id, k=config.VENTANA_K):
        self.sesion_id = sesion_id
        self.k = k
        self.ruta = config.SESIONES_DIR / f"{sesion_id}.json"
        self.turnos = []  # lista de {"tecnico": str, "agente": str}
        if self.ruta.exists():
            try:
                self.turnos = json.loads(self.ruta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.turnos = []

    def agregar(self, mensaje_tecnico, respuesta_agente):
        self.turnos.append({"tecnico": mensaje_tecnico, "agente": respuesta_agente})
        config.SESIONES_DIR.mkdir(exist_ok=True)
        self.ruta.write_text(json.dumps(self.turnos, ensure_ascii=False, indent=2), encoding="utf-8")

    def mensajes(self):
        """Últimos K turnos como mensajes de LangChain, listos para reinyectar al modelo."""
        salida = []
        for t in self.turnos[-self.k:]:
            salida += [HumanMessage(t["tecnico"]), AIMessage(t["agente"])]
        return salida

    def ultima_respuesta(self):
        return self.turnos[-1]["agente"] if self.turnos else ""
