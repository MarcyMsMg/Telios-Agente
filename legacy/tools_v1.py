import os
import re
import json
import glob
import ollama
import chromadb
from langchain_core.tools import tool

EMBED_MODEL = "nomic-embed-text"
UMBRAL_DISTANCIA = 0.30
MARGEN = 0.08

_cliente = chromadb.PersistentClient(path="chroma_db")
_coleccion = _cliente.get_or_create_collection("telios_kb", metadata={"hnsw:space": "cosine"})

def _embed(textos):
    return ollama.embed(model=EMBED_MODEL, input=textos).embeddings

def _indexar_si_vacia():
    if _coleccion.count() > 0:
        return
    docs, ids = [], []
    for ruta in glob.glob("data/*.txt"):
        with open(ruta, encoding="utf-8-sig") as f:
            docs.append(f.read().strip())
        ids.append(os.path.basename(ruta))
    if docs:
        _coleccion.add(ids=ids, documents=docs, embeddings=_embed(docs))

@tool
def AnonimizerTool(texto: str) -> str:
    """Sanitiza y elimina datos personales sensibles (RUT, email, teléfono) de una consulta antes de procesarla."""
    texto_limpio = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_OCULTO]', texto)
    texto_limpio = re.sub(r'\b\d{1,2}\.\d{3}\.\d{3}-[\dkK]\b|\b\d{7,8}-[\dkK]\b', '[RUT_OCULTO]', texto_limpio)
    texto_limpio = re.sub(r'(\+?56)?\s?9\s?\d{8}', '[TELEFONO_OCULTO]', texto_limpio)
    return texto_limpio

@tool
def RAGQueryTool(query: str) -> str:
    """Busca semánticamente en ChromaDB manuales técnicos y catálogo de errores de Transbank/NubePyme."""
    _indexar_si_vacia()
    if _coleccion.count() == 0:
        return "NO_INFO: La base de conocimiento está vacía."
    res = _coleccion.query(query_embeddings=_embed([query]), n_results=2)
    resultados = []
    for doc_id, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
        if dist <= UMBRAL_DISTANCIA and dist <= res["distances"][0][0] + MARGEN:
            resultados.append(f"[FUENTE: {doc_id}] (distancia {dist:.2f})\n{doc}")
    if not resultados:
        return "NO_INFO: No se encontró contexto suficiente en la base de conocimiento con similitud aceptable."
    return "\n\n".join(resultados)

@tool
def TicketWriterTool(input_data: str) -> str:
    """Registra y guarda el borrador de respuesta y estado final de la incidencia en un archivo JSON local."""
    try:
        os.makedirs("tickets_output", exist_ok=True)
        import datetime; filename = f"tickets_output/ticket_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        data = {
            "agente": "Telios NubePyme Support Agent",
            "resultado": input_data,
            "estado": "ESCALADO" if "Escalar el caso" in input_data else "BORRADOR_RESPUESTA"
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"ÉXITO: Registro guardado correctamente en '{filename}'."
    except Exception as e:
        return f"ERROR: No se pudo guardar el registro: {str(e)}"




