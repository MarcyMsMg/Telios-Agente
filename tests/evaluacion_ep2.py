"""Pruebas de la Evaluación Parcial 2 - Agente Telios.

Ejecuta 9 casos que cubren herramientas, memoria, planificación y decisiones
adaptativas. Requiere Ollama corriendo con llama3.2:3b y nomic-embed-text.

  python tests/evaluacion_ep2.py

Al terminar escribe tests/resultados_pruebas.md con la tabla de resultados
(caso, qué se esperaba, qué hizo el agente, OK/FALLA) para usarla como evidencia.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import config, memory, tools  # noqa: E402
from agent.core import AgenteTelios  # noqa: E402

SESION_PRUEBA = f"prueba_{datetime.now():%Y%m%d_%H%M%S}"


def herramientas_usadas(agente):
    return [t.split("(")[0] for e, t in agente.traza if e == "Action"]


def ticket_mas_reciente():
    archivos = sorted(config.TICKETS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return archivos[-1] if archivos else None


def leer_estado(ruta):
    import json
    return json.loads(ruta.read_text(encoding="utf-8")).get("estado") if ruta else None


# --------------------------------------------------------------------------- casos
def caso_1_anonimizacion(_):
    texto = "Cliente RUT 12.345.678-9, correo ana@pyme.cl, fono +56 9 87654321 reporta error"
    salida = tools.AnonimizerTool.invoke(texto)
    ok = all(x not in salida for x in ["12.345.678-9", "ana@pyme.cl", "87654321"])
    return ok, "Datos personales reemplazados por marcadores", salida


def caso_2_priorizacion(_):
    cola = ["¿Cómo cambio la razón social?", "Error 422 al anular con OneClick",
            "URGENTE: Webpay da error 401 en producción, no puedo vender"]
    orden = sorted(range(len(cola)), key=lambda i: ({"ALTA": 0, "MEDIA": 1, "BAJA": 2}[tools.clasificar(cola[i])[0]], i))
    prioridades = [tools.clasificar(cola[i])[0] for i in orden]
    ok = prioridades == ["ALTA", "MEDIA", "BAJA"]
    return ok, "Cola ordenada ALTA > MEDIA > BAJA", f"Orden de atención: {orden} -> {prioridades}"


def caso_3_escenario_a(agente):
    resp = agente.atender("El cliente no puede cobrar: Webpay responde error HTTP 401 en producción.", usar_historial=False)
    usadas = herramientas_usadas(agente)
    ok = "RAGQueryTool" in usadas and "manual_webpay.txt" in " ".join(t for _, t in agente.traza)
    return ok, "Clasifica, consulta RAG y cita manual_webpay.txt", f"Herramientas: {usadas}\n{resp[:300]}"


def caso_4_escenario_b_escala(agente):
    antes = set(config.TICKETS_DIR.glob("*.json"))
    resp = agente.atender("Cliente reporta error 508 de servidor al cargar el catálogo de productos.", usar_historial=False)
    nuevos = set(config.TICKETS_DIR.glob("*.json")) - antes
    estados = [leer_estado(p) for p in nuevos]
    ok = "ESCALADO" in estados
    return ok, "Sin información + prioridad MEDIA -> ticket ESCALADO a Nivel 3", f"Tickets nuevos: {estados}\n{resp[:300]}"


def caso_5_baja_sin_info_no_escala(agente):
    antes = set(config.TICKETS_DIR.glob("*.json"))
    resp = agente.atender("¿Cómo puedo exportar el listado de ventas a Excel?", usar_historial=False)
    nuevos = set(config.TICKETS_DIR.glob("*.json")) - antes
    estados = [leer_estado(p) for p in nuevos]
    ok = "ESCALADO" not in estados and "Prioridad: BAJA" in resp and "Escalar el caso" not in resp
    return ok, "Sin información + prioridad BAJA -> NO escala, pide más detalles", f"Tickets nuevos: {estados}\n{resp[:300]}"


def caso_6_memoria_corto_plazo(agente):
    agente.atender("Cliente recibe error 422 al anular una compra con OneClick.")
    resp = agente.atender("Hazlo más breve, solo el mensaje para el cliente en 2 líneas.")
    usadas = herramientas_usadas(agente)
    ok = ("RAGQueryTool" not in usadas and tools.estado_turno.get("prioridad") is None and len(resp) > 0
          and "1. Tipo de Problema" not in resp)
    return ok, "2º turno se reconoce como repregunta: responde breve con la memoria, sin reclasificar ni consultar RAG", f"Herramientas 2º turno: {usadas}\n{resp[:300]}"


def caso_7_escenario_c_guardar(agente):
    antes = set(config.TICKETS_DIR.glob("*.json"))
    resp = agente.atender("Perfecto, guarda ese borrador en el sistema de tickets.")
    nuevos = set(config.TICKETS_DIR.glob("*.json")) - antes
    consulta_guardada = ""
    if nuevos:
        import json
        consulta_guardada = json.loads(next(iter(nuevos)).read_text(encoding="utf-8")).get("consulta", "")
    ok = "TicketWriterTool" in herramientas_usadas(agente) and len(nuevos) == 1 and "422" in consulta_guardada
    return ok, "El agente decide llamar a TicketWriterTool cuando el técnico lo pide", f"Tickets nuevos: {[p.name for p in nuevos]} | consulta guardada: {consulta_guardada}\n{resp[:200]}"


def caso_8_memoria_largo_plazo(_):
    obs = tools.BuscarCasosPreviosTool.invoke("error 508 de servidor al cargar el catálogo de productos")  # escalado en el caso 4
    ok = obs.startswith("[CASO PREVIO")
    return ok, "Recupera un caso guardado en sesiones anteriores (ChromaDB telios_memoria)", obs[:300]


def caso_9_retomar_sesion(_):
    agente2 = AgenteTelios(sesion_id=SESION_PRUEBA, verbose=False)
    n = len(agente2.memoria.turnos)
    ok = n >= 2
    return ok, "Una sesión nueva con el mismo ID recupera la conversación guardada", f"Turnos recuperados: {n}"


CASOS = [
    ("1. Anonimización (Ley 21.719)", caso_1_anonimizacion),
    ("2. Priorización de la cola", caso_2_priorizacion),
    ("3. Escenario A: error 401 Webpay", caso_3_escenario_a),
    ("4. Escenario B: sin información, prioridad MEDIA", caso_4_escenario_b_escala),
    ("5. Sin información, prioridad BAJA", caso_5_baja_sin_info_no_escala),
    ("6. Memoria de corto plazo (repregunta)", caso_6_memoria_corto_plazo),
    ("7. Escenario C: guardar a pedido del técnico", caso_7_escenario_c_guardar),
    ("8. Memoria de largo plazo (casos previos)", caso_8_memoria_largo_plazo),
    ("9. Continuidad entre ejecuciones (sesión)", caso_9_retomar_sesion),
]


def main():
    memory.sincronizar_base_conocimiento()
    agente = AgenteTelios(sesion_id=SESION_PRUEBA, verbose=True)
    filas = []
    for nombre, funcion in CASOS:
        print(f"\n{'=' * 70}\n{nombre}\n{'=' * 70}")
        try:
            ok, esperado, obtenido = funcion(agente)
        except Exception as e:
            ok, esperado, obtenido = False, "-", f"Excepción: {e}"
        print(f"\n>>> {'OK' if ok else 'FALLA'} | {esperado}")
        filas.append((nombre, esperado, obtenido, ok))

    total_ok = sum(1 for f in filas if f[3])
    lineas = [f"# Resultados de pruebas - Agente Telios ({datetime.now():%Y-%m-%d %H:%M})", "",
              f"Modelo: {config.MODELO_LLM} | Embeddings: {config.EMBED_MODEL} | Sesión: {SESION_PRUEBA}", "",
              f"**{total_ok}/{len(filas)} casos OK**", "",
              "| Caso | Comportamiento esperado | Resultado | Estado |", "|---|---|---|---|"]
    for nombre, esperado, obtenido, ok in filas:
        celda = obtenido.replace("|", "/").replace("\n", "<br>")[:400]
        lineas.append(f"| {nombre} | {esperado} | {celda} | {'OK' if ok else 'FALLA'} |")
    salida = Path(__file__).resolve().parent / "resultados_pruebas.md"
    salida.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\n{total_ok}/{len(filas)} casos OK. Resultados guardados en {salida}")


if __name__ == "__main__":
    main()
