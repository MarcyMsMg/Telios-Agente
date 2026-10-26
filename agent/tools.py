"""Herramientas (tools) del Agente Telios.

Consulta:      RAGQueryTool, BuscarCasosPreviosTool
Razonamiento:  ClasificarPrioridadTool (prioriza y propone un plan de pasos)
Escritura:     TicketWriterTool
Seguridad:     AnonimizerTool (guardrail que el orquestador ejecuta SIEMPRE antes
               de que el modelo vea el texto, por la Ley N° 21.719)
"""
import json
import re
from datetime import datetime

from langchain_core.tools import tool

from agent import config, memory

# Estado del turno actual (lo fija el orquestador). Permite que TicketWriterTool
# guarde la consulta original anonimizada aunque el modelo no la repita.
estado_turno = {"consulta": "", "prioridad": None, "tickets_guardados": [], "incidencia_previa": None,
                "ultima_respuesta": ""}

ESTADOS_VALIDOS = {"BORRADOR_RESPUESTA", "ESCALADO", "EN_ESPERA_INFO", "RESUELTO"}


# ---------------------------------------------------------------------------
# Seguridad
# ---------------------------------------------------------------------------
@tool
def AnonimizerTool(texto: str) -> str:
    """Sanitiza y elimina datos personales sensibles (RUT, email, teléfono) de una consulta antes de procesarla."""
    texto_limpio = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_OCULTO]', texto)
    texto_limpio = re.sub(r'\b\d{1,2}\.\d{3}\.\d{3}-[\dkK]\b|\b\d{7,8}-[\dkK]\b', '[RUT_OCULTO]', texto_limpio)
    texto_limpio = re.sub(r'(\+?56)?\s?9\s?\d{8}', '[TELEFONO_OCULTO]', texto_limpio)
    return texto_limpio


# ---------------------------------------------------------------------------
# Razonamiento: priorización y plan
# ---------------------------------------------------------------------------
REGLAS_PRIORIDAD = [
    ("ALTA", ["no puedo vender", "no podemos vender", "no puedo cobrar", "pagos caídos", "caída", "caido", "caído",
              "rechaz", "error 401", "http 401", "401", "error 500", "500", "producción", "produccion",
              "urgente", "todos los clientes", "no funciona", "bloquead"],
     "Afecta la capacidad de cobrar o vender del cliente."),
    ("MEDIA", ["422", "anulación", "anulacion", "anular", "reembolso", "devolución", "oneclick", "webpay",
               "transbank", "error"],
     "Error en una operación puntual; el cliente puede seguir operando."),
    ("BAJA", ["configur", "cambiar", "actualizar", "razón social", "razon social", "cómo", "como puedo",
              "consulta", "duda"],
     "Consulta o ajuste de configuración sin impacto en ventas."),
]

PLANES = {
    "ALTA": ["BuscarCasosPreviosTool: revisar si ya ocurrió antes",
             "RAGQueryTool: buscar el procedimiento oficial",
             "Si hay información: proponer solución y guardar el ticket como BORRADOR_RESPUESTA",
             "Si NO hay información: escalar de inmediato a Soporte Nivel 3 (TicketWriterTool con estado ESCALADO)"],
    "MEDIA": ["RAGQueryTool: buscar el procedimiento oficial",
              "BuscarCasosPreviosTool: complementar con casos anteriores si hace falta",
              "Proponer solución; guardar el ticket solo si el técnico lo pide",
              "Si NO hay información: escalar a Soporte Nivel 3"],
    "BAJA": ["RAGQueryTool: buscar la guía de configuración",
             "Responder al técnico; guardar el ticket solo si el técnico lo pide",
             "Si NO hay información: pedir más detalles al técnico (estado EN_ESPERA_INFO), sin escalar"],
}


def clasificar(texto):
    """Clasificación por reglas. Devuelve (prioridad, motivo)."""
    t = texto.lower()
    for prioridad, palabras, motivo in REGLAS_PRIORIDAD:
        if any(p in t for p in palabras):
            return prioridad, motivo
    return "MEDIA", "Sin indicadores claros; se asigna prioridad media por defecto."


# Pasos que el supervisor del plan exige antes de aceptar una respuesta, según prioridad.
PASOS_OBLIGATORIOS = {
    "ALTA": ["BuscarCasosPreviosTool", "RAGQueryTool"],
    "MEDIA": ["RAGQueryTool"],
    "BAJA": ["RAGQueryTool"],
}

PALABRAS_GUARDAR = ["guarda", "guárda", "registra", "regístra", "graba"]
PALABRAS_SEGUIMIENTO = ["más breve", "mas breve", "más corto", "mas corto", "resume", "resúme", "acorta",
                        "acórta", "reformula", "más formal", "mas formal", "otro tono", "cambia el tono",
                        "hazlo", "solo el mensaje", "sólo el mensaje", "tradu", "explica mejor", "y eso"]


def tipo_de_mensaje(texto):
    """Distingue una incidencia nueva de una instrucción sobre la respuesta anterior."""
    t = texto.lower()
    if any(p in t for p in PALABRAS_GUARDAR):
        return "GUARDAR"
    if any(p in t for p in PALABRAS_SEGUIMIENTO):
        return "SEGUIMIENTO"
    return "INCIDENCIA"


@tool
def ClasificarPrioridadTool(incidencia: str) -> str:
    """Clasifica la prioridad de una incidencia (ALTA, MEDIA o BAJA) y devuelve el plan de pasos que el agente debe seguir. Úsala primero ante cualquier incidencia nueva."""
    tipo = tipo_de_mensaje(estado_turno.get("consulta") or incidencia)
    if estado_turno.get("incidencia_previa") and tipo == "GUARDAR":
        return ("NO_ES_INCIDENCIA_NUEVA: el técnico pide guardar el caso anterior. "
                "Llama a TicketWriterTool con el último borrador; no clasifiques de nuevo.")
    if estado_turno.get("incidencia_previa") and tipo == "SEGUIMIENTO":
        anterior = (estado_turno.get("ultima_respuesta") or "")[:1500]
        return ("NO_ES_INCIDENCIA_NUEVA: el técnico pide ajustar tu respuesta anterior. No llames más herramientas.\n"
                f"Instrucción del técnico: {estado_turno.get('consulta', '')}\n"
                f"Tu respuesta anterior fue:\n{anterior}\n"
                "Responde SOLO con el texto reescrito según la instrucción, sin el formato de 5 secciones.")
    prioridad, motivo = clasificar(incidencia)
    estado_turno["prioridad"] = prioridad
    pasos = "\n".join(f"  {i}. {p}" for i, p in enumerate(PLANES[prioridad], 1))
    return f"PRIORIDAD: {prioridad}\nMotivo: {motivo}\nPlan sugerido:\n{pasos}"


# ---------------------------------------------------------------------------
# Consulta
# ---------------------------------------------------------------------------
@tool
def RAGQueryTool(query: str) -> str:
    """Busca semánticamente en ChromaDB manuales técnicos y catálogo de errores de Transbank/NubePyme. Devuelve los fragmentos con su [FUENTE] o NO_INFO si no hay información suficiente."""
    memory.sincronizar_base_conocimiento()
    encontrados = memory.buscar_conocimiento(query)
    if not encontrados:
        return ("NO_INFO: No se encontró contexto suficiente en la base de conocimiento con similitud aceptable. "
                "No inventes una solución: sigue el plan para casos sin información.")
    return "\n\n".join(f"[FUENTE: {e['fuente']}] (distancia {e['distancia']:.2f})\n{e['texto']}" for e in encontrados)


@tool
def BuscarCasosPreviosTool(query: str) -> str:
    """Busca en la memoria de largo plazo incidencias parecidas que el agente ya atendió en sesiones anteriores."""
    memory.sincronizar_tickets_previos()
    casos = memory.buscar_casos(query)
    if not casos:
        return "SIN_CASOS_PREVIOS: No hay incidencias similares registradas."
    return "\n\n".join(
        f"[CASO PREVIO: {c['ticket_id']}] estado={c.get('estado')} prioridad={c.get('prioridad')} "
        f"(distancia {c['distancia']:.2f})\n{c['texto'][:600]}" for c in casos)


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------
def respuesta_sin_informacion(prioridad):
    """Respuesta de protocolo cuando la base de conocimiento no tiene información."""
    if prioridad == "BAJA":
        return ("Prioridad: BAJA\n"
                "1. Tipo de Problema: Consulta sin cobertura en la base de conocimiento.\n"
                "2. Posible Causa Raíz: No determinada; no hay documentación sobre este tema.\n"
                "3. Pasos de Solución Recomendados: Solicitar al cliente más detalles (pantalla, módulo y pasos que siguió). "
                "Por ser prioridad baja no se escala todavía.\n"
                "4. Fuentes Consultadas: Sin información en la base de conocimiento.\n"
                "5. Borrador de Respuesta para el Cliente:\nEstimado cliente,\n\nPara ayudarle mejor, ¿podría indicarnos en qué "
                "sección de la plataforma realiza esta acción y qué resultado obtiene? Con esa información le responderemos a la brevedad.\n\n"
                "Equipo de Soporte Telios")
    return (f"Prioridad: {prioridad or 'MEDIA'}\n"
            "1. Tipo de Problema: Incidencia sin cobertura en la base de conocimiento.\n"
            "2. Posible Causa Raíz: No determinada; no hay información documentada.\n"
            "3. Pasos de Solución Recomendados: Escalar el caso a Soporte Nivel 3.\n"
            "4. Fuentes Consultadas: Sin información en la base de conocimiento.\n"
            "5. Borrador de Respuesta para el Cliente:\nEstimado cliente,\n\nHemos recibido su solicitud y la estamos derivando "
            "a un especialista. Le contactaremos a la brevedad.\n\nEquipo de Soporte Telios")


def guardar_ticket(borrador, estado, prioridad=None, consulta=None):
    """Lógica de guardado (la usan la tool y el guardrail del orquestador)."""
    # Si el modelo pasa un borrador vacío o genérico ("el último borrador"), se usa la
    # última respuesta que el agente le dio al técnico (memoria de corto plazo).
    if (not borrador or len(borrador.strip()) < 60) and estado_turno.get("ultima_respuesta"):
        borrador = estado_turno["ultima_respuesta"]
    estado = (estado or "BORRADOR_RESPUESTA").strip().upper()
    if estado not in ESTADOS_VALIDOS:
        estado = "BORRADOR_RESPUESTA"
    # Si el técnico pide guardar en un turno posterior ("guarda ese borrador"), se usa la
    # incidencia original que el agente recuerda de la sesión, no la frase "guarda...".
    previa = estado_turno.get("incidencia_previa") or {}
    es_seguimiento = estado_turno.get("prioridad") is None and previa
    prioridad = (prioridad or estado_turno.get("prioridad") or previa.get("prioridad") or "MEDIA").upper()
    consulta = consulta or (previa.get("consulta") if es_seguimiento else estado_turno.get("consulta", ""))
    config.TICKETS_DIR.mkdir(exist_ok=True)
    ticket_id = f"ticket_{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]  # milisegundos: evita sobrescribir
    ruta = config.TICKETS_DIR / f"{ticket_id}.json"
    datos = {
        "ticket_id": ticket_id,
        "agente": "Telios NubePyme Support Agent",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "consulta": consulta,
        "prioridad": prioridad,
        "estado": estado,
        "borrador": borrador,
    }
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    memory.recordar_caso(ticket_id, consulta, borrador[:1500], prioridad, estado)
    estado_turno["tickets_guardados"].append(ticket_id)
    estado_turno["ultimo_ticket"] = {"id": ticket_id, "estado": estado, "consulta": consulta}
    return ticket_id, ruta


@tool
def TicketWriterTool(borrador: str, estado: str = "BORRADOR_RESPUESTA", prioridad: str = "") -> str:
    """Guarda el ticket en el sistema de soporte (JSON) y en la memoria de largo plazo. estado debe ser BORRADOR_RESPUESTA, ESCALADO, EN_ESPERA_INFO o RESUELTO."""
    try:
        ticket_id, ruta = guardar_ticket(borrador, estado, prioridad or None)
        return f"ÉXITO: Ticket {ticket_id} guardado con estado {estado.upper()} en '{ruta.name}'."
    except Exception as e:  # el error vuelve al modelo como observación para que se adapte
        return f"ERROR: No se pudo guardar el ticket: {e}"


HERRAMIENTAS_AGENTE = [ClasificarPrioridadTool, RAGQueryTool, BuscarCasosPreviosTool, TicketWriterTool]
