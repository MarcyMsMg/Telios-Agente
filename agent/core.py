"""Orquestador del Agente Telios: bucle ReAct con tool calling (LangChain + Ollama).

En cada iteración el MODELO decide si llama a una herramienta (y con qué
argumentos) o si ya puede responder. El orquestador solo:
  1. aplica guardrails (anonimizar antes, escalar si corresponde después),
  2. ejecuta las herramientas que el modelo pidió y le devuelve la observación,
  3. supervisa el plan: si el modelo intenta responder sin haber hecho los pasos
     obligatorios de la prioridad asignada (p. ej. consultar RAG), los ejecuta antes,
  4. si no hay información documentada, detiene la generación (respuesta de protocolo),
  5. limita el número de iteraciones y maneja errores para que el modelo se adapte.
"""
import json
import re
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from agent import config, tools
from agent.memory import MemoriaCortoPlazo

PROMPT_SISTEMA = """Eres Telios, un agente de soporte técnico de Nivel 2 para la plataforma NubePyme. Apoyas a técnicos humanos que atienden incidencias de clientes (pymes) con Webpay/Transbank y con la plataforma.

Trabajas con el ciclo ReAct: piensas qué necesitas, usas una herramienta, observas el resultado y decides el siguiente paso.

Reglas:
- Si el mensaje describe una incidencia NUEVA: llama primero a ClasificarPrioridadTool y luego sigue el plan que te devuelve.
- Si el mensaje pide cambiar tu respuesta anterior (más breve, más formal, otro tono) o pregunta algo sobre ella: responde usando la conversación previa, SIN llamar herramientas.
- Si el técnico pide guardar o registrar el caso: llama a TicketWriterTool con el último borrador.
- No inventes soluciones. Usa solo la información que devuelven las herramientas. Si RAGQueryTool devuelve NO_INFO, sigue el plan para casos sin información.
- Los datos personales ya vienen anonimizados ([RUT_OCULTO], [EMAIL_OCULTO], [TELEFONO_OCULTO]). No intentes recuperarlos.

Si es una repregunta, entrega SOLO el texto que pide el técnico (por ejemplo, el mensaje breve para el cliente), sin el formato de secciones.

Cuando respondas una incidencia NUEVA usa este formato:
Prioridad: <ALTA/MEDIA/BAJA>
1. Tipo de Problema:
2. Posible Causa Raíz:
3. Pasos de Solución Recomendados:
4. Fuentes Consultadas: [FUENTE: archivo]
5. Borrador de Respuesta para el Cliente:

Responde siempre en español."""

ORDEN_PRIORIDAD = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}


class AgenteTelios:
    def __init__(self, sesion_id=None, llm=None, verbose=True):
        self.llm = llm or ChatOllama(model=config.MODELO_LLM, temperature=config.TEMPERATURA,
                                     num_predict=config.NUM_PREDICT)
        self.llm_con_tools = self.llm.bind_tools(tools.HERRAMIENTAS_AGENTE)
        self.herramientas = {t.name: t for t in tools.HERRAMIENTAS_AGENTE}
        self.sesion_id = sesion_id or f"sesion_{datetime.now():%Y%m%d_%H%M%S}"
        self.memoria = MemoriaCortoPlazo(self.sesion_id)
        self.verbose = verbose
        self.traza = []  # registro de decisiones del último turno (lo usan las pruebas)
        self.ultima_incidencia = None  # última incidencia clasificada en la sesión (para guardados posteriores)

    # ------------------------------------------------------------------ utilidades
    def _log(self, etiqueta, texto=""):
        self.traza.append((etiqueta, texto))
        if self.verbose:
            texto = texto if len(texto) < 700 else texto[:700] + " [...]"
            print(f"[{etiqueta}] {texto}".rstrip())

    def _normalizar_args(self, herramienta, args):
        """Los modelos pequeños a veces usan otro nombre de argumento. Si la tool
        tiene un solo parámetro obligatorio y llegó un solo valor, se reasigna."""
        params = list(herramienta.args.keys())
        if not isinstance(args, dict):
            args = {params[0]: str(args)}
        if set(args) <= set(params):
            return args
        obligatorios = [p for p in params if p in (herramienta.args_schema.model_json_schema().get("required") or [])]
        if len(obligatorios) == 1 and len(args) == 1:
            return {obligatorios[0]: str(next(iter(args.values())))}
        return {k: v for k, v in args.items() if k in params}

    def _tool_calls_desde_texto(self, contenido):
        """Respaldo: llama3.2 a veces escribe la llamada como JSON en el texto en vez de usar tool_calls."""
        m = re.search(r"\{.*\}", contenido or "", re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        nombre = data.get("name")
        args = data.get("parameters") or data.get("arguments") or data.get("args") or {}
        if nombre in self.herramientas:
            return [{"name": nombre, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}]
        return []

    # ------------------------------------------------------------------ bucle ReAct
    def atender(self, consulta, usar_historial=True):
        self.traza = []
        tools.estado_turno.update({"consulta": "", "prioridad": None, "tickets_guardados": [], "ultimo_ticket": None,
                                   "incidencia_previa": self.ultima_incidencia,
                                   "ultima_respuesta": self.memoria.ultima_respuesta() if usar_historial else ""})

        # Guardrail 1: el modelo nunca ve datos personales.
        limpia = tools.AnonimizerTool.invoke(consulta)
        tools.estado_turno["consulta"] = limpia
        if limpia != consulta:
            self._log("Guardrail", f"AnonimizerTool -> {limpia}")

        historial = self.memoria.mensajes() if usar_historial else []
        if historial:
            self._log("Memoria", f"Se reinyectan {len(historial) // 2} turno(s) previos de la sesión")
        mensajes = [SystemMessage(PROMPT_SISTEMA), *historial, HumanMessage(limpia)]

        self._ejecutadas = {}
        self._usadas = set()
        self._rag = {"sin_info": False, "con_info": False}
        self._fuente = None  # primer fragmento recuperado por RAG en este turno
        respuesta = None

        for iteracion in range(1, config.MAX_ITERACIONES + 1):
            ai = self.llm_con_tools.invoke(mensajes)
            llamadas = ai.tool_calls or self._tool_calls_desde_texto(ai.content)
            if ai.tool_calls == [] and llamadas:
                ai.content, ai.tool_calls = "", llamadas  # normaliza el respaldo

            if not llamadas:
                # Supervisor del plan: el modelo quiere responder, pero ¿cumplió los pasos
                # obligatorios para la prioridad asignada? Si no, se ejecutan antes de aceptar.
                pendientes = self._pasos_pendientes()
                if pendientes:
                    self._log("Supervisor", f"El modelo quiso responder sin completar el plan; faltan: {', '.join(pendientes)}. Se ejecutan.")
                    forzadas = [{"name": n, "args": {"query": limpia}, "id": f"plan_{uuid.uuid4().hex[:8]}"} for n in pendientes]
                    mensajes.append(AIMessage(content="", tool_calls=forzadas))
                    for llamada in forzadas:
                        self._ejecutar(llamada, mensajes)
                else:
                    mensajes.append(ai)
                    respuesta = (ai.content or "").strip()
                    self._log("Decisión", f"Iteración {iteracion}: el modelo responde sin más herramientas")
                    break
            else:
                mensajes.append(ai)
                if ai.content and ai.content.strip():
                    self._log("Thought", ai.content.strip())
                for llamada in llamadas:
                    self._ejecutar(llamada, mensajes)

            # Sin información documentada: no se deja generar al modelo (evita alucinaciones).
            if self._rag["sin_info"] and not self._rag["con_info"]:
                if not tools.estado_turno.get("prioridad"):
                    # El modelo buscó sin clasificar: el supervisor completa ese paso del plan,
                    # porque la decisión (escalar o pedir detalles) depende de la prioridad.
                    prioridad, _ = tools.clasificar(limpia)
                    tools.estado_turno["prioridad"] = prioridad
                    self._log("Supervisor", f"La incidencia no se había clasificado; prioridad asignada: {prioridad}")
                respuesta = tools.respuesta_sin_informacion(tools.estado_turno.get("prioridad"))
                self._log("Decisión", "RAG sin información: se detiene la generación y se usa la respuesta de protocolo")
                break
        else:
            self._log("Límite", f"Se alcanzó el máximo de {config.MAX_ITERACIONES} iteraciones")

        if not respuesta:
            respuesta = ("No pude completar el análisis de forma confiable. "
                         "Se recomienda revisión manual por un técnico de Nivel 2.")

        prioridad = tools.estado_turno.get("prioridad")
        if self._rag["con_info"] and prioridad and self._fuente:
            # Respuesta anclada a la fuente: las secciones técnicas salen del documento
            # recuperado y el modelo solo redacta el mensaje al cliente con esa información.
            respuesta = self._respuesta_anclada(prioridad)
            self._log("Decisión", f"Respuesta estructurada desde {self._fuente['fuente']} (sin agregar información externa)")
        elif tools.estado_turno.get("ultimo_ticket") and not prioridad:
            t = tools.estado_turno["ultimo_ticket"]
            respuesta = (f"Listo. El ticket {t['id']} quedó guardado con estado {t['estado']}.\n"
                         f"Incidencia: {t['consulta']}")

        respuesta = self._guardrail_escalamiento(respuesta, self._rag["sin_info"] and not self._rag["con_info"])
        if tools.estado_turno.get("prioridad"):
            self.ultima_incidencia = {"consulta": limpia, "prioridad": tools.estado_turno["prioridad"]}
        self.memoria.agregar(limpia, respuesta)
        return respuesta

    @staticmethod
    def _separar_secciones(texto):
        """Los documentos siguen el formato 'Problema: causa. Solución: pasos'."""
        tipo, resto = texto.split(":", 1) if ":" in texto else (texto, "")
        if "Solución:" in resto:
            causa, solucion = resto.split("Solución:", 1)
        else:
            causa, solucion = "No especificada en la documentación.", resto
        return tipo.strip(), causa.strip(), solucion.strip()

    def _redactar_mensaje_cliente(self, tipo, causa, solucion):
        prompt = ("Eres un técnico de soporte de Telios (plataforma NubePyme). Escribe en español, en 2 o 3 oraciones, "
                  "el cuerpo de un mensaje para el cliente explicando el problema y cómo solucionarlo.\n"
                  "Usa SOLO esta información, sin agregar nada más:\n"
                  f"Problema: {tipo}\nCausa: {causa}\nSolución: {solucion}\n"
                  "No incluyas saludo, despedida, firma ni corchetes. Responde directamente.")
        for _ in range(2):
            try:
                salida = (self.llm.invoke([HumanMessage(prompt)]).content or "").strip().strip('"')
            except Exception:
                salida = ""
            if salida and not salida.lower().startswith(("lo siento", "no puedo", "i can")):
                return re.sub(r"\[.*?\]", "", salida).split("\n\n")[0].strip()
        return f"Hemos identificado el problema ({tipo}). {solucion}"

    def _respuesta_anclada(self, prioridad):
        tipo, causa, solucion = self._separar_secciones(self._fuente["texto"])
        cuerpo = self._redactar_mensaje_cliente(tipo, causa, solucion)
        return (f"Prioridad: {prioridad}\n"
                f"1. Tipo de Problema: {tipo}\n"
                f"2. Posible Causa Raíz: {causa}\n"
                f"3. Pasos de Solución Recomendados: {solucion}\n"
                f"4. Fuentes Consultadas: [FUENTE: {self._fuente['fuente']}]\n"
                f"5. Borrador de Respuesta para el Cliente:\nEstimado cliente,\n\n{cuerpo}\n\nEquipo de Soporte Telios")

    def _pasos_pendientes(self):
        prioridad = tools.estado_turno.get("prioridad")
        if not prioridad:
            consulta = tools.estado_turno.get("consulta", "")
            es_incidencia = tools.tipo_de_mensaje(consulta) == "INCIDENCIA" and len(consulta.split()) >= 4
            if es_incidencia and "ClasificarPrioridadTool" not in self._usadas:
                return ["ClasificarPrioridadTool"]  # incidencia nueva que el modelo no clasificó
            return []  # repregunta, guardado o mensaje corto: puede responder directo
        return [n for n in tools.PASOS_OBLIGATORIOS[prioridad] if n not in self._usadas]

    def _ejecutar(self, llamada, mensajes):
        nombre, args = llamada["name"], llamada.get("args", {})
        self._log("Action", f"{nombre}({json.dumps(args, ensure_ascii=False)})")
        clave = (nombre, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if nombre not in self.herramientas:
            obs = f"ERROR: la herramienta '{nombre}' no existe. Disponibles: {', '.join(self.herramientas)}."
        elif clave in self._ejecutadas:
            obs = "Ya ejecutaste esta herramienta con los mismos argumentos. Usa la observación anterior y continúa."
        else:
            herramienta = self.herramientas[nombre]
            try:
                obs = herramienta.invoke(self._normalizar_args(herramienta, args))
            except Exception as e:
                obs = f"ERROR: argumentos inválidos para {nombre} ({e}). Corrige los parámetros."
            self._ejecutadas[clave] = obs
            self._usadas.add(nombre)
        obs = str(obs)
        if nombre == "RAGQueryTool":
            if obs.startswith("NO_INFO"):
                self._rag["sin_info"] = True
            elif obs.startswith("[FUENTE"):
                self._rag["con_info"] = True
                if self._fuente is None:
                    primer = obs.split("\n\n")[0]
                    m = re.match(r"\[FUENTE: ([^\]]+)\]", primer)
                    texto = primer.split("\n", 1)[1].strip() if "\n" in primer else ""
                    self._fuente = {"fuente": m.group(1) if m else "desconocida", "texto": texto}
        self._log("Observation", obs)
        mensajes.append(ToolMessage(content=obs, tool_call_id=llamada["id"]))

    def _guardrail_escalamiento(self, respuesta, rag_sin_info):
        """Guardrail 2: una incidencia ALTA o MEDIA sin información documentada
        siempre queda escalada a Nivel 3, aunque el modelo olvide guardarla."""
        prioridad = tools.estado_turno.get("prioridad")
        if rag_sin_info and prioridad in ("ALTA", "MEDIA") and not tools.estado_turno["tickets_guardados"]:
            ticket_id, _ = tools.guardar_ticket(respuesta, "ESCALADO", prioridad)
            self._log("Guardrail", f"Sin información y prioridad {prioridad}: ticket {ticket_id} escalado a Nivel 3")
            respuesta += f"\n\n[Ticket {ticket_id} registrado con estado ESCALADO]"
        return respuesta

    # ------------------------------------------------------------------ cola priorizada
    def procesar_cola(self, incidencias):
        """Recibe varias incidencias, las ordena por prioridad (ALTA > MEDIA > BAJA,
        respetando el orden de llegada en empates) y las atiende en ese orden."""
        clasificadas = []
        for i, texto in enumerate(incidencias):
            limpia = tools.AnonimizerTool.invoke(texto)
            prioridad, motivo = tools.clasificar(limpia)
            clasificadas.append({"orden_llegada": i + 1, "texto": texto, "prioridad": prioridad, "motivo": motivo})
        plan = sorted(clasificadas, key=lambda c: (ORDEN_PRIORIDAD[c["prioridad"]], c["orden_llegada"]))

        if self.verbose:
            print("\n[Plan] Orden de atención según prioridad:")
            for paso, c in enumerate(plan, 1):
                print(f"  {paso}. [{c['prioridad']}] (llegó #{c['orden_llegada']}) {tools.AnonimizerTool.invoke(c['texto'])[:80]}")

        resultados = []
        for paso, c in enumerate(plan, 1):
            if self.verbose:
                print(f"\n{'-' * 60}\n[Cola] Atendiendo {paso}/{len(plan)} - prioridad {c['prioridad']}\n{'-' * 60}")
            respuesta = self.atender(c["texto"], usar_historial=False)
            if self.verbose:
                print(f"\n[Respuesta]\n{respuesta}")
            resultados.append({**c, "respuesta": respuesta, "traza": list(self.traza)})
        return resultados
