import sys
import re
from langchain_ollama import OllamaLLM
from agent.tools import AnonimizerTool, RAGQueryTool, TicketWriterTool

MODELO = "llama3.2:3b"

def es_rechazo(texto):
    inicio = texto.strip().lower()[:120]
    return any(p in inicio for p in ["lo siento", "no puedo", "i can't", "i cannot", "sorry"])

def extraer_primer_documento(contexto):
    for bloque in contexto.split("\n\n"):
        m = re.match(r"\[FUENTE: ([^\]]+)\]", bloque.strip())
        if m:
            texto = bloque.strip().split("\n", 1)[1].strip() if "\n" in bloque.strip() else ""
            return m.group(1), texto
    return None, ""

def separar_secciones(texto):
    if ":" in texto:
        tipo, resto = texto.split(":", 1)
    else:
        tipo, resto = texto, ""
    if "Solución:" in resto:
        causa, solucion = resto.split("Solución:", 1)
    else:
        causa, solucion = "No especificada en el manual.", resto
    return tipo.strip(), causa.strip(), solucion.strip()

def redactar_cuerpo(llm, tipo, causa, solucion):
    prompt = f"""Eres un técnico de soporte de la empresa Telios (plataforma NubePyme). Escribe en español, en 2 o 3 oraciones, el cuerpo de un mensaje para el cliente explicando el problema y cómo solucionarlo.
Usa SOLO esta información, sin agregar nada más:
Problema: {tipo}
Causa: {causa}
Solución: {solucion}
No incluyas saludo, despedida, firma ni corchetes. Responde directamente."""
    for _ in range(3):
        salida = llm.invoke(prompt).strip().strip('"')
        if salida and not es_rechazo(salida):
            salida = re.sub(r"\[.*?\]", "", salida)
            return salida.split("\n\n")[0].strip()
    return f"Hemos identificado el problema ({tipo}). {solucion}"

def ejecutar_agente_telios():
    print("="*60)
    print("      TELIOS - AGENTE FUNCIONAL DE SOPORTE NUBEPYME")
    print("="*60)

    try:
        llm = OllamaLLM(model=MODELO, temperature=0.0)
    except Exception as e:
        print(f"[ERROR] No se pudo conectar con Ollama: {e}")
        sys.exit(1)

    print("\n[SISTEMA OK] Agente Telios iniciado. Escribe 'salir' para finalizar.\n")

    while True:
        consulta = input("\nTécnico de Soporte > ")
        if consulta.lower() in ["salir", "exit"]:
            print("\nFinalizando sesión del Agente Telios. ¡Hasta luego!")
            break
        if not consulta.strip():
            continue

        print("\n> [Agente Telios]: Iniciando flujo de razonamiento y ejecución de herramientas...")

        print("\n[Thought]: Sanitizando datos personales sensibles...")
        consulta_sanitizada = AnonimizerTool.invoke(consulta)
        print(f"[Action: AnonimizerTool]\n[Observation]: {consulta_sanitizada}")

        print("\n[Thought]: Consultando base de conocimiento...")
        contexto = RAGQueryTool.invoke(consulta_sanitizada)

        if contexto.startswith("NO_INFO"):
            print("[Action: RAGQueryTool]\n[Observation]: Sin contexto suficiente en la base de conocimiento.")
            print("\n[Thought]: Sin contexto suficiente; se escala el caso sin invocar al modelo.")
            respuesta = ("1. Tipo de Problema: Incidencia sin cobertura en la base de conocimiento.\n"
                         "2. Posible Causa Raíz: No determinada; no hay información documentada.\n"
                         "3. Pasos de Solución Recomendados: Escalar el caso a Soporte Nivel 3.\n"
                         "4. Fuentes Consultadas: Sin información en la base de conocimiento; se recomienda escalar el caso.\n"
                         "5. Borrador de Respuesta para el Cliente:\nEstimado cliente,\n\nHemos recibido su solicitud y la estamos derivando a un especialista. Le contactaremos a la brevedad.\n\nEquipo de Soporte Telios")
        else:
            print("[Action: RAGQueryTool]\n[Observation]: Contexto recuperado:\n" + contexto)
            fuente, texto = extraer_primer_documento(contexto)
            tipo, causa, solucion = separar_secciones(texto)
            print("\n[Thought]: Redactando mensaje al cliente con el modelo local...")
            cuerpo = redactar_cuerpo(llm, tipo, causa, solucion)
            respuesta = (f"1. Tipo de Problema: {tipo}\n"
                         f"2. Posible Causa Raíz: {causa}\n"
                         f"3. Pasos de Solución Recomendados: {solucion}\n"
                         f"4. Fuentes Consultadas: [FUENTE: {fuente}]\n"
                         f"5. Borrador de Respuesta para el Cliente:\nEstimado cliente,\n\n{cuerpo}\n\nEquipo de Soporte Telios")

        print(f"\n[Resultado del Agente Telios]:\n{respuesta}")

        print("\n[Thought]: Registrando borrador en el sistema...")
        print(f"[Action: TicketWriterTool]\n[Observation]: {TicketWriterTool.invoke(respuesta)}")

if __name__ == "__main__":
    ejecutar_agente_telios()
