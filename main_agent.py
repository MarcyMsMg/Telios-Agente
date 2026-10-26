"""Punto de entrada del Agente Telios (CLI interactiva).

Uso:
  python main_agent.py                       # nueva sesión
  python main_agent.py --sesion ID           # retoma una sesión guardada (memoria de corto plazo)
  python main_agent.py --cola archivo.json   # atiende varias incidencias ordenadas por prioridad

Comandos dentro de la CLI:
  memoria   muestra los turnos que el agente recuerda en esta sesión
  cola      atiende la cola de ejemplo tests/cola_incidencias.json
  salir     termina la sesión
"""
import argparse
import json
import sys

from agent import config, memory
from agent.core import AgenteTelios

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # tildes correctas en la consola de Windows


def cargar_cola(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Agente Telios - soporte NubePyme")
    parser.add_argument("--sesion", help="ID de una sesión anterior para retomarla")
    parser.add_argument("--cola", help="Archivo JSON con una lista de incidencias")
    args = parser.parse_args()

    print("=" * 60)
    print("      TELIOS - AGENTE FUNCIONAL DE SOPORTE NUBEPYME")
    print("=" * 60)

    try:
        n_kb = memory.sincronizar_base_conocimiento()
        n_mem = memory.sincronizar_tickets_previos()
        agente = AgenteTelios(sesion_id=args.sesion)
    except Exception as e:
        print(f"[ERROR] No se pudo iniciar el agente (¿está Ollama corriendo con {config.MODELO_LLM} "
              f"y {config.EMBED_MODEL}?): {e}")
        sys.exit(1)

    print(f"\n[SISTEMA OK] Modelo: {config.MODELO_LLM} | Base de conocimiento: {n_kb} fragmentos | "
          f"Casos en memoria de largo plazo: {n_mem}")
    print(f"[SESIÓN] {agente.sesion_id} ({len(agente.memoria.turnos)} turnos recordados)")

    if args.cola:
        agente.procesar_cola(cargar_cola(args.cola))
        return

    print("Escribe 'memoria', 'cola' o 'salir'.\n")
    while True:
        try:
            consulta = input("\nTécnico de Soporte > ").strip()
        except (EOFError, KeyboardInterrupt):
            consulta = "salir"
        if not consulta:
            continue
        if consulta.lower() in ("salir", "exit"):
            print(f"\nSesión guardada como '{agente.sesion_id}'. Para retomarla: "
                  f"python main_agent.py --sesion {agente.sesion_id}")
            break
        if consulta.lower() == "memoria":
            for i, t in enumerate(agente.memoria.turnos[-config.VENTANA_K:], 1):
                print(f"  {i}. Técnico: {t['tecnico'][:80]}\n     Agente: {t['agente'][:80]}")
            continue
        if consulta.lower() == "cola":
            agente.procesar_cola(cargar_cola(config.BASE_DIR / "tests" / "cola_incidencias.json"))
            continue

        print("\n> [Agente Telios]: razonando...")
        respuesta = agente.atender(consulta)
        print(f"\n[Resultado del Agente Telios]:\n{respuesta}")


if __name__ == "__main__":
    main()
