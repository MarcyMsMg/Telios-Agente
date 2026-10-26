"""Diagrama de orquestación del Agente Telios, con el estilo del diagrama original del equipo."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams["font.family"] = "Liberation Sans"

W, H = 1500, 615
fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

GRIS, MORADO, VERDE, BORDE = "#f3f3f3", "#b4a7d6", "#b6d7a8", "#666666"
BW, BH = 270, 68
A, B, C, D = 165, 545, 925, 1290          # columnas (centros)
R1, R2, R3, R4, R5 = 55, 175, 300, 425, 560  # filas (centros)

cajas = {}


def caja(clave, x, y, titulo, sub, color=GRIS, w=BW, h=BH):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor=BORDE, lw=1.2, zorder=2))
    ax.text(x, y - 11, titulo, ha="center", va="center", fontsize=12.5, fontweight="bold", zorder=3)
    ax.text(x, y + 13, sub, ha="center", va="center", fontsize=11.5, zorder=3)
    cajas[clave] = (x, y, w, h)


def borde(clave, lado, desp=0):
    x, y, w, h = cajas[clave]
    return {"izq": (x - w / 2, y + desp), "der": (x + w / 2, y + desp),
            "arr": (x + desp, y - h / 2), "abj": (x + desp, y + h / 2)}[lado]


def flecha(puntos, etiqueta=None, pos=None, ha="center", va="center", estilo="-|>", ls="-"):
    for p, q in zip(puntos[:-2], puntos[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="black", lw=3, solid_capstyle="butt", ls=ls, zorder=1)
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle=f"{estilo},head_length=9,head_width=7",
                                 color="black", lw=3, shrinkA=0, shrinkB=1, mutation_scale=1, ls=ls, zorder=1))
    if etiqueta:
        ax.text(*pos, etiqueta, ha=ha, va=va, fontsize=11, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


# ---------------------------------------------------------------- leyenda
ax.add_patch(Rectangle((22, 22), 26, 22, facecolor=MORADO, edgecolor=BORDE))
ax.text(58, 33, "Herramienta", va="center", fontsize=12)
ax.add_patch(Rectangle((22, 56), 26, 22, facecolor=VERDE, edgecolor=BORDE))
ax.text(58, 67, "Memoria / datos", va="center", fontsize=12)

# ---------------------------------------------------------------- cajas
caja("tec", B, R1, "Técnico de soporte", "Ingresa ticket o consulta")
caja("nube", C, R1, "NubePyme", "Sistema de soporte (tickets)")
caja("anon", A, R2, "AnonimizerTool", "Oculta RUT, correo, teléfono", MORADO)
caja("agente", B, R2, "Agente orquestador", "LangChain · ciclo ReAct")
caja("tw", C, R2, "TicketWriterTool", "Escribe ticket JSON", MORADO)
caja("ms", A, R3, "Memoria de sesión", "Ventana de 5 turnos", VERDE)
caja("eval", B, R3, "Evalúa herramienta", "requerida (máx. 6 pasos)")
caja("rag", C, R3, "RAGQueryTool", "Distancia coseno ≤ 0,30", MORADO)
caja("kb", D, R3, "Base de conocimiento", "ChromaDB · nomic-embed-text", VERDE)
caja("cp", A, R4, "ClasificarPrioridadTool", "ALTA / MEDIA / BAJA + plan", MORADO)
caja("sup", B, R4, "Supervisor del plan", "¿Pasos obligatorios hechos?")
caja("bc", C, R4, "BuscarCasosPreviosTool", "Incidencias ya atendidas", MORADO)
caja("mc", D, R4, "Memoria de casos", "ChromaDB · telios_memoria", VERDE)
caja("llm", B, R5, "LLM local", "llama3.2:3b vía Ollama")

# ---------------------------------------------------------------- flechas
# técnico <-> agente
flecha([borde("tec", "abj", -45), borde("agente", "arr", -45)], "Ticket", (B - 45, (R1 + R2) / 2), ha="right")
flecha([borde("agente", "arr", 45), borde("tec", "abj", 45)], "Resultado", (B + 45, (R1 + R2) / 2), ha="left")
# guardrail y memoria de sesión
flecha([borde("agente", "izq"), borde("anon", "der")], "Primero", ((A + B) / 2, R2 - 14))
flecha([borde("anon", "abj"), borde("ms", "arr")])
flecha([borde("ms", "der"), borde("eval", "izq")], "Historial", ((A + B) / 2, R3 - 14))
# agente -> evalúa
flecha([borde("agente", "abj"), borde("eval", "arr")], "Thought", (B + 8, (R2 + R3) / 2), ha="left")
# escritura
flecha([borde("agente", "der"), borde("tw", "izq")], "Guardar", ((B + C) / 2, R2 - 14))
flecha([borde("tw", "arr"), borde("nube", "abj")])
xr = D + BW / 2 + 28
flecha([borde("tw", "der"), (xr, R2), (xr, R4), borde("mc", "der")], "Guarda caso", (xr - 8, R2 + 22), ha="right")
# consulta RAG
flecha([borde("eval", "der", -12), borde("rag", "izq", -12)], "Búsqueda", ((B + C) / 2, R3 - 26))
flecha([borde("rag", "izq", 14), borde("eval", "der", 14)], "Observación", ((B + C) / 2, R3 + 28))
flecha([borde("rag", "der", -12), borde("kb", "izq", -12)], "Consulta", ((C + D) / 2, R3 - 26))
flecha([borde("kb", "izq", 14), borde("rag", "der", 14)], "Chunks", ((C + D) / 2, R3 + 28))
# casos previos
flecha([borde("bc", "der", -12), borde("mc", "izq", -12)], "Consulta", ((C + D) / 2, R4 - 26))
flecha([borde("mc", "izq", 14), borde("bc", "der", 14)], "Casos", ((C + D) / 2, R4 + 28))
yc = (R3 + R4) / 2 + 2  # pasillo entre filas 3 y 4
flecha([borde("eval", "abj", -95), (B - 95, yc), (A, yc), borde("cp", "arr")], "Prioridad y plan", ((A + B) / 2 - 20, yc - 13))
flecha([borde("eval", "abj", 95), (B + 95, yc), (C, yc), borde("bc", "arr")], "Casos previos", ((B + C) / 2 + 20, yc - 13))
# supervisor y LLM
flecha([borde("eval", "abj", -20), borde("sup", "arr", -20)])
flecha([borde("sup", "arr", 25), borde("eval", "abj", 25)], estilo="-|>", ls=(0, (3, 2)))
ax.text(B + 33, yc + 14, "Faltan pasos", ha="left", va="center", fontsize=10, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))
flecha([borde("sup", "abj"), borde("llm", "arr")], "Genera propuesta", (B + 8, (R4 + R5) / 2), ha="left")
fig.savefig(__import__("sys").argv[1], dpi=200, facecolor="white")
print("ok")
