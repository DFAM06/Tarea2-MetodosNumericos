"""Tarea II - Métodos Numéricos y Análisis Computacional.
Herramienta de programación lineal con simplex (M simbólica) y método gráfico.
Ejecución: streamlit run tarea_2_programacion_lineal.py
"""

# -- Parte 1 - Bibliotecas, constantes y ejemplos para comprobar el programa

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-9
TOL_GRAFICA = 1e-7
MAX_ITERACIONES = 200

EJEMPLOS = {
    "1. Maximización (sin artificiales)": {
        "tipo": "Maximizar", "c": [3.0, 5.0],
        "restricciones": [(1.0, 0.0, "≤", 4.0),
                          (0.0, 2.0, "≤", 12.0),
                          (3.0, 2.0, "≤", 18.0)],
    },
    "2. Minimización (con artificiales)": {
        "tipo": "Minimizar", "c": [2.0, 3.0],
        "restricciones": [(1.0, 1.0, "≥", 4.0),
                          (1.0, 2.0, "≥", 6.0),
                          (1.0, 1.0, "≤", 8.0)],
    },
    "3. Maximización con igualdad (artificial)": {
        "tipo": "Maximizar", "c": [3.0, 2.0],
        "restricciones": [(1.0, 1.0, "=", 4.0),
                          (1.0, 0.0, "≤", 3.0),
                          (0.0, 1.0, "≤", 3.0)],
    },
    "4. Problema infactible": {
        "tipo": "Maximizar", "c": [1.0, 1.0],
        "restricciones": [(1.0, 1.0, "≤", 1.0),
                          (1.0, 1.0, "≥", 4.0),
                          (1.0, 0.0, "≤", 5.0)],
    },
    "5. Problema no acotado": {
        "tipo": "Maximizar", "c": [1.0, 1.0],
        "restricciones": [(1.0, -1.0, "≥", 0.0),
                          (0.0, 1.0, "≥", 1.0),
                          (1.0, 1.0, "≥", 1.0)],
    },
}


# -- Parte 2 - Formato de números, expresiones y modelos matemáticos

def numero(valor):
    """Presenta números pequeños como cero y evita cifras innecesarias."""
    if abs(valor) < EPS:
        return "0"
    return f"{valor:.7g}"


def expresion(terminos):
    """Convierte pares (coeficiente, nombre) en una expresión legible."""
    partes = []
    for coef, nombre in terminos:
        if abs(coef) < EPS:
            continue
        magnitud = "" if abs(abs(coef) - 1.0) < EPS else numero(abs(coef))
        termino = f"{magnitud}{nombre}"
        if not partes:
            partes.append(("-" if coef < 0 else "") + termino)
        else:
            partes.append((" - " if coef < 0 else " + ") + termino)
    return "".join(partes) if partes else "0"


def par_m(m, constante):
    """Representa m*M + constante, sin asignar un valor numérico a M."""
    m = 0.0 if abs(m) < EPS else m
    constante = 0.0 if abs(constante) < EPS else constante
    if m == 0:
        return numero(constante)
    signo = "-" if m < 0 else ""
    magnitud = "" if abs(abs(m) - 1.0) < EPS else numero(abs(m))
    texto = f"{signo}{magnitud}M"
    if constante > 0:
        texto += f" + {numero(constante)}"
    elif constante < 0:
        texto += f" - {numero(-constante)}"
    return texto


def texto_modelo(tipo, c, restricciones):
    objetivo = expresion(zip(c, ("x1", "x2")))
    lineas = [f"{tipo.upper()} Z = {objetivo}", "Sujeto a:"]
    for a1, a2, signo, b in restricciones:
        lineas.append(f"  {expresion([(a1, 'x1'), (a2, 'x2')])} {signo} {numero(b)}")
    lineas.append("  x1 ≥ 0,  x2 ≥ 0")
    return "\n".join(lineas)


# -- Parte 3 - Conversión a ecuaciones con holguras, superávit y artificiales

def transformar(c, restricciones, tipo):
    """Construye Ax=b, base inicial y costos simbólicos Cj=m*M+k."""
    if len(c) != 2 or len(restricciones) != 3:
        raise ValueError("Se requieren exactamente 2 variables y 3 restricciones.")

    filas = [[float(r[0]), float(r[1])] for r in restricciones]
    rhs = [float(r[3]) for r in restricciones]
    signos = [r[2] for r in restricciones]
    nombres = ["x1", "x2"]
    base = [-1, -1, -1]
    artificiales = []
    cambios = []
    if tipo not in ("Maximizar", "Minimizar"):
        raise ValueError("El tipo debe ser Maximizar o Minimizar.")

    for i in range(3):
        if signos[i] not in ("≤", "≥", "="):
            raise ValueError("Cada restricción debe usar ≤, ≥ o =.")
        if rhs[i] < -EPS:
            rhs[i] *= -1
            filas[i][0] *= -1
            filas[i][1] *= -1
            signos[i] = {"≤": "≥", "≥": "≤", "=": "="}[signos[i]]
            cambios.append(f"R{i + 1}: se multiplicó por -1 para obtener el lado derecho no negativo.")
        if abs(rhs[i]) <= EPS:
            rhs[i] = 0.0

    def columna(nombre, fila, valor):
        pos = len(nombres)
        nombres.append(nombre)
        for j in range(3):
            filas[j].append(float(valor) if j == fila else 0.0)
        return pos

    for i, signo in enumerate(signos):
        if signo == "≤":
            base[i] = columna(f"h{i + 1}", i, 1.0)
        elif signo == "≥":
            columna(f"e{i + 1}", i, -1.0)
            base[i] = columna(f"a{i + 1}", i, 1.0)
            artificiales.append(base[i])
        else:
            base[i] = columna(f"a{i + 1}", i, 1.0)
            artificiales.append(base[i])

    ecuaciones = [
        f"R{i + 1}: {expresion(zip(filas[i], nombres))} = {numero(rhs[i])}"
        for i in range(3)
    ]

    # La minimización se convierte en maximización de -Z.
    cambio_signo = 1.0 if tipo == "Maximizar" else -1.0
    costos_m = np.zeros(len(nombres), dtype=float)
    costos_c = np.zeros(len(nombres), dtype=float)
    costos_c[:2] = cambio_signo * np.array(c, dtype=float)
    for k in artificiales:
        costos_m[k] = -1.0  # M simbólica positiva: penalización -M.

    return {
        "A": np.array(filas, dtype=float), "b": np.array(rhs, dtype=float),
        "base": base, "nombres": nombres, "artificiales": artificiales,
        "cm": costos_m, "cc": costos_c,
        "ecuaciones": ecuaciones, "cambios": cambios,
    }


# -- Parte 4 - Cálculo de Cj-Zj, preparación y visualización de la tabla simplex

def reducidos(A, b, base, cm, cc):
    """Calcula Cj - Zj con coeficientes de M y términos independientes."""
    cbm = cm[base]
    cbc = cc[base]
    return (cm - cbm @ A, cc - cbc @ A,
            float(cbm @ b), float(cbc @ b))


def tabla_simplex(A, b, base, nombres, cm, cc):
    rm, rc, zm, zc = reducidos(A, b, base, cm, cc)
    filas = []
    for i in range(len(base)):
        fila = {
            "Base": nombres[base[i]],
            "C_B": par_m(cm[base[i]], cc[base[i]]),
            "LD": numero(b[i]),
        }
        fila.update({nombre: numero(A[i, j]) for j, nombre in enumerate(nombres)})
        filas.append(fila)
    ultima = {"Base": "Cj - Zj", "C_B": "", "LD": "Z = " + par_m(zm, zc)}
    ultima.update({nombre: par_m(rm[j], rc[j]) for j, nombre in enumerate(nombres)})
    filas.append(ultima)
    return filas


# -- Parte 5 - Algoritmo simplex con método de la M y pivoteo Gauss-Jordan

def resolver_simplex(c, restricciones, tipo):
    """Resuelve sin llamar a optimizadores externos: M simbólica, pivotes y razón mínima."""
    modelo = transformar(c, restricciones, tipo)
    A, b = modelo["A"].copy(), modelo["b"].copy()
    base = list(modelo["base"])
    cm, cc = modelo["cm"], modelo["cc"]
    nombres, artificiales = modelo["nombres"], modelo["artificiales"]
    inicial = tabla_simplex(A, b, base, nombres, cm, cc)
    pasos = []
    estado = ""
    explicacion = ""

    for it in range(1, MAX_ITERACIONES + 1):
        rm, rc, _, _ = reducidos(A, b, base, cm, cc)
        suma_artificial = sum(max(0.0, b[i]) for i in range(3) if base[i] in artificiales)
        no_basicas = [j for j in range(A.shape[1]) if j not in base]

        # Si aún existen artificiales positivas, se prioriza estrictamente
        # disminuir su suma (coeficiente de M); M nunca es un número arbitrario.
        if suma_artificial > 1e-7:
            posibles = [j for j in no_basicas if rm[j] > EPS]
            if not posibles:
                estado = "Infactible"
                explicacion = "No es posible llevar a cero todas las variables artificiales."
                break
        else:
            posibles = [j for j in no_basicas
                        if rm[j] > EPS or (abs(rm[j]) <= EPS and rc[j] > EPS)]
            if not posibles:
                estado = "Óptimo"
                explicacion = "No quedan costos reducidos positivos; se alcanzó el óptimo."
                break

        # Regla de Bland: primera columna elegible; previene ciclos por degeneración.
        entrada = min(posibles)
        razones = []
        for i in range(3):
            if A[i, entrada] > EPS:
                razones.append(max(0.0, float(b[i])) / A[i, entrada])
            else:
                razones.append(None)

        candidatas = [i for i, r in enumerate(razones) if r is not None]
        if not candidatas:
            estado = "No acotado" if suma_artificial <= 1e-7 else "Infactible"
            explicacion = ("Existe una dirección que mejora indefinidamente el objetivo."
                           if estado == "No acotado" else
                           "No puede eliminarse una variable artificial básica positiva.")
            break

        minimo = min(razones[i] for i in candidatas)
        empates = [i for i in candidatas if abs(razones[i] - minimo) <= 1e-8]
        salida_fila = min(empates, key=lambda i: (base[i], i))
        sale = base[salida_fila]
        pivote = float(A[salida_fila, entrada])
        tabla_antes = tabla_simplex(A, b, base, nombres, cm, cc)

        # Operaciones elementales de Gauss-Jordan, implementadas explícitamente.
        operaciones = [f"F{salida_fila + 1} ← F{salida_fila + 1} / ({numero(pivote)})"]
        A[salida_fila, :] /= pivote
        b[salida_fila] /= pivote
        for i in range(3):
            if i == salida_fila:
                continue
            factor = float(A[i, entrada])
            if abs(factor) > EPS:
                A[i, :] -= factor * A[salida_fila, :]
                b[i] -= factor * b[salida_fila]
                operaciones.append(
                    f"F{i + 1} ← F{i + 1} - ({numero(factor)}) F{salida_fila + 1}"
                )
        A[np.abs(A) < EPS] = 0.0
        b[np.abs(b) < EPS] = 0.0
        base[salida_fila] = entrada

        pasos.append({
            "iteracion": it, "entra": nombres[entrada], "sale": nombres[sale],
            "pivote": pivote, "fila": salida_fila + 1,
            "razones": razones, "bases_antes": [fila["Base"] for fila in tabla_antes[:-1]],
            "columna_entrada": [fila[nombres[entrada]] for fila in tabla_antes[:-1]],
            "tabla_antes": tabla_antes, "tabla_despues": tabla_simplex(A, b, base, nombres, cm, cc),
            "operaciones": operaciones,
        })
    else:
        estado = "Límite de iteraciones"
        explicacion = "Se alcanzó el límite de seguridad de iteraciones."

    solucion = np.zeros(len(nombres), dtype=float)
    for i, j in enumerate(base):
        solucion[j] = float(b[i])
    solucion[np.abs(solucion) < EPS] = 0.0

    # Verificación adicional: ninguna artificial debe quedar positiva.
    if estado == "Óptimo" and any(solucion[j] > 1e-7 for j in artificiales):
        estado = "Infactible"
        explicacion = "Al finalizar quedó una variable artificial positiva."

    return {
        "estado": estado, "explicacion": explicacion,
        "modelo": modelo, "inicial": inicial, "pasos": pasos,
        "solucion": solucion, "base_final": base,
        "x": solucion[:2] if estado == "Óptimo" else None,
        "z": float(np.dot(c, solucion[:2])) if estado == "Óptimo" else None,
    }


# -- Parte 6 - Método gráfico: intersecciones, factibilidad y vértices

def es_factible(x, restricciones, tol=TOL_GRAFICA):
    if min(x) < -tol:
        return False
    for a1, a2, signo, b in restricciones:
        lhs = a1 * x[0] + a2 * x[1]
        margen = tol * max(1.0, abs(lhs), abs(b))
        if signo == "≤" and lhs > b + margen:
            return False
        if signo == "≥" and lhs < b - margen:
            return False
        if signo == "=" and abs(lhs - b) > margen:
            return False
    return True


def vertices_factibles(restricciones):
    # Se agregan x1=0 y x2=0 como fronteras por no negatividad.
    fronteras = [(float(r[0]), float(r[1]), float(r[3])) for r in restricciones]
    fronteras.extend([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    vertices = []
    for primera, segunda in itertools.combinations(fronteras, 2):
        matriz = np.array([[primera[0], primera[1]],
                           [segunda[0], segunda[1]]], dtype=float)
        if abs(float(np.linalg.det(matriz))) <= EPS:
            continue
        vector = np.array([primera[2], segunda[2]], dtype=float)
        punto = np.linalg.solve(matriz, vector)
        if not np.all(np.isfinite(punto)) or not es_factible(punto, restricciones):
            continue
        punto[np.abs(punto) < EPS] = 0.0
        if not any(np.linalg.norm(punto - v, ord=np.inf) < 1e-6 for v in vertices):
            vertices.append(punto)
    return sorted(vertices, key=lambda v: (v[0], v[1]))


def resultado_grafico(c, restricciones, estado):
    vertices = vertices_factibles(restricciones)
    tabla = []
    for v in vertices:
        tabla.append({"x1": float(v[0]), "x2": float(v[1]),
                      "Z": float(np.dot(c, v))})
    if estado != "Óptimo" or not tabla:
        return vertices, tabla, []
    valores = [fila["Z"] for fila in tabla]
    # El óptimo se coteja con simplex posteriormente; el vértice puede empatar.
    return vertices, tabla, valores


# -- Parte 7 - Dibujo de las rectas, región factible y solución óptima

def dibujar_grafico(c, restricciones, resultado, vertices):
    valores_escala = [5.0]
    for v in vertices:
        valores_escala.extend(float(t) for t in v if 0 <= t < 1e9)
    for a1, a2, _, b in restricciones:
        for a in (a1, a2):
            if abs(a) > EPS:
                corte = b / a
                if 0 < corte < 1e9:
                    valores_escala.append(corte)
    escala = max(valores_escala)
    maximo = max(5.0, 1.20 * escala)

    xs = np.linspace(0.0, maximo, 320)
    ys = np.linspace(0.0, maximo, 320)
    X, Y = np.meshgrid(xs, ys)
    mascara = np.ones_like(X, dtype=bool)
    for a1, a2, signo, b in restricciones:
        lhs = a1 * X + a2 * Y
        tol = 1e-7 * max(1.0, abs(b))
        if signo == "≤":
            mascara &= lhs <= b + tol
        elif signo == "≥":
            mascara &= lhs >= b - tol
        else:
            mascara &= np.abs(lhs - b) <= tol

    fig, ax = plt.subplots(figsize=(7, 5))
    if np.any(mascara):
        ax.contourf(X, Y, mascara.astype(float), levels=[0.5, 1.5],
                    colors=["#83d0b6"], alpha=0.46)

    colores = ["#3478c0", "#d47731", "#8c4ac0"]
    for i, (a1, a2, signo, b) in enumerate(restricciones):
        nombre = f"R{i + 1}: {expresion([(a1, 'x1'), (a2, 'x2')])} {signo} {numero(b)}"
        if abs(a2) > EPS:
            ax.plot(xs, (b - a1 * xs) / a2, color=colores[i], lw=2, label=nombre)
        elif abs(a1) > EPS:
            ax.axvline(b / a1, color=colores[i], lw=2, label=nombre)
        else:
            # 0 ≤ b, 0 ≥ b o 0 = b: no define una recta.
            continue

    # Con igualdades, la región factible puede ser un segmento (área cero).
    # Se destaca la porción factible de la recta para que siga siendo visible.
    hay_segmento = False
    for a1, a2, signo, limite in restricciones:
        if signo != "=" or (abs(a1) < EPS and abs(a2) < EPS):
            continue
        coordenadas = np.linspace(0.0, maximo, 1200)
        if abs(a2) > EPS:
            xe = coordenadas
            ye = (limite - a1 * xe) / a2
        else:
            ye = coordenadas
            xe = np.full_like(ye, limite / a1)
        valido = np.array([
            (0 <= xx <= maximo and 0 <= yy <= maximo and
             es_factible((xx, yy), restricciones))
            for xx, yy in zip(xe, ye)
        ])
        if np.any(valido):
            ax.plot(np.where(valido, xe, np.nan),
                    np.where(valido, ye, np.nan),
                    color="#17856b", linewidth=5, alpha=0.85,
                    label="Región factible (segmento)" if not hay_segmento else None)
            hay_segmento = True

    for i, v in enumerate(vertices):
        ax.scatter(v[0], v[1], c="#246d65", s=45, zorder=5,
                   label="Vértices factibles" if i == 0 else None)
        ax.annotate(f"({numero(v[0])}, {numero(v[1])})", v,
                    xytext=(6, 7), textcoords="offset points", fontsize=9)

    if resultado["estado"] == "Óptimo":
        x1, x2 = resultado["x"]
        ax.scatter([x1], [x2], marker="*", c="#c72431", s=280,
                   zorder=10, label=f"Óptimo: Z={numero(resultado['z'])}")
        if abs(c[1]) > EPS:
            ax.plot(xs, (resultado["z"] - c[0] * xs) / c[1], "--",
                    color="#c72431", alpha=0.65, label="Recta objetivo óptima")
        elif abs(c[0]) > EPS:
            ax.axvline(resultado["z"] / c[0], ls="--", color="#c72431",
                       alpha=0.65, label="Recta objetivo óptima")

    ax.set_xlim(0, maximo)
    ax.set_ylim(0, maximo)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Método gráfico: restricciones y región factible")
    ax.grid(alpha=0.24)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# -- Parte 8 - Resultados, holguras, superávits e interpretación matemática

def informe_restricciones(x, restricciones):
    datos = []
    for i, (a1, a2, signo, b) in enumerate(restricciones, 1):
        lhs = a1 * x[0] + a2 * x[1]
        if signo == "≤":
            nombre = f"Holgura h{i}"
            cantidad = b - lhs
        elif signo == "≥":
            nombre = f"Superávit e{i}"
            cantidad = lhs - b
        else:
            nombre = f"Residuo de igualdad R{i}"
            cantidad = lhs - b
        if abs(cantidad) < 1e-7:
            cantidad = 0.0
        datos.append({"Restricción": f"R{i} ({signo})", "Variable/medida": nombre,
                      "Valor": float(cantidad)})
    return datos


def interpretacion(resultado):
    estado = resultado["estado"]
    if estado == "Óptimo":
        x1, x2 = resultado["x"]
        return (f"La solución óptima es x1 = {numero(x1)}, x2 = {numero(x2)} "
                f"y Z = {numero(resultado['z'])}. Se cumplen las tres restricciones "
                "y las condiciones de no negatividad.")
    if estado == "Infactible":
        return ("No hay ninguna combinación de x1 y x2 no negativas que cumpla "
                "simultáneamente las tres restricciones. No existe solución factible.")
    if estado == "No acotado":
        return ("La región es factible, pero la función objetivo puede mejorar "
                "indefinidamente en alguna dirección factible; no existe valor óptimo finito.")
    return resultado["explicacion"]


# -- Parte 9 - Interfaz interactiva para capturar los datos y ejecutar los métodos

def iniciar_interfaz():
    import streamlit as st

    st.set_page_config(page_title="Tarea II - Programación lineal", layout="wide")
    st.title("Programación lineal: simplex y método gráfico")
    st.caption("IF0017 · Tarea II · 2 variables de decisión y exactamente 3 restricciones")
    st.write("Ingrese el problema, resuélvalo y revise las transformaciones, "
             "todas las iteraciones simplex y su representación gráfica.")

    primero = next(iter(EJEMPLOS.values()))
    st.session_state.setdefault("tipo", primero["tipo"])
    st.session_state.setdefault("c1", primero["c"][0])
    st.session_state.setdefault("c2", primero["c"][1])
    for i, (a1, a2, signo, b) in enumerate(primero["restricciones"], 1):
        for sufijo, valor in (("a1", a1), ("a2", a2), ("signo", signo), ("b", b)):
            st.session_state.setdefault(f"r{i}_{sufijo}", valor)

    st.subheader("1. Ingreso de datos y formulación")
    elegir, cargar = st.columns([3, 1])
    with elegir:
        ejemplo = st.selectbox("Ejemplos para las pruebas", list(EJEMPLOS.keys()))
    with cargar:
        st.write("")
        st.write("")
        if st.button("Cargar ejemplo", use_container_width=True):
            datos = EJEMPLOS[ejemplo]
            st.session_state["tipo"] = datos["tipo"]
            st.session_state["c1"], st.session_state["c2"] = datos["c"]
            for i, (a1, a2, signo, b) in enumerate(datos["restricciones"], 1):
                for sufijo, valor in (("a1", a1), ("a2", a2), ("signo", signo), ("b", b)):
                    st.session_state[f"r{i}_{sufijo}"] = valor
            st.session_state.pop("resultado_guardado", None)
            st.rerun()

    with st.form("formulario_modelo"):
        st.radio("Objetivo", ["Maximizar", "Minimizar"], horizontal=True, key="tipo")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Coeficiente de x1 en Z", key="c1", step=1.0, format="%.4f")
        with col2:
            st.number_input("Coeficiente de x2 en Z", key="c2", step=1.0, format="%.4f")
        st.markdown("**Restricciones** (pueden combinarse ≤, ≥ y =):")
        for i in range(1, 4):
            st.markdown(f"**Restricción {i}**")
            cols = st.columns([2, 2, 1.4, 2])
            with cols[0]:
                st.number_input(f"Coef. de x1 · R{i}", key=f"r{i}_a1",
                                step=1.0, format="%.4f")
            with cols[1]:
                st.number_input(f"Coef. de x2 · R{i}", key=f"r{i}_a2",
                                step=1.0, format="%.4f")
            with cols[2]:
                st.selectbox(f"Signo · R{i}", ["≤", "≥", "="], key=f"r{i}_signo")
            with cols[3]:
                st.number_input(f"Lado derecho · R{i}", key=f"r{i}_b",
                                step=1.0, format="%.4f")
        st.caption("Se consideran automáticamente x1 ≥ 0 y x2 ≥ 0.")
        resolver = st.form_submit_button("Resolver con simplex y método gráfico", type="primary")

    if resolver:
        tipo = st.session_state["tipo"]
        c = [float(st.session_state["c1"]), float(st.session_state["c2"])]
        restricciones = [
            (float(st.session_state[f"r{i}_a1"]),
             float(st.session_state[f"r{i}_a2"]),
             st.session_state[f"r{i}_signo"],
             float(st.session_state[f"r{i}_b"]))
            for i in range(1, 4)
        ]
        try:
            resultado = resolver_simplex(c, restricciones, tipo)
            st.session_state["resultado_guardado"] = (tipo, c, restricciones, resultado)
        except (ValueError, OverflowError, np.linalg.LinAlgError) as ex:
            st.error(f"No se pudo resolver el modelo: {ex}")
            st.stop()

    if "resultado_guardado" not in st.session_state:
        st.info("Seleccione un ejemplo o introduzca sus datos y presione «Resolver».")
        return

    tipo, c, restricciones, resultado = st.session_state["resultado_guardado"]
    modelo = resultado["modelo"]
    st.code(texto_modelo(tipo, c, restricciones), language="text")

    st.subheader("2. Transformación y método simplex")
    if tipo == "Minimizar":
        st.write("Para aplicar simplex se maximiza **−Z**; al final se recupera el valor original de Z.")
    if modelo["artificiales"]:
        st.write("Se aplica el **método de la M**: cada artificial tiene coeficiente **−M** "
                 "en la función de maximización transformada, con M positiva y simbólica. "
                 "No se sustituye M por un número elegido arbitrariamente.")
    else:
        st.write("Todas las restricciones se prepararon con variables de holgura; "
                 "no se requieren variables artificiales.")
    if modelo["cambios"]:
        for cambio in modelo["cambios"]:
            st.write(cambio)
    st.markdown("**Sistema de ecuaciones resultante**")
    st.code("\n".join(modelo["ecuaciones"]), language="text")

    st.markdown("**Coeficientes de la función objetivo transformada (Cj)**")
    st.dataframe(pd.DataFrame({
        "Variable": modelo["nombres"],
        "Cj": [par_m(modelo["cm"][j], modelo["cc"][j])
               for j in range(len(modelo["nombres"]))],
    }), hide_index=True, use_container_width=True)
    st.markdown("**Tabla simplex inicial**")
    st.dataframe(pd.DataFrame(resultado["inicial"]), hide_index=True, use_container_width=True)

    if not resultado["pasos"]:
        st.write("No fue necesario realizar pivoteos.")
    for paso in resultado["pasos"]:
        with st.expander(
            f"Iteración {paso['iteracion']}: entra {paso['entra']} y sale {paso['sale']}",
            expanded=(paso["iteracion"] == 1),
        ):
            st.markdown("**Tabla antes del pivote**")
            st.dataframe(pd.DataFrame(paso["tabla_antes"]),
                         hide_index=True, use_container_width=True)
            st.write(f"**Variable de entrada:** {paso['entra']} "
                     "(costo reducido positivo; regla de Bland).")
            st.markdown("**Prueba de razones: LD / coeficiente positivo de la columna entrante**")
            st.dataframe(pd.DataFrame({
                "Fila": ["F1", "F2", "F3"],
                "Variable básica": paso["bases_antes"],
                "Coeficiente entrante": paso["columna_entrada"],
                "Razón": ["No aplica" if r is None else numero(r) for r in paso["razones"]],
            }), hide_index=True, use_container_width=True)
            st.write(f"**Variable de salida:** {paso['sale']}. "
                     f"**Pivote:** {numero(paso['pivote'])} (fila {paso['fila']}).")
            st.markdown("**Operaciones de Gauss-Jordan**")
            st.code("\n".join(paso["operaciones"]), language="text")
            st.markdown("**Tabla obtenida después del pivote**")
            st.dataframe(pd.DataFrame(paso["tabla_despues"]),
                         hide_index=True, use_container_width=True)

    st.subheader("3. Método gráfico")
    vertices, tabla_vertices, _ = resultado_grafico(c, restricciones, resultado["estado"])
    fig = dibujar_grafico(c, restricciones, resultado, vertices)
    st.pyplot(fig, width=700)
    plt.close(fig)
    if tabla_vertices:
        st.markdown("**Evaluación de la función objetivo en cada vértice factible**")
        st.dataframe(pd.DataFrame(tabla_vertices), hide_index=True, use_container_width=True)
    else:
        st.write("No se identificaron vértices factibles.")

    st.subheader("4. Resultados e interpretación")
    estado = resultado["estado"]
    if estado == "Óptimo":
        x1, x2 = resultado["x"]
        a, b, d = st.columns(3)
        a.metric("x1 óptimo", numero(x1))
        b.metric("x2 óptimo", numero(x2))
        d.metric("Z óptimo", numero(resultado["z"]))
        st.markdown("**Holguras, superávits y residuos en el modelo original**")
        st.dataframe(pd.DataFrame(informe_restricciones(resultado["x"], restricciones)),
                     hide_index=True, use_container_width=True)
        if tabla_vertices:
            valores = [fila["Z"] for fila in tabla_vertices]
            z_grafico = max(valores) if tipo == "Maximizar" else min(valores)
            if abs(z_grafico - resultado["z"]) <= 1e-6 * max(1.0, abs(z_grafico)):
                st.success(f"Comparación: simplex y método gráfico coinciden en Z = {numero(z_grafico)}.")
            else:
                st.warning("Los valores del simplex y de los vértices difieren. Revise la precisión de los datos.")
    elif estado == "Infactible":
        st.error("Problema infactible: las restricciones no admiten una solución común.")
    elif estado == "No acotado":
        st.warning("Problema no acotado: no existe un óptimo finito.")
        st.write("La evaluación de los vértices por sí sola no permite detectar "
                 "que la función objetivo mejora sin límite.")
    else:
        st.warning(resultado["explicacion"])
    st.write("**Interpretación matemática:** " + interpretacion(resultado))

    st.subheader("5. Verificación con problemas de prueba")
    st.write("Los ejemplos incluyen maximización, minimización, igualdad con "
             "variable artificial, infactibilidad y no acotación. Puede cargarlos "
             "individualmente arriba o comprobarlos todos aquí.")
    if st.button("Ejecutar las cinco pruebas automáticas"):
        esperados = [
            ("Óptimo", 36.0), ("Óptimo", 10.0), ("Óptimo", 11.0),
            ("Infactible", None), ("No acotado", None),
        ]
        comprobaciones = []
        for (nombre, datos), (estado_esperado, z_esperado) in zip(EJEMPLOS.items(), esperados):
            obtenido = resolver_simplex(datos["c"], datos["restricciones"], datos["tipo"])
            correcto = obtenido["estado"] == estado_esperado
            if z_esperado is not None:
                correcto = correcto and obtenido["z"] is not None and abs(obtenido["z"] - z_esperado) < 1e-7
            comprobaciones.append({
                "Caso": nombre, "Estado obtenido": obtenido["estado"],
                "Valor Z": "No existe" if obtenido["z"] is None else numero(obtenido["z"]),
                "Verificación": "Correcta" if correcto else "Revisar",
            })
        st.dataframe(pd.DataFrame(comprobaciones), hide_index=True, use_container_width=True)
        if all(fila["Verificación"] == "Correcta" for fila in comprobaciones):
            st.success("Las cinco pruebas produjeron los resultados esperados.")
        else:
            st.error("Al menos una prueba no coincidió con el resultado esperado.")


# -- Parte 10 - Punto de entrada para ejecutar la aplicación con Streamlit

if __name__ == "__main__":
    iniciar_interfaz()
