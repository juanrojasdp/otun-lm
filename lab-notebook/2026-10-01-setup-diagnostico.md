# Setup del repo y diagnóstico inicial

## Fecha

2026-10-01 — Semana 1, sesión 0

## Hipótesis / Objetivo

Montar el repo y el entorno de Otún-LM, verificar que PyTorch funciona, y diagnosticar mi punto de partida en cálculo, álgebra lineal y redes neuronales para ajustar el plan de la semana 1.

## Configuración

- Código: estructura inicial del repo (primer commit)
- Hardware: Windows 11, CPU (6 threads), sin GPU NVIDIA
- Entorno: Python 3.14.4, PyTorch 2.14.1+cpu en `.venv`

## Qué hice

1. Creé la estructura del repo: `phase1/01..08`, `lab-notebook/`, `data/`, `writeups/`, `README.md`, `requirements.txt`, `.gitignore`, `TEMPLATE.md`.
2. Instalé PyTorch CPU en un venv. Falló con `WinError 1114` al cargar `c10.dll`: el Visual C++ Runtime era de 2015 (`14.00.23026`) y faltaba `vcruntime140_1.dll`. Lo arreglé instalando `Microsoft.VCRedist.2015+.x64` (14.51) con winget.
3. Respondí 5 preguntas de diagnóstico.

## Resultados

- PyTorch funciona: autograd calcula d/dx(x³ + 2x) en x=2 = 14.0 (correcto). Matmul 1024×1024 ≈ 30 ms (~72 GFLOPS).
- Diagnóstico:
  - Derivadas: `f'(2) = 8` para `f(x) = 3x² − 4x + 5`, correcto. Intuición de razón de cambio (posición/velocidad) sólida.
  - Regla de la cadena: no la recuerdo.
  - Gradiente con varias rutas (`L = x·y + x`): no lo recuerdo.
  - Álgebra lineal: sé que el producto punto da un escalar; no recuerdo matriz × vector ni shapes.
  - Redes neuronales: no conozco formalmente loss, gradient ni learning rate.

## Qué no entendí

- Regla de la cadena y cómo se acumulan los gradientes cuando una variable se usa varias veces (pendiente: sesiones 1–2).

## Conclusiones

El punto de partida es la intuición de derivada como "razón de cambio". La regla de la cadena es la base de backpropagation, así que el plan se ajusta con una sesión de repaso de derivadas (numéricas, por "empujón") antes del video de Karpathy. Álgebra lineal se deja para la semana 2, porque micrograd es escalar.

## Siguiente paso

Sesión 1: aproximar derivadas numéricamente con `(f(x+h) − f(x)) / h`, explorar el efecto de `h`, y derivadas parciales en `d = a*b + c`.
