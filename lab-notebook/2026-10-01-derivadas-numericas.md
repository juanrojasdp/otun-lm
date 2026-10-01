# Derivadas numéricas por "empujón" (EN PROGRESO)

## Fecha

2026-10-01 — Semana 1, sesión 1 (continúa el 2026-10-02)

## Hipótesis / Objetivo

Estimar derivadas numéricamente con `(f(x+h) − f(x)) / h` sin saber derivar, verificar contra la derivada en papel, y entender cómo el signo de la derivada indica hacia dónde moverse para minimizar una función.

## Configuración

- Código: `phase1/01-micrograd/derivatives.ipynb`
- Función: `f(x) = 3x² − 4x + 5`, derivada analítica `6x − 4`

## Qué hice

1. Creé el notebook y definí `f(x)`.
2. Derivada numérica en `x = 2` con `h = 0.001` → `8.003000000000426`.
3. Probé `h` más pequeños: el sobrante se reduce en la misma proporción que `h`.
4. Derivada en `x = 0` ≈ −4, en `x = −1` ≈ −10, en `x = 0.66` ≈ −0.04.
5. En papel: expandí `(f(x+h) − f(x)) / h` y obtuve `6x − 4 + 3h` (primer intento salió `−1` en vez de `−4` por un error de signo en `−4(x+h) − (−4x)`; lo detecté comparando contra el valor del notebook).

## Resultados

- El sobrante de la aproximación es exactamente `3h`: es error del método, predecible, no ruido.
- Cuando `h → 0`, queda la derivada exacta `6x − 4` (definición formal de derivada).
- Derivada = 0 en `x = 2/3`: es el **mínimo** de la parábola (no un punto de inflexión, como dije primero).
- El signo de la derivada dice si `f` sube o baja al mover `x` a la derecha.
- **Para minimizar, hay que moverse en sentido contrario al signo de la derivada** (x=0, derivada −4 → derecha; x=3, derivada +14 → izquierda).

## Qué no entendí

- Me enredé escribiendo la regla de actualización de `x` (pendiente).
- Pendiente de la sesión: el `...000000426` al final del resultado (se explora en el paso de "romper h").

## Conclusiones

Comparar papel contra código atrapa errores (gradient check). La idea de moverse contra el signo de la derivada es la base de gradient descent.

## Siguiente paso

1. Terminar la regla de actualización: ¿qué operación con la derivada (−4) y el learning rate (0.1) da un paso de +0.4?
2. Loop de gradient descent de 20 pasos desde `x = 0`. Predecir antes: ¿a qué `x` llega? ¿Qué pasa con learning rate 0.5 y 0.01?
3. Paso 3 original: tabla de `h` desde `1e-1` hasta `1e-17` (predecir antes).
4. Paso 4: derivadas parciales en `d = a*b + c`.
5. Paso 5: el misterio de `L = x·y + x` (gradientes por varias rutas).
