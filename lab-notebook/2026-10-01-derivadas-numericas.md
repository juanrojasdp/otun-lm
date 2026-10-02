# Derivadas numéricas y gradient descent (EN PROGRESO)

## Fecha

2026-10-01 y 2026-10-02 — Semana 1, sesión 1

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
6. (2026-10-02) Escribí la regla de gradient descent: `x_new = x − lr · f'(x)`, y un loop de 20 pasos desde `x = 0` con `lr` = 0.1, 1 y 0.01.
7. Bug: no reiniciaba `x` entre experimentos, así que cada uno arrancaba donde terminó el anterior y los resultados de `lr = 1` y `lr = 0.01` salieron sin sentido. Lo detecté porque no cuadraban con el cálculo a mano. Arreglo: refactoricé el experimento en una función `gradient_descent(lr, steps=20, x0=0.0)` para que cada llamada empiece limpia.
8. En papel: con `d = x − 2/3` (distancia al mínimo) derivé `d_new = (1 − 6·lr) · d`.

## Resultados

- El sobrante de la aproximación es exactamente `3h`: es error del método, predecible, no ruido.
- Cuando `h → 0`, queda la derivada exacta `6x − 4` (definición formal de derivada).
- Derivada = 0 en `x = 2/3`: es el **mínimo** de la parábola (no un punto de inflexión, como dije primero).
- El signo de la derivada dice si `f` sube o baja al mover `x` a la derecha.
- **Para minimizar, hay que moverse en sentido contrario al signo de la derivada** (x=0, derivada −4 → derecha; x=3, derivada +14 → izquierda).

Gradient descent, 20 pasos desde `x = 0`:

| lr | `x` final | Comportamiento |
|---|---|---|
| 0.1 | 0.6666666593365892 | Converge; el paso se achica solo porque la derivada → 0 cerca del mínimo |
| 1 | — | Oscila entre positivos y negativos con valor absoluto creciente: diverge |
| 0.01 | 0.47326250591235874 | Converge pero muy lento; no llega en 20 pasos |

Análisis con `d_new = (1 − 6·lr) · d`, que predice exactamente los tres resultados (0.4²⁰ · 2/3 ≈ 7·10⁻⁹; 2/3 · (1 − 0.94²⁰) ≈ 0.473; factor −5 → oscila y explota):

| lr | Factor `1 − 6·lr` | Comportamiento |
|---|---|---|
| `0 < lr < 1/6` | entre 0 y 1 | Converge sin oscilar |
| `lr = 1/6` | 0 | Llega en un solo paso, desde cualquier punto |
| `1/6 < lr < 1/3` | entre −1 y 0 | Converge oscilando |
| `lr = 1/3` | −1 | Oscila para siempre |
| `lr > 1/3` | menor que −1 | Diverge |

**Converge si y solo si `|1 − 6·lr| < 1` ⟺ `0 < lr < 1/3`.** Primero planteé la desigualdad como `0 < 1 − 6·lr < 1` y obtuve `0 < lr < 1/6`, que es solo el rango sin oscilación; el signo del factor puede ser negativo, lo que importa es su valor absoluto.

## Qué no entendí

- Pendiente de la sesión: el `...000000426` al final del resultado (se explora en el paso de "romper h").

## Conclusiones

- Comparar papel contra código atrapa errores (gradient check), tanto en el papel (el `−1`) como en el código (el `x` sin reiniciar).
- Gradient descent: `x_new = x − lr · f'(x)`. El learning rate es un hyperparameter: lo elijo yo, no sale de ninguna fórmula.
- Tres regímenes: lr muy alto → diverge; lr adecuado → converge rápido; lr muy bajo → converge pero desperdicia pasos (cómputo).
- El lr perfecto (1/6) solo se puede calcular porque conozco la fórmula completa de la parábola. En una red neuronal solo conozco la derivada en el punto actual, por eso el lr se busca experimentalmente. El mejor suele estar cerca del límite de divergencia, pero por debajo.
- El estado oculto de los notebooks causa bugs: encapsular experimentos en funciones.

## Siguiente paso

1. Verificar la tabla de regímenes en el notebook (lr = 0.15, 0.25, 0.33, 0.34, imprimiendo la trayectoria).
2. Tabla de `h` desde `1e-1` hasta `1e-17` (predecir antes): ¿más pequeño siempre es mejor?
3. Derivadas parciales en `d = a*b + c`.
4. El misterio de `L = x·y + x` (gradientes por varias rutas).
