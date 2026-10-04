# Derivadas numéricas, gradient descent y derivadas parciales

## Fecha

2026-10-01, 2026-10-02 y 2026-10-04 — Semana 1, sesión 1

## Hipótesis / Objetivo

Estimar derivadas numéricamente con `(f(x+h) − f(x)) / h` sin saber derivar, verificar contra la derivada en papel, y entender cómo el signo de la derivada indica hacia dónde moverse para minimizar una función. Después, extenderlo a varias variables: derivadas parciales y qué pasa cuando una variable influye en el resultado por varios caminos.

## Configuración

- Código: `phase1/01-micrograd/derivatives.ipynb`
- Función: `f(x) = 3x² − 4x + 5`, derivada analítica `6x − 4`
- Varias variables: `d = a*b + c` con `a, b, c = 2, −3, 10`; `L = x*y + x` con `x, y = 3, −2`

## Qué hice

1. Creé el notebook y definí `f(x)`.
2. Derivada numérica en `x = 2` con `h = 0.001` → `8.003000000000426`.
3. Probé `h` más pequeños: el sobrante se reduce en la misma proporción que `h`.
4. Derivada en `x = 0` ≈ −4, en `x = −1` ≈ −10, en `x = 0.66` ≈ −0.04.
5. En papel: expandí `(f(x+h) − f(x)) / h` y obtuve `6x − 4 + 3h` (primer intento salió `−1` en vez de `−4` por un error de signo en `−4(x+h) − (−4x)`; lo detecté comparando contra el valor del notebook).
6. (2026-10-02) Escribí la regla de gradient descent: `x_new = x − lr · f'(x)`, y un loop de 20 pasos desde `x = 0` con `lr` = 0.1, 1 y 0.01.
7. Bug: no reiniciaba `x` entre experimentos, así que cada uno arrancaba donde terminó el anterior y los resultados de `lr = 1` y `lr = 0.01` salieron sin sentido. Lo detecté porque no cuadraban con el cálculo a mano. Arreglo: refactoricé el experimento en una función `gradient_descent(lr, steps=20, x0=0.0)` para que cada llamada empiece limpia.
8. En papel: con `d = x − 2/3` (distancia al mínimo) derivé `d_new = (1 − 6·lr) · d`.
9. Escribí `numerical_derivative(fn, x, h)` genérica (recibe la función como argumento) y la evalué en `x = 2` con `h` desde `1e-1` hasta `1e-17`.
10. (2026-10-04) Derivadas parciales de `d = a*b + c`: predije en papel `∂d/∂a = b`, `∂d/∂b = a`, `∂d/∂c = 1` y verifiqué numéricamente con `numerical_derivative` + `lambda`. Primera versión sustituía los valores a mano en cada `lambda` (`lambda a: -3*a + 10`); la corregí para usar la expresión original (`lambda a: a*b + c`) con las otras variables leídas del entorno, para que la verificación sea independiente de mi álgebra.
11. `L = x*y + x`: predije que las contribuciones de los dos caminos de `x` se suman. Lo verifiqué a mano (con dos errores: se me corrió un cero, `−0.001` en vez de `−0.0001`, y perdí el signo al dividir por `h`), con código y con la derivada tradicional (`y + 1`).

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

Derivada numérica de `f` en `x = 2` (exacta: 8) según `h`:

| h | Resultado |
|---|---|
| 1e-1 | 8.3 |
| 1e-3 | 8.003000000000426 |
| 1e-5 | 8.000030000054892 |
| 1e-8 | 7.999999951380232 |
| 1e-12 | 8.000711204658728 |
| 1e-15 | 7.105427357601001 |
| 1e-17 | 0.0 |

Mejora hasta `1e-8` y después se degrada. Dos fuentes de error:

- **Error del método:** `3h`, baja cuando `h` baja.
- **Error de redondeo:** float64 tiene ~16 dígitos significativos. `f(2+h) − f(2)` resta dos números casi iguales (~9) y se cancelan casi todos los dígitos (**catastrophic cancellation**); queda una "suciedad" de ~`1e-15` que no depende de `h`, y dividir por `h` la amplifica: ~`1e-15 / h`.
- Total ≈ `3h + 1e-15/h` → mínimo cerca de `h ≈ √1e-15 ≈ 1e-8`. Con `1e-17`, `2 + h` ni siquiera se distingue de `2` → resta = 0.
- **Rango ≠ precisión:** `1e-17` sí es representable (float64 llega a ~`1e-308`); lo que no cabe es `2.00000000000000001` (18 dígitos). Primero lo atribuí al rango, y era la precisión. El `...000000426` de `8.003000000000426` también es redondeo.

Derivadas parciales de `d = a*b + c` (`a, b, c = 2, −3, 10`):

| | Fórmula | Valor | Numérico |
|---|---|---|---|
| `∂d/∂a` | `b` | −3 | −3.000000000010772 |
| `∂d/∂b` | `a` | 2 | 2.0000000000042206 |
| `∂d/∂c` | `1` | 1 | 0.9999999999976694 |

- Subir `a` hace **bajar** `d`, porque `b` es negativo (mi primera intuición, "se incrementa", ignoraba el signo).
- `∂d/∂a` depende de `b`, no de `a`: en una multiplicación, la derivada respecto a cada input es *el otro input*. En una suma, es `1`.

`L = x*y + x`, `x` influye por dos caminos (`x*y` aporta `y`, `+x` aporta `1`):

| y | Camino `x*y` | Camino `+x` | Suma | Numérico |
|---|---|---|---|---|
| −2 | −2 | +1 | −1 | −1.0000000000021103 |
| 0.5 | 0.5 | +1 | 1.5 | 1.500000000005386 |

- Las contribuciones de los caminos **se suman**. Coincide con la derivada tradicional `∂L/∂x = y + 1`.
- Mi intuición de que "la multiplicación siempre pesa más" era falsa: con `y = 0.5` pesa más el camino de la suma. Lo que decide son los valores, no el tipo de operación.

## Qué no entendí

- Nada pendiente de esta sesión. A vigilar: sigo cometiendo errores de signo y de ceros en aritmética a mano; revisar siempre que el signo del resultado cuadre con lo observado.

## Conclusiones

- Comparar papel contra código atrapa errores (gradient check), tanto en el papel (el `−1`) como en el código (el `x` sin reiniciar).
- Gradient descent: `x_new = x − lr · f'(x)`. El learning rate es un hyperparameter: lo elijo yo, no sale de ninguna fórmula.
- Tres regímenes: lr muy alto → diverge; lr adecuado → converge rápido; lr muy bajo → converge pero desperdicia pasos (cómputo).
- El lr perfecto (1/6) solo se puede calcular porque conozco la fórmula completa de la parábola. En una red neuronal solo conozco la derivada en el punto actual, por eso el lr se busca experimentalmente. El mejor suele estar cerca del límite de divergencia, pero por debajo.
- El estado oculto de los notebooks causa bugs: encapsular experimentos en funciones.
- La derivada numérica tiene un `h` óptimo (~`1e-8` en float64) por el balance entre error del método y error de redondeo. La precisión numérica importa: los modelos grandes se entrenan en `bf16`, con mucha menos precisión.
- Una verificación numérica solo sirve si es independiente de mi razonamiento: usar la expresión original, no la versión que ya simplifiqué a mano.
- Reglas locales: multiplicación → la derivada es el otro input; suma → 1.
- **Si una variable influye por varios caminos, su derivada es la suma de las contribuciones de cada camino.** En backprop esto significa que los gradientes se acumulan (`+=`), no se asignan (`=`).

## Siguiente paso

1. (Opcional) Verificar la tabla de regímenes de lr en el notebook (0.15, 0.25, 0.33, 0.34, imprimiendo la trayectoria).
2. Sesión 2: regla de la cadena. Resolver en papel la pregunta 2 del diagnóstico: `L = (a·b + c)²` con `a, b, c = 2, −3, 10`, calcular `∂L/∂a` nodo por nodo y verificar numéricamente.
