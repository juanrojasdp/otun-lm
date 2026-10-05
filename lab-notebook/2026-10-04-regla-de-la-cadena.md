# Regla de la cadena y backpropagation a mano

## Fecha

2026-10-04 — Semana 1, sesión 2

## Hipótesis / Objetivo

Resolver la pregunta 2 del diagnóstico (`∂L/∂a` para `L = (a·b + c)²`) partiendo la expresión en un grafo de operaciones simples, descubrir cómo se combinan las derivadas locales a lo largo de una cadena, y describir el algoritmo de backpropagation con mis palabras.

## Configuración

- Código: `phase1/01-micrograd/derivatives.ipynb` (verificación numérica con `numerical_derivative`, `h = 0.0001`)
- Expresión: `L = (a·b + c)²` con `a, b, c = 2, −3, 10`, descompuesta como `e = a·b`, `d = e + c`, `L = d²`
- Referencia: Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd*, hasta el minuto ~50

## Qué hice

1. Descompuse la expresión en un grafo:
   ```
   a ──┐
       (·)── e ──┐
   b ──┘         (+)── d ──(²)── L
             c ──┘
   ```
2. Forward pass en papel.
3. Derivadas locales de cada nodo respecto a sus inputs directos.
4. Analogía del carro, la bicicleta y la persona (2× y 4× → 8×): las razones de cambio encadenadas se multiplican.
5. Calculé `∂L/∂a`, `∂L/∂b` y `∂L/∂c` por la cadena. Primero expandí en fórmulas y en ambos casos se me perdió un factor al distribuir (`2ab² + 2c` en vez de `2ab² + 2bc`; `2a²b + 2c` en vez de `2a²b + 2ac`). Con números, multiplicando las derivadas locales, no hubo dónde equivocarse.
6. Verifiqué los tres gradientes con `numerical_derivative` sobre la expresión original `(a*b + c)**2`.
7. Vi el video de Karpathy hasta el final del backprop manual.

## Resultados

Forward: `e = −6`, `d = 4`, `L = 16`.

Derivadas locales: `∂e/∂a = b = −3`, `∂e/∂b = a = 2`, `∂d/∂e = 1`, `∂d/∂c = 1`, `∂L/∂d = 2d = 8`.

| Gradient | Cadena | Valor | Numérico |
|---|---|---|---|
| `∂L/∂a` | `8 · 1 · (−3)` | −24 | −23.9991 |
| `∂L/∂b` | `8 · 1 · 2` | 16 | ✓ |
| `∂L/∂c` | `8 · 1` | 8 | ✓ |

- **Regla de la cadena:** las derivadas locales a lo largo de un camino se multiplican: `∂L/∂a = ∂L/∂d · ∂d/∂e · ∂e/∂a`. Con esto respondí la pregunta 2 del diagnóstico: `∂L/∂a = −24`.
- Los tres gradientes comparten el tramo `∂L/∂d = 8` y dos comparten `∂d/∂e = 1`: calculando desde `L` hacia atrás, cada tramo se calcula una sola vez.
- `∂L/∂d` depende de `d = 4`, que sale del forward: por eso el forward va primero y sus valores se guardan (eso también explica por qué entrenar usa tanta memoria).
- La semilla del backward es `∂L/∂L = 1`.
- El sobrante de la derivada numérica de `∂L/∂a` fue `0.0009 = 9h` (en `f` era `3h`).

### Resumen del algoritmo (con mis palabras)

> El algoritmo de back propagation tiene dos fases principales: Forward y backward. En la fase de forward (se ejecuta primero), el grafo se construye. Básicamente, en estos ejemplos el objetivo es convertir expresiones matemáticas multi variables en grafos, en los que cada nodo es un valor y las uniones entre ellos son operaciones que generan nuevos valores. En el forward, generamos el grafo de izquierda a derecha y le damos valor a todos los nodos, construyendo también las aristas. Después de llegar al último nodo (L en los ejemplos), empieza la fase de backward. En esta fase el objetivo es, usando la regla de la cadena, obtener el gradient de cada uno de los nodos. Se empieza con el caso semilla (dL/dL en los ejemplos) en los que el gradient es 1. De ahí se aplica la ley de la cadena en cálculo, en la que el valor de cada nodo (recorriendo de derecha a izquierda) va a ser el gradient del nodo anterior (nuevamente de derecha a izquierda) por su derivada local, es decir, la razón de cambio de ese nodo respecto al anterior. De esa manera, de derecha a izquierda, se pueden calcular todas las derivadas de la variable principal (el último nodo a la derecha) respecto a cada una de las expresiones o variables. Al final del backward, cada Value (o nodo del grafo) tendrá seteado su gradient.

**Revisión:**

1. Donde dice "el **valor** de cada nodo va a ser el gradient..." debe decir **gradient**. Cada `Value` guarda las dos cosas por separado: `data` (del forward) y `grad` (del backward).
2. La derivada local es la razón de cambio **de la salida de la operación respecto a este nodo** (`∂salida/∂nodo`), no al revés. Ejemplo: `∂e/∂a = −3` mide cuánto cambia `e` al mover `a`.
3. Falta la regla de los caminos: si un nodo alimenta a varios nodos, su gradient es la **suma** de las contribuciones de cada uno (`+=` en código, no `=`).

Regla completa:

```
gradient del nodo = Σ sobre cada nodo que lo consume (gradient de ese nodo × derivada local de la conexión)
```

Además, el orden importa: antes de calcular el gradient de un nodo hay que terminar todos los nodos que lo consumen (ordenamiento topológico, sesión 3).

## Qué no entendí

- Nada bloqueante. Patrón a vigilar: errores al distribuir en álgebra (pierdo el factor que multiplica al último término). Mitigación: trabajar con números y usar fórmulas solo para entender.
- Pendiente: la derivada de `tanh`, que aparece en el video justo después del minuto 50 (ejemplo de la neurona).

## Conclusiones

- Backpropagation = forward (calcula y guarda valores) + backward desde `∂L/∂L = 1` aplicando regla de la cadena (multiplicar) y regla de caminos (sumar).
- Backprop trabaja con **números**, no con expresiones simbólicas: cada nodo conoce su derivada local evaluada en el valor actual. Expandir fórmulas fue justo donde me equivoqué.
- Calcular de atrás hacia adelante reutiliza los tramos compartidos: por eso backprop es eficiente.
- El grafo dibujado (como el de Karpathy) fue lo que terminó de aclarar el algoritmo: usar más visualizaciones.

## Siguiente paso

Sesión 3: escribir mi propia clase `Value` (`__add__`, `__mul__`, `tanh`, `_backward`, `backward()` con ordenamiento topológico), dibujar el grafo con Graphviz y verificar los gradientes contra PyTorch y contra `numerical_derivative`.
