# Neuron, Layer, MLP, loss y primer training loop (sesión 4)

## Fecha

2026-10-27 → 2026-11-02 — Semana 1, sesión 4

## Hipótesis / Objetivo

Construir una red neuronal (MLP) completa encima de mi `Value`, de forma que el forward pass arme el grafo y `backward()` pueda calcular los gradients de todos sus parámetros.

Experimento lateral, con hipótesis: si dos neuronas empiezan con los mismos pesos y bias, reciben gradients idénticos y nunca se diferencian.

## Configuración

- Código: `phase1/01-micrograd/micrograd.ipynb` (commits `62c2a7b`, `2dccdb6` y el de esta entrada)
- Red de prueba: `MLP(3, [4, 4, 1])` → 41 parámetros
- Pesos y biases inicializados con `random.uniform(-1, 1)`
- Hardware: CPU local

## Qué hice

1. **Experimento de simetría:** dos neuronas A y B con pesos `(1, 1)`, bias `0`, mismas entradas `(1, 2)`, `L = oA + oB`, `L.backward()`.
2. **`Neuron`:** guarda una lista de pesos (uno por entrada) y un bias; `__call__` calcula `tanh(Σ wi·xi + b)`. La verifiqué con pesos fijos `(1, 2, −1)`, bias `0.5` y entrada `(0.5, −1, 1)`, primero el forward y después el backward.
3. **`Value` con floats:** `__add__` y `__mul__` convierten `other` en `Value` si no lo es; `__rmul__` delega en `__mul__` (`return self * other`).
4. **`Layer(nin, nout)`:** lista de `nout` neuronas que reciben la misma entrada; devuelve una lista de `nout` `Value`.
5. **`MLP(nin, nouts)`:** `sizes = [nin] + nouts`, una `Layer(sizes[i], sizes[i+1])` por par consecutivo; `__call__` pasa la salida de cada capa a la siguiente.
6. **`parameters()`** en las tres clases, con list comprehensions anidadas.
7. **Loss:** dataset de 4 ejemplos de Karpathy (`ys = [1, −1, −1, 1]`). Comparé `|pred − y|` contra `(pred − y)²` graficando los dos losses y sus gradients, y elegí la suma de errores al cuadrado.
8. **Operaciones nuevas en `Value`:** `__neg__` (`self * -1`) y `__sub__` (`self + (-other)`) delegan; `__pow__` tiene su propio `_backward` (regla de la potencia, exponente solo `int`/`float`, hijo solo `self`); `__radd__` delega en `__add__`.
9. **Training loop:** `forward_mlp` hace forward + zero_grad + backward y devuelve el loss; después viene el update `p.data -= lr * p.grad`. Primero hice pasos a mano y después 100 pasos en un `for` con `lr = 0.05`, guardando la curva del loss.

## Resultados

**Simetría:** `wA1.grad = wB1.grad = 0.009866`. Las dos neuronas siguen siendo copias exactas después de cada update.

**Neurona con pesos fijos:** `Σ wi·xi + b = −2` → salida `tanh(−2) = −0.9640` ✓

| | `xi` | Esperado `0.0707 · xi` | Obtenido |
|---|---|---|---|
| `w1` | 0.5 | 0.0353 | 0.0353 ✓ |
| `w2` | −1.0 | −0.0707 | −0.0707 ✓ |
| `w3` | 1.0 | 0.0707 | 0.0707 ✓ |
| `b` | — | 0.0707 | 0.0707 ✓ |

(`0.0707 = 1 − tanh²(−2)`)

**MLP(3, [4, 4, 1]):** formas `(nout, nin)` por capa `[(4, 3), (4, 4), (1, 4)]`; `len(mlp.parameters()) = 41` (16 + 20 + 5); la salida es una lista con un solo `Value`.

**Primer intento de training (con bugs):** el loss bajó de ~5.9 a ~1.07, saltó a 7.2 y se quedó en **8.0**. Causa: el forward + backward quedó *dentro* del loop de update (41 `backward()` por paso) y sin zero_grad, así que los grads se acumularon, dieron un paso gigante y saturaron la red a una salida constante ±1. Con objetivos `[1, −1, −1, 1]`, una salida constante de +1 o de −1 da `4 + 4 = 8`, y con `tanh` saturada el gradient ≈ 0 la dejó atascada.

**Training corregido, 100 pasos, `lr = 0.05`:** loss final **0.008**.

| target | pred |
|---|---|
| +1.0 | +0.9647 |
| −1.0 | −0.9688 |
| −1.0 | −0.9447 |
| +1.0 | +0.9483 |

**Bugs que cometí en el camino:**
- Olvidé sumar el bias en la neurona: corría y daba un número razonable, pero estaba mal. Con pesos aleatorios, "dio un número" no prueba nada.
- Convertí a float solo el resultado del forward y dejé `other` como float, así que el float terminaba como hijo en el grafo y `backward()` habría fallado. Después escribí `Value(other.data)` sobre un float.
- En `__rmul__` revisé `isinstance(self, Value)` (siempre `True`) y reimplementé la multiplicación sin `_backward`, en vez de delegar.
- Edité `Value` sin volver a correr su celda: el kernel seguía usando la versión vieja.
- `Layer`: `[Neuron for _ in ...]` (la clase, no instancias), después `Neuron()` sin `nin`, y `__call__` sin `return`.
- `MLP`: faltó `self`, `range(nin - 1)` y después un off-by-one con `range(len(sizes))`.
- Loss: usé `.data` dos veces (primero en la predicción, después al sumar `total_loss`) y saqué el loss del grafo, así que `backward()` no habría llegado a los pesos.
- `__pow__`: primero metí el exponente como `Value` hijo (un nodo cuyo grad nadie calcula); después dejé el `int` en la tupla de hijos, y `build()` falló con `'int' object has no attribute '_prev'`.
- Training: el forward quedó indentado dentro del loop de parámetros, faltó zero_grad, y `mlp = MLP(...)` estaba en la misma celda que el paso (cada ejecución creaba una red nueva).

## Qué no entendí

- Al principio pensé que los pesos debían empezar en 1 "para no alterar los valores", y luego que debían ser proporcionales a las diferencias entre entradas. La inicialización no depende de los datos: tiene que ser aleatoria (romper la simetría) y pequeña (no saturar `tanh`).
- Confundí comparar dos pesos de la misma neurona (`wA1` contra `wA2`) con comparar el mismo peso en dos neuronas (`wA1` contra `wB1`).
- "Parámetros" en ML significa pesos y biases (lo que se aprende), no los argumentos de una función.
- Me costó pasar a código "la salida de cada capa es la entrada de la siguiente". Me ayudó escribirlo primero sin loop (`out1`, `out2`, `out3`), reemplazar los nombres por una sola variable `x` y solo entonces convertirlo en `for`.
- Pensé que el loss debía tener signo para saber hacia dónde ajustar. La dirección la da el **gradient**; el loss solo dice cuánto me equivoqué. Con `loss = pred − y`, minimizar empuja `pred` hacia −∞.
- Pensé que "correr el forward" era volver a crear el `MLP`. El forward es usar la red que ya existe para predecir.
- Terminé la sesión del 11-01 saturado; el training loop junta todo lo de la semana a la vez. Lo retomé al día siguiente con la secuencia zero_grad → forward → backward → update escrita explícitamente.

## Conclusiones

- **Simetría:** neuronas con los mismos pesos reciben gradients idénticos para siempre; 100 neuronas iguales equivalen a una sola. Además, pesos en 1 hacen que las sumas crezcan (`n = 3`) y saturen `tanh` (grad ≈ 0.01). Por eso la inicialización es aleatoria y pequeña. Se confirmó la hipótesis.
- Una neurona siempre produce un solo número; `nout` es cuántas neuronas tiene la capa, y no depende de `nin`. Parámetros de `Layer(nin, nout)` = `nout · (nin + 1)`.
- Python nunca voltea operandos: `3.0 * v` necesita `__rmul__`. Escribir `return self * other` declara que la operación es conmutativa y reutiliza la lógica (y el `_backward`) de `__mul__`.
- Python no revisa tipos: `Value(Value(2.0))` no falla al crearse, sino después y en otro lugar.
- Cuidado con `[Foo()] * n`: crea un solo objeto repetido n veces.
- **Loss:** tiene que ser ≥ 0 y valer 0 solo con predicción perfecta. El cuadrado es mejor que el valor absoluto porque su gradient `2(pred − y)` es proporcional al error (pasos más pequeños cerca del objetivo), y además su signo indica la dirección.
- `backward()` necesita un solo nodo raíz: los errores de todos los ejemplos se combinan sumando (o promediando) en un `Value`.
- El update se hace sobre `.data`, fuera del grafo (en PyTorch, `torch.no_grad()` / `optimizer.step()`). Va en contra del gradient: `-=`.
- Solo hay que reiniciar los grads de los **parámetros**: los nodos intermedios se crean nuevos en cada forward.
- Mi autograd entrenó una red real: el ciclo zero_grad → forward → backward → update es el mismo que se usa para entrenar un GPT. Ojo: con 4 ejemplos la red memorizó; sin datos de validación no sé si generaliza (overfitting, semana 2).

## Experimento: learning rate sweep

### Hipótesis

Con `lr` muy pequeño (0.001) el loss baja demasiado lento en 100 pasos; con `lr` muy grande (1.0) el loss puede llegar a subir después de algunos pasos; existe un `lr` óptimo intermedio, alrededor de 0.01–0.05.

### Diseño

- `train(lr, steps, seed, xs, ys)`: fija `random.seed(seed)` antes de crear `MLP(3, [4, 4, 1])`, entrena y devuelve `losses` y el `mlp`.
- **Métrica:** loss final después de 100 pasos, o sea un presupuesto fijo de cómputo (así se razona también a gran escala: la mejor configuración dentro de un presupuesto dado).
- **Misma semilla para todos los `lr`** → comparación justa (mismos pesos iniciales). **Varias semillas** → comprobar que el resultado no dependa de la inicialización.
- Verificación de la semilla: dos corridas con `seed=42` dieron `a == b → False`, pero con una diferencia máxima de **8.9e-16**. Causa: `_prev` es un `set`, cuyo orden depende de las direcciones de memoria; cambia el orden de los `_backward` y la suma en punto flotante no es exactamente asociativa. No afecta las conclusiones; los floats se comparan con tolerancia (`math.isclose`).

### Resultados

Loss final después de 100 pasos (💀 = la red quedó saturada / atascada):

| seed | 0.001 | 0.01 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| 42 | 1.59 | 0.087 | 0.012 | 0.0049 | 0.0019 | 0.0011 | 8.00 💀 | 8.00 💀 |
| 15 | 2.10 | 0.055 | 0.010 | 0.0049 | 0.0020 | 0.0051 | 0.0003 | 4.00 💀 |
| 7 | 2.16 | 0.050 | 0.0065 | 0.0029 | 0.0005 | 7.999 💀 | 0.0003 | 0.0002 |
| 123 | 2.46 | 0.091 | 0.0096 | 0.0038 | 0.0004 | 0.0009 | 8.00 💀 | 8.00 💀 |

(Yo corrí las semillas 42 y 15; las columnas 0.2 y 0.3 y las semillas 7 y 123 las corrió Claude con la misma función `train`.)

- Con `seed=42`, en 0.5 y 1.0 el loss **saltó de 5.23 a ~8.0 en el primer paso** y quedó plano: predicciones constantes (+1 con 0.5, −1 con 1.0). En la gráfica en escala log el salto casi no se ve y parecía "constante desde el inicio".
- `seed=15`, `lr = 1.0`: el loss bajó, se quedó en **4.0** exacto, con predicciones `[+1, −1, +1, +1]` contra `[+1, −1, −1, +1]`: un ejemplo saturado en el signo contrario, `(1 − (−1))² = 4`, sin gradient para corregirlo.
- Hasta 0.2: resultado estable en las 4 semillas, cada `lr` más grande es mejor. Desde 0.3: lotería; el mismo `lr` da el mejor loss de la tabla o mata la red según la semilla.

### Conclusiones (con mis palabras)

> Se confirmó que los `lr` demasiado pequeños hacen que el loss decrezca demasiado lento y van a exigir mucho cómputo porque exigen muchos pasos. Sí puede ser cierto que los mejores `lr` sean puntos intermedios, pero toca encontrarlos: toca ensayar con muchos seeds y distintos valores de `lr` para encontrar el valor más alto posible que no falle.
>
> Si tuviera que entrenar una sola vez, elegiría 0.2: con los datos que tenemos, es el que más minimiza el loss sin romper la red con los seeds ensayados. Tal vez ensayaría con más seeds, pero con esta tabla me iría con 0.2.

Contra la hipótesis:
- **`lr` pequeño lento:** ✓ confirmado.
- **`lr` grande hace subir el loss:** ✓ confirmado, pero el mecanismo fue un salto inmediato que **satura `tanh` y mata la red** (gradient ≈ 0), no una subida gradual.
- **Óptimo en 0.01–0.05:** ✗ refutado; el rango confiable fue 0.1–0.2.

Aprendizajes:
- Con una sola semilla habría concluido "0.5 mata la red" (seed 42) o "0.5 es el mejor" (seed 15); las dos conclusiones eran incorrectas.
- Mejor `lr` posible ≠ mejor `lr` **confiable**. En un entrenamiento real solo hay un intento, así que conviene el `lr` más alto que sea estable. El óptimo suele estar cerca del borde de la inestabilidad (0.2 funciona, 0.3 ya falla una vez), y por eso en los labs se usan técnicas como el *warmup*.

## Siguiente paso

Cerrar la semana 1 con el resumen de lo aprendido. Semana 2: makemore (bigram + MLP) con nombres de municipios y apellidos colombianos, con split train/validation para medir overfitting.
