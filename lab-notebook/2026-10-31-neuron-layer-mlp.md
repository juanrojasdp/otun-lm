# Neuron, Layer, MLP, loss y primer training loop (sesión 4)

## Fecha

2026-10-27 → 2026-11-02 — Semana 1, sesión 4 (en progreso: falta el experimento de learning rate)

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

## Siguiente paso

Experimento de learning rate. **Hipótesis:** con `lr` muy pequeño (0.001) el loss baja demasiado lento en 100 pasos; con `lr` muy grande (1.0) el loss puede llegar a subir después de algunos pasos; existe un `lr` óptimo intermedio, alrededor de 0.01–0.05. Probar `0.001, 0.01, 0.05, 0.1, 0.5, 1.0` y comparar las curvas del loss.
