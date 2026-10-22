# Clase `Value` y `backward()` con ordenamiento topológico

## Fecha

2026-10-09 → 2026-10-22 — Semana 1, sesión 3 (interrumpida varias veces por cortes de luz: 10-09, 10-15, 10-20, 10-22)

## Hipótesis / Objetivo

Construir mi propio autograd mínimo: una clase `Value` que arme el grafo en el forward y calcule todos los gradients con una sola llamada a `backward()`. Hipótesis: si cada operación sabe su derivada local y los nodos se procesan en el orden correcto, los gradients coinciden con los de PyTorch.

## Configuración

- Código: `phase1/01-micrograd/derivatives.ipynb` (commits `84feb7a`, `a3b0050`, `2c35459`, `3320f3a`), `phase1/01-micrograd/viz.py` (`draw_dot`, Graphviz)
- Grafo de prueba (neurona de Karpathy): `x = 2`, `w = −3`, `b = 6.8813735870195432`, `i = x·w`, `n = i + b`, `o = tanh(n)`
- Referencia: Karpathy, video de micrograd, minutos ~50:00–1:10 (backprop manual por una neurona y `_backward` por operación)
- Hardware: CPU local

## Qué hice

1. **Derivada de `tanh`:** `d/dn tanh(n) = 1 − tanh²(n)`. La grafiqué junto a `tanh` y la verifiqué con `numerical_derivative` en `n = 0` y `n = 4`.
2. **Clase `Value`** con `data`, `grad`, `_prev` (hijos), `_op` y `_backward`. Escribí `__add__`, `__mul__` y `tanh`, cada una con su closure `_backward`, que lee `out.grad` y acumula con `+=` en los grads de sus hijos.
3. **Pruebas pequeñas:** `a*b`, `a*a` (acumulación por dos caminos) y `tanh` en 0 y en 4.
4. **Backward manual en la neurona:** llamé `_backward()` nodo por nodo y vi que, en el orden equivocado, los gradients quedan en 0.
5. **Diseño del orden (en palabras, antes de escribir código):**
   - Al llamar `v._backward()`, el `grad` de `v` tiene que estar **completo**, o sea que ya corrieron los `_backward()` de todos los nodos que usan a `v`.
   - Orden de construcción de `topo` (los `append`): hijos antes que padres. Orden de procesamiento: el inverso. Es un stack: `o` entra de último y sale de primero.
   - Propuse una versión iterativa con un stack de pendientes: mirar el tope **sin sacarlo**, validar si sus hijos ya están en `topo` y, si no, bajar a ellos. Es correcta, y es lo mismo que hace la recursión con el call stack.
6. **Implementé `backward()`** de forma recursiva con una función anidada `build(v)`; `topo` y `visited` son locales de `backward()` y `build` los ve por closure:
   ```python
   def backward(self):
       visited = set()
       topo = []
       def build(v):
           if v not in visited:
               for child in v._prev:
                   build(child)
               topo.append(v)
               visited.add(v)
       build(self)
       self.grad = 1.0
       for i in reversed(topo):
           i._backward()
   ```
7. **Experimento:** llamar `o.backward()` dos veces sobre el mismo grafo, prediciendo antes los grads.
8. **Verificación contra PyTorch:** mismo grafo con `torch.tensor(..., requires_grad=True)`, `retain_grad()` en los intermedios y `o.backward()`.

## Resultados

**`tanh`:** derivada 1.0 en `n = 0` y 0.00134 en `n = 4` (saturación).

**Neurona, una sola llamada a `o.backward()`:**

| Nodo | data | grad (mi `Value`) | grad (PyTorch) |
|---|---|---|---|
| `o` | 0.7071 | 1.0 | — |
| `n` | 0.8814 | 0.5 | 0.5 |
| `i` | −6.0 | 0.5 | 0.5 |
| `b` | 6.8814 | 0.5 | 0.5 |
| `x` | 2.0 | −1.5 | −1.5 |
| `w` | −3.0 | 1.0 | 1.0 |

En float32, PyTorch da `x.grad = −1.5000003576278687`; la diferencia es de precisión, no de cálculo.

**`o.backward()` dos veces seguidas (sin reiniciar grads):** predicción correcta (con un desliz al cruzar los valores viejos de `x` y `w`, que corregí).

| Nodo | 1ª llamada | 2ª llamada | Factor |
|---|---|---|---|
| `o` | 1.0 | 1.0 | ×1 (la semilla usa `=`) |
| `n` | 0.5 | 1.0 | ×2 |
| `i`, `b` | 0.5 | 1.5 | ×3 |
| `x` | −1.5 | −6.0 | ×4 |
| `w` | 1.0 | 4.0 | ×4 |

El error crece hacia las hojas: cada `_backward` lee un `out.grad` ya inflado y además lo suma a lo que el nodo ya tenía.

**Bugs que cometí en el camino:**
- Olvidé `out._backward = _backward`, olvidé sembrar `grad = 1` y llamé `c._backward` sin `()`.
- Primera versión de `backward()`: me llamaba a mí mismo con `child.backward()`, así que cada llamada creaba **su propio** `topo` y `visited` vacíos.
- `build()` sin parámetro y usando `self`: siempre procesaba la raíz. Después, `topo.append(self)` en vez de `v`.
- Creí que `backward()` solo tenía que construir el orden; faltaban la semilla y el loop de `_backward()`.
- Tenía un `if child not in visited` redundante: `build(child)` ya revisa eso al entrar. Lo quité y los resultados no cambiaron.
- `x.grad.item` sin paréntesis imprime el método, no el número.

## Qué no entendí

- Al principio confundí "no se activa" con "no aprende" en la saturación de `tanh`. Lo que pasa es que con gradient ≈ 0 los pesos anteriores casi no se actualizan (vanishing gradient).
- Confundí el orden en que se calcula el gradient (`o` primero) con el orden de los `append` en `topo` (`o` de último).
- Pensé que gradients inflados darían pasos *más pequeños*. En `w.data -= lr * w.grad`, un grad más grande da un paso más grande.

## Conclusiones

- Mi `Value` calcula los mismos gradients que PyTorch en la neurona: la hipótesis se confirmó.
- `backward()` = construir el orden topológico (recursivo) + sembrar `grad = 1` + `_backward()` en orden inverso. Son dos responsabilidades, y solo la primera es recursiva.
- La recursión es un stack implícito: la columna de "pendientes" de mi versión iterativa es el call stack de Python. PyTorch usa una versión iterativa para no chocar con el límite de recursión en grafos profundos.
- Variables compartidas por llamadas recursivas: locales de la función externa + función anidada (closure). Atributos de clase no sirven (estado global que sobrevive entre llamadas) y atributos de instancia tampoco (uno por nodo).
- **`+=` acumula entre llamadas:** si no se reinician los grads, el paso queda mal en **tamaño** (grads inflados) y en **dirección** (se mezclan gradients de pesos viejos que pueden tener el signo contrario). Por eso existe `optimizer.zero_grad()`. Pero acumular a propósito sobre los **mismos** pesos es correcto: es *gradient accumulation*, que usaré en la semana 7.
- En PyTorch, `requires_grad` va en las hojas que se entrenan (pesos, biases); los intermedios lo heredan y tienen `grad_fn`, pero no guardan `.grad` salvo con `retain_grad()`.

## Siguiente paso

Sesión 4: añadir `zero_grad`, construir `Neuron`, `Layer` y `MLP` sobre `Value`, y escribir el training loop. Primer experimento con hipótesis: efecto del learning rate en la convergencia del loss.
