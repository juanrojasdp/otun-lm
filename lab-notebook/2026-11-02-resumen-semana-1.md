# Resumen de la semana 1: micrograd

## Fecha

2026-11-02 — cierre de la semana 1 (sesiones 1–4, del 2026-10-01 al 2026-11-02)

## Resumen

1. Una derivada se puede calcular numéricamente (empujando una entrada con `h`), y backpropagation aplica la regla de la cadena sobre un grafo: el forward le da `data` a cada nodo, y el backward, desde `grad = 1` en el loss, multiplica derivadas locales y suma (`+=`) cuando un nodo alimenta varios caminos. El grad de cada nodo es la derivada parcial del loss respecto a ese nodo.
2. Construí mi propio autograd: una clase `Value` cuyo `backward()` arma el orden topológico y llama los `_backward()` en orden inverso; sus gradients coinciden con los de PyTorch.
3. Una red (`Neuron` → `Layer` → `MLP`) se entrena con un loss ≥ 0 (suma de errores al cuadrado) y un ciclo zero_grad → forward → backward → update (`p.data -= lr * p.grad`). Sin zero_grad los grads se acumulan entre pasos; con pesos iniciales iguales las neuronas nunca se diferencian, por eso se inicializan aleatorios y pequeños.
4. El learning rate decide si el training funciona: muy pequeño es lento y muy grande satura `tanh` y mata la red. El mejor es el más alto que sea estable con varias semillas, no el que ganó una vez.

## Reflexión (con mis palabras)

En la primera sesión hicimos un repaso de derivadas. Ya las había visto en la universidad hace 11 años, pero con un enfoque distinto: allá también vimos derivadas numéricas, pero todo estaba enfocado en las reglas del cálculo para resolverlas en papel de forma analítica. El enfoque numérico es diferente y lo tuve que reforzar: derivadas con aproximación numérica (usando `h`), con una y con múltiples variables.

En la sesión 2 vimos la regla de la cadena, también con el enfoque numérico, y empezamos a asociar las funciones con grafos. Así aprendí qué es el forward (darle `data` a todos los nodos, donde cada nodo es un componente de la función al que se le aplica la regla de la cadena) y el backward (recorrer el grafo hacia atrás para llenar el `grad` de cada nodo). El grad, en términos de mis clases de matemáticas de la universidad, es la derivada parcial de la función respecto a ese término.

En la sesión 3 trabajamos con la derivada de `tanh`, una función que suele usarse como activación en las redes neuronales. También implementamos la clase `Value`, nuestra primera clase de micrograd, que tiene `data`, `grad` y un `backward()`. Ese `backward()` construye internamente el orden en el que hay que calcular los grads (el orden topológico) y recorre los nodos en orden inverso llamando el `_backward()` de cada uno, que llena los grads: eso es backpropagation. También vimos la necesidad de `zero_grad`: poner todos los grads de los parámetros en 0 antes de volver a hacer backpropagation. Si no se hace, los grads se acumulan entre llamadas a `backward()` (entre pasos de training) y el update queda mal en tamaño y en dirección.

En la sesión 4 llevamos todo más allá. Hicimos un experimento de simetría con dos neuronas idénticas y vimos que sus grads son los mismos, así que siguen iguales para siempre: por eso los pesos se inicializan aleatorios y pequeños. Implementamos las clases `Neuron` (una función de la forma `tanh(w1·x1 + w2·x2 + … + wn·xn + b)`), `Layer` (una capa de neuronas con cierta cantidad de entradas y de salidas, una salida por neurona) y `MLP` (una red de varias capas con una cantidad de entradas y una de salidas, normalmente 1 en nuestros experimentos). Vimos el loss: informalmente, qué tan lejos están los valores que produce el MLP de los que esperamos; se mide con la suma de las diferencias al cuadrado para que siempre sea ≥ 0, y la dirección de ajuste la da el gradient. Vimos el training loop: un ciclo de zero_grad → forward → backward → update que se repite para que el loss disminuya; backpropagation solo calcula los gradients, y lo que baja el loss es el update.

Por último hicimos el sweep del learning rate (el valor por el que se multiplica el grad de cada parámetro en el update). Con 100 pasos fijos, probamos varios `lr` y varias semillas (que fijan los números aleatorios con que se inicializan los pesos) y concluimos que el `lr` óptimo es el más alto que no falló con las semillas probadas. Es importante saber que en el training loop lo que se modifica son los parámetros (pesos y biases), porque son los que tienen la capacidad de modificar la salida de la red; de ahí que se trabaje con su grad.

## Nota sobre el ritmo

La "semana 1" tomó un mes de calendario (con cuatro cortes de luz en el camino). Las semanas del plan son bloques de contenido, no semanas de calendario: prefiero avanzar constante que rápido.

## Siguiente paso

Semana 2: makemore (bigram + MLP) con nombres de municipios y apellidos colombianos, con split train/validation para medir overfitting.
