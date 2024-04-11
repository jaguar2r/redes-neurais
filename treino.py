import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

"""
TODO:
    - Baixar arquivo
    - Comparar  ReLU e GeLU
    - Treinar a rede
"""


"""
# Introdução

As equações para as funções de ativação RELU (Rectified Linear Unit) e GELU (Gaussian Error Linear Unit) podem ser expressas em LaTeX da seguinte maneira:

### RELU

A função RELU é definida como:

$$
f(x) = \max(0, x)
$$

Essa função retorna \(x\) se \(x\) for maior que 0; caso contrário, retorna 0.

### GELU

A função GELU, por outro lado, é um pouco mais complexa e é definida como:

$$
f(x) = x \cdot \Phi(x)
$$

onde \(\Phi(x)\) é a função de distribuição cumulativa (CDF) da distribuição normal padrão, que pode ser expressa como:

$$
\Phi(x) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

Aqui, \(\text{erf}\) é a função de erro, que é uma função especial usada em probabilidade e estatísticas.

Por vezes, para simplificar o cálculo da GELU em implementações práticas, utiliza-se uma aproximação baseada em tanh, resultando na seguinte expressão aproximada:

$$
f(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right]\right)
$$

Essa aproximação busca equilibrar a precisão e a eficiência computacional, permitindo que a função GELU seja utilizada em modelos de deep learning sem um custo computacional excessivamente alto.
"""


# Definição da função RELU
def relu(x):
    return np.maximum(0, x)

# Definição da função GELU
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# Gerando um conjunto de dados para plotar
start_time = time.time()
x = np.linspace(-3, 3, 100)
y_relu = relu(x)
y_gelu = gelu(x)
total_time = time.time() - start_time
print(f"Tempo total de execução: {total_time:.4f} segundos")

# Plotando as funções
plt.figure(figsize=(10, 5))
plt.plot(x, y_relu, label='RELU')
plt.plot(x, y_gelu, label='GELU', linestyle='--')
plt.xlim(-4, 4)
plt.ylim(-1, 4)
plt.title('RELU vs GELU')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.savefig('relu_vs_gelu.png', bbox_inches='tight', pad_inches=0.1)


# Funções do próprio Pytorch

# Gerando um conjunto de dados para plotar
start_time_torch = time.time()
x = torch.linspace(-3, 3, 100)

# Calculando as ativações usando PyTorch
y_relu = F.relu(x)
y_gelu = F.gelu(x)

# Convertendo as saídas do PyTorch para NumPy para plotagem
x_np = x.numpy()
y_relu_np = y_relu.numpy()
y_gelu_np = y_gelu.numpy()
total_time = time.time() - start_time_torch
print(f"Tempo total de execução com PyTorch: {total_time:.4f} segundos")
# Plotando as funções
plt.figure(figsize=(10, 5))
plt.plot(x_np, y_relu_np, label='ReLU - PyTorch')
plt.plot(x_np, y_gelu_np, label='GELU - PyTorch', linestyle='--')

# Adicionando títulos e legendas
plt.title('ReLU vs GELU in PyTorch')
plt.xlabel('x')
plt.ylabel('Activation Value')
plt.legend()
plt.grid(True)
plt.savefig('relu_vs_gelu_pytorch.png', bbox_inches='tight', pad_inches=0.1)


# Iniciando treinamento
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
