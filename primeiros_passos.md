# Bora lá ...

# Introdução

## Redes Neurais

As redes neurais são um pilar fundamental da inteligência artificial, inspiradas no funcionamento do cérebro humano. Elas são capazes de aprender padrões complexos a partir de grandes quantidades de dados. Vamos explorar os princípios fundamentais por trás das redes neurais, entender como funcionam, e discutir diferentes tipos de arquiteturas como CNNs (Redes Neurais Convolucionais) e RNNs (Redes Neurais Recorrentes), destacando por que certas escolhas de arquitetura são feitas para resolver problemas específicos.

### Princípios Fundamentais

- **Neurônios e Camadas:** Uma rede neural é composta de neurônios (ou unidades) organizados em camadas. Um neurônio recebe entradas, as processa aplicando pesos (importância) e uma função de ativação (para introduzir não-linearidades), e produz uma saída.
- **Aprendizado:** O processo de aprendizado em redes neurais envolve a ajuste dos pesos sinápticos com base nos dados de entrada, de modo que a rede possa prever ou classificar dados com precisão. Isso é geralmente alcançado através de um processo chamado backpropagation, combinado com um algoritmo de otimização (como SGD ou Adam), para minimizar uma função de perda.

### Como Funcionam

1. **Propagação para Frente (Forward Propagation):** Os dados de entrada passam pela rede, de camada em camada, até a camada de saída. Cada neurônio processa a entrada com base em seus pesos e função de ativação.
2. **Cálculo da Perda:** A saída é comparada com o resultado esperado, utilizando uma função de perda (como a entropia cruzada para classificação) para medir o erro.
3. **Retropropagação (Backpropagation):** O erro é propagado de volta pela rede, permitindo que os pesos sejam ajustados para reduzir o erro.
4. **Atualização dos Pesos:** Os pesos são atualizados para minimizar o erro, usando um algoritmo de otimização.

### Diferentes Tipos de Arquiteturas

- **Redes Neurais Convolucionais (CNNs):** São especializadas em processar dados com uma topologia de grade, como imagens. Utilizam camadas convolucionais que aplicam filtros para capturar características espaciais. **Por que CNNs?** São escolhidas para visão computacional e tarefas relacionadas porque podem eficientemente identificar padrões visuais escaláveis e invariantes à posição.

- **Redes Neurais Recorrentes (RNNs):** São projetadas para processar sequências de dados, como texto ou séries temporais. Elas têm conexões em loop, permitindo que informações de entradas anteriores afetem a saída atual. **Por que RNNs?** São usadas para problemas sequenciais, como tradução de linguagem ou previsão de séries temporais, porque podem manter um "estado" ou memória dos dados anteriores.

### Escolhas de Arquitetura para Problemas Específicos

- **Visão Computacional:** CNNs são preferidas devido à sua habilidade de extrair características visuais hierárquicas dos dados de imagem.
- **Processamento de Linguagem Natural (PLN) e Séries Temporais:** RNNs ou suas variantes como LSTM (Long Short-Term Memory) e GRU (Gated Recurrent Units) são adequadas, pois podem lidar com a natureza sequencial dos dados. Contudo, para algumas tarefas de PLN, modelos baseados em atenção, como o Transformer, têm mostrado um desempenho superior, devido à sua capacidade de lidar com dependências de longo alcance dentro do texto.
- **Classificação e Regressão de Dados Tabulares:** Redes neurais densamente conectadas (ou redes totalmente conectadas) são comumente usadas, embora a escolha específica de arquitetura possa depender da complexidade do problema e da natureza dos dados.

### Em resumo

As redes neurais oferecem uma poderosa ferramenta para modelar problemas complexos em uma variedade de domínios. A escolha da arquitetura depende intrinsecamente da natureza dos dados e do problema específico a ser resolvido. Entender as capacidades e limitações de cada tipo de rede é crucial para aplicá-las efetivamente.

## Criptografia com um Foco em Curvas Elípticas

A criptografia é uma área da matemática e da ciência da computação focada em técnicas de segurança da informação, especialmente na proteção de comunicações e dados. Um dos conceitos mais fundamentais na criptografia moderna é a criptografia de chave pública, e as curvas elípticas desempenham um papel crucial nesse contexto. Vamos explorar esses conceitos passo a passo.

### Fundamentos da Criptografia

A criptografia permite que informações sejam transformadas de uma forma que somente quem possua a chave correta possa lê-las, mesmo que alguém consiga interceptar a mensagem. Existem dois tipos principais:

- **Criptografia Simétrica:** Onde a mesma chave é usada para criptografar e descriptografar a mensagem.
- **Criptografia de Chave Pública (ou Assimétrica):** Utiliza um par de chaves, uma pública e uma privada. A chave pública pode ser compartilhada com todos, mas apenas quem possui a chave privada correspondente pode descriptografar a mensagem.

### Criptografia de Chave Pública

A ideia por trás da criptografia de chave pública é permitir comunicações seguras em um ambiente inseguro sem a necessidade de compartilhar uma chave secreta. As aplicações incluem não apenas a criptografia de mensagens, mas também a assinatura digital, que permite verificar a autenticidade de uma mensagem.

### Curvas Elípticas e Criptografia

- **Curvas Elípticas?** Em matemática, uma curva elíptica é o conjunto de pontos que satisfazem uma equação do tipo \(y^2 = x^3 + ax + b\), junto com um ponto "no infinito". Estas curvas têm propriedades matemáticas únicas que as tornam adequadas para uso em criptografia.
- **Por que são úteis para criptografia?** A segurança dos sistemas de criptografia baseados em curvas elípticas deriva da dificuldade de resolver o problema do logaritmo discreto em grupos formados pelos pontos da curva. Simplificando, dadas dois pontos \(A\) e \(B\) na curva, é computacionalmente viável calcular \(A + A + A + \dots + A = B\) (operação chamada de multiplicação escalar), mas extremamente difícil fazer a operação inversa sem conhecer o número de vezes \(A\) foi somado a si mesmo para obter \(B\).
- **Benefícios em termos de segurança e eficiência:**
  - **Segurança:** Para um dado tamanho de chave, a criptografia baseada em curvas elípticas oferece um nível de segurança maior do que as chaves RSA equivalentes, o que significa que chaves menores podem ser usadas sem reduzir a segurança.
  - **Eficiência:** Chaves menores e operações mais rápidas tornam os algoritmos baseados em curvas elípticas mais eficientes em termos de uso de recursos computacionais e energéticos. Isso os torna particularmente valiosos para dispositivos móveis e hardware com recursos limitados.

A segurança dos sistemas de criptografia que utilizam curvas elípticas, como o ECDH (Elliptic Curve Diffie-Hellman) para troca de chaves e o ECDSA (Elliptic Curve Digital Signature Algorithm) para assinaturas digitais, se baseia fundamentalmente na dificuldade computacional de resolver o problema do logaritmo discreto em grupos formados pelos pontos sobre curvas elípticas.

### Grupos e Operações em Curvas Elípticas

Um grupo em matemática é um conjunto de elementos combinados com uma operação que satisfaz quatro condições básicas (fechamento, associatividade, elemento neutro e existência de inverso). No contexto de curvas elípticas, os "elementos" são os pontos que pertencem à curva, e a "operação" é a adição de pontos.

A adição de dois pontos em uma curva elíptica segue regras geométricas específicas: para somar dois pontos \(A\) e \(B\), você traça uma linha reta que passa por \(A\) e \(B\), e esta linha irá intersectar a curva em um terceiro ponto \(C\). O ponto resultante da adição \(A + B\) é então definido como o ponto "oposto" de \(C\) sobre a curva.

### O Problema do Logaritmo Discreto

O problema do logaritmo discreto, em termos simples, é o seguinte: dado um ponto \(B\) e um ponto \(G\) em uma curva elíptica, encontrar o número \(n\) tal que \(B = nG\) (ou seja, \(B\) é o resultado da adição de \(G\) a si mesmo \(n\) vezes). Embora seja computacionalmente fácil realizar a operação de multiplicação escalar (adicionar \(G\) a si mesmo \(n\) vezes para obter \(B\)), o processo inverso (determinar \(n\) a partir de \(G\) e \(B\)) é extremamente difícil para valores grandes de \(n\). Este problema é conhecido como o logaritmo discreto.

### Por Que Isso Garante Segurança?

A dificuldade de resolver o problema do logaritmo discreto em curvas elípticas é o que torna a criptografia baseada em curvas elípticas segura. Mesmo com computadores poderosos, resolver esse problema para curvas elípticas de tamanho adequado (por exemplo, com um número suficiente de pontos) é impraticável com a tecnologia atual, garantindo a segurança das chaves criptográficas.

A segurança de muitos protocolos criptográficos, incluindo a troca de chaves e assinaturas digitais, depende da dificuldade de resolver problemas matemáticos específicos. No caso da criptografia baseada em curvas elípticas, a segurança é assegurada pela dificuldade do problema do logaritmo discreto em grupos formados pelos pontos da curva, fazendo com que seja uma escolha robusta para proteger informações sensíveis contra adversários que tentam quebrar a criptografia por meio de força bruta ou outros métodos computacionais.

### Por Que é Difícil?

A dificuldade do problema do logaritmo discreto em grupos gerais, e mais especificamente em grupos formados pelos pontos de curvas elípticas, deriva do fato de que, enquanto é relativamente fácil computar \(g^x\) para um dado \(x\) (um processo chamado de exponenciação rápida), o processo inverso, ou seja, encontrar \(x\) dado \(g\) e \(h\), é considerado computacionalmente inviável para valores grandes de \(x\) e grupos grandes. Não existe, até o momento, um algoritmo eficiente que resolva o problema do logaritmo discreto para todos os grupos em tempo polinomial, o que significa que, com os recursos computacionais atuais, resolver este problema para certos grupos levaria um tempo impraticável.

### Importância na Criptografia

A segurança de muitos sistemas de criptografia de chave pública, como o Diffie-Hellman (e sua variante baseada em curvas elípticas, o ECDH) e o DSA (Digital Signature Algorithm) e sua variante ECDSA (Elliptic Curve Digital Signature Algorithm), depende diretamente da dificuldade do problema do logaritmo discreto.

- **Em sistemas de troca de chaves**, como o Diffie-Hellman, a segurança do método de troca de chaves depende da dificuldade de um atacante em calcular a chave secreta compartilhada, mesmo conhecendo os outros componentes públicos da troca.
- **Para assinaturas digitais**, a segurança da assinatura depende da dificuldade de um atacante em forjar uma assinatura válida sem conhecer a chave privada, o que também se reduz ao problema do logaritmo discreto.

### Em resumo

A dificuldade inerente ao problema do logaritmo discreto em certos grupos é o que torna a criptografia baseada nestes princípios segura contra ataques. A escolha de parâmetros seguros, como a escolha de um grupo e um gerador adequados, é crucial para garantir que a solução do problema do logaritmo discreto seja impraticável com a tecnologia atual, protegendo assim as comunicações e dados cifrados com esses métodos. A criptografia baseada em curvas elípticas é uma tecnologia poderosa que permite comunicações seguras com eficiência e segurança elevadas. Ela se tornou um padrão na indústria para muitas aplicações, desde a segurança da web até moedas digitais como o Bitcoin, que se baseia em curvas elípticas para suas assinaturas digitais.

## NLP (Processamento de Linguagem Natural)

O Processamento de Linguagem Natural (NLP, do inglês Natural Language Processing) é uma área de estudo na interseção da ciência da computação, inteligência artificial e linguística, focada em como as máquinas podem entender e interpretar a linguagem humana.

### Processamento de Texto

O processamento de texto é a base do NLP e envolve a manipulação ou análise de texto para realizar tarefas específicas. Aqui estão alguns passos fundamentais:

- **Tokenização:** Divide o texto em unidades menores, como palavras ou frases.
- **Remoção de Stop Words:** Elimina palavras comuns (e.g., "e", "o", "a") que têm pouco valor na análise.
- **Stemming e Lemmatização:** Reduz palavras às suas formas base. Por exemplo, "correndo" e "correu" podem ser reduzidos a "correr".
- **Análise de Partes do Discurso (POS Tagging):** Identifica a função gramatical das palavras no texto (substantivos, verbos, adjetivos, etc.).

### Análise de Sentimentos

A análise de sentimentos é uma aplicação de NLP que identifica e extrai opiniões dentro de um texto. O objetivo é determinar a atitude do autor em relação a certos tópicos ou a polaridade geral do texto. Isso é geralmente classificado como positivo, negativo ou neutro. Empresas usam análise de sentimentos para monitorar sentimentos sobre produtos ou serviços em mídias sociais e avaliações de clientes.

### Classificação de Texto

A classificação de texto envolve a atribuição de uma ou mais categorias a um documento de texto. Usos comuns incluem:

- **Filtragem de Spam:** Classificar emails como "spam" ou "não spam".
- **Categorização de Notícias:** Classificar artigos de notícias em categorias como política, esportes, entretenimento, etc.
- **Análise de Sentimento:** Como mencionado acima, é uma forma de classificação de texto.

Técnicas comuns incluem modelos de aprendizado de máquina como Naive Bayes, Support Vector Machines (SVM) e redes neurais.

### Geração de Linguagem

A geração de linguagem é o processo de produzir texto que é indistinguível do texto humano. Aplicações incluem:

- **Chatbots e Assistente Virtuais:** Responder a perguntas de usuários de forma natural.
- **Geração de Notícias Automatizadas:** Produzir artigos de notícias a partir de dados.
- **Tradução Automática:** Converter texto de uma língua para outra.

Modelos de linguagem como o GPT (Generative Pretrained Transformer) da OpenAI são exemplos avançados de tecnologia de geração de linguagem, treinados em grandes conjuntos de dados para produzir texto coerente e relevante.

### Em resumo

NLP é uma área vasta e em rápido crescimento, com aplicações que vão desde a simplificação de tarefas do dia a dia até o avanço de como interagimos com a tecnologia. A capacidade de processar e entender a linguagem humana abre novas portas para a comunicação entre humanos e máquinas, oferecendo insights valiosos a partir de grandes quantidades de texto e melhorando a interação usuário-máquina em muitos aspectos da vida cotidiana.


## Referências
- [Deep Residual Learning for Image Recognition] https://arxiv.org/abs/1512.03385
- [Uma introdução a Redes Neurais] https://matheusjorge.github.io/introducao-redes-neurais/
- [Funções de ativação] https://matheusjorge.github.io/funcoes-ativacao/
- [Regressão Logística] https://matheusfacure.github.io/2017/02/25/regr-log/
- [Neural Networks From Scratch - Lec 15 - GeLU Activation Function] https://www.youtube.com/watch?v=kMpptn-6jaw
- [CNN Fundamental 3- Why Residual Networks ResNet Works] https://www.youtube.com/watch?v=CwNlWRKW8fo

## Glossário
- ReLU (Rectified Linear Unit): Unidade Linear Retificada -> ReLU(x)=max(0,x)
  . A função de ativação ReLU é uma função não linear que mapeia os valores de entrada para 0 se forem negativos ou para o próprio valor de entrada se forem positivos. Essa função é amplamente utilizada em redes neurais profundas devido à sua simplicidade e eficácia na superação do problema de desaparecimento do gradiente.
- GELU (Gaussian Error Linear Unit): Unidade Linear de Erro Gaussiano -> f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  . Função de ativação não linear que aproxima a função de distribuição Gaussiana acumulada. Essa função de ativação tem a vantagem de ser suave e diferenciável em todo o seu domínio, o que é importante para o treinamento eficiente de redes neurais. A Activation GELU também é conhecida por sua capacidade de lidar com gradientes instáveis, o que é um desafio comum em redes neurais profundas. Isso a torna uma escolha atraente para muitos pesquisadores e praticantes de deep learning.

## Deep Residual Learning for Image Recognition
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- Microsoft Research

"Is learning better networks as easy as stacking more layers?"

Esse questionamento toca em um dos tópicos centrais da pesquisa em deep learning, particularmente no que diz respeito ao desenvolvimento de redes neurais profundas. A "profundidade" de uma rede neural refere-se ao número de camadas ocultas entre a camada de entrada e a camada de saída. Historicamente, percebeu-se que adicionar mais camadas (ou seja, aumentar a profundidade) pode significativamente melhorar a capacidade da rede de aprender representações complexas dos dados e, consequentemente, melhorar o desempenho em tarefas de aprendizado de máquina, como classificação de imagens, processamento de linguagem natural, entre outros.

No entanto, o questionamento “Is learning better networks as easy as stacking more layers?” (É tão fácil aprender redes melhores apenas empilhando mais camadas?) aponta para a complexidade subjacente em simplesmente adicionar mais camadas a uma rede neural na esperança de obter melhores resultados. Existem várias questões e desafios que surgem com redes mais profundas, incluindo:

### Problemas do Gradiente Desaparecendo/Explodindo
- À medida que o número de camadas aumenta, a rede pode sofrer do problema do gradiente desaparecendo ou explodindo, tornando o treinamento extremamente difícil. Isso ocorre porque os gradientes calculados durante o backpropagation podem se tornar tão pequenos (desaparecer) ou tão grandes (explodir) que a atualização dos pesos se torna ineficaz ou instável.

### Dificuldade de Otimização
- Redes mais profundas apresentam uma superfície de erro mais complexa, com muitos mínimos locais, tornando o processo de otimização mais desafiador. Encontrar o conjunto ótimo de pesos para uma rede muito profunda pode ser uma tarefa difícil.

### Overfitting
- Com o aumento da profundidade, a rede pode se tornar excessivamente complexa em relação ao conjunto de dados, resultando em overfitting. Isso significa que a rede aprende a representar o ruído dos dados de treinamento em vez de generalizar a partir das características subjacentes, prejudicando seu desempenho em dados novos, não vistos.

### Soluções e Avanços
Em resposta a esses desafios, temos várias técnicas e arquiteturas para mitigar os problemas associados com redes profundas, tais como:
- **Inicialização Cuidadosa:** Técnicas de inicialização de pesos para prevenir gradientes desaparecendo/explodindo no início do treinamento.
- **Normalização por Lotes (Batch Normalization):** Normaliza as ativações da camada para acelerar o treinamento e combater o desaparecimento do gradiente.
- **Residual Networks (ResNets):** Introduzem conexões de salto ou resíduas, permitindo que o sinal de gradiente flua diretamente através de camadas alternativas, facilitando o treinamento de redes muito profundas.

Portanto, enquanto aumentar a profundidade de uma rede neural tem o potencial de melhorar sua capacidade de aprendizado, não é uma solução mágica e traz consigo um conjunto de desafios que precisam ser abordados de forma cuidadosa e deliberada. A inovação contínua em técnicas de modelagem, arquiteturas de rede e estratégias de treinamento é crucial para aproveitar os benefícios das redes neurais profundas.



## Instalação ambiente conda

```
conda create -n env_pytorch python=3.10
conda activate env_pytorch
conda install pytorch::pytorch
```
