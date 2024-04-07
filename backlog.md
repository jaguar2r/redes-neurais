# First Thing First

## ISSUE #1: Modelagem do Problema

- **Objetivo:** Familiarizar-se com os conceitos fundamentais do desafio, garantindo uma compreensão introdutória dos objetivos e das tecnologias envolvidas.
- **Desafios:** Encontrar boas referências bibliográficas que forneçam insights relevantes para o problema em questão.
- **Diretrizes:**
  1. **Compreensão:** Dado a quantidade de tempo disponível, priorizar a compreensão abrangente do problema, dedicando tempo para pesquisar e absorver informações críticas que influenciarão a abordagem de modelagem.
  2. **Integração Tecnológica:** Explorar como combinar diferentes tecnologias de maneira eficaz para atingir os objetivos do projeto.

### Ambiente de Desenvolvimento:

- **Local:** Criar um ambiente local utilizando o Conda para gerenciamento de dependências, facilitando a prototipagem. Posteriormente, encapsular o ambiente e a aplicação utilizando Docker para garantir a consistência do ambiente de desenvolvimento.

### Controle de Versão:

- Utilizar o Git para controle de versão, permitindo o rastreamento de alterações.
- O repositório será hospedado no GitHub.

## ISSUE #2. Implementar ResNet no PyTorch

- **Objetivo:** Mostrar habilidade em modificar e implementar redes neurais profundas usando PyTorch, uma biblioteca de aprendizado de máquina para Python.
- **Desafios:** Substituir a função de ativação, customizar a inicialização da camada de normalização, e analisar o aprendizado da rede.
- **Diretrizes:**
  1. **Estudar a arquitetura ResNet** para entender o funcionamento.
  2. **Modificar a função de ativação para GELU** em todas as camadas que originalmente usam ReLU.
  3. **Implemente uma camada de normalização personalizada** e utilizar na rede.
  4. **Treinar a rede** no dataset ImageNet ou Tiny ImageNet e **monitorar o aprendizado**, observando o platô de aprendizado.
  5. **Exportar o modelo** para ONNX no ponto ótimo de aprendizado.

## ISSUE #3. Desenvolver um frontend web que utiliza a rede ResNet

- **Objetivo:** Criar uma interface web que possa classificar imagens em tempo real, usando a câmera do dispositivo e o modelo treinado anteriormente.
- **Desafios:** Hospedar o site e implementar feedback visual das detecções.
- **Diretrizes:**
  1. **Aprender NextJS e wonnx**, ferramentas para desenvolvimento web e execução de modelos ONNX em ambientes web, respectivamente.
  2. **Integrar o modelo ResNet com a interface web**, capturando vídeo da câmera e classificando em tempo real.
  3. **Hospedar o site** usando CloudFlare Pages para demonstrar o uso aplicação.
  4. **Implementar feedback visual** para mostrar os resultados das detecções na interface.

## ISSUE #4. Adicionar um backend em Rust para processamento

- **Objetivo:** Construir um backend robusto para processar dados de detecção e gerenciar imagens.
- **Desafios:** Implementar lógica de backend em Rust, armazenamento de imagens e autenticação segura para acesso aos dados.
- **Diretrizes:**
  1. **Desenvolver o backend** usando Rust, focando em performance e segurança.
  2. **Utilizar Cloudflare Workers** para a lógica serverless e **Cloudflare R2** para armazenar imagens detectadas.
  3. **Criar um endpoint REST** para download das imagens, com autenticação via API key assinada por HMAC.

**Tecnologias:**

1. PyTorch para treinamento de redes neurais;
2. Next.JS e TailwindCSS para desenvolvimento frontend;
3. Rust para backend;
4. PostgreSQL para gerenciamento de banco de dados.

## Estratégia Geral para Resolução:

1. **Familiarizar-se** com as tecnologias e conceitos listados nas instruções;
2. **Estudar cada tecnologia**;
3. **Começar com tarefas isoladas**, quebrar em ISSUES menores;
4. **Testar cada componente** e no final mergear em uma branch principal;
5. **Documentar cada etapa de desenvolvimento**;
6. **Utilizar os príncipios de Clean Code**.
