1. **Utilizando PyTorch implementar ResNet no dataset ImageNet com os seguintes desafios:**
   1. Substituir ativação ReLU por GELU
   2. Inicialização explícita da camada de normalização (implementação própria)
   3. Mostrar platô de aprendizado
   4. Gerar ONNX da rede GELU no ponto ótimo de aprendizado
   5. **Nota:** Você pode utilizar também o dataset Tiny ImageNet (240MB) caso GPU/NET/HD sejam uma restrição para utilizar o ImageNet completo (167G)

2. **Utilizando NextJS e wonnx implementar um frontend web (interface gráfica) que utiliza a rede ResNET desenvolvida em (1) classifica o que está aparecendo na câmera acessada via capacitor com os seguintes desafios:**
   1. Host site via CloudFlare Pages
   2. Feedback visual das detecções

3. **Adicionar ao projeto desenvolvido em (2) um backend em Rust que recebe todas as pessoas detectadas no feed da câmera com os seguintes desafios:**
   1. Lógica via Cloudflare Workers (em Rust)
   2. Storage das imagens via Cloudflare R2
   3. Endpoint REST para download das imagens com auth via API key assinada em HMAC.
