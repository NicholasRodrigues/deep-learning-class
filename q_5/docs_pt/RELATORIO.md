# Exercício 5 — Análise de Sentimento com RNNs

## 1. Pré-processamento dos Dados

O dataset IMDB contém 50k resenhas em HTML com bastante ruído. Aplicamos o seguinte pipeline de limpeza:

1. **Decodificação de entidades HTML** — `&amp;` → `&`, `&lt;` → `<`, etc.
2. **Remoção de tags HTML** — `<br />`, `<p>`, e similares via regex `<[^>]+>`
3. **Lowercase** — colapsa "Great", "GREAT", "great" em um único token
4. **Remoção de caracteres não-alfabéticos** — pontuação, dígitos, caracteres especiais
5. **Tokenização por whitespace** — `split()` simples

**Decisão: sem remoção de stopwords.** Palavras como "not", "no", "never" carregam informação de negação crítica para sentimento — removê-las inverteria o significado de frases como "not good".

**Decisão: sem stemming/lematização.** Com vocabulário de 20k palavras e 50k resenhas, o modelo tem dados suficientes para aprender formas flexionadas. Stemming poderia fundir palavras com sentimento oposto (ex: "caring" vs "careless" → "care").

### Vocabulário e Codificação

- **Vocabulário:** 20.000 palavras mais frequentes (cobertura de 97,6% dos tokens)
- **Tokens especiais:** `<pad>` (índice 0) e `<unk>` (índice 1)
- **Comprimento fixo:** 300 tokens (média das resenhas: 231, mediana: ~175)
  - Resenhas maiores são truncadas, menores recebem zero-padding
- **Split estratificado:** 80% treino (40k) / 10% validação (5k) / 10% teste (5k)

---

## 2. Arquitetura Implementada

Implementamos 4 variantes: **BiLSTM** e **BiGRU**, cada uma com embeddings aprendidos (100d) e pré-treinados GloVe (300d).

### Visão Geral

```
Entrada (batch, 300)
  │
  ▼
Embedding (20002 × dim) + Dropout(0.3)
  │
  ▼
BiLSTM ou BiGRU (2 camadas, hidden=128, dropout=0.5 entre camadas)
  │
  ├──► Self-Attention → (batch, 256)
  ├──► Mean Pooling   → (batch, 256)
  └──► Max Pooling    → (batch, 256)
  │
  ▼
Concatenar → (batch, 768) → Dropout(0.5) → Linear(768, 1) → Sigmoid
```

### O que fizemos além do baseline

Um baseline típico de RNN para sentimento usa apenas o **último hidden state** como representação da sequência. Nós combinamos três representações complementares:

- **Self-Attention** — vetor de pesos aprendível que calcula um score por posição, normaliza via softmax, e produz uma soma ponderada dos hidden states. Posições de padding são mascaradas com -∞. Isso permite ao modelo focar nas palavras mais relevantes para sentimento.
- **Mean Pooling** — média dos hidden states (excluindo padding). Captura o tom geral.
- **Max Pooling** — máximo por dimensão (excluindo padding). Captura o sinal mais forte.

A concatenação desses três vetores (256 × 3 = 768 dims) alimenta a camada de classificação.

### Embeddings

Testamos duas estratégias:

- **Aprendidos (100d):** inicializados aleatoriamente, treinados junto com o modelo
- **GloVe-6B-300d:** vetores pré-treinados do Stanford NLP (6B tokens, Wikipedia + Gigaword), com fine-tuning durante o treinamento. Cobertura de 99,5% do nosso vocabulário. Palavras não encontradas inicializadas com N(0, 0.6).

### Inicialização

- **Pesos recorrentes (W_hh):** inicialização ortogonal — preserva a norma dos vetores e melhora o fluxo de gradientes em sequências longas
- **Biases:** inicializados em zero
- **Embedding do `<pad>`:** fixo em zero (congelado)

---

## 3. Parâmetros de Treinamento

| Parâmetro             | Valor              |
| --------------------- | ------------------ |
| Otimizador            | Adam (lr=0.001, β₁=0.9, β₂=0.999) |
| Função de perda       | Binary Cross-Entropy (BCELoss) |
| Batch size            | 64                 |
| Épocas máximas        | 15                 |
| Gradient clipping     | norma máx. = 1.0   |
| LR scheduler          | ReduceLROnPlateau (fator=0.5, paciência=2) |
| Early stopping        | paciência = 5 épocas |
| Dropout embedding     | 0.3                |
| Dropout entre camadas RNN | 0.5            |
| Dropout pré-classificador | 0.5            |
| Seed                  | 42                 |
| Device                | MPS (Apple Silicon) |

O early stopping monitora a perda de validação e restaura o checkpoint com menor perda. Na prática, os modelos com embeddings aprendidos pararam na época ~9 (melhor: 4) e os com GloVe na época ~7 (melhor: 2).

---

## 4. Resultados

### Acurácia e Perda no Teste

| Modelo     | Embeddings     | Acurácia   | Perda   | Melhor Época | Parâmetros  |
| ---------- | -------------- | ---------- | ------- | ------------ | ----------- |
| BiLSTM     | Aprendido-100d | 90,24%     | 0,2554  | 4            | 2.632.009   |
| BiGRU      | Aprendido-100d | 90,06%     | 0,2630  | 4            | 2.474.313   |
| **BiLSTM** | **GloVe-300d** | **91,30%** | **0,2229** | **2**     | **6.837.209** |
| BiGRU      | GloVe-300d     | 91,02%     | 0,2347  | 2            | 6.628.313   |

**Melhor modelo: BiLSTM + GloVe-300d com 91,30% de acurácia no teste.**

### Curvas de Treinamento

![Curvas de Treinamento](../plots/training_curves.png)

Os gráficos mostram perda e acurácia por época nos conjuntos de treino e validação para os 4 modelos. Observa-se:

- **GloVe acelera convergência:** melhor validação já na época 2 (vs época 4 para aprendidos)
- **Overfitting nos modelos GloVe:** acurácia de treino atinge 99,5% enquanto validação estabiliza em ~91% — o early stopping é essencial para parar no ponto certo
- **LSTM e GRU têm comportamento muito similar**, com LSTM levemente superior

### Comparação de Modelos

![Comparação de Modelos](../plots/model_comparison.png)

### Observações

- **GloVe vs Aprendido (+1%):** embeddings pré-treinados fornecem uma inicialização semântica que beneficia o modelo, especialmente nas primeiras épocas
- **LSTM vs GRU (~0,2%):** diferença marginal; LSTM tem um gate a mais (forget e input separados) que dá controle mais fino sobre a memória
- **Atenção + pooling:** a combinação das 3 representações supera baselines RNN típicas (87-88%) por ~3 pontos percentuais

---

## 5. Exemplos de Predições no Conjunto de Teste

5 resenhas do conjunto de teste com predição do melhor modelo (BiLSTM-GloVe):

### Exemplo 1

> "'The Adventures Of Barry McKenzie' started life as a satirical comic strip in 'Private Eye', written by Barry Humphries and based on an idea by Peter Cook. McKenzie ('Bazza' to his friends) is a lanky, loud, hat-wearing Australian whose two main interests in life are sex (despite never having had..."

| Rótulo verdadeiro | Predição | Confiança |
| ----------------- | -------- | --------- |
| positivo          | positivo ✓ | 81,5%   |

### Exemplo 2

> "For a while it seemed like this show was on 24/7. Then apparently there was a second season or some other kind of continuation of this horrible show about the two most vapid and conceited people who have ever been filmed. All the other comments have captured the essence of these two selfish, haggish..."

| Rótulo verdadeiro | Predição | Confiança |
| ----------------- | -------- | --------- |
| negativo          | negativo ✓ | 99,0%   |

### Exemplo 3

> "Well it's been a long year and I'm down to reviewing the final film for 2004. Panaghoy Sa Suba (Call of The River) placed second in the recent Metro Manila Film Festival. As expected, it didn't do so well at the box office as it was too artsy for the common moviegoers especially since MMFF is the se..."

| Rótulo verdadeiro | Predição | Confiança |
| ----------------- | -------- | --------- |
| positivo          | positivo ✓ | 95,9%   |

### Exemplo 4

> "Call me adolescent but I really do think that this is a great series. If you haven't had a chance to experience a few episodes of the latest Star Trek series, you should definitely watch this one. Perhaps more compelling than that of Voyager's Caretaker, which launched the series with Cpt. Janeway, ..."

| Rótulo verdadeiro | Predição | Confiança |
| ----------------- | -------- | --------- |
| positivo          | positivo ✓ | 99,5%   |

### Exemplo 5

> "A scientist (John Carradine--sadly) finds out how to bring the dead back to life. However they come back with faces of marble. Eventually this all leads to disaster. Boring, totally predictable 1940s outing. This scared me silly when I was a kid but just bores me now. I had to struggle to..."

| Rótulo verdadeiro | Predição | Confiança |
| ----------------- | -------- | --------- |
| negativo          | negativo ✓ | 99,8%   |

### Resumo

| # | Verdadeiro | Predição   | Confiança |
|---|------------|------------|-----------|
| 1 | positivo   | positivo ✓ | 81,5%     |
| 2 | negativo   | negativo ✓ | 99,0%     |
| 3 | positivo   | positivo ✓ | 95,9%     |
| 4 | positivo   | positivo ✓ | 99,5%     |
| 5 | negativo   | negativo ✓ | 99,8%     |

Todos os 5 exemplos classificados corretamente. O modelo mostra maior confiança em resenhas com sentimento explícito (exemplos 2, 4, 5) e confiança mais moderada em resenhas com tom mais sutil (exemplo 1).
