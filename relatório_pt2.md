# Regras Associativas — Sumário Executivo

Base
- Usuários: 5.891
- Cesta: presença/ausência por usuário
- Modo: pares 1→1 (rápido e estável)
- Parâmetros (produção):  
  - Categoria: support ≥ 4%, confidence ≥ 35%, lift ≥ 1.25 (max_len=2)  
  - Produto: top 40 mais presentes; support ≥ 2%, confidence ≥ 35%, lift ≥ 1.30 (max_len=2)
- Deduplicação: mantemos a direção com MAIOR confiança para cada par {A,B}.

Arquivos de origem (que você enviou)
- outputs_assoc/category/category_item_support.csv
- outputs_assoc/category/category_pair_rules_all.csv (e top200)
- outputs_assoc/product/product_item_support.csv
- outputs_assoc/product/product_pair_rules_all.csv (e top200)

## 1) Principais pares por Categoria (lift alto e suporte ≥ 4%)

| antecedent | consequent | support | confidence | lift |
|---|---|---:|---:|---:|
| cat_17 | cat_18 | 0.0402 | 0.5563 | 2.5525 |
| cat_14 | cat_18 | 0.0706 | 0.4284 | 1.9656 |
| cat_9 | cat_15 | 0.0599 | 0.8610 | 2.0787 |
| cat_9 | cat_10 | 0.0528 | 0.7585 | 1.9195 |
| cat_18 | cat_7  | 0.1030 | 0.4727 | 1.9062 |
| cat_14 | cat_13 | 0.1181 | 0.7168 | 1.8585 |
| cat_17 | cat_13 | 0.0516 | 0.7136 | 1.8503 |
| cat_18 | cat_12 | 0.1073 | 0.4922 | 1.8504 |
| cat_9 | cat_13 | 0.0484 | 0.6951 | 1.8024 |
| cat_18 | cat_10 | 0.1463 | 0.6713 | 1.6988 |

Observações
- cat_9 (categoria 9) é um ótimo "gatilho": quando aparece, há forte associação com cat_10 e cat_15 (lifts ~1.92 e ~2.08) com altas confianças.
- cat_18 conecta-se bem com múltiplas (10, 12, 7) — pode servir de hub de cross-sell.
- cat_17 → (18, 13) tem lifts elevados; suporte já acima de 4%, bom para campanhas específicas.

## 2) Principais pares por Produto
Muito fortes (lift ≥ 2.0, suporte ≥ 6%):

| antecedent | consequent | support | confidence | lift |
|---|---|---:|---:|---:|
| P00111142 | P00112542 | 0.0771 | 0.4355 | 2.2081 |
| P00111142 | P00114942 | 0.0793 | 0.4483 | 2.1824 |
| P00052842 | P00111142 | 0.0636 | 0.3717 | 2.1001 |
| P00005042 | P00028842 | 0.0711 | 0.4211 | 2.0173 |

Fortes (1.90 ≤ lift < 2.0, suporte ≥ 6%):

| antecedent | consequent | support | confidence | lift |
|---|---|---:|---:|---:|
| P00270942 | P00114942 | 0.0814 | 0.4054 | 1.9736 |
| P00270942 | P00057642 | 0.1008 | 0.5017 | 1.9575 |
| P00111142 | P00270942 | 0.0675 | 0.3813 | 1.8981 |

Observações
- Há um “hub” em P00111142 (aparece em várias regras como consequente); se o catálogo permitir, é um candidato a “produto âncora” em bundles/landing.
- P00270942 é outro hub interessante (regras para P00114942 e P00057642, com boa cobertura).

## 3) Ações recomendadas (prioridade)

1. Cross-sell imediato por categoria (site/app)
   - Se usuário exibe/compra cat_9, recomendar itens de cat_10 e cat_15; se cat_18, recomendar cat_10/12/7.  
   - Posição: shelf de “compre também” com peso por score (confidence × lift).

2. Bundles e cupons
   - cat_14 + cat_13/cat_18 (lifts > 1.85) — bundle de linha complementar.
   - Produtos: (P00111142 ↔ P00112542/P00114942), (P00005042 ↔ P00028842).

3. Campanhas de e-mail/Whats com gatilho
   - Se usuário comprou cat_17, enviar oferta de cat_18/13 em até 48h.
   - Se comprou P00270942, ofertar P00057642 (conf 0.50, lift ~1.96).

4. Páginas de produto
   - “Frequentemente comprados juntos” usando as regras de produto com lift ≥ 1.8 e support ≥ 7%.

5. Segmentação por valor (da Parte 1)
   - No cluster 2 (alto ticket), priorizar pares com lift alto e ticket médio maior (mapear produtos premium quando o catálogo estiver disponível).

6. Governança
   - Exclua consequentes já comprados recentemente (janela de 30–60 dias) e itens fora de estoque.
   - Monitore CTR/CR por regra; mantenha um “top 50” ativo.

## 4) Como operacionalizar (score)

Score recomendado por regra para ranqueamento:  
`score = confidence × lift × (0.5 + 0.5 × support)`  
- Dá peso ao lift e à confiança, sem ignorar cobertura.
- Agregue scores por consequente quando várias regras disparam.

## 5) Próximos passos
- Rodar também “por cluster” (categoria) para campanhas segmentadas: suporte mínimo adaptativo = max(5%, 40 usuários).
- Enriquecer com nomes/margem dos produtos para priorizar recomendações rentáveis.
- Atualização mensal das regras; armazenar métricas de performance por regra (CTR, CVR, AOV).
