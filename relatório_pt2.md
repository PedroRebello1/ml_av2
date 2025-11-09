```text
Relatório Final — Regras Associativas (Categorias e Produtos)
Autor: Pedro Rebéllo
Data: 2025-11-07
Resumo: este relatório documenta o módulo de regras associativas aplicado ao dataset de transações (walmart.csv), produzindo associações 1→1 entre categorias e entre produtos para suportar iniciativas de cross‑sell, bundles e recomendação. A versão analisada usa a saída em outputs_assoc/.
```

# 1. Descrição Geral
O módulo de Regras Associativas (Parte 2) visa descobrir padrões de co‑ocorrência entre categorias e produtos mais presentes nas cestas agregadas por usuário para gerar ações de marketing e recomendação. Foi adotado um enfoque pragmático orientado a pares (1→1) com métricas suporte, confiança e lift, priorizando interpretabilidade e execução rápida.

Resumo do fluxo final (produção)
- Agregação de transações por usuário → cesta binária (presença/ausência).
- Dois níveis de mineração: Categorias (20) e Produtos (top 40 por presença).
- Estratégia primária: Pair Mode (enumerar pares 1→1) com deduplicação de direções (mantém a de maior confiança).
- Thresholds calibrados para gerar conjuntos de regras enxutos e de alto impacto.
- Geração de relatórios, tabelas e figuras de distribuição suporte × confiança (colorido por lift).
- Preparação de fórmula de score para ranqueamento (confidence × lift × fator de suporte).

# 2. Objetivo do Módulo 2 — Regras Associativas
Extrair padrões acionáveis de associação entre itens para:
- Viabilizar recomendações “frequentemente comprados juntos” (site / app).
- Criar bundles e cupons dirigidos a pares de alta complementaridade.
- Priorizar campanhas baseadas em gatilhos (se usuário compra X, ofertar Y).
- Alimentar uma camada de recomendação simples escalável antes de modelos mais complexos.

# 3. Arquivos analisados (outputs_assoc)
Categoria:
- category_item_support.csv — suporte (contagem de usuários) por categoria.
- category_pair_rules_all.csv — todas as regras 1→1 categoria (após filtros e deduplicação).
- category_pair_rules_top200.csv — top regras por ordenação (lift, confiança).
- category_pair_rules_scatter.png — suporte × confiança (cor = lift).

Produto:
- product_item_support.csv — suporte por produto (top 40).
- product_pair_rules_all.csv — todas as regras 1→1 produto.
- product_pair_rules_top200.csv — top pares produto.
- product_pair_rules_scatter.png — suporte × confiança (cor = lift).

Geral:
- association_rules_report.md — relatório gerado automaticamente pelo pipeline.

(Não foram geradas nesta rodada as saídas por cluster; podem ser incluídas em execução futura.)

# 4. Pré‑processamento e construção das cestas
- Leitura de walmart.csv com normalização de tipos (User_ID, Product_ID, Product_Category).
- Criação de uma variável categórica padronizada de categoria: Product_Category_str = "cat_<n>".
- Remoção de duplicatas (User_ID, Item) para presença binária.
- Conversão para matriz usuário × item (valores {0,1}).
- Seleção de itens para produto: top 40 por número de usuários distintos (garante foco em itens com massa crítica).
- Filtros de suporte mínimo inicial (categoria ≥ 4% dos usuários; produto ≥ 2% dos usuários) aplicados na fase de construção da cesta.

# 5. Implementação do algoritmo e parâmetros finais
Modo adotado: Pair Mode (enumerar pares e calcular métricas diretamente).
- Métricas calculadas:
  - support(X,Y) = usuários com X e Y / total de usuários
  - confidence(X→Y) = usuários com X e Y / usuários com X
  - lift(X→Y) = confidence(X→Y) / support(Y)

Parâmetros finais (produção):
- Categorias:
  - min_support (par) implícito via suporte mínimo dos itens (≥ 4%); pares aparecem se ambos atendem.
  - min_conf = 0.35
  - min_lift = 1.25
  - max_len = 2 (apenas pares)
- Produtos:
  - top_n_items = 40
  - min_support itens ≥ 2%
  - min_conf = 0.35
  - min_lift = 1.30
  - max_len = 2
- Deduplicação de direção: para cada conjunto {A,B} mantém a regra (A→B ou B→A) com maior confiança.
- Score sugerido (não armazenado nas tabelas originais, mas definido): score = confidence × lift × (0.5 + 0.5 × support) para ranqueamento final de recomendações.

Motivos da escolha:
- Pair Mode evita explosão combinatória de itemsets e facilita controle direto sobre interpretabilidade.
- Limitar a produtos mais presentes melhora estabilidade estatística das métricas (reduz pares com suporte frágil).
- Lift & Confiança moderados filtram correlações triviais de popularidade.

# 6. Resultados numéricos essenciais
Estatísticas base:
- Usuários totais: 5.891
- Categorias distintas (após filtro): 20
- Produtos analisados: 40 (mais presentes)

Output de regras (após filtros e deduplicação):
- Categorias: 71 regras 1→1
- Produtos: 227 regras 1→1

Exemplos de regras fortes (Categorias):
| antecedent | consequent | support | confidence | lift |
|------------|------------|---------|-----------:|-----:|
| cat_17 | cat_18 | 0.0402 | 0.5563 | 2.5525 |
| cat_9  | cat_15 | 0.0599 | 0.8610 | 2.0787 |
| cat_9  | cat_10 | 0.0528 | 0.7585 | 1.9195 |
| cat_18 | cat_10 | 0.1463 | 0.6713 | 1.6988 |
| cat_14 | cat_13 | 0.1181 | 0.7168 | 1.8585 |

Exemplos de regras fortes (Produtos):
| antecedent  | consequent  | support | confidence | lift |
|-------------|-------------|---------|-----------:|-----:|
| P00111142 | P00112542 | 0.0771 | 0.4355 | 2.2081 |
| P00111142 | P00114942 | 0.0793 | 0.4483 | 2.1824 |
| P00052842 | P00111142 | 0.0636 | 0.3717 | 2.1001 |
| P00005042 | P00028842 | 0.0711 | 0.4211 | 2.0173 |
| P00270942 | P00057642 | 0.1008 | 0.5017 | 1.9575 |

Interpretação breve:
- Alguns itens (ex.: P00111142, P00270942) funcionam como hubs (aparecem em múltiplas regras com lift alto).
- Em categoria, cat_9 e cat_18 atuam como gatilhos de cross-sell fortes.

# 7. Visualizações (incluídas)
- category_pair_rules_scatter.png:
  - Distribuição suporte vs confiança; pontos com lift > ~1.8 destacados (cores mais quentes). Observa-se cluster de regras com suporte baixo-moderado e confiança alta.
- product_pair_rules_scatter.png:
  - Regras de produto apresentam suporte menor (≈6–12%) e confianças concentradas entre 0.35–0.50; lifts > 1.9 são mais raros. Indica complementaridade específica.

Utilização:
- Inserir scatter de categorias em apresentações executivas para justificar seleção de pares de maior lift.
- Scatter de produtos para mostrar diversidade de oportunidades de bundle.

# 8. Interpretação qualitativa — perfis de associação
Categorias (padrões principais):
- Gatilhos: cat_9 e cat_17 (quando presentes, ativam consequentes de alto lift — cat_10, cat_15, cat_18).
- Hubs: cat_18 e cat_13 agregam conexões multiplas (rede de ofertas cruzadas).
- Complementaridade nicho: pares envolvendo cat_14, cat_17 (lift alto porém suporte mais baixo) — bons para campanhas especializadas.

Produtos:
- Hub premium/alto envolvimento: P00111142 (diversos consequentes com lift > 2).
- Combos consolidados: P00005042 ↔ (P00028842, P00059442, P00148642) sugerem bundles baseados em complementaridade funcional ou ciclo de uso.
- Conjunto de prolongamento: P00270942 com consequentes P00057642 e P00114942 — potencial upsell se usuário já possui um dos itens.

# 9. Recomendações práticas (priorizadas)
1. Cross-sell dinâmico:
   - Se usuário tem cat_9, recomendar itens de cat_10 e cat_15 em bloco “Compre também”.
   - Em produto, se usuário tem P00111142, priorizar recomendação de P00112542 / P00114942.

2. Bundles e kits:
   - Pacotes (P00005042, P00028842) reforçados por lift > 2.0.
   - Kits de categorias cat_18 + cat_10 (alta cobertura e lift > 1.6).

3. Campanhas gatilho (e-mail / push):
   - Compra de cat_17 → oferta direcionada de cat_18 (lift > 2.5).
   - Compra de P00270942 → voucher para P00057642 (conf 0.50, lift 1.96).

4. Página de produto / PDP:
   - “Frequentemente comprados juntos” baseado nas top 50 regras por score (filtrar itens já no carrinho).

5. Integração com segmentação (Parte 1):
   - Em cluster de alto ticket (Cluster 2), destacar pares de produtos premium (ex.: P00111142 hubs).
   - No cluster “massa” (Cluster 0), usar pares de alta cobertura (cat_18 ↔ cat_10) para elevar AOV.

6. Governança e validação:
   - Monitorar CTR e conversão de cada regra — desativar regras com CTR < baseline após 2 ciclos.
   - Excluir recomendação de itens recém-comprados (janela de saturação).

# 10. Limitações e riscos
- Cesta agregada por usuário não diferencia temporalidade; uma associação pode refletir compras feitas em momentos distintos.
- Foco em pares: não captura relações de conjuntos maiores (triplet combos); pode perder sinal de pacotes avançados.
- Lift inflado em itens de suporte baixo — mitigado por thresholds, mas ainda exige validação de negócio.
- Ausência de nomes/margens no dataset limita priorização econômica automática.
- Sem cluster-level regras nesta versão — perde nuances segmentadas (planejado como extensão).

# 11. Próximos passos técnicos (curto/médio prazo)
- Extensão por cluster: repetir Pair Mode com suporte adaptativo (≥ max(5%, 40 usuários)).
- Implementar pipeline de produção diário/semanal para atualizar regras (script de extração + export JSON para serviço de recomendação).
- Adicionar camada de filtragem por disponibilidade/estoque e margem.
- Integrar com modelo de ranking (ex.: LightGBM) usando features adicionais (popularidade recente, position bias).
- Evoluir de pares para itemsets >2 com FP-Growth controlado (max_len=3) apenas em categorias de maior densidade.

# 12. Apêndice A — Métricas exatas extraídas
- Contagem de usuários por categoria (top 5): cat_1: 5.767; cat_5: 5.751; cat_8: 5.659; cat_2: 4.296; cat_6: 4.085.
- Regras categoria total: 71 (após filtro e deduplicação).
- Regras produto total: 227 (top 40 produtos).
- Exemplos de lifts extremos:
  - cat_17 → cat_18: lift 2.5525
  - cat_14 → cat_12: lift 2.1139
  - P00111142 → P00112542: lift 2.2081
  - P00111142 → P00114942: lift 2.1824
  - P00005042 → P00028842: lift 2.0173

Score sugerido (não na base):
- Fórmula: score = confidence × lift × (0.5 + 0.5 × support)

# 13. Apêndice B — Lista de arquivos incluídos (outputs_assoc)
- association_rules_report.md
- category/category_item_support.csv
- category/category_pair_rules_all.csv
- category/category_pair_rules_top200.csv
- category/category_pair_rules_scatter.png
- product/product_item_support.csv
- product/product_pair_rules_all.csv
- product/product_pair_rules_top200.csv
- product/product_pair_rules_scatter.png

# 14. Instruções de reprodução (comando)
Pré‑requisitos:
- Python 3.8+; pacotes: pandas, numpy, mlxtend, seaborn, matplotlib, networkx (opcional).
Passos:
1. Instalar dependências:
   - pip install mlxtend seaborn networkx
2. Executar pipeline (versão produção Pair Mode):
   - python association_rules_pipeline_colab.py
3. Ver resultados:
   - Abrir outputs_assoc/category_pair_rules_top200.csv (categorias)
   - Abrir outputs_assoc/product_pair_rules_top200.csv (produtos)
   - Visualizar figures: category_pair_rules_scatter.png, product_pair_rules_scatter.png
4. (Opcional) Aplicar recomendador simples:
   - python apply_rules_recommender.py (ou importar funções e usar recommend())

# 15. Observações finais
O módulo entrega um conjunto compacto de regras com bom equilíbrio entre cobertura e lift, pronto para alimentar componentes simples de recomendação e iniciativas de cross‑sell. A integração com a segmentação (Parte 1) amplia poder de personalização. Próxima evolução natural: adicionar camada por cluster e inserção de métricas de performance para governança contínua.
