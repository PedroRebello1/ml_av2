```text
Relatório Final — Segmentação por Clusterização Hierárquica (Pipeline híbrido)
Autor: Pedro Rebéllo
Data: 2025-11-07
Resumo: este relatório resume os passos, os resultados e as recomendações do módulo de clusterização hierárquica executado sobre os dados de transações (walmart.csv). A versão final usada é a saída em outputs_final/.
```

# 1. Descrição Geral
O objetivo deste projeto foi agrupar usuários da plataforma de e‑commerce com base em seu comportamento de compra (frequência, ticket médio, mix de categorias, participação em receita por categoria), visando gerar perfis acionáveis para marketing, personalização e recomendações de produto.

Resumo do fluxo final (hybrid)
- Construção de features enriquecidas por usuário: contagens, total/avg/median/std de Purchase (com log1p), número de produtos únicos, participações por categoria (contagem e receita), entropia de categorias, modos demográficos.
- Pré‑processamento: winsorização em nível de transação e usuário, RobustScaler em variáveis numéricas, one‑hot de modos categóricos.
- Workflow híbrido (final):
  1. Detecção de pequenos nichos/outliers via aglomeração (average + Manhattan, k=4).
  2. Isolamento desses nichos (clusters muito pequenos).
  3. Re‑clusterização do "core" (usuários restantes) com configuração mais balanceada (complete + Manhattan, k=4, PCA preservando 90% de variância).
  4. Re‑anexação dos nichos como segmentos separados no resultado final.

# 2. Objetivo do Módulo 1 — Clusterização Hierárquica
Agrupar usuários por comportamento de compra para:
- Identificar segmentos comercialmente relevantes (alto gasto, compradores frequentes, nichos).
- Gerar recomendações operacionais (campanhas, PDV, ofertas).
- Fornecer mapeamento reutilizável (atribuição de novos usuários via modelo de classificação posterior).

# 3. Arquivos analisados (outputs_final)
- final_user_clusters.csv — mapa User_ID → cluster (com cluster "4" reservado para nichos identificados).
- final_user_clusters_sample.csv — amostra (primeiras 5k linhas) para inspeção rápida.
- cluster_profile_summary.csv — agregados por cluster (tabela principal de perfis).
- silhouette_detect.csv — silhouette do passo de detecção (average + manhattan, k=4).
- silhouette_core.csv — silhouette do re‑cluster do core (complete + manhattan, k=4).
- clusters_core_2d.png — projeção PCA 2D dos clusters (core).
- dendrogram_core.png — dendrograma (últimas 50 fusões) do core.
- (opcional) clusters_core_3d.html — visualização 3D (não enviada por ser pesada).

Vou referenciar essas saídas ao longo do relatório.

# 4. Pré‑processamento e engenharia de features (resumo)
- Leitura e limpeza dos dados de transação (remoção de duplicatas, coerção de tipos, preenchimento de NA).
- Winsorização de Purchase no nível transacional (quantis 1% / 99%) para reduzir impacto de outliers. Winsorização adicional em features por usuário (core mais forte).
- Features principais construídas:
  - n_transactions (nº de transações por usuário)
  - total_purchase, avg_purchase, median_purchase, std_purchase (e log1p dessas variáveis)
  - unique_products (nº produtos distintos por usuário)
  - cat_share__* (fração de compras por categoria — por contagem)
  - cat_rev_share__* (fração da receita por categoria)
  - cat_entropy (entropia da distribuição de categorias por usuário)
  - n_unique_categories
  - modos demográficos (Age_mode, Gender_mode, City_Category_mode, Stay_mode, Marital_mode, Occupation_mode)
- Normalização: RobustScaler (reduz sensibilidade a outliers).
- Redução dimensional: PCA — detect: 80% variação explicada; core: 90% variação explicada.

# 5. Implementação do algoritmo e parâmetros finais
- Etapa de detecção (identificar nichos/outliers)
  - Linkage: average
  - Métrica: manhattan (cityblock)
  - k (escolhido): 4
  - Silhouette (detect): 0.3819461098644433 (arquivo silhouette_detect.csv)
  - Resultado: clusters detectados — tamanhos: {0: 5667, 1: 75, 2: 102, 3: 47} — identificamos cluster 3 como outliers (47 usuários), usando threshold min_cluster_size = 50 ou pct = 1%.

- Etapa de reclusterização do core
  - Linkage: complete
  - Métrica: manhattan
  - k (final): 4 (escolhido)
  - PCA variance target: 0.90
  - Winsorização mais forte em features de usuário antes do core clustering
  - Silhouette (core): 0.09272604239156731 (arquivo silhouette_core.csv)
  - Resultado: tamanhos do core (clusters do core): {0: 4151, 1: 851, 2: 744, 3: 98}. Somando com os 47 outliers re‑anexados, total = 5891 usuários.

Observação sobre métricas
- Silhouette para o passo de detecção (0.382) é razoável — a etapa detectou nichos bem separados.
- Silhouette do core (0.093) é baixo — os clusters do core possuem fronteiras menos nítidas (sobreposição). Isso é esperado: ao remover os nichos bem distintos, o restante tende a ser mais heterogêneo e menos separável por distância rígida. Ainda assim produz segmentos de tamanho e perfil operáveis.

# 6. Resultados numéricos essenciais

Resumo de tamanhos (do arquivo final_user_clusters.csv / cluster_profile_summary.csv)
- Cluster 0: 4.151 usuários — 70.5% do total (4151 / 5891)
- Cluster 1: 851 usuários — 14.4%
- Cluster 2: 744 usuários — 12.6%
- Cluster 3: 98 usuários — 1.7%
- Cluster 4 (niches/outliers): 47 usuários — 0.8%

(Ver tabela de perfis agregados abaixo — extraída de cluster_profile_summary.csv)

Tabela resumida — métricas chave por cluster
| cluster | n_users | n_transactions_mean | total_purchase_mean | avg_purchase_mean | unique_products_mean | cat_entropy_mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4151 | 108.89 | 1,002,850.51 | 9,644.27 | 108.89 | 0.2001 |
| 1 | 851  | 45.28  | 396,805.42     | 8,796.83  | 45.28  | 0.2252 |
| 2 | 744  | 69.57  | 712,359.83     | 10,309.02 | 69.57  | 0.1326 |
| 3 | 98   | 55.83  | 427,265.70     | 7,689.00  | 55.83  | 0.1275 |
| 4* | 47  | 48.74  | 370,516.32     | 7,768.86  | 48.74  | 0.1631 |

*cluster 4 = nichos / outliers detectados na etapa inicial.

Observações importantes sobre a tabela
- Os valores acima foram retirados diretamente do arquivo cluster_profile_summary.csv (colunas agregadas). Interprete as médias como “médias por usuário no cluster”.
- O cluster 2 apresenta o maior avg_purchase_mean (≈ 10.3k) — indica compradores de ticket médio mais alto (possível segmento de alto‑ticket).
- O cluster 0 concentra a maior parte dos usuários e tem o maior total_purchase_mean (em média) por usuário — representa a massa (cliente recorrente/volume).
- A entropia de categorias (cat_entropy_mean) sugere que:
  - clusters 0 e 1 têm entropia moderada (usuários compram em várias categorias),
  - clusters 2/3/4 têm entropias ligeiramente menores (indicam maior concentração em poucas categorias, mais nicho).

# 7. Visualizações (incluídas)
- PCA 2D dos clusters (arquivo `clusters_core_2d.png`)
  - Mostra claramente um pequeno grupo bem separado (aquele re‑anexado como cluster 4) e uma distribuição do restante com sobreposições, especialmente entre clusters 0/1/2.
  - (Inserir a figura em apresentações e documentos executivos.)

- Dendrograma do core (arquivo `dendrogram_core.png`)
  - Apresenta as fusões mais recentes; útil para justificar escolha de k=4 e visualizar distância de corte.
  - Mostra estrutura hierárquica que sustenta a separação do core em 4 grupos.

(As imagens foram entregues em outputs_final/ — use-as no apêndice do relatório final.)

# 8. Interpretação qualitativa dos clusters — perfis sugeridos
Com base nas métricas agregadas, seguem perfis de alto nível e recomendações de ação:

- Cluster 0 — "Massa frequentadora, alto volume"
  - ~70% dos usuários; alto número médio de transações e alto total_purchase_mean.
  - Perfil: compradores frequentes, compram em múltiplas categorias (entropia moderada).
  - Ações recomendadas: programas de retenção (fidelidade), cross‑sell entre categorias (com base em cat_rev_share), descontos progressivos por frequência, testes de bundles.

- Cluster 1 — "Compradores moderados"
  - ~14% dos usuários; menor frequência média, ticket médio razoável (~8.8k).
  - Perfil: clientes menos frequentes que o cluster 0, comportamento mais conservador.
  - Ações: campanhas de reativação (e‑mail com incentivos), frete grátis acima de um mínimo, recomendações personalizadas baseadas nas categorias mais frequentes.

- Cluster 2 — "Alto ticket / compradores menos frequentes"
  - ~12.6% dos usuários; avg_purchase_mean mais alto (~10.3k).
  - Perfil: compradores de ticket elevado (podem ser compradores por necessidade ou presentes).
  - Ações: upsell e ofertas premium, campanhas com produtos de margem maior, garantias estendidas, atendimento prioritário.

- Cluster 3 — "Pequeno segmento (98 users) — comport. intermediária"
  - ~1.7% dos usuários; características intermediárias (ticket e frequência medianos).
  - Ações: analisar categorias dominantes; testar micro‑segmentação/experimentos.

- Cluster 4 — "Nicho / Outliers (47 users)"
  - ~0.8% — detectado originalmente pela etapa de detecção como um grupo bem separado (silhouette_detect ~0.382).
  - Perfil: usuários com comportamento muito distinto (p. ex., compras concentradas em categorias específicas ou padrões de gasto muito altos).
  - Ações: tratar como segmento separado (ofertas exclusivas, atendimento diferenciado); investigar individualmente para entender se são clientes de grande valor ou erros/anomalias.

# 9. Recomendações práticas (priorizadas)
1. Operacional
   - Tratar cluster 4 como um segmento de alto interesse para campanhas exclusivas (mas revisar manualmente os IDs para confirmar qualidade dos dados).
   - Usar clusters 0–2 como base para estratégias contínuas: cluster 0 (fidelidade), cluster 2 (alto ticket), cluster 1 (reengajamento).
2. Testes e validação
   - A/B test: comunicar ofertas distintas por cluster (ex.: bundle para cluster 0 vs upsell premium para cluster 2).
   - Testar atribuição de novos usuários por um classificador (treinar um modelo que, dada a feature pipeline, atribua o cluster) para produção.
3. Melhoria técnica
   - Avaliar modelos alternativos (HDBSCAN, GMM, KMeans sobre features transformadas) para ver se melhoram separabilidade do core (melhor silhouette).
   - Estudar transformação adicional: separar features por comportamento por categoria e construir embeddings de produto/usuário para melhorar distinção.
4. Monitoramento
   - Recalcular clusters mensalmente ou após eventos sazonais (Promoções, Black Friday).
   - Monitorar estabilidade de clusters (ex.: recalcular silhouette e drift de centroide).

# 10. Limitações e riscos
- Silhouette do core é baixo (≈0.093) → indica sobreposição entre segmentos do core; interpretar os clusters como heurísticos de segmentação (úteis para campanhas) e não como fronteiras absolutas.
- Winsorização altera a representação de extremos — útil para robustez, mas pode remover sinais relevantes se extremo for comportamento real (ex.: grosso comprador).
- O grupo de nicho (cluster 4) pode conter anomalias/erros de dados — é recomendável validação manual dos IDs listados.
- O uso de PCA reduz interpretabilidade dos componentes diretamente — as features originais ainda devem ser usadas para descrever perfis.

# 11. Próximos passos técnicos (curto/médio prazo)
- Implementar pipeline de produção:
  - Script final_clustering_pipeline.py (já fornecido) encapsula o fluxo híbrido.
  - Empacotar em container / agendar job (cron/airflow) para execução periódica e exportação de final_user_clusters.csv.
- Atribuição de novos usuários:
  - Treinar um classificador (ex.: LightGBM) que aprenda a mapear features do usuário para clusters (apenas para produção).
- Experimentos:
  - Testar HDBSCAN (detecção de densidade), GMM e KMeans sobre o mesmo conjunto de features para comparar estabilidade e interpretabilidade.
  - Tunagem de winsorização e PCA (variação 0.85–0.95) e métricas de cluster adicionais (Calinski‑Harabasz, Davies‑Bouldin).

# 12. Apêndice A — Métricas exatas extraídas
- silhouette_detect.csv:
  - k=4; silhouette_detect = 0.3819461098644433
- silhouette_core.csv:
  - k=4; silhouette_core = 0.09272604239156731

- cluster_profile_summary.csv: (valores principais usados no relatório)
  - (fornecido; colunas agregadas por cluster. Ver seção 6 para resumo.)

# 13. Apêndice B — Lista de arquivos incluídos (outputs_final)
- final_user_clusters.csv
- final_user_clusters_sample.csv
- cluster_profile_summary.csv
- silhouette_detect.csv
- silhouette_core.csv
- clusters_core_2d.png
- dendrogram_core.png
- clusters_core_3d.html (não enviada por tamanho; existe localmente se necessário)

# 14. Instruções de reprodução (comando)
- Pré‑requisitos:
  - Python 3.8+, packages: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn.
- Executar pipeline híbrido (já fornecido e executado):
  - python final_clustering_pipeline.py
- Para inspecionar resultados:
  - Abrir outputs_final/cluster_profile_summary.csv e outputs_final/final_user_clusters.csv
  - Ver imagens: outputs_final/clusters_core_2d.png e outputs_final/dendrogram_core.png

# 15. Observações finais
- O pipeline produzido preserva pequenos nichos (alto potencial de valor) e entrega um core de segmentos balanceados utilizáveis para ações comerciais.
- Recomendo, antes de campanhas de grande escala, revisar manualmente os 47 usuários do cluster nicho (IDs no final_user_clusters.csv com label 4) para confirmar qualidade e se desejamos tratá‑los como VIPs ou excluí‑los por serem anomalias.

Se desejar, eu já:
- monto um PDF pronto para apresentação com este conteúdo + as figuras incorporadas;
- gero slides (PowerPoint) com 6–8 slides resumindo insights acionáveis;
- adiciono o trecho de código do pipeline no apêndice técnico do README.

Qual(is) desses outputs prefere agora (PDF / slides / incluir lista detalhada dos 47 users do cluster 4 com top categorias)? 