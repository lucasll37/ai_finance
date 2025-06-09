- Criar sinal dos indicadores de volume
- Entender Ichimoku
- Cluesterizar por curva de retorno sob ação do algoritmo
- Extender funcionalidade de plot de retorno acumulado para demais gráficos (proto em agent_nn.naive_choice)



É feito o download das séries históricas dos ativos que compõem o Indice Bovespa
Para cada um dos ativos é feito o seguinte:
- Cria-se uma variavel target sintética baseada em tendencia (alta, baixa ou neutra). Isso é feito com base em médias moveis. use-se tres médias móveis de periodos diferentes. quando existe a evidencia de alta ou baixa, recorta-se o intervalo em que se manteve essa tendencia. a variavel target passa a ser alta (ou baixa) no subintervalo entre os extremos de preço dessa região. O restante do intervalo é marcado como neutro.

- Cria uma série de sinais features baseados em indicadores técnicos. Esses sinais são baseados em médias móveis, RSI, MACD, Bollinger Bands, etc. A ideia é que esses sinais ajudem a identificar padrões de comportamento do ativo.

- A seguir cria-se uma série temporal dos mesmos indicadores para uma janela de tempo fixa. Essa janela é usada para treinar o modelo de aprendizado de máquina.

Com o intuito de tornar o modelo especialista em operações de swing trade do mercado braasileiro, os dados preprocessados de cada ativo são unificados para que se desfrute de quantidade vasta de dados.

Existem tres benchmarks:
- Operar segundo o sinal target
- Operar segundo um sinal aleatório de tendencia
- Operar segundo regras simples de compra baseados unicamente em um indicador técnico e um politica simplificado (exemplo: comprar quando o RSI estiver abaixo de 30 e vender quando estiver acima de 70)

E será comparado com um modelo de aprendizado de máquina que aprende a operar com base nos dados preprocessados.

Ao fim, será feito a comparação entre os resultados dos benchmarks e do modelo de aprendizado de máquina e a performance do marcado (investir no indice Bovespa) para o mesmo período. Sera usando o beta e o alfa de cada modelo para comparar a performance.

Também sera feita análise de importancia de features para entender quais indicadores técnicos são mais relevantes para o modelo de aprendizado de máquina.