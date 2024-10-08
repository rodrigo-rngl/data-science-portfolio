<h1><p align= "center"><b>Bank Customers Churn Forecast (Churn Problem)</b></p></h1>

<p align= "center">
<a href="https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers"><img src= "img/bank-customers-churn-forecast-cover.png" alt= "top bank churn predict cover"></a>
</p>

> Status: *Em progresso* ⚠️

<p align="right"><i> Este projeto foi sugerido pela Comunidade DS.</i></p>

<h2 align= "center"><p><a href= "https://nbviewer.org/github/rodrigo-rngl/data-science-portfolio/blob/master/projects/bank-customers-churn-forecast/notebooks/Bank%20Customers%20Churn%20Forecast%20%28pt-br%29.ipynb"><u>Clique aqui para visualizar o projeto!</u></a></p></h2> 

<div style= "margin: 40px;"></div>

# Objetivos do Projeto

O principal objetivo neste notebook é <u>demonstrar o meu conhecimento sobre construção de modelos supervisionados de classificação e sobre a disponibilização e utilização de modelos preditivos em sistemas web</u>. Por esse motivo, escolhi um conjunto de dados desbalanceados de usuários de contas bancárias de uma empresa fictícia (Top Bank) para prever a rotatividade dos clientes - <i>Churn Prediction</i>. Além da criação do modelo de previsão e do sistema web para a utilização do modelo, criei um contexto e questionamentos que devem ser respondidos através da análise do conjunto de dados e do desempenho do modelo.

## Problema de Negócio

**Descrição**: *"A empresa Top Bank atua na Europa tendo como principal produto uma conta bancária. Este produto pode manter o salário do cliente e efetuar pagamentos. Essa conta não tem nenhum custo nos primeiros 12 meses, porém, após esse período, o cliente precisa recontratar o banco para os próximos 12 meses e refazer esse processo todos os anos. Recentemente, a equipe de análise notou que a taxa de churn está aumentando."*

**Objetivo**

Como Cientista de Dados, você precisa: 
- Criar um plano de ação para diminuir o número de clientes churn.
- Criar relatório mostrando o desempenho do modelo criado e o impacto da solução, repondendo as seguintes perguntas:
    1. Qual é a atual taxa de churn do Top Bank?
    2. Qual é o desempenho do modelo na classificação de clientes como churns?
    3. Qual é o retorno esperado, em termos de receita, se a empresa utilizar seu modelo para evitar o churn de clientes?
<hr> 
<div style= "margin: 20px;"></div>

# Problemas/Suposições de Negócio

Através da descrição dos dados, pude perceber que não existe variável temporal identificando o momento da extração dos dados. Com isso, nasce alguns questionamentos:

	1. Qual ponto no tempo a extração de dados foi feita? Qual é a relevância desse período?

	2. Se os dados foi extraído como o resumo de 1 ano, o valor de 'saldo' representa o valor presente na conta(s) no momento da extração, o somátorio de saldo presente na(s) conta(s) durante 1 ano ou o valor máximo do saldo presente nesse período.

	3. A variável 'Churn' significa que o cliente de fato saiu da empresa ou possui a vontade de sair. Pois se o cliente de fato saiu da empresa, existe clientes que saíram do banco, mas que ainda possuiam saldo em sua(s) conta(s).

	4. O que seria de fato 'membro_ativo', clientes que fizeram transações em todos os meses do ano, ou pelo menos uma vez durante o período de extração?

Pra esse projeto, irei modelar sem a resposta dessas questões, embora, ter o contexto melhora o entendimento do problema e possivelmente melhores resultados na modelagem dos dados.


# Estágios da Solução

Meus passos estratégicos para desenvolver a solução do Problema.

1) **Descrição e Manipulação de Dados**: Nesta etapa, busquei entender e validar os dados brutos para análise. Aqui, identifiquei problemas referentes ao perído da extração de dados e assumi que os dados estão consistentes para a construção do modelo. Também validei os tipos das variáveis e criei variávies que poderiam ser úteis para a análise de dados.

2) **Análise Exploratória de Dados**: Na Análise Exploratória de Dados busquei explorar e resumir os principais aspectos dos dados através de visualizações personalizadas.

3) **Modelagem Preditiva**: Criei, otimizei e avaliei o modelo XGBoost, obtendo uma performance de 83% na validação cruzada. Também na modelagem, criei pipelines de transformações para que as etapas de otimização e validação não sofressem vazamento de dados. Ao testar o modelo com dados não vistos, reafirmei a sua performance, verificando que não há overfitting do modelo. 

<hr> 
<div style= "margin: 20px;"></div>

# Performance do Modelo

Na validação cruzada, o modelo desenvolvido conseguiu prever corretamente 83% dos valores previstos (acurácia) e 67% de prever corretamente os clientes sujeitos a rotatividade (recall).
Ao testar o modelo com dados não antes vistos, o modelo atingiu a mesma performance: 84% de acurácia e 67% de recall.

<hr> 
<div style= "margin: 20px;"></div>

# Resultados do Negócio


**1.** Qual é a atual taxa de churn do Top Bank?
	
	R: Clientes que irão dar Churn correspondem a 20.37% da base de dados e clientes que não irão dar Churn correspondem a 79.63% da base de dados.

**2.** Qual é o desempenho do modelo na classificação de clientes como churns?
	
	R: O modelo XGBoost possui uma performance real de 67% para indentificar churns.

**3.** Qual é o retorno esperado, em termos de receita, se a empresa utilizar seu modelo para evitar o churn de clientes?
	
	Considerando que a empresa vá converter todos os clientes que foram identificados pelo modelo desenvolvido com um recall de 67% para clientes com rotatividade positiva, a receita média anual aproximada será de US$933.356.055,48
<hr> 
<div style= "margin: 20px;"></div>

# Lições Aprendidas

- Conhecer os dados é primordial para o planejamento da solução.
- O balanceamento dos dados pode aumentar consideravelmente a chance de overfitting.
- Não tenha medo de experimentar, afinal, é **Ciência** de Dados!
<hr> 
<div style= "margin: 20px;"></div>

#  Referências

GÉRON, Aurélion. **Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow: Conceitos, Ferramentas e Técnicas Para a Construção de Sistemas Inteligentes**. Alta Books, 2021

BROWNLEE, Jason. **SMOTE for Imbalanced Classification with Python**. Disponível em: <https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/>. Acesso em: 08/10/2024
<hr> 
<div style= "margin: 20px;"></div>

<p align= "center">Para acessar as versões do projeto, acesse a pasta <a href= "https://github.com/rodrigo-rngl/data-science-portfolio/tree/master/projects/bank-customers-churn-forecast/notebooks">notebooks</a>.</p>
<p align= "center">Obrigado!</p>
