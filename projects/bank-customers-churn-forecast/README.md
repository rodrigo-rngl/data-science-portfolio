<h1><p align= "center"><b>Bank Customers Churn Forecast</b></p></h1>

<p align= "center">
<a href="https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers"><img src= "img/bank-customers-churn-forecast-cover.png" alt= "top bank churn predict cover"></a>
</p>

> Status: Desenvolvimento por versões. Em progresso ⚠️

<p align="right"><i> Este projeto foi sugerido pela Comunidade DS.</i></p>

<div style= "margin: 40px;"></div>

# 1 - Problema de Negócio

**Descrição**: *"A empresa Top Bank atua na Europa tendo como principal produto uma conta bancária. Este produto pode manter o salário do cliente e efetuar pagamentos. Essa conta não tem nenhum custo nos primeiros 12 meses, porém, após esse período, o cliente precisa recontratar o banco para os próximos 12 meses e refazer esse processo todos os anos. Recentemente, a equipe de análise notou que a taxa de churn está aumentando."*

**Objetivo**

Como Cientista de Dados, você precisa: 
- Criar um plano de ação para diminuir o número de clientes churn.
- Criar relatório mostrando o desempenho do modelo criado e o impacto da solução, repondendo as seguintes perguntas:
    1. Qual é a atual taxa de churn do Top Bank?
    2. Qual é o desempenho do modelo na classificação de clientes como churns?
    3. Qual é o retorno esperado, em termos de receita, se a empresa utilizar seu modelo para evitar o churn de clientes?
<hr> 
<div style= "margin: 15px;"></div>

# 2 - Problemas/Suposições de Negócio

Através da descrição dos dados, pude perceber que não existe variável temporal identificando o momento da extração dos dados. Com isso, nasce alguns questionamentos:

**1.** Qual ponto no tempo a extração de dados foi feita? Qual é a relevância desse período?
	
	R: Vou considerar que extração dos dados foi feita no período de 1 ano.

**2.** Se os dados foi extraído como o resumo de 1 ano, o valor de 'saldo' representa o valor presente na conta(s) no momento da extração, o somátorio de saldo presente na(s) conta(s) durante 1 ano ou o valor máximo do saldo presente nesse período.
	
	R: Sem suposições no momento. 

**3.** A variável 'Churn' significa que o cliente de fato saiu da empresa ou possui a vontade de sair. Pois se o cliente de fato saiu da empresa, existe clientes que saíram do banco, mas que ainda possuiam saldo em sua(s) conta(s).

	R: Vamos trabalhar com a idéia de que a variável 'Churn' representa a vontade de rotatividade do cliente no próximo ano.

**4.** O que seria de fato 'membro_ativo', clientes que fizeram transações em todos os meses do ano, ou pelo menos uma vez durante o período de extração?

	R: Vou considerar positivo para 'membro_ativo' clientes que realizaram transações em todos os meses.

Pra esse projeto, irei modelar sem a resposta dessas questões, embora, ter o contexto melhora o entendimento do problema e possivelmente melhores resultados na modelagem dos dados.


# 3 - Estágios da Solução

Meus passos estratégicos para desenvolver a solução do Problema.

1) **Data Description**

2) **Problem Understanding and Solution Planning**

3) **Exploratory Data Analysis**

4) **Feature Engineering**

5) **Feature Selection**

6) **Machine Lerning and Model Metrics**

7) **Business Translation**

<hr> 
<div style= "margin: 15px;"></div>

# 4 - Machine Learning e Métricas

- No primeiro ciclo, foi criado um modelo **Random Forest com 86% de acurácia**;
- No segundo ciclo, foi criado um modelo **XGBoost com 96% de acurácia**;

<hr> 
<div style= "margin: 15px;"></div>

# 5 - Resultados do Negócio

- **V1**

	**1.** Qual é a atual taxa de churn do Top Bank?
    	
		R: Clientes que irão dar Churn correspondem a 20.37% da base de dados e clientes que não irão dar Churn correspondem a 79.63% da base de dados.
	**2.** Qual é o desempenho do modelo na classificação de clientes como churns?
    	
		R: O modelo possui uma performance real de 86% para indentificar churns neste primeiro ciclo.
	**3.** Qual é o retorno esperado, em termos de receita, se a empresa utilizar seu modelo para evitar o churn de clientes?
    	
		R: Se a empresa usar o modelo desenvolvido com um recall de 55.05% (métrica extraída a partir de dados de teste) para clientes com rotatividade positiva, a receita média anual aproximada será de US$908.006.389,32 

		Obs.: Considerando que a empresa vá converter todos os clientes que foram identificados.
        
<hr> 
<div style= "margin: 15px;"></div>

- **V2**

	**1.** Qual é a atual taxa de churn do Top Bank?
    	
		R: Clientes que irão dar Churn correspondem a 20.37% da base de dados e clientes que não irão dar Churn correspondem a 79.63% da base de dados.
	**2.** Qual é o desempenho do modelo na classificação de clientes como churns?
    	
		R: R: O modelo escolhido (XGBoost) possui uma performance real de 96% nesse segundo ciclo.
	**3.** Qual é o retorno esperado, em termos de receita, se a empresa utilizar seu modelo para evitar o churn de clientes?
    	
		R: Se a empresa usar o modelo desenvolvido com um recall de 72.14% para clientes com rotatividade positiva, a receita média anual aproximada será de US$943.316.806,57

		Obs.: Considerando que a empresa vá converter todos os clientes que foram identificados.
<hr> 
<div style= "margin: 15px;"></div>

# 7 - Lessons Learned

- Conhecer os dados é primordial para o planejamento da solução.
- O balanceamento dos dados pode aumentar consideravelmente a chance de overfitting.
- Não tenha medo de experimentar, afinal, é **Ciência** de Dados!

<hr> 
<div style= "margin: 15px;"></div>

<center>Para acessar as versões do projeto, acesse a pasta <b>notebooks</b>.</center>
<center>Obrigado!</center>
<center>Rodrigo Rangel</center>
