import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
import matplotlib.colors as mcolors
from matplotlib import cm
import random

def _n_bins(numeric_variable):
    n = numeric_variable.shape[0]
    k = round(1 + (3.3 * np.log10(n)))    
        
    return k


def _comparison_test_for_ordinal_or_quantitative_vars(qualitative_var, quant_var= None, numeric_ordinal_var= None):
    """
    Realiza testes estatísticos de comparação entre variáveis qualitativas e quantitativas ou ordinais, 
    com base na normalidade dos dados e na natureza das variáveis. 

    Dependendo dos tipos de variáveis passadas, a função seleciona e aplica o teste estatístico adequado, 
    incluindo T-Test, ANOVA, Mann-Whitney U, ou Kruskal-Wallis.

    Parâmetros:
    ----------
    qualitative_var : pandas.Series
        Variável qualitativa (categórica) usada para agrupar a variável quantitativa ou ordinal. Deve conter pelo menos 
        duas categorias para a comparação.
    
    quant_var : pandas.Series, opcional
        Variável quantitativa contínua. Se fornecida, a função realiza um teste paramétrico ou não-paramétrico, 
        dependendo da normalidade dos dados.
    
    numeric_ordinal_var : pandas.Series, opcional
        Variável numérica ordinal. Se fornecida, a função realiza um teste não-paramétrico adequado para dados ordinais.

    Retorno:
    --------
    p_value : float
        O valor p do teste estatístico aplicado.
    
    test_name : str
        O nome do teste estatístico aplicado. Pode ser 'T-Test', 'ANOVA Test', 'Mann Whitney U Test' ou 'Kruskal-Wallis Test'.

    Erros:
    -------
    AttributeError
        É levantado se não for passada uma variável numérica (quantitativa ou ordinal) ou se ambas forem fornecidas 
        simultaneamente.

    Notas:
    ------
    - A função assume que se a variável quantitativa não for normal na análise univariada, serão usados testes não-paramétricos.
    - Para a análise bivariada (variável qualitativa vs. quantitativa), a normalidade de cada grupo é verificada separadamente.
    - Se todos os grupos forem normais, são aplicados testes paramétricos (T-Test ou ANOVA). Caso contrário, são usados 
      testes não-paramétricos (Mann-Whitney U ou Kruskal-Wallis).
    - Quando uma variável ordinal é fornecida, a função sempre aplica testes não-paramétricos, como Mann-Whitney U ou 
      Kruskal-Wallis.

    Exemplos de uso:
    ----------------
    >>> p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(qualitative_var, quant_var=quant_var)
    >>> p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(qualitative_var, numeric_ordinal_var=numeric_ordinal_var)
    """

    if (quant_var is not None) & (numeric_ordinal_var is None):
        # Padroniza a variável quantitativa
        standardized_variable = sts.zscore(quant_var)
        
        # Verifica normalidade para a variável quantitativa (univariada)
        if len(quant_var) <= 50:
            _, p_value = sts.shapiro(standardized_variable)
        else:
            _, p_value = sts.kstest(standardized_variable, 'norm')
        
        is_normal = p_value > 0.05
        
        # Se a variável quantiativa não for normal na análise univariada, usar testes não paramétricos
        if is_normal:
            # Verifica normalidade da variável contínua por grupo da variável qualitativa (bivariada)
            group_normality_results = {}
            for category in qualitative_var.unique():
                group_data = standardized_variable[qualitative_var == category]
                if len(group_data) <= 50:
                    _, p_value = sts.shapiro(group_data)
                else:
                    _, p_value = sts.kstest(group_data, 'norm')

                group_normality_results[category] = p_value > 0.05
            
            # Escolha do teste estatístico baseado na normalidade dos grupos
            if all(group_normality_results.values()):
                # Caso todos os grupos possuam distribuições normais, utiliza testes paramétricos
                if qualitative_var.nunique() == 2:
                    _, p_value = sts.ttest_ind(*[quant_var[qualitative_var == cat] for cat in qualitative_var.unique()])
                    test_name = 'T-Test'
                else:
                    _, p_value = sts.f_oneway(*[quant_var[qualitative_var == cat] for cat in qualitative_var.unique()])
                    test_name = 'ANOVA Test'
            else:
                # Caso algum dos grupos não seja normal, utiliza testes não paramétricos
                if qualitative_var.nunique() == 2:
                    _, p_value = sts.mannwhitneyu(*[quant_var[qualitative_var == cat] for cat in qualitative_var.unique()])
                    test_name = 'Mann Whitney U Test'
                else:
                    _, p_value = sts.kruskal( *[quant_var[qualitative_var == cat] for cat in qualitative_var.unique()])
                    test_name = 'Kruskal-Wallis Test'
        else:
            # Caso algum dos grupos não seja normal, utiliza testes não paramétricos
            if qualitative_var.nunique() == 2:
                _, p_value = sts.mannwhitneyu(*[quant_var[qualitative_var == cat] for cat in qualitative_var.unique()])
                test_name = 'Mann Whitney U Test'
            else:
                _, p_value = sts.kruskal(*[quant_var[qualitative_var == cat] for cat in qualitative_var.unique()])
                test_name = 'Kruskal-Wallis Test'

        return p_value, test_name
    elif (quant_var is None) & (numeric_ordinal_var is not None):
        if qualitative_var.nunique() == 2:
            _, p_value = sts.mannwhitneyu(*[numeric_ordinal_var[qualitative_var == cat] for cat in qualitative_var.unique()])
            test_name = 'Mann Whitney U Test'
        elif (qualitative_var.nunique() < 2) | (len(np.unique(numeric_ordinal_var)) < 2):
            p_value = float(1)
            test_name = None

            return p_value, test_name
        else:
            _, p_value = sts.kruskal(*[numeric_ordinal_var[qualitative_var == cat] for cat in qualitative_var.unique()])
            test_name = 'Kruskal-Wallis Test'
        return p_value, test_name
    else:
        raise AttributeError("Passe pelo menos e apenas uma variavel numérica, ou 'quant_var' ou 'numeric_ordinal_var'")



def _calculate_continuous_variable_metrics(numeric_variable):
    """
    Calcula várias métricas descritivas para uma variável contínua.

    Parâmetros:
    ----------
    numeric_variable : pd.Series
        Uma série do Pandas contendo valores numéricos contínuos.

    Retorno:
    -------
    metrics_dict : dict
        Um dicionário contendo as seguintes métricas:
        - 'minimum': valor mínimo
        - 'maximum': valor máximo
        - 'mean': valor médio
        - 'median': valor mediano
        - 'mode': valor modal (se houver mais de um, retorna o menor)
        - 'first_quartile': primeiro quartil (25º percentil)
        - 'third_quartile': terceiro quartil (75º percentil)
        - 'lower_fence': limite inferior para detecção de outliers
        - 'upper_fence': limite superior para detecção de outliers
    """
    # Calcula valor mínimo e máximo
    minimum_value = numeric_variable.min()
    maximum_value = numeric_variable.max()

    # Calcula medidas de posição
    mean_value = numeric_variable.mean()
    median_value = numeric_variable.median()
    mode_value = numeric_variable.mode().min()  # Retorna o menor valor em caso de múltiplos modos

    # Calcula quartis
    first_quartile_value = numeric_variable.quantile(0.25)
    third_quartile_value = numeric_variable.quantile(0.75)
    
    # Calcula intervalo interquartil
    interquartile_range = third_quartile_value - first_quartile_value
    
    # Calcula "fences" para detecção de outliers
    lower_fence_value = max(first_quartile_value - (1.5 * interquartile_range), minimum_value)
    upper_fence_value = min(third_quartile_value + (1.5 * interquartile_range), maximum_value)
    
    # Compila todas as métricas em um dicionário
    metrics_dict = {
        'minimum': minimum_value,
        'maximum': maximum_value,
        'mean': mean_value,
        'median': median_value,
        'mode': mode_value,
        'first_quartile': first_quartile_value,
        'third_quartile': third_quartile_value,
        'lower_fence': lower_fence_value,
        'upper_fence': upper_fence_value
    }
    
    return metrics_dict


def plot_continuous_variables_distributions(continuous_numeric_vars_df):
    """
    Plota a distribuição de variáveis contínuas em um DataFrame utilizando histogramas e boxplots.

    Parâmetros:
    -----------
    continuous_numeric_variables_dataframe : pd.DataFrame
        Um DataFrame do Pandas contendo variáveis numéricas contínuas.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    list_continuous_variables = list(continuous_numeric_vars_df.columns)
    
    for var_name in list_continuous_variables:
        variable = continuous_numeric_vars_df[var_name]

        # Chama n_bins para definir a quantidade de intervalos nas distribuições
        bins = _n_bins(variable)
        
        # Calcula as métricas da variável
        metrics_dict = _calculate_continuous_variable_metrics(variable)

        # Cria figura com duas área de plotagem
        fig, axes = plt.subplots(2, 1, figsize= (16, 4.5), gridspec_kw= {'height_ratios': [2.5, 1]})

        # Cria histograma com linhas representando as métricas
        axes[0].hist(variable, color= '#34673e', edgecolor= 'black', bins= bins)

        # Cria boxplot e adiciona customizações
        axes[1].boxplot(variable, vert= False)

        # Adiciona customizações aos gáficos
        ## Adiciona título a figura
        fig.suptitle(f"Distribution ({var_name})", fontsize= 14, fontweight= 'bold')
        ## Adiciona linhas verticais para cada métrica no histograma
        axes[0].axvline(x=metrics_dict['minimum'], color='black', linestyle='dashed', linewidth=2, label=f"Minimum: {metrics_dict['minimum']:.1f}")
        axes[0].axvline(x=metrics_dict['lower_fence'], color='gray', linestyle='dashed', linewidth=2, label=f"Lower Fence: {metrics_dict['lower_fence']:.1f}")
        axes[0].axvline(x=metrics_dict['mean'], color='cyan', linestyle='dashed', linewidth=2, label=f"Mean: {metrics_dict['mean']:.1f}")
        axes[0].axvline(x=metrics_dict['median'], color='red', linestyle='dashed', linewidth=2, label=f"Median: {metrics_dict['median']:.1f}")
        axes[0].axvline(x=metrics_dict['mode'], color='yellow', linestyle='dashed', linewidth=2, label=f"Mode: {metrics_dict['mode']:.1f}")
        axes[0].axvline(x=metrics_dict['upper_fence'], color='gray', linestyle='dashed', linewidth=2, label=f"Upper Fence: {metrics_dict['upper_fence']:.1f}")
        axes[0].axvline(x=metrics_dict['maximum'], color='black', linestyle='dashed', linewidth=2, label=f"Maximum: {metrics_dict['maximum']:.1f}")
        ## Adiciona legendas ao histograma
        axes[0].legend(loc='upper right', fontsize= 'x-small', fancybox= True, framealpha= 0.9, shadow= True, borderpad= 1)
        ## Adiciona e customiza grades
        axes[0].grid(color= "gray", linestyle= "dotted", linewidth= 0.5)
        axes[1].grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'x')
        axes[0].set_axisbelow(True) # A grade fica atrás das barras
        axes[1].set_axisbelow(True)
        ## Customiza labels e intervalos nos eixos verticiais dos gráficos
        axes[0].set_ylabel('Absolute Frequencies (count)')
        axes[0].yaxis.label.set_size(10)  # Ajusta o tamanho da label
        axes[0].yaxis.label.set_fontstyle('italic') # Fonte itálico para label
        axes[0].tick_params(axis='y', labelsize= 9, labelrotation=0)
        axes[1].yaxis.set_ticks([])
        ## Customiza labels e intervalos nos eixos horizontais dos gráficos
        axes[1].set_xlabel(var_name)  
        axes[1].xaxis.label.set_fontstyle('italic') # Define a label do eixo x como itálico
        axes[1].tick_params(axis='x', labelsize= 9, labelrotation=0)
        
        # Exibe o gráfico
        plt.show()

    return None


def plot_discrete_variables_distributions(discrete_numeric_vars_df):
    """
    Plota a distribuição de variáveis discretas em um DataFrame utilizando gráficos de barras.

    Parâmetros:
    -----------
    discrete_numeric_vars_df : pd.DataFrame
        Um DataFrame do Pandas contendo variáveis numéricas discretas.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    # Lista de variáveis discretas
    list_discrete_vars = list(discrete_numeric_vars_df.columns)
    
    for var_name in list_discrete_vars:
        variable = discrete_numeric_vars_df[var_name]
        n_unique_values = variable.nunique()

        # Cria figura com uma área de plotagem
        fig, ax = plt.subplots(1, 1, figsize= (16, 4.5))

        # Cria do countplot
        sns.countplot(x= variable, edgecolor= 'black', color= '#34673e', ax= ax)

        # Adiciona customizações ao gráfico
        fig.suptitle(f"Frequencies ({var_name})", fontsize= 14, fontweight= 'bold')
        ## Adiciona e customiza grades da horizontal
        ax.grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
        ax.set_axisbelow(True) # A grade fica atrás das barras
        ## Customiza labels e intervalos nos eixos verticiais dos gráficos
        ax.set_ylabel('Absolute Frequencies (count)')
        ax.yaxis.label.set_fontstyle('italic') # Fonte itálico para label
        ax.yaxis.label.set_size(10)  # Ajusta o tamanho da label
        ax.tick_params(axis= 'y', labelsize= 9, labelrotation= 0)
        ## Customiza labels e intervalos nos eixos horizontais dos gráficos
        ax.set_xlabel(var_name)  
        ax.xaxis.label.set_fontstyle('italic') # Define a label do eixo x como itálico
        ax.xaxis.label.set_size(10)
        ax.tick_params(axis='x', labelsize= 9, labelrotation= 0)
        # Rotaciona os rótulos no eixo x se houver mais de 12 categorias
        if variable.value_counts().shape[0] > 12:
            ax.tick_params(axis= 'x', labelrotation= 90)
            
        # Exibe o gráfico
        plt.show()

    return None


def plot_nominal_variables_distributions(nominal_categorical_vars_df):
    """
    Plota a distribuição de variáveis categóricas nominais em um DataFrame utilizando gráficos de contagem.

    Parâmetros:
    -----------
    nominal_categorical_vars_df : pd.DataFrame
        Um DataFrame do Pandas contendo variáveis categóricas nominais.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    # Lista de variáveis nominais
    list_nominal_vars = list(nominal_categorical_vars_df.columns)
    
    for var_name in list_nominal_vars:
        n_unique_values = nominal_categorical_vars_df[var_name].nunique()

        # Cria figura com uma área de plotagem
        fig, ax = plt.subplots(1, 1, figsize=(16, 4.5))

        # Define a paleta de cores e a embaralhas
        palette = sns.color_palette('cubehelix', n_unique_values)
        random.shuffle(palette)

        # Criação do countplot
        sns.countplot(data= nominal_categorical_vars_df, x= var_name, edgecolor='black', hue= var_name, palette= palette, legend= False, ax=ax)
        
        # Adiciona customizações ao gráfico
        fig.suptitle(f"Frequencies ({var_name})", fontsize= 14, fontweight= 'bold')
        ## Adiciona e customiza grades da horizontal
        ax.grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
        ax.set_axisbelow(True) # A grade fica atrás das barras
        ## Customiza labels e intervalos nos eixos verticiais dos gráficos
        ax.set_ylabel('Absolute Frequencies (count)')
        ax.yaxis.label.set_fontstyle('italic') # Fonte itálico para label
        ax.yaxis.label.set_size(10)  # Ajusta o tamanho da label
        ax.tick_params(axis= 'y', labelsize= 9, labelrotation= 0)
        ## Customiza labels e intervalos nos eixos horizontais dos gráficos
        ax.set_xlabel(var_name)  
        ax.xaxis.label.set_fontstyle('italic') # Define a label do eixo x como itálico
        ax.xaxis.label.set_size(10)
        ax.tick_params(axis='x', labelsize= 9, labelrotation= 0)

        if len(nominal_categorical_vars_df[var_name].unique()) > 12:
            ax.tick_params(axis='x', labelsize= 9, labelrotation= 50)
        
        # Exibe o gráfico
        plt.show()

    return None


def plot_ordinal_variables_distributions(ordinal_categorical_vars_df, dict_ordinal_vars):
    """
    Plota a distribuição de variáveis categóricas ordinais em um DataFrame utilizando gráficos de contagem.

    Parâmetros:
    -----------
    ordinal_categorical_vars_df : pd.DataFrame
        Um DataFrame do Pandas contendo variáveis categóricas ordinais.

    dict_ordinal_vars : dict
        Um dicionário onde as chaves são os nomes das variáveis ordinais e os valores são listas que definem a ordem das categorias.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    # Lista de variáveis ordinais a partir das chaves do dicionário
    list_ordinal_vars = list(dict_ordinal_vars.keys())
    
    for var_name in list_ordinal_vars:
        # Cria figura com uma área de plotagem
        fig, ax = plt.subplots(1, 1, figsize=(16, 4.5))
        
        # Obtendo a ordem das categorias associada à variável no dicionário
        order = dict_ordinal_vars.get(var_name)

        # Cria countplot com as categorias na ordem correta
        sns.countplot(data= ordinal_categorical_vars_df, x= var_name, order= order, edgecolor='black', hue= var_name, palette= 'cubehelix', legend= False, ax=ax)

        # Adiciona customizações ao gráfico
        fig.suptitle(f"Frequencies ({var_name})", fontsize= 14, fontweight= 'bold')
        ## Adiciona e customiza grades da horizontal
        ax.grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
        ax.set_axisbelow(True) # A grade fica atrás das barras
        ## Customiza labels e intervalos nos eixos verticiais dos gráficos
        ax.set_ylabel('Absolute Frequencies (count)')
        ax.yaxis.label.set_fontstyle('italic') # Fonte itálico para label
        ax.yaxis.label.set_size(10)  # Ajusta o tamanho da label
        ax.tick_params(axis= 'y', labelsize= 9, labelrotation= 0)
        ## Customiza labels e intervalos nos eixos horizontais dos gráficos
        ax.set_xlabel(var_name)  
        ax.xaxis.label.set_fontstyle('italic') # Define a label do eixo x como itálico
        ax.xaxis.label.set_size(10)
        ax.tick_params(axis='x', labelsize= 9, labelrotation= 0)
            
        # Exibe o gráfico    
        plt.show()
        
    return None
 
    
def plot_bivariate_analysis_quantitative_variables(numeric_independent_vars_df, numeric_target_var_df, list_discrete_var_names):
    """
    Plota a análise bivariada entre variáveis quantitativas independentes e uma variável-alvo quantitativa.

    Parâmetros:
    -----------
    numeric_independent_vars_dataframe : pd.DataFrame
        DataFrame contendo as variáveis quantitativas independentes.

    numeric_target_var_dataframe : pd.DataFrame
        DataFrame contendo a variável-alvo quantitativa.

    list_discrete_var_names : list
        Lista de nomes de variáveis discretas dentro das variáveis quantitativas independentes.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    # Variáveis externas
    numeric_target_var_name = numeric_target_var_df.columns[0] # Nome da variável alvo
    dict_target_metrics = _calculate_continuous_variable_metrics(numeric_target_var_df[numeric_target_var_name]) # Dicionário de métricas da variável alvo
    value_target_upper_fence = dict_target_metrics.get('upper_fence') # Valor da cerca superior da variável alvo
    
    list_numeric_independent_vars = list(numeric_independent_vars_df.columns) # Lista de variáveis independentes
    
    for numeric_independent_var_name in list_numeric_independent_vars:
        # Variáveis internas
        numeric_independent_var = numeric_independent_vars_df[numeric_independent_var_name]
        n_unique_values_var = numeric_independent_var.nunique()
        dict_independent_metrics = _calculate_continuous_variable_metrics(numeric_independent_var) # Dicionário de métricas da variável independente
        value_independent_upper_fence = dict_independent_metrics.get('upper_fence') # Valor da cerca superior da variável independente
        
        # Calcula a correlação de Spearman entre variável independedente e variável alvo
        dataframe_for_correlation = pd.merge(numeric_independent_var, numeric_target_var_df, right_index=True, left_index=True)
        spearman_correlation = dataframe_for_correlation.corr(method='spearman')
        spearman_correlation.drop(index=numeric_independent_var_name, columns=numeric_target_var_name, inplace=True)
        
        # Cria figura com duas área de plotagem
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

        # Define a paleta de cores e a embaralha
        palette = sns.color_palette('cubehelix', n_unique_values_var)
        random.shuffle(palette)

        # Cria primeiro gráfico
        if numeric_independent_var_name in list_discrete_var_names and n_unique_values_var <= 12:
            sns.boxplot(x= numeric_independent_var, y= numeric_target_var_df[numeric_target_var_name],
                        orient= 'v', palette= palette, ax= axes[0])
        else:
            sns.scatterplot(data= dataframe_for_correlation, x= numeric_independent_var_name, 
                            y= numeric_target_var_name, color= "black", alpha= 0.5, s= 50, ax= axes[0])
            if numeric_independent_var_name not in list_discrete_var_names:
                axes[0].axvline(x= value_independent_upper_fence, color= '#008080', linestyle= 'dashed', linewidth= 1, 
                            label= f"Upper Fence ({numeric_independent_var_name}): {float(value_independent_upper_fence):.1f}")
        
        # Cria segundo gráfico
        icefire_r = cm.get_cmap('icefire_r', 256)
        new_icefire_r = mcolors.LinearSegmentedColormap.from_list('icefire_r_10_90', icefire_r(np.linspace(0.10, 0.90, 256)))

        sns.heatmap(spearman_correlation, annot=True, cmap= new_icefire_r, linewidths=1, linecolor='black', vmin=-1.01, vmax=1.01, ax= axes[1])

        # Adiciona customizações aos subplots
        fig.suptitle(f"Correlation Plot ({numeric_independent_var_name} x {numeric_target_var_name})", fontsize= 14, fontweight= 'bold')
        ## Adciona linha e legenda para cerca superior da variável alvo no primeiro subplot
        axes[0].axhline(y=value_target_upper_fence, color='#483D8B', linestyle='dashed', linewidth=1, 
                    label=f"Upper Fence ({numeric_target_var_name}): {float(value_target_upper_fence):.1f}")
        axes[0].legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1)
        ## Adiciona nome do teste de correlação no segundo subplot
        axes[1].text(0.5, 0.95, 'Spearman Correlation', transform=axes[1].transAxes, fontsize=12, verticalalignment='top', 
                 horizontalalignment= 'center', bbox=dict(facecolor='white', alpha=0.5))
        ## Adiciona e customiza grades no primeiro subplot
        axes[0].grid(color="gray", linestyle="dotted", linewidth=0.5)
        axes[0].set_axisbelow(True)
        ## Customiza labels e intervalos nos eixos verticiais dos gráficos
        axes[0].set_ylabel(numeric_target_var_name)
        axes[0].yaxis.label.set_fontstyle('italic') # Fonte itálico para label
        axes[0].yaxis.label.set_size(10)  # Ajusta o tamanho da label
        axes[0].tick_params(axis= 'y', labelsize= 9, labelrotation= 0)
        axes[1].yaxis.set_ticks([])
        axes[1].set_ylabel(numeric_target_var_name)
        axes[1].yaxis.label.set_fontstyle('italic') # Fonte itálico para label
        axes[1].yaxis.label.set_size(10)  # Ajusta o tamanho da label
        ## Customiza labels e intervalos nos eixos horizontais dos gráficos
        axes[0].set_xlabel(numeric_independent_var_name)
        axes[0].xaxis.label.set_fontstyle('italic') # Define a label do eixo x como itálico
        axes[0].xaxis.label.set_size(10)
        axes[0].tick_params(axis='x', labelsize= 9, labelrotation= 0)
        axes[1].xaxis.set_ticks([])
        axes[1].set_xlabel(numeric_independent_var_name)
        axes[1].xaxis.label.set_fontstyle('italic') # Define a label do eixo x como itálico
        axes[1].xaxis.label.set_size(10)

        # Exibe os gráficos
        plt.show()
    
    return None
              

def plot_bivariate_analysis_continuous_target_and_qualitative_independent_vars(continuous_target_var_df, categorical_vars_df, dict_ordinal_vars):
    """
    Plota a análise bivariada entre uma variável-alvo contínua e variáveis independentes qualitativas.

    Parâmetros:
    -----------
    continuous_target_var_df : pd.DataFrame
        DataFrame contendo a variável-alvo contínua.

    categorical_vars_df : pd.DataFrame
        DataFrame contendo as variáveis qualitativas independentes.

    dict_ordinal_vars : dict
        Um dicionário onde as chaves são os nomes das variáveis ordinais e os valores são listas que definem a ordem das categorias.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    # Nome da variável-alvo
    continuous_target_var_name = continuous_target_var_df.columns[0]
    continuous_target_var = continuous_target_var_df[continuous_target_var_name]
    
    for var_name in categorical_vars_df:
        variable = categorical_vars_df[var_name]
        n_unique_values_var = variable.nunique()

        #Realiza o teste de comparação adequado
        p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(variable, quant_var= continuous_target_var)
        
        # Cria igura com uma área de plotagem
        fig, ax = plt.subplots(1, 1, figsize=(16, 4.5))

        list_ordinal_vars = list(dict_ordinal_vars.keys())

        if var_name in  list_ordinal_vars:
            order = dict_ordinal_vars.get(var_name)

            # Cria boxplot
            sns.boxplot(x= variable, y= continuous_target_var, order= order, palette= 'cubehelix', ax= ax)
        else:
            palette = sns.color_palette('cubehelix', n_unique_values_var)
            random.shuffle(palette)
            
            # Cria boxplot
            sns.boxplot(x= variable, y= continuous_target_var, palette= palette, ax= ax)
        
        # Adiciona customizações ao gráfico
        ## Adiciona título a figura
        fig.suptitle(f"Distributions ({var_name} x {continuous_target_var_name})", fontsize= 14, fontweight= 'bold')
        ## Adiciona o valor p do teste de Kruskal-Wallis ao gráfico
        ax.text(0.99, 0.95, f"{test_name} (p-value): {p_value}", fontsize= 12, horizontalalignment='right',
                verticalalignment= 'top', bbox= dict(facecolor='white', alpha=0.5), transform= ax.transAxes)
        ## Adiciona e customiza grades ao gráfico
        ax.grid(color="gray", linestyle="dotted", linewidth=0.5, axis= 'y')
        ax.set_axisbelow(True)
        ## Customiza labels e intervalos nos eixos verticiais dos gráficos
        ax.set_ylabel(continuous_target_var_name)
        ax.yaxis.label.set_fontstyle('italic') # Fonte itálico para label
        ax.yaxis.label.set_size(10)  # Ajusta o tamanho da label
        ax.tick_params(axis= 'y', labelsize= 9, labelrotation= 0)
        ## Customiza labels e intervalos nos eixos horizontais dos gráficos
        ax.set_xlabel(var_name)
        ax.xaxis.label.set_fontstyle('italic') # Define a label do eixo x como itálico
        ax.xaxis.label.set_size(10)
        if n_unique_values_var > 12:
            ax.tick_params(axis='x', labelsize= 9, labelrotation= 50)
        else:
            ax.tick_params(axis='x', labelsize= 9, labelrotation= 0)

        # Exibe o gráfico
        plt.show()
    
    return None


def plot_bivariate_analysis_qualitative_target_and_nominal_independent_vars(qualitative_target_var_df, nominal_vars_df):
    """
    Plota a análise bivariada entre uma variável-alvo qualitativa e variáveis independentes nominais.

    Parâmetros:
    -----------
    qualitative_target_var_df : pd.DataFrame
        DataFrame contendo a variável-alvo qualitativa.

    nominal_vars_df : pd.DataFrame
        DataFrame contendo as variáveis independentes nominais.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    cubehelix = cm.get_cmap('cubehelix', 256)
    new_cubehelix = mcolors.LinearSegmentedColormap.from_list('cubehelix_30_70', cubehelix(np.linspace(0.30, 0.70, 256)))

    # Nome da variável-alvo
    qualitative_target_var_name = qualitative_target_var_df.columns[0]
    qualitative_target_var = qualitative_target_var_df[qualitative_target_var_name]
    for var_name in nominal_vars_df:
        nominal_var = nominal_vars_df[var_name]
        
        # Criação da tabela de contingência
        contingency_table = pd.crosstab(nominal_var, qualitative_target_var)

        # Realiza o teste qui-quadrado
        _, p_value, _, _ = sts.chi2_contingency(contingency_table)
        
        # Cria figura com subplots
        fig, axes = plt.subplots(1, 2, figsize= (16, 4.5))

        # Gráfico de barras múltiplas agrupadas (frequência absoluta)
        contingency_table.plot(kind= 'bar', stacked= False, ax= axes[0], colormap= new_cubehelix, edgecolor='black')
       
        # Gráfico de barras empilhadas (frequência relativa percentual)
        contingency_table.div(contingency_table.sum(1), axis= 0).plot(kind= 'bar', stacked= True, ax= axes[1], colormap= new_cubehelix, edgecolor='black')
        
        # Adiciona e ajusta customizações
        fig.suptitle(f"Absolute and Relative Frequencies ({var_name} x {qualitative_target_var_name})", fontsize=14, fontweight='bold')
        ## Adiciona e customiza labels os eixos verticais dos gráficos
        axes[0].set_ylabel("Absolute Frequencies (count)")
        axes[1].set_ylabel("Relative Frequencies (%)")
        axes[0].yaxis.label.set_size(10)  # Ajusta o tamanho da label
        axes[1].yaxis.label.set_size(10)
        axes[0].yaxis.label.set_fontstyle('italic') # Fonte itálico para label
        axes[1].yaxis.label.set_fontstyle('italic')
        ## Adciona e customiza grades da horizontal
        axes[0].grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
        axes[1].grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
        axes[0].set_axisbelow(True) # A grade fica atrás das barras
        axes[1].set_axisbelow(True)
        ## Customiza labels e nome dos grupos nos eixos horizontais dos gráficos
        axes[0].tick_params(axis='x', labelsize= 9, labelrotation=0)  # Ajusta o tamanho da fonte e rotação dos nomes dos grupos
        axes[1].tick_params(axis='x', labelsize= 9, labelrotation=0)
        axes[0].xaxis.label.set_fontstyle('italic')  # Define a label do eixo x como itálico
        axes[1].xaxis.label.set_fontstyle('italic')
        ## Remove a legenda do primeiro gráfico e a move para fora do segundo gráfico
        axes[0].legend().set_visible(False)
        axes[1].legend(title='Churn', loc='upper right', bbox_to_anchor=(1.175, 1), fancybox=True, framealpha=1, shadow=True, borderpad=1)
        ## Adiciona valor p do teste qui-quadrado ao gráfico
        plt.figtext(0.5, 0.01, f"Chi-Square Test (p-value): {p_value}", ha= "center", fontsize= 10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Exibe o gráfico
        plt.show()
    
    return None


def plot_bivariate_analysis_qualitative_target_and_ordinal_independent_vars(qualitative_target_var_df, ordinal_vars_df, ordinal_vars_dict):
    """
    Plota a análise bivariada entre uma variável-alvo qualitativa e variáveis independentes ordinais.

    Parâmetros:
    -----------
    qualitative_target_var_df : pd.DataFrame
        DataFrame contendo a variável-alvo qualitativa.

    ordinal_vars_df : pd.DataFrame
        DataFrame contendo as variáveis independentes ordinais.

    ordinal_vars_dict : dict
        Dicionário com o nome das variáveis ordinais e suas categorias ordenadas em formato de lista.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    cubehelix = cm.get_cmap('cubehelix', 256)
    new_cubehelix = mcolors.LinearSegmentedColormap.from_list('cubehelix_30_70', cubehelix(np.linspace(0.30, 0.70, 256)))

    # Nome da variável-alvo
    qualitative_target_var_name = qualitative_target_var_df.columns[0]
    qualitative_target_var = qualitative_target_var_df[qualitative_target_var_name]

    for var_name in ordinal_vars_df:
        ordinal_var = ordinal_vars_df[var_name]

        # Cria a categoria ordenada da variável ordinal
        ordinal_var_test = pd.Categorical(ordinal_var, categories= ordinal_vars_dict[var_name], ordered= True).codes
        
        #Realiza o teste de comparação adequado
        p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(qualitative_target_var, numeric_ordinal_var= ordinal_var_test)

        # Criação da tabela de contingência
        contingency_table = pd.crosstab(ordinal_var, qualitative_target_var)
        
        # Cria figura com subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
        
        # Gráfico de barras múltiplas agrupadas (frequência absoluta)
        contingency_table.plot(kind= 'bar', stacked= False, ax=axes[0], colormap= new_cubehelix, edgecolor='black')
        
        # Gráfico de barras empilhadas (frequência relativa percentual)
        contingency_table.div(contingency_table.sum(1), axis= 0).plot(kind= 'bar', stacked= True, ax= axes[1], colormap= new_cubehelix, edgecolor='black')
        
        # Adciona e ajusta customizações nos gráficos
        fig.suptitle(f"Absolute and Relative Frequencies ({var_name} x {qualitative_target_var_name})", fontsize=14, fontweight='bold')
        plt.figtext(0.5, 0.01, f"{test_name} (p_value): {p_value}", ha= "center", fontsize= 10, bbox=dict(facecolor='white', alpha=0.5))
        ## Customiza labels e nome dos grupos nos eixos horizontais dos gráficos
        axes[0].tick_params(axis='x', labelsize= 9, labelrotation=0)  # Ajusta o tamanho da fonte e rotação dos nomes dos grupos
        axes[1].tick_params(axis='x', labelsize= 9, labelrotation=0)
        axes[0].xaxis.label.set_fontstyle('italic')  # Define a label do eixo x como itálico
        axes[1].xaxis.label.set_fontstyle('italic')
        axes[0].set_ylabel("Frequência Absoluta") # Define o nome da label do eixo y
        axes[1].set_ylabel("Frequência Relativa (%)")
        axes[0].yaxis.label.set_size(10)  # Ajusta o tamanho da label
        axes[1].yaxis.label.set_size(10)
        axes[0].yaxis.label.set_fontstyle('italic') # Fonte itálico para as labels
        axes[1].yaxis.label.set_fontstyle('italic')
        ## Adiciona e customiza grades da horizontal
        axes[0].grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
        axes[1].grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
        axes[0].set_axisbelow(True) # A grade fica atrás das barras
        axes[1].set_axisbelow(True)
        ## Remove a legenda do primeiro gráfico e a move para fora do segundo gráfico
        axes[0].legend().set_visible(False)
        axes[1].legend(title='Churn', loc='upper right', bbox_to_anchor=(1.175, 1), fancybox=True, framealpha=1, shadow=True, borderpad=1)
        
        # Exibe o gráfico
        plt.show()
    
    return None


def plot_bivariate_analysis_qualitative_target_and_discrete_independent_vars(qualitative_target_var_df, discrete_vars_df):
    """
    Plota a análise bivariada entre uma variável-alvo qualitativa e variáveis independentes discretas.

    Parâmetros:
    -----------
    qualitative_target_var_df : pd.DataFrame
        DataFrame contendo a variável-alvo qualitativa.

    discrete_vars_df : pd.DataFrame
        DataFrame contendo as variáveis independentes discretas.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    cubehelix = cm.get_cmap('cubehelix', 256)
    new_cubehelix = mcolors.LinearSegmentedColormap.from_list('cubehelix_30_70', cubehelix(np.linspace(0.30, 0.70, 256)))

    # Nome da variável-alvo
    qualitative_target_var_name = qualitative_target_var_df.columns[0]
    qualitative_target_var = qualitative_target_var_df[qualitative_target_var_name]
    for var_name in discrete_vars_df:
        discrete_var = discrete_vars_df[var_name]

        #Realiza o teste de comparação adequado
        p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(qualitative_target_var, quant_var= discrete_var)
        
        if discrete_var.nunique() > 12:
            # Criação dos boxplots
            fig, ax = plt.subplots(figsize= (16, 4.5))
            sns.boxplot(x= qualitative_target_var, y= discrete_var, hue= discrete_var, palette= new_cubehelix, ax= ax)
            
            # Adicionando customizações ao gráfico
            fig.suptitle(f"Distributions ({var_name} x {qualitative_target_var_name})", fontsize=14, fontweight='bold')
            plt.text(0.5, 0.95, f"{test_name} (p-value): {p_value}", fontsize= 10, horizontalalignment='center',
                     verticalalignment= 'center', bbox= dict(facecolor='white', alpha=0.5), transform= ax.transAxes)
            ## Customiza labels
            ax.set_xlabel(qualitative_target_var_name)
            ax.set_ylabel(var_name)
            ax.yaxis.label.set_size(10) # Ajusta o tamanho da label
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_fontstyle('italic') # Fonte itálico para as labels
            ax.xaxis.label.set_fontstyle('italic')
            ## Adiciona e customiza grades da horizontal
            ax.grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
            ax.set_axisbelow(True) # A grade fica atrás das barras
        
        else:
            # Criação da tabela de contingência
            contingency_table = pd.crosstab(discrete_var, qualitative_target_var)
            
            # Cria figura com subplots
            fig, axes = plt.subplots(1, 2, figsize= (16, 4.5))
            
            # Gráfico de barras múltiplas agrupadas (frequência absoluta)
            contingency_table.plot(kind= 'bar', stacked= False, ax= axes[0], colormap= new_cubehelix, edgecolor='black')
            
            # Gráfico de barras empilhadas (frequência relativa percentual)
            contingency_table.div(contingency_table.sum(1), axis= 0).plot(kind= 'bar', stacked= True, ax= axes[1], colormap= new_cubehelix, edgecolor='black')
            
            # Adiciona e customiza os gráficos
            fig.suptitle(f"Absolute and Relative Frequencies ({var_name} x {qualitative_target_var_name})", fontsize=14, fontweight='bold')
            plt.figtext(0.5, 0.01, f"{test_name} (p-value): {p_value}", ha= "center", fontsize= 10, bbox=dict(facecolor= 'white', alpha= 0.5))
            ## Customiza labels e nome dos grupos nos eixos horizontais dos gráficos
            axes[0].tick_params(axis='x', labelsize= 9, labelrotation=0)  # Ajusta o tamanho da fonte e rotação dos nomes dos grupos
            axes[1].tick_params(axis='x', labelsize= 9, labelrotation=0)
            axes[0].xaxis.label.set_fontstyle('italic')  # Define a label do eixo x como itálico
            axes[1].xaxis.label.set_fontstyle('italic')
            axes[0].set_ylabel("Frequência Absoluta") # Define o nome da label do eixo y
            axes[1].set_ylabel("Frequência Relativa (%)")
            axes[0].yaxis.label.set_size(10)  # Ajusta o tamanho da label
            axes[1].yaxis.label.set_size(10)
            axes[0].yaxis.label.set_fontstyle('italic') # Fonte itálico para as labels
            axes[1].yaxis.label.set_fontstyle('italic')
            ## Adiciona e customiza grades da horizontal
            axes[0].grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
            axes[1].grid(color= "gray", linestyle= "dotted", linewidth= 0.5, axis= 'y')
            axes[0].set_axisbelow(True) # A grade fica atrás das barras
            axes[1].set_axisbelow(True)
            ## Remove a legenda do primeiro gráfico e a move para fora do segundo gráfico
            axes[0].legend().set_visible(False)
            axes[1].legend(title='Churn', loc='upper right', bbox_to_anchor=(1.175, 1), fancybox=True, framealpha=1, shadow=True, borderpad=1)

        # Exibe o gráfico
        plt.show()

    return None


def plot_bivariate_analysis_qualitative_target_and_continuous_independent_vars(qualitative_target_var_df, continuous_vars_df):
    """
    Plota a análise bivariada entre uma variável-alvo qualitativa e variáveis independentes contínuas.

    Parâmetros:
    -----------
    qualitative_target_var_df : pd.DataFrame
        DataFrame contendo a variável-alvo qualitativa.

    continuous_vars_df : pd.DataFrame
        DataFrame contendo as variáveis independentes contínuas.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    # Nome da variável-alvo
    qualitative_target_var_name = qualitative_target_var_df.columns[0]
    qualitative_target_var = qualitative_target_var_df[qualitative_target_var_name]
    
    for var_name in continuous_vars_df:
        continuous_var = continuous_vars_df[var_name]
        
        #Realiza o teste de comparação adequado
        p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(qualitative_target_var, quant_var= continuous_var)
        
        # Criação de figura e 1 subplot
        fig, ax = plt.subplots(figsize= (16, 4.5))

        # Cria boxplot
        flierprops = dict(marker= 'o', markerfacecolor='none', markersize= 6)

        sns.boxplot(x= continuous_var, y= qualitative_target_var, hue= qualitative_target_var, palette= 'cubehelix', flierprops= flierprops, ax= ax)

        fig.suptitle(f"Distributions ({var_name} x {qualitative_target_var_name})", fontsize=14, fontweight='bold')
        plt.text(0.99, 0.95, f"{test_name} (p-value): {p_value}", fontsize= 10, horizontalalignment='right',
                    verticalalignment= 'top', bbox= dict(facecolor='white', alpha=0.5), transform= ax.transAxes)
        ## Customiza labels
        ax.set_xlabel(var_name)
        ax.set_ylabel(qualitative_target_var_name)
        ax.yaxis.label.set_size(10) # Ajusta o tamanho da label
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_fontstyle('italic') # Fonte itálico para as labels
        ax.xaxis.label.set_fontstyle('italic')
        ## Adiciona e customiza grades da horizontal
        ax.grid(color= "gray", linestyle= "dotted", linewidth= 0.5)
        ax.set_axisbelow(True) # A grade fica atrás das barras
        
        # Exibe o gráfico
        plt.show()
    
    return None


def plot_multivariate_heatmap_qualitative_vars(qualitative_vars_df, ordinal_vars_dict= dict()):
    """
    Gera um heatmap multivariado que visualiza os p-valores das comparações entre variáveis qualitativas e ordinais, 
    utilizando testes estatísticos apropriados para cada combinação de variáveis.

    Dependendo da natureza das variáveis (qualitativas nominais ou ordinais), a função aplica o teste estatístico 
    adequado e preenche uma matriz de p-valores, que é então exibida como um heatmap.

    Parâmetros:
    -----------
    qualitative_vars_df : pandas.DataFrame
        DataFrame contendo variáveis qualitativas (nominais ou ordinais) para análise. Cada coluna deve representar uma 
        variável qualitativa e cada linha uma observação.

    ordinal_vars_dict : dict, opcional
        Dicionário contendo as variáveis ordinais e suas respectivas ordens. As chaves do dicionário são os nomes das 
        variáveis ordinais, e os valores são listas que especificam a ordem das categorias da variável ordinal.

    Retorno:
    --------
    None
        A função não retorna nenhum valor, mas exibe um gráfico de heatmap com os p-valores das comparações entre as 
        variáveis qualitativas.

    Processo:
    ---------
    1. Para cada par de variáveis qualitativas, a função verifica se elas são variáveis ordinais ou nominais.
    2. Se ambas as variáveis forem ordinais, ou se uma delas for ordinal e a outra nominal, a função utiliza a função 
       `_comparison_test_for_ordinal_or_quantitative_vars` para realizar o teste adequado (como Mann-Whitney ou Kruskal-Wallis).
    3. Se ambas as variáveis forem nominais, é realizada uma análise de contingência com o teste qui-quadrado.
    4. Todos os p-valores resultantes dos testes são armazenados em uma matriz de p-valores.
    5. Um heatmap é gerado a partir dessa matriz de p-valores, exibindo visualmente a significância estatística das comparações.

    Notas:
    ------
    - O heatmap utiliza uma paleta de cores personalizada ("icefire") para representar os p-valores. Valores menores 
      (mais significativos) são destacados em cores diferentes.
    - As comparações entre variáveis ordinais e nominais são tratadas com testes não-paramétricos adequados.
    - As variáveis nominais são comparadas utilizando o teste qui-quadrado de independência.
    - A função presume que as variáveis ordinais são fornecidas com uma ordem categórica definida no dicionário `ordinal_vars_dict`.

    Exemplos de uso:
    ----------------
    >>> qualitative_vars_df = pd.DataFrame({
    >>>     'var1': ['A', 'B', 'A', 'C'],
    >>>     'var2': ['X', 'Y', 'X', 'Y'],
    >>>     'var3': ['Low', 'Medium', 'High', 'Low']
    >>> })
    >>> ordinal_vars_dict = {'var3': ['Low', 'Medium', 'High']}
    >>> plot_multivariate_heatmap_qualitative_vars(qualitative_vars_df, ordinal_vars_dict)
    """
    qualitative_vars_list = qualitative_vars_df.columns
    ordinals_vars_list = list(ordinal_vars_dict.keys())

    # Inicializa o DataFrame para armazenar os p-valores
    pvalues_matrix_df = pd.DataFrame(index=qualitative_vars_list, columns=qualitative_vars_list)

    for qual_var_name1 in qualitative_vars_list:
        for qual_var_name2 in qualitative_vars_list:
            if (qual_var_name1 in ordinals_vars_list) & (qual_var_name2 in ordinals_vars_list):
                ordinal_var1 = qualitative_vars_df[qual_var_name1]
                ordinal_var2 = qualitative_vars_df[qual_var_name2]

                # Cria a categoria ordenada da variável ordinal
                ordinal_var2 = pd.Categorical(ordinal_var2, categories= ordinal_vars_dict[qual_var_name2], ordered= True).codes
                
                #Realiza o teste de comparação adequado
                p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(ordinal_var1, numeric_ordinal_var= ordinal_var2)
            elif (qual_var_name1 in ordinals_vars_list) & (qual_var_name2 not in ordinals_vars_list):
                ordinal_var = qualitative_vars_df[qual_var_name1]
                nominal_var = qualitative_vars_df[qual_var_name2]

                # Cria a categoria ordenada da variável ordinal
                ordinal_var = pd.Categorical(ordinal_var, categories= ordinal_vars_dict[qual_var_name1], ordered= True).codes
                
                #Realiza o teste de comparação adequado
                p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(nominal_var, numeric_ordinal_var= ordinal_var)
            elif (qual_var_name1 not in ordinals_vars_list) & (qual_var_name2 in ordinals_vars_list):
                ordinal_var = qualitative_vars_df[qual_var_name2]
                nominal_var = qualitative_vars_df[qual_var_name1]

                # Cria a categoria ordenada da variável ordinal
                ordinal_var = pd.Categorical(ordinal_var, categories= ordinal_vars_dict[qual_var_name2], ordered= True).codes
                
                #Realiza o teste de comparação adequado
                p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(nominal_var, numeric_ordinal_var= ordinal_var)
            else:
                # Criação da tabela de contingência
                contingency_table = pd.crosstab(qualitative_vars_df[qual_var_name1], qualitative_vars_df[qual_var_name2])

                # Realiza o teste qui-quadrado
                _, p_value, _, _ = sts.chi2_contingency(contingency_table)

            # Armazena o p-valor na matriz
            pvalues_matrix_df.loc[qual_var_name1, qual_var_name2] = p_value

    # Converte todos os valores para float
    pvalues_matrix_df = pvalues_matrix_df.astype(float)

    # Cria figura com uma área de plotagem
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    icefire = cm.get_cmap('icefire', 256)
    new_icefire = mcolors.LinearSegmentedColormap.from_list('icefire_50_80', icefire(np.linspace(0.5, 0.8, 256)))

    # Cria heatmap da matrix
    sns.heatmap(pvalues_matrix_df, annot=True, fmt=".2f", cmap= new_icefire, annot_kws={"size": 6}, ax= ax);

    # Exibe o gráfico
    plt.show()

    return None


def plot_multivariate_heatmap_quantitative_vars(quantitative_vars_df):
    """
    Calcula o score de correlação entre variáveis quantitativas e plota um heatmap das variáveis associadas.

    Parâmetros:
    -----------
    quantitative_vars_df : pd.DataFrame
        DataFrame contendo as variáveis quantitativas.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe os gráficos gerados.
    """
    correlations = quantitative_vars_df.corr(method= 'spearman')

    plt.figure(figsize= (20, 10))

    icefire_r = cm.get_cmap('icefire_r', 256)
    new_icefire_r = mcolors.LinearSegmentedColormap.from_list('icefire_r_10_90', icefire_r(np.linspace(0.10, 0.90, 256)))
    
    sns.heatmap(correlations, cmap= new_icefire_r, vmin=-1.01, vmax=1.01, annot=True, fmt=".2f", annot_kws={"size": 8});
    
    plt.show()

    return None


def plot_multivariate_heatmap_quantitative_qualitative_vars(quantitative_vars_df, qualitative_vars_df):
    """
    Gera um heatmap multivariado visualizando os p-valores das comparações entre variáveis quantitativas e qualitativas, 
    utilizando testes estatísticos adequados com base nos dados.

    A função itera sobre todas as combinações de variáveis quantitativas e qualitativas, realizando testes apropriados 
    para avaliar a relação entre essas variáveis. Os p-valores resultantes são representados em um heatmap.

    Parâmetros:
    -----------
    quantitative_vars_df : pandas.DataFrame
        DataFrame contendo as variáveis quantitativas contínuas. Cada coluna representa uma variável quantitativa e 
        cada linha representa uma observação.

    qualitative_vars_df : pandas.DataFrame
        DataFrame contendo as variáveis qualitativas (categóricas). Cada coluna representa uma variável qualitativa e 
        cada linha representa uma observação.

    Retorno:
    --------
    None
        A função não retorna nenhum valor, mas exibe um gráfico de heatmap com os p-valores das comparações entre as 
        variáveis quantitativas e qualitativas.

    Processo:
    ---------
    1. A função percorre cada combinação de variável quantitativa e qualitativa.
    2. Para cada combinação, utiliza a função `_comparison_test_for_ordinal_or_quantitative_vars` para realizar o teste 
       estatístico adequado, como T-Test, ANOVA, Mann-Whitney U, ou Kruskal-Wallis, dependendo da natureza dos dados.
    3. Os p-valores são armazenados em uma matriz com variáveis quantitativas nas linhas e variáveis qualitativas nas colunas.
    4. Um heatmap é gerado a partir dessa matriz de p-valores, fornecendo uma visualização gráfica dos resultados dos testes.

    Notas:
    ------
    - A função assume que as variáveis quantitativas são contínuas e que as variáveis qualitativas são categóricas.
    - Os p-valores indicam a significância estatística das relações entre as variáveis, com valores menores indicando 
      maior significância.
    - O heatmap usa uma paleta de cores personalizada ("icefire") para representar os p-valores.

    Exemplos de uso:
    ----------------
    >>> quantitative_vars_df = pd.DataFrame({
    >>>     'var1': [10, 20, 30, 40],
    >>>     'var2': [5, 15, 25, 35]
    >>> })
    >>> qualitative_vars_df = pd.DataFrame({
    >>>     'cat1': ['A', 'B', 'A', 'B'],
    >>>     'cat2': ['X', 'X', 'Y', 'Y']
    >>> })
    >>> plot_multivariate_heatmap_quantitative_qualitative_vars(quantitative_vars_df, qualitative_vars_df)
    """
    quantitative_vars_list = quantitative_vars_df.columns
    qualitative_vars_list = qualitative_vars_df.columns

    # Inicializa o DataFrame para armazenar os p-valores
    pvalues_matrix_df = pd.DataFrame(index=quantitative_vars_list, columns=qualitative_vars_list)

    # Itera sobre cada combinação de variáveis quantitativas e qualitativas
    for quant_var_name in quantitative_vars_list:
        quant_var = quantitative_vars_df[quant_var_name]

        for qual_var_name in qualitative_vars_list:
            qual_var = qualitative_vars_df[qual_var_name]

            p_value, test_name = _comparison_test_for_ordinal_or_quantitative_vars(qual_var, quant_var= quant_var)

            # Armazena o p-valor na matriz
            pvalues_matrix_df.loc[quant_var_name, qual_var_name] = p_value

    # Converte todos os valores para float
    pvalues_matrix_df = pvalues_matrix_df.astype(float)

    # Cria figura com uma área de plotagem
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    icefire = cm.get_cmap('icefire', 256)
    new_icefire = mcolors.LinearSegmentedColormap.from_list('icefire_50_80', icefire(np.linspace(0.5, 0.8, 256)))

    # Cria heatmap da matrix
    sns.heatmap(pvalues_matrix_df, annot=True, fmt=".2f", cmap= new_icefire, annot_kws={"size": 6}, ax= ax);

    # Exibe o gráfico
    plt.show()

    return None