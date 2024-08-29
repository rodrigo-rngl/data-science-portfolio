import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.stats as sts


def print_missing_values_in_dataframe(dataframe):
    """
    Exibe o número e a porcentagem de valores faltantes em cada variável de um DataFrame.

    Parâmetros:
    -----------
    dataframe : pd.DataFrame
        O DataFrame do Pandas que será analisado para valores faltantes.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe as informações sobre valores faltantes diretamente na saída.
    """
    # Cria uma máscara booleana para valores ausentes
    na_mask = dataframe.isna()
    
    # Calcula o número e a porcentagem de valores faltantes para cada variável
    na_counts = na_mask.sum()
    na_percentages = (na_counts / len(dataframe)) * 100

    # Filtra variáveis que possuem valores faltantes
    variables_with_na_values = na_counts[na_counts > 0].index.tolist()

    if len(variables_with_na_values) == 0:
         print("O dataframe não possui valores faltantes")
    else:
        for variable in variables_with_na_values:
            amount_na_values_in_variable = na_counts[variable]
            percentage_na_values = na_percentages[variable]
            print(f"'{variable}' possui {amount_na_values_in_variable} registros faltantes ({percentage_na_values:.2f}%)")
            
    return None


def print_zero_values_in_dataframe(numeric_df):
    """
    Exibe o número e a porcentagem de valores zerados em cada variável numérica de um DataFrame.

    Parâmetros:
    -----------
    numeric_df : pd.DataFrame
        O DataFrame do Pandas que será analisado para valores zerados.

    Retorno:
    --------
    None
        A função não retorna nenhum valor. Ela exibe as informações sobre valores zerados diretamente na saída.
    """
    # Seleciona apenas variáveis numéricas
    numerical_variables = numeric_df.select_dtypes(include=['int64', 'float64'])

    # Cria uma máscara booleana para valores zerados
    zero_mask = numerical_variables == 0
    
    # Calcula o número e a porcentagem de valores zerados para cada variável
    zero_counts = zero_mask.sum()
    zero_percentages = (zero_counts / len(numerical_variables)) * 100

    # Filtra variáveis que possuem valores zerados
    variables_with_zeros = zero_counts[zero_counts > 0].index.tolist()
    
    if len(variables_with_zeros) == 0:
        print("O dataframe não possui valores faltantes")
    else:
        for variable in variables_with_zeros:
            amount_zero_values_in_variable = zero_counts[variable]
            percentage_zero_values = zero_percentages[variable]
            print(f"'{variable}' possui {amount_zero_values_in_variable} registros zerados ({percentage_zero_values:.2f}%)")
            
    return None


def descriptive_statistics_continuous_variables(continuous_vars_df):
    """
    Calcula estatísticas descritivas para variáveis numéricas, incluindo teste de normalidade.

    Parâmetros:
    -----------
    continuous_vars_df : pd.DataFrame
        DataFrame contendo as variáveis numéricas para as quais as estatísticas descritivas serão calculadas.

    Retorno:
    --------
    pd.DataFrame
        DataFrame contendo as estatísticas descritivas para cada variável numérica, incluindo valores únicos,
        desvio padrão, variância, assimetria, curtose e p-valor do teste de normalidade de Kolmogorov-Smirnov.
    """
    # Padroniza os dados para o teste de Kolmogorov-Smirnov
    df_standardized = pd.DataFrame(StandardScaler().fit_transform(continuous_vars_df), columns=continuous_vars_df.columns)
    
    # Calcula estatísticas descritivas
    unique = continuous_vars_df.apply(lambda x: len(x.unique()))
    standard_deviation = continuous_vars_df.std()
    variance = continuous_vars_df.var()
    skewness = continuous_vars_df.skew()
    kurtosis = continuous_vars_df.kurtosis()
    
    # Teste de Kolmogorov-Smirnov para normalidade
    kolmogorov = df_standardized.apply(lambda x: sts.kstest(x, 'norm').pvalue)
    
    # Cria o DataFrame com as estatísticas
    stats_df = pd.DataFrame({
        'Valores Únicos': unique,
        'Desv. Padrão': standard_deviation,
        'Variância': variance,
        'Assimetria': skewness,
        'Curtose': kurtosis,
        'Normalidade (p-value)': kolmogorov
    })
    
    # Arredonda os valores para melhor apresentação
    cols_to_round = ['Valores Únicos', 'Desv. Padrão', 'Variância', 'Assimetria', 'Curtose']
    stats_df[cols_to_round] = stats_df[cols_to_round].round(3)
    
    return stats_df


def treat_outliers_by_percentile_capping(continuous_vars_df, target_vars, percentile_inf_value= 0, percentile_sup_value= 99.5):
    """
    Trata outliers em variáveis contínuas por meio de capping percentual.

    Parâmetros:
    -----------
    continuous_vars_df : pd.DataFrame
        DataFrame contendo as variáveis contínuas a serem tratadas.
    
    target_vars : list
        Lista de nomes das variáveis contínuas que serão tratadas.

    percentile_inf_value : float, opcional
        Percentil inferior para o capping (padrão é 0.5).

    percentile_sup_value : float, opcional
        Percentil superior para o capping (padrão é 99.5).

    Retorno:
    --------
    None
        O DataFrame de entrada é modificado in-place.
    """
    for var in target_vars:
        # Calcula os percentis inferior e superior
        percentile_inf = np.percentile(continuous_vars_df[var], percentile_inf_value)
        percentile_sup = np.percentile(continuous_vars_df[var], percentile_sup_value)
        median = continuous_vars_df[var].median()

        # Substitui valores fora dos percentis por mediana
        continuous_vars_df[var] = continuous_vars_df[var].apply(lambda x: median if (x < percentile_inf) or (x > percentile_sup) else x)
    
    return None