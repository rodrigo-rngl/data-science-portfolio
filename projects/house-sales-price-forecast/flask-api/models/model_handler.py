import pandas as pd
import joblib

class ModelHandler:
    def __init__(self, model_path: str):
        """
        Inicializa o manipulador do modelo.

        Args:
            model_path (str): Caminho para o arquivo do modelo salvo (.pkl).
        """
        self.model = joblib.load(model_path)

    def predict(self, df: pd.DataFrame) -> list:
        """
        Faz predições usando o modelo.

        Args:
            df (pandas.DataFrame): Dataframe dos dados.

        Returns:
            list: Lista de predições.
        """

        predictions = self.model.predict(df)
        return predictions.tolist()