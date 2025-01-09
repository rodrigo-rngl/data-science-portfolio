import requests
import pandas as pd
import json

# URL da API
url = "http://127.0.0.1:5000/predict"  # Atualize o URL se necessário

# Dados de entrada para teste (substitua com os dados apropriados)
test_data = [
    {"MSSubClass":20,"MSZoning":"RL","LotFrontage":81.0,"LotArea":14267,"Alley": "None",
     "LotShape":"IR1","LandContour":"Lvl","LotConfig":"Corner","LandSlope":"Gtl","Neighborhood":"NAmes",
     "Condition1":"Norm","BldgType":"1Fam","HouseStyle":"1Story","OverallQual":6,
     "OverallCond":6,"YearBuilt":1958,"YearRemodAdd":1958,"RoofStyle":"Hip",
     "Exterior1st":"Wd Sdng","Exterior2nd":"Wd Sdng","MasVnrType":"BrkFace",
     "MasVnrArea":108.0,"ExterQual":"TA","ExterCond":"TA","Foundation":"CBlock","BsmtQual":"TA"
     ,"BsmtCond":"TA","BsmtExposure":"No","BsmtFinType1":"ALQ","BsmtFinSF1":923.0,"BsmtFinType2":"Unf"
     ,"BsmtUnfSF":406.0,"TotalBsmtSF":1329.0,"HeatingQC":"TA","CentralAir":1,"Electrical":"SBrkr"
     ,"1stFlrSF":1329,"2ndFlrSF":0,"GrLivArea":1329,"BsmtFullBath":0.0,"FullBath":1,"HalfBath":1,
     "BedroomAbvGr":3,"KitchenAbvGr":1,"KitchenQual":"Gd","TotRmsAbvGrd":6,"Functional":"Typ",
     "Fireplaces":0,"FireplaceQu":"None","GarageType":"Attchd","GarageYrBlt":1958,"GarageFinish":"Unf",
     "GarageCars":1.0,"GarageArea":312.0,"GarageQual":"TA","GarageCond":"TA","PavedDrive":"Y",
     "WoodDeckSF":393,"OpenPorchSF":36,"EnclosedPorch":0,"ScreenPorch":0,"Fence":"None","SaleType":"WD",
     "SaleCondition":"Normal","HasPorch":1,"CountPorch":1}
]

# Enviando a requisição POST
response = requests.post(url, json=test_data)

# Verificando a resposta
if response.status_code == 200:
    # Converte a resposta JSON em dicionário
    response_data = response.json()
    
    # Imprime as predições
    print(response)
else:
    # Caso a API retorne erro
    print(f"Erro: Código de status {response.status_code}")
    print("Resposta:", response.text)