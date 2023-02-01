from sklearn.model_selection import train_test_split
import pandas as pd

# Importamos nuestro modeulo para graficar
from graphic_methods import Graphics

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class Analysis(Graphics):
    def __init__(self, df: pd.DataFrame, show_info: bool = True):
        # -- Instanciamos graphics para acceder a los metodos de graficacion
        super().__init__()

        self.df = df
        self.show_info = show_info

    def run(self) -> dict:
        # -- 1: Dividimos en train y test    ---------------------------------------------------------------------------
        self.introPrint("Analysis 1: Separamos en train y test")

        train, test = self.splitTrainTest()
        print(f'Dimensiones train: {train.shape}\n')
        print(f'Dimensiones test: {test.shape}\n')

        # -- 2: Echamos un vistazo a los datos    ----------------------------------------------------------------------
        self.introPrint("Analysis 2: Primer vistazo a los datos: dtypes, head, describe")

        if self.show_info:
            print("\n-----------------------   Dtypes   --------------------------------\n")
            print(train.dtypes)

            print("\n-----------------------   Head     --------------------------------\n")
            print(train.head())

            print("\n-----------------------   Describe     ----------------------------\n")
            print(train.describe().T)

        # -- 3: Creamos mapa de calor y lo guardamos en imgAnalisis    -------------------------------------------------
        self.introPrint("Analysis 3: Creamos un mapa de calor respecto a Price y lo almaceanmos como 01-AnalisisHeatmap")
        self.createAndSaveHeatmap(train, "Price", "01-AnalisisHeatmap.png")

        # -- 4: Crear histogramas de todas las variables    ------------------------------------------------------------
        self.introPrint("Analysis 4: Creamos histograma con todas las variables numericas: 02-AnalisisHistograma")
        self.createAndSaveHistplot(train, "02-AnalisisHistograma.png", type(train["Listing Url"].dtype))

        # --5: Ya tenemos informacion para eliminar varios campos, fundamentalmente los descriptivos -> a preprocesar
        self.introPrint("Analysis 5: Ya sabemos que campos vamos a desechar, return: -> dict[train, test]")
        return {"train": train, "test": test}

    def splitTrainTest(self, test_size: float = 0.2, shuffle: bool = True, random_state: int = 0):
        return train_test_split(self.df, test_size=test_size, shuffle=shuffle, random_state=random_state)



