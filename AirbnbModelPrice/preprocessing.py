import numpy as np
import pandas as pd
from sklearn import preprocessing

# Importamos nuestro modeulo para graficar
from graphic_methods import Graphics

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class Preprocessing(Graphics):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, show_info: bool = True):
        # 1-- Instanciamos graphics para acceder a los metodos de graficacion
        super().__init__()

        self.df = train
        self.test_df = test
        self.show_info = show_info

    def run(self, test_solution=False) -> dict:
        self.introPrint("Preprocessing 1: Recogemos los datos de Analysis para empezar la criba")

        # -- 2: Eliminamos los campos no relevantes (usamos inplace para modificar el df en lugar de generar otro)
        self.introPrint("Preprocessing 2: Eliminamos campos no relevantes (Descriptivos y URLs) en Train y Test")

        drop_cols = ['ID', 'Listing Url', 'Scrape ID', 'Last Scraped', 'Name', 'Summary',
                     'Space', 'Description', 'Experiences Offered', 'Neighborhood Overview',
                     'Notes', 'Transit', 'Access', 'Interaction', 'House Rules',
                     'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url',
                     'Host ID', 'Host URL', 'Host Name', 'Host Location',
                     'Host About', 'Host Response Time',
                     'Host Acceptance Rate', 'Host Thumbnail Url', 'Host Picture Url',
                     'Host Neighbourhood', 'Host Verifications', 'Street',
                     'State', 'Market', 'Smart Location', 'Country Code', 'Country',
                     'Cleaning Fee', 'Calendar Updated',
                     'Has Availability', 'Calendar last Scraped',
                     'First Review', 'Last Review', 'License', 'Jurisdiction Names',
                     'Geolocation', 'Features']

        self.df.drop(drop_cols, axis=1, inplace=True)
        self.test_df.drop(drop_cols, axis=1, inplace=True)

        # -- 3: Eliminamos columnas con demasiados nulos    ------------------------------------------------------------
        self.introPrint("Preprocessing 3: Eliminamos columnas con demasiados nulos: NOTA: Calculamos con train "
                        "que columnas tienen muchos nulos y tambien borramos esas en test")

        null_drop_col_list = self.checkAndDropNullCols(self.df)

        self.df.drop(null_drop_col_list, axis=1, inplace=True)
        train = self.df

        self.test_df.drop(null_drop_col_list, axis=1, inplace=True)
        test = self.test_df

        # -- 4: Como solo usamos Madrid, intentamos reformatear los CP que podamos y eliminamos las filas restantes-----
        self.introPrint("Preprocessing 4: Reformateamos y limpiamos zipcodes en train y test. Eliminamos col City")
        train = self.reformatAndDropZipcodes(train)
        test = self.reformatAndDropZipcodes(test)

        train.drop("City", axis=1, inplace=True)
        test.drop("City", axis=1, inplace=True)

        # -- 5: Volvemos a analizar los campos que tenemos ahora con un scatterMatrix  ---------------------------------
        self.introPrint("Preprocessing 5: Visualizamos los datos de train con un scatterMatrix: 03-ScatterMatrix_1.png")
        self.createAndSaveScatterMatrix(train, "03-ScatterMatrix_1.png")

        # -- 6: Buscamos outliers   ------------------------------------------------------------------------------------
        self.introPrint("Preprocessing 6: Buscamos outliers en los campos numericos de train")
        object_type = train["Neighbourhood"].dtype
        numeric_cols = [z for z in train.columns if train[z].dtype != object_type]
        numeric_cols.remove("Latitude")
        numeric_cols.remove("Longitude")

        self.findOutliers(train, numeric_cols)

        print("\nA pesar de que existen valores dudosos, no está claro que sean outliers, pueden ser casas rurales"
              " etc. Por tanto, no voy a filtrar los datos mas extremos (de momento)")

        # -- 7: Vamos a ver que variables categoricas tenemos:   -------------------------------------------------------
        self.introPrint("Preprocessing 7: Exploramos las variables categoricas en train para ver como abordarlas")
        categoric_vars = [z for z in train.columns if train[z].dtype == object_type]

        print(f'\nNuestras variables categoricas son: {categoric_vars}')

        # -- 8: Codificacion de variables categoricas   ----------------------------------------------------------------
        self.introPrint("Preprocessing 8: Hacemos Mean Encodding con nuestras variables categoricas en train y test")
        train = self.meanEncoder(train, categoric_vars)
        test = self.meanEncoder(test, categoric_vars)

        print(train.head())

        # -- 9: Graficamos otro scatter y otro mapa de calor para ver si eliminamos mas campos   -----------------------
        self.introPrint("Preprocessing 9: Graficamos otro scatter y otro mapa de calor para ver correlaciones en train:"
                        "04-ScatterMatrix_Prep09.png, 05-ProcessingHeatmap_Prep09.png")
        self.createAndSaveScatterMatrix(train, "04-ScatterMatrix_Prep09.png")
        self.createAndSaveHeatmap(train, "Price", "05-ProcessingHeatmap_Prep09.png")

        # -- 10: Listamos los campos correlacionados
        self.introPrint("Preprocessing 10: Observamos correlaciones relevantes entre campos")
        print("Hay >=0.8 correlacion entre todos los campos 'Listings Count'")
        print("Hay >=0.8 correlacion entre todos los campos 'Availability' salvo 'Availability 365'")
        print("Hay >=0.8 correlacion entre todos los campos 'Review Scores'")
        print("Hay >=0.8 correlacion entre 'Beds' y 'Accommodates'")
        print("Hay >=0.8 correlacion entre 'Reviews per month' y 'Number of Reviews'")
        print("Hay >=0.8 correlacion 'NeighBourhood', 'Neighbourhood Cleansed','Neighbourhood Group Cleansed', 'Zipcode'")

        # -- 11: Comparamos los campos correlacionados para escoger y eliminamos los descartados   ---------------------
        self.introPrint("Preprocessing 11: Comparamos los campos que tienen mucha correlacion entre si")
        self.compareCorrelatedFields(train, ["Calculated host listings count", "Host Listings Count"])
        self.compareCorrelatedFields(train, ["Availability 30", "Availability 60", "Availability 90"])
        self.compareCorrelatedFields(train, ["Reviews per Month", "Number of Reviews"])
        self.compareCorrelatedFields(train, ["Beds", "Accommodates"])
        self.compareCorrelatedFields(train, ["Neighbourhood", "Neighbourhood Cleansed","Neighbourhood Group Cleansed", "Zipcode"])

        print("\nAhora podriamos escoger el campo mas representativo de cada subconjunto altamente correlacionado, "
              "pero en lugar de hacerlo a mano vamos a dejar que el modelo vaya eligiendo los campos por nosotros.\n"
              "NOTA IMPORTANTE: A mano, Voy a escoger las siguientes:\n"
              "---   'Calculated host listings count'\n"
              "---   'Availability 30'\n"
              "---   'Availability 30'\n"
              "---   'Number of Reviews'\n"
              "---   'Accommodates'\n"
              "---   'Zipcode'\n"
              "Y el resultado del modelo es muy similar. Si se comenta el codigo del paso 11 preprocessing.py donde"
              " se hacen los drops (lineas 135-140 mas o menos) el modelo final usa 9 campos incluyendo beds "
              "y accomodates. \nDe lo contrario, usa 8 campos, los mismos sin Beds")

        correlated_cols_to_drop = ['Host Listings Count', 'Availability 60', 'Availability 90', 'Reviews per Month',
                                   'Beds', 'Neighbourhood', 'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed']

        train.drop(correlated_cols_to_drop, axis=1, inplace=True)
        test.drop(correlated_cols_to_drop, axis=1, inplace=True)

        # -- 12: Volvemos a buscar outliers en el dataset resultante, para evaluar los resultados de MeanEncoding ------
        self.introPrint("Preprocessing 12: Buscamos outliers en el nuevo dataset (Con Mean Encoding realizado)")
        col_list = list(train.columns)
        col_list.remove("Latitude")
        col_list.remove("Longitude")
        col_list.remove("Minimum Nights")
        col_list.remove("Maximum Nights")

        self.findOutliers(train, col_list)

        # -- 13: Sustituimos los valores nulos de las columnas Bathrooms, Bedrooms y Review Scores Rating con la media
        self.introPrint("Preprocessing 13: Buscamos las columnas que tienen nulos y los sustituimos con la media")

        train = self.meanFillNulls(train, list(train.columns))
        test = self.meanFillNulls(test, list(train.columns))

        print("Usamos el metodo meanFillNulls de preprocessing.py en train y test -> df sin nulos")

        # --14: Sacamos la variable objetivo, creamos el scaler y devolvemos y, X , scaler_model en un diccionario------
        self.introPrint("Preprocessing 14: Ccreamos el scaler y devolvemos x,y en train y test ademas del scaler")
        data_dict = {"y_train": train["Price"], "y_test": test["Price"]}

        train.drop("Price", axis=1, inplace=True)
        test.drop("Price", axis=1, inplace=True)

        if test_solution:
            solution_cols = ['Latitude', 'Longitude', 'Accommodates', 'Bathrooms', 'Bedrooms', 'Beds', 'Amenities', 'Review Scores Communication', 'Review Scores Value']
            data_dict["X_train"] = train[solution_cols]
            data_dict["X_test"] = test[solution_cols]
            data_dict["scaler"] = preprocessing.StandardScaler().fit(train[solution_cols])

        else:
            data_dict["X_train"] = train
            data_dict["X_test"] = test
            data_dict["scaler"] = preprocessing.StandardScaler().fit(train)

        print("Separamos los datasets de train y test de su columna objetivo. \n"
              "Retornamos data_dict{'X_train': train, 'X_test': test, 'y_train': train['Price'], "
              "y_test': test['Price'], 'scaler: preprocessing.StandardScaler().fit(train)}")

        return data_dict

    @staticmethod
    def checkAndDropNullCols(df: pd.DataFrame, drop_limit: float = 0.4) -> list:
        """
        Cuenta los valores nulos de cada columna y en caso de que superen el drop_limit/1 elimina la columna
        :param df: Dataframe que recibe (train)
        :param drop_limit: Limite establecido en tanto por uno, por defecto 0.4
        :return: Un nuevo dataframe
        """
        # -- Descomponemos en nombres de columna y numero de nulls de cada columna
        total_rows = int(df.shape[0])

        null_series = df.isnull().sum()
        null_df = pd.DataFrame(null_series).reset_index()
        col_names = tuple(null_df["index"])
        col_null_count = tuple(null_df[0])

        # -- Buscamos una lista de nombres de columna que no pasen la criba
        cols_to_drop = [col_names[z] for z in range(len(col_names)) if
                        (int(col_null_count[z]) / total_rows) >= drop_limit]

        print(f'Las columnas con un {drop_limit * 100}% de nulos o mas son: {cols_to_drop}\n')

        # -- Hacemos el drop y devolvemos el df
        return cols_to_drop

    def compareCorrelatedFields(self, df: pd.DataFrame, cols: list) -> None:
        """
        Vamos a comprobar que campo quedarnos entre una lista de campos
        :param df: Dataframe
        :param cols: Lista de variables
        :return: None
        """
        if self.show_info:
            print(f"\n------------  compareCorrelatedFields: {cols}   --------------")
            for z in cols:
                # --: Sacamos informacion agrupada por elemento
                obj = z
                prom_df = pd.DataFrame()
                prom_df["Mean"] = df[["Price", obj]].groupby(obj).mean()
                prom_df["Max"] = df[["Price", obj]].groupby(obj).max()
                prom_df["Min"] = df[["Price", obj]].groupby(obj).min()
                prom_df["Count"] = df[["Price", obj]].groupby(obj).count()

                print(f'Hay {prom_df.shape[0]} {obj} diferentes y {df[obj].isnull().sum()} valores nulos')

    def findOutliers(self, df: pd.DataFrame, col_list: list):
        if self.show_info:
            for z in col_list:
                print(f'\nValores Maximos de {z}: {df[z].nlargest(5).values}')
                print(f'Valores Minimos de {z}: {df[z].nsmallest(5).values}')

    @staticmethod
    def meanEncoder(df: pd.DataFrame, categoric_cols: list) -> pd. DataFrame:
        """
        Asignamos valores numericos a las variables categoricas
        :param df: Dataframe
        :param categoric_cols: Columnas con tipo object (categoricas)
        :return: Dataframe
        """
        # Sustituimos categorias por floats en base a la media de la columna objetivo: Price
        mean_map = {}
        for c in categoric_cols:
            mean = df.groupby(c)['Price'].mean()
            df[c] = df[c].map(mean)
            mean_map[c] = mean

        return df

    @staticmethod
    def meanFillNulls(df: pd.DataFrame, col_list: list) -> pd.DataFrame:
        for z in col_list:
            df[z].fillna(df[z].mean(), inplace=True)
        return df

    @staticmethod
    def reformatAndDropZipcodes(df: pd.DataFrame):
        """
        Evalua los datos de codigo postal para que tengan en formato deseado, si no es capaz, elimina las filas
        :param df: Dataframe que recibe
        :return: Dataframe con la columna Zipcode homogeneizada y sin valores np.nan
        """
        df = df[df["Zipcode"].notna()]
        df = df[df["Zipcode"].str.contains('28')]

        def getNumber(x):
            start = x
            end = ""
            numbers = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

            for i in range(len(start)):
                if start[i] in numbers:
                    end += start[i]
            if len(end) == 5:
                return end
            if len(end) == 4:
                return f'{end[0:3]}0{end[3:]}'
            else:
                return np.nan

        df["Zipcode"] = df["Zipcode"].apply(lambda x: getNumber(x))
        df = df[df["Zipcode"].notna()]
        return df


