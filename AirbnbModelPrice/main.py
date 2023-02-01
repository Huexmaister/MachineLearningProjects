import pandas as pd

from analisis import Analysis
from preprocessing import Preprocessing
from modeling import Modeling


if __name__ == '__main__':
    """
    *** -- VARIABLES A RELLENAR:
    - test_solution: bool -> Si es true, se ejecutara una prueba para validar que los resultados del modelo son iguales
                            pasando el scaler y usando np.ndarrays que no pasandolo y usando pd.DataFrames.
                            
    - show_info: bool -> Si es true, muestra en consola en mas detalle las operaciones que se van realizando durante la 
                        ejecucion.
                            
    - METHOD: str -> Elige el tipo de modelo que deseas evaluar: Todos me dan r2~0.9 en train y r2~0.93 en test
    
    
    *** -- FUNCIONAMIENTO DEL PROGRAMA:
    - Hay un modulo para cada parte del proceso y un modulo graphic_methods para graficar con matplotlib y seaborn
    
    - El desarrollo de la ejecucion se ira mostrando en consola, asi como las decisiones que se van tomando en cada
    paso. 
    
    - Activa o desactiva show_info para ver con mas o menos detalle los resultados de cada metodo
    
    """
    def run():
        # -- 0: Variables de control
        show_info = False
        test_solution = False

        choose_method = ["RIDGE", "LASSO", "RF"]
        METHOD = choose_method[0]

        if METHOD not in choose_method:
            print("La constante METHOD debe contener un metodo valido")
            return

        # -- 1: Cargamos Datos
        airbnb = pd.read_csv("airbnb-listings.csv", sep=";")

        # -- 2: Instanciamos Analysis  y recibimos train y test
        analisis_result_dict = Analysis(airbnb, show_info=show_info).run()

        train_df = analisis_result_dict["train"]
        test_df = analisis_result_dict["test"]

        # -- 3: Instanciamos Preprocessing pasandole los datos de train (para probar sin bucle test_solution=True)

        preprocessing_result_dict = Preprocessing(train_df, test_df, show_info=show_info).run(test_solution=test_solution)

        # -- 4: Instanciamos Modeling pasandole los datos anteriores y el df_test

        if test_solution:
            Modeling(preprocessing_result_dict).noScalerTest()
        else:
            Modeling(preprocessing_result_dict).run(chosen_method=METHOD)
    run()
