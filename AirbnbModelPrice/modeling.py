import math

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

from graphic_methods import Graphics


class Modeling(Graphics):
    def __init__(self, data_dict: dict, show_info: bool = True):
        # -- : Instanciamos graphics para acceder a los metodos de graficacion
        super().__init__()
        self.show_info = show_info

        # -- : Almacenamos en objetos de clase los datos de train y test

        self.X_train = data_dict["X_train"]
        self.y_train = data_dict["y_train"]

        self.X_test = data_dict["X_test"]
        self.y_test = data_dict["y_test"]

        self.scaler = data_dict["scaler"]

    def noScalerTest(self):
        # Ojo cuidao que hay que cambiar el scaler de preprocessing
        XtrainScaled = self.scaler.transform(self.X_train)
        XtestScaled = self.scaler.transform(self.X_test)
        # print(self.X_train.head(10))
        # print(self.y_train.head(10))

        ridge_best_alpha = self.ridgeGridSearchCV(XtrainScaled, self.y_train)

        print(f'----------   Tenemos {XtrainScaled.shape[1]} columnas en training   -----------------')
        print(f'----------   Tenemos {XtestScaled.shape[1]} columnas en testing   -----------------')
        ridge = Ridge(alpha=ridge_best_alpha)
        ridge.fit(XtrainScaled, self.y_train)

        y_pred_train = ridge.predict(XtrainScaled)
        mse_train = mean_squared_error(self.y_train, y_pred_train)

        y_pred_test = ridge.predict(XtestScaled)
        mse_test = mean_squared_error(self.y_test, y_pred_test)

        r2_train = ridge.score(XtrainScaled, self.y_train)
        r2_test = ridge.score(XtestScaled, self.y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

    def run(self, chosen_method: str):
        # -- 1: Aplicamos el scaler que tenemos de los datos de train a ambos conjuntos
        self.introPrint(f"Modeling 1: Aplicamos el scaler a train y llamamos al metodo {chosen_method}")
        print(
            "NOTA IMPORTANTE:  No aplico el scaler que tenemos de los datos de train a ambos conjuntos, porque me acepta el"
            " formato df.\nNo obstante, se puede probar con scaler aplicado llamando al metodo "
            "Modeling.pruebaAMano() en lugar de Modeling.run().\n"
            "La diferencia estriba en que:\n"
            "--  pruebaAMano existe porque los resultados del modelo me parecian demasiado buenos "
            "para ser ciertos.\nYA SABIENDO LAS COLUMNAS QUE RIDGE ESCOGE TRAS APLICAR run() "
            "Y QUE EJECUTE EL BUCLE. HAGO TRAMPA PARA PROBAR SI EFECTIVAMENTE EL MODELO ESTA BIEN.\n"
            " Entonces Entreno el modelo conociendo los campos que mejor predicen y  AQUI SI AMBOS CONJUNTOS"
            " ESTAN ESCALADOS (CON EL SCALER DE TRAIN)\n"
            "--  run usa el X_scaled para calcular el alpha_optimo pero trabaja directamente con el df "
            " no con el ndarray del scaler.\n"
            "Se puede probar para ver que los resultados son iguales trabajando con scaled ndarray que con pd.df")

        XtrainScaled = self.scaler.transform(self.X_train)
        XtestScaled = self.scaler.transform(self.X_test)

        if chosen_method == "RIDGE":
            self.ridgeApp(XtrainScaled)

        elif chosen_method == "LASSO":
            self.lassoApp(XtrainScaled)

        else:
            self.randomForestApp(XtrainScaled)

    # --- RIDGE
    def ridgeApp(self, XtrainScaled):
        self.introPrint("Modeling 2: Aplicamos RIDGE e iteramos hasta encontrar el mejor numero de predictores")
        ridge_best_alpha = self.ridgeGridSearchCV(XtrainScaled, self.y_train)

        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.ridgeInfo(self.X_train[curr_cols], self.y_train, self.X_test[curr_cols],
                                                  self.y_test, ridge_best_alpha)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

    def ridgeGridSearchCV(self, X_train: np.array, y_train: np.array, logspace=np.logspace(-5, 1.8, 25),
                          cv_number: int = 5):
        alpha_vector = logspace
        param_grid = {'alpha': alpha_vector}
        grid = GridSearchCV(Ridge(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=cv_number)
        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = -1 * np.array(grid.cv_results_['mean_test_score'])
        self.plotAlphaValues(alpha_vector, scores, f"AlphaRidge{X_train.shape[1]}.png")

        # Devolvemos el mejor aplpha
        return grid.best_params_['alpha']

    @staticmethod
    def ridgeInfo(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, alpha: float):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)

        y_pred_train = ridge.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)

        y_pred_test = ridge.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = ridge.score(X_train, y_train)
        r2_test = ridge.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        sfm = SelectFromModel(ridge, threshold=0.25)
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features

    # -- LASSO
    def lassoApp(self, XtrainScaled):
        self.introPrint("Modeling 2: Aplicamos LASSO e iteramos hasta encontrar el mejor numero de predictores")
        lasso_best_alpha = self.lassoGridSearchCV(XtrainScaled, self.y_train)

        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.lassoInfo(self.X_train[curr_cols], self.y_train, self.X_test[curr_cols],
                                                  self.y_test, lasso_best_alpha)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

    def lassoGridSearchCV(self, X_train: np.array, y_train: np.array, logspace=np.logspace(-5, 1.8, 25),
                          cv_number: int = 5):
        alpha_vector = logspace
        param_grid = {'alpha': alpha_vector}
        grid = GridSearchCV(Lasso(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=cv_number)
        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = -1 * np.array(grid.cv_results_['mean_test_score'])
        self.plotAlphaValues(alpha_vector, scores, f"AlphaLasso{X_train.shape[1]}.png")

        # Devolvemos el mejor aplpha
        return grid.best_params_['alpha']

    @staticmethod
    def lassoInfo(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, alpha: float):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)

        y_pred_train = lasso.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)

        y_pred_test = lasso.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = lasso.score(X_train, y_train)
        r2_test = lasso.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        sfm = SelectFromModel(lasso, threshold=0.25)
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features

    # -- RANDOM FOREST
    def randomForestApp(self, XtrainScaled):
        self.introPrint("Modeling 2: Aplicamos RandomForest e iteramos hasta encontrar el mejor numero de predictores")
        rf_best_alpha = self.randomForestGridSearchCV(XtrainScaled, self.y_train)

        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.randomForestInfo(self.X_train[curr_cols], self.y_train, self.X_test[curr_cols],
                                                         self.y_test, rf_best_alpha)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

    def randomForestGridSearchCV(self, X_train: np.array, y_train: np.array, maxDepth=range(1, 18), cv_number: int = 5):

        param_grid = {'max_depth': maxDepth}
        grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200, max_features='sqrt'),
                            param_grid=param_grid, cv=cv_number, verbose=2)
        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = np.array(grid.cv_results_['mean_test_score'])
        self.plotAlphaValues(maxDepth, scores, f"AlphaRf{X_train.shape[1]}.png")

        # Devolvemos el mejor depth
        return grid.best_params_['max_depth']

    @staticmethod
    def randomForestInfo(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array,
                         maxDepth: float):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')
        rf = RandomForestRegressor(max_depth=maxDepth, n_estimators=200, max_features='sqrt')
        rf.fit(X_train, y_train)

        y_pred_train = rf.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)

        y_pred_test = rf.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = rf.score(X_train, y_train)
        r2_test = rf.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        sfm = SelectFromModel(rf, threshold=0.25)
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features
