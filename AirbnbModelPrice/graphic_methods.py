import math

import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


class Graphics:
    def __init__(self, master_path: str = "./imgAnalisis/"):
        self.master_path = master_path

    def createAndSaveHeatmap(self, df: pd.DataFrame, target_col_name: str, file_name: str) -> None:
        """
        Devuelve una imagen con el correlation-heatmap del df sin la columna objetivo
        :param df: Dataframe
        :param target_col_name: Nombre de la columna objetivo
        :param file_name: Nombre y extension del archivo
        :return: None
        """
        if not os.path.exists(f"{self.master_path}{file_name}"):
            corr = np.abs(df.drop([target_col_name], axis=1).corr())

            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True

            plt.subplots(figsize=(12, 10))

            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(corr, mask=mask, vmin=0.0, vmax=1.0, center=0.5,
                        linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

            plt.savefig(f"{self.master_path}{file_name}", bbox_inches="tight")
            print(f"{self.master_path}{file_name} creado correctamente")
        else:
            print(f"{self.master_path}{file_name} ya existe")

    def createAndSaveHistplot(self, df: pd.DataFrame, file_name: str, object_type) -> None:
        """
        Crea y guarda un archivo png de los histogramas de los campos con valor numerico (si no existe ya)
        Tambien aporta informacion sobre el numero de valores respecto a nulos.
        :param object_type: Tipo objeto de los tipos de datos de un df
        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :return: None
        """

        if not os.path.exists(f"{self.master_path}{file_name}"):
            col_list = [z for z in df.columns if type(df[z].dtype) != object_type]
            counter_primary = int(len(col_list) / 4) + 1
            counter_second = 4
            counter_third = 0

            for z in col_list:
                counter_third += 1

                value_number = f'{df[z].count()}/{df.shape[0]}'
                plt.subplot(counter_primary, counter_second, counter_third)
                df[z].plot.hist(alpha=0.5, bins=25, grid=True)
                plt.xlabel(f'{z} {value_number}')
                plt.tight_layout()

            plt.savefig(f"{self.master_path}{file_name}")
            print(f"{self.master_path}{file_name} creado correctamente")
        else:
            print(f"{self.master_path}{file_name} ya existe")

    def createAndSaveScatterMatrix(self, df: pd.DataFrame, file_name: str) -> None:
        """
        Crea y almacena en png(si no existe ya) un scatter_matrix de los campos actuales

        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :return: None
        """
        if not os.path.exists(f"{self.master_path}{file_name}"):
            pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(30, 30), diagonal='hist')
            plt.savefig(f"{self.master_path}{file_name}")
            print(f"{self.master_path}{file_name} creado correctamente")
        else:
            print(f"{self.master_path}{file_name} ya existe")

    def plotAlphaValues(self, alpha_vector, scores, file_name):
        if not os.path.exists(f"{self.master_path}{file_name}"):
            fig, ax = plt.subplots()
            r2 = [math.sqrt(abs(z)) for z in scores]
            ax.semilogx(alpha_vector, scores, '-o')
            plt.xlabel('alpha', fontsize=16)
            plt.ylabel('5-Fold RMSE')
            #plt.ylim((0, 1))
            plt.savefig(f"{self.master_path}{file_name}")
            print(f"{self.master_path}{file_name} creado correctamente")
        else:
            print(f"{self.master_path}{file_name} ya existe")

    @staticmethod
    def plotFeaturesRank(X, f_test, feature_names, mi):
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(X.shape[1]), f_test, align="center")
        plt.xticks(range(X.shape[1]), feature_names, rotation=90)
        plt.xlabel('features')
        plt.ylabel('Ranking')
        plt.title('$F-info$ score')

        plt.subplot(1, 2, 2)
        plt.bar(range(X.shape[1]), mi, align="center")
        plt.xticks(range(X.shape[1]), feature_names, rotation=90)
        plt.xlabel('features')
        plt.ylabel('Ranking')
        plt.title('Mutual information score')

        plt.show()

    @staticmethod
    def introPrint(text_to_print: str):
        print(f'\n#######################################################################################\n'
              f'----    {text_to_print}\n'
              f'#######################################################################################\n')