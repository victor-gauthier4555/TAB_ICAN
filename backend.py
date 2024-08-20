import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind, chi2_contingency



def preprocess_dataframe(df):
    def replace_nulls_with_mean(df):
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
        return df

    def encode_non_numeric_columns(df):
        label_encoders = {}
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
        return df, label_encoders

    def remove_outliers(df):
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    df = replace_nulls_with_mean(df)
    df, _ = encode_non_numeric_columns(df)
    df = remove_outliers(df)
    return df





def median_for_condition(df, column1, column2, element):

    # Filtrer le DataFrame pour ne conserver que les lignes où column2 == element
    filtered_df = df[df[column2] == element]

    if filtered_df.empty:
        return "N/A"

    # Calculer la médiane, le premier quartile et le troisième quartile
    median_value = round(filtered_df[column1].median(), 3)
    first_quartile = round(filtered_df[column1].quantile(0.25), 3)
    third_quartile = round(filtered_df[column1].quantile(0.75), 3)

    # Retourner le texte avec la médiane et les quartiles
    return f"{median_value} ({first_quartile}-{third_quartile})"


def is_continuous(df, threshold=10):
    """
    Détermine si les colonnes d'un dataframe sont des variables continues.

    :param df: pandas DataFrame
    :param threshold: Seuil de cardinalité pour déterminer une variable continue (par défaut 20)
    :return: Dictionnaire avec les colonnes comme clés et un booléen indiquant si elles sont continues
    """
    continuous_columns = {}

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].nunique()
            if unique_values > threshold:
                continuous_columns[column] = True
            else:
                continuous_columns[column] = False
        else:
            continuous_columns[column] = False

    return continuous_columns


def calculate_p_values(df, result_df, texte):
    p_values = []

    unique_values = df[texte].unique()
    print(unique_values)

    # Vérifier si la variable texte a exactement deux catégories pour effectuer un t-test
    if len(unique_values) != 2:
        raise ValueError("Le t-test nécessite exactement deux catégories dans la variable 'texte'.")

    group1 = df[df[texte] == unique_values[0]]
    group2 = df[df[texte] == unique_values[1]]

    for index, row in result_df.iterrows():
        variable = row['Variable']

        if "=" in variable:
            # Cas des variables non continues scindées
            variable_name, value = variable.split("=")
            variable_name = variable_name.strip('"')
            value = value.strip('"')

            # Recréer une colonne binaire temporaire pour effectuer le test du chi-carré
            temp_column = (df[variable_name] == value).astype(int)
            contingency_table = pd.crosstab(temp_column, df[texte])
            _, p_value, _, _ = chi2_contingency(contingency_table)
        else:
            # Cas des variables continues
            stat, p_value = ttest_ind(group1[variable], group2[variable], nan_policy='omit')

        p_values.append(p_value)

    result_df['p-value'] = p_values
    return result_df


def decompose_discontinuous_variables(df, continuous_columns, texte):
    decomposed_rows = []

    # Supprimer la première colonne du DataFrame
    df.drop(df.columns[0], axis=1, inplace=True)

    # Supprimer la colonne spécifiée par le texte
    df.drop(columns=texte, inplace=True)

    # Ajouter les variables discontinues décomposées en lignes
    for column in df.columns:
        if not continuous_columns.get(column, True):  # Si la colonne est discontinue
            unique_values = df[column].unique()
            for value in unique_values:
                row = {
                    'Variable': f'{column}="{value}"',
                    'Is Continuous': False
                }
                decomposed_rows.append(row)

    # Ajouter les colonnes continues en lignes sans les changer
    for column in df.columns:
        if continuous_columns.get(column, False):  # Si la colonne est continue
            row = {
                'Variable': column,
                'Is Continuous': True
            }
            decomposed_rows.append(row)

    # Créer le DataFrame final à partir de la liste des lignes décomposées
    decomposed_df = pd.DataFrame(decomposed_rows)
    return decomposed_df





def calculate_frequencies(df, result_df, texte):
    """
    Calcule les fréquences et les pourcentages pour les lignes discontinues de result_df en fonction des colonnes du DataFrame original.
    """
    for index, row in result_df.iterrows():
        if not row['Is Continuous']:
            variable, value = row['Variable'].split('=')
            variable = variable.strip('"')
            value = value.strip('"')

            # Convertir value au bon type en fonction du type de la colonne dans le DataFrame original
            if pd.api.types.is_numeric_dtype(df[variable]):
                value = pd.to_numeric(value)

            # Filtrer le DataFrame df pour ne conserver que les lignes où variable == value
            filtered_df = df[df[variable] == value]

            # Calculer le total pour la condition de la ligne actuelle
            total_count = filtered_df.shape[0]

            # Calculer la fréquence pour chaque colonne de result_df
            for col in result_df.columns[2:]:  # Ignorer les colonnes 'Variable' et 'Is Continuous'
                column_condition, column_value = col.split('=')
                column_condition = column_condition.strip('"')
                column_value = column_value.strip('"')

                if column_condition in df.columns:
                    # Convertir column_value au bon type en fonction du type de la colonne dans le DataFrame original
                    if pd.api.types.is_numeric_dtype(df[column_condition]):
                        column_value = pd.to_numeric(column_value)

                    count = filtered_df[filtered_df[column_condition] == column_value].shape[0]

                    # Calculer le pourcentage par rapport au total
                    if total_count > 0:
                        percentage = (count / total_count) * 100
                    else:
                        percentage = 0

                    # Mettre à jour la cellule avec le format "fréquence (pourcentage%)"
                    result_df.at[index, col] = f"{count} ({percentage:.2f}%)"

    return result_df


def final_df(df, df2, texte):
    # Décomposition des variables discontinues et ajout des colonnes continues


    result_df = decompose_discontinuous_variables(df2, is_continuous(df), texte)

    # Ajout des colonnes pour les valeurs uniques de la variable texte
    unique_values = df[texte].unique()
    for value in unique_values:
        new_col_name = f'{texte}="{value}"'
        result_df[new_col_name] = ""  # Ajout de colonnes vides pour chaque catégorie

        for i in range(len(result_df[new_col_name])):
            if result_df['Is Continuous'][i]:
                result_df.at[i, new_col_name] = median_for_condition(df, result_df['Variable'][i], texte, value)

    # Calcul des fréquences pour les variables discontinues
    result_df = calculate_frequencies(df, result_df, texte)

    # Calcul des p-values et ajout de la colonne p-value
    result_df = calculate_p_values(df, result_df, texte)

    return result_df


# Exemple d'utilisation avec un fichier CSV
df = pd.read_csv('uploads/exemple_multivariate.csv', delimiter=";")
df2 = pd.read_csv('uploads/exemple_multivariate.csv', delimiter=";")
result_df = final_df(df, df2, 'Status')

# Affichage du DataFrame final
print(result_df)


