import pandas as pd

from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler

def stratify(df: pd.DataFrame, frac: float, random_state: int) -> pd.DataFrame:
    return df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=random_state))


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    unwanted_columns = [
        'MAWILAB_taxonomy', 'MAWILAB_label', 
        'MAWILAB_nbDetectors', 'MAWILAB_distance',
        'VIEGAS_numberOfDifferentServices_A', 'VIEGAS_numberOfDifferentDestinations_A',
        'ORUNADA_numberOfDifferentDestinations', 'ORUNADA_numberOfDifferentServices'
    ]
    return df.drop(unwanted_columns, axis=1, errors='ignore')

def normalize_dataset(df: pd.DataFrame, view='NIGEL') -> pd.DataFrame:
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = RobustScaler()
    features = df.drop(["NIGEL_protocol", 'class'], axis=1, errors='ignore').columns.values
    df[features] = scaler.fit_transform(df[features])

    if view == 'NIGEL':
        encoder = OneHotEncoder(categories='auto', sparse_output=False)

        # Encoding into n features 
        temp = encoder.fit_transform(df["NIGEL_protocol"].to_numpy().reshape(-1, 1))

        # converting to dataframe to concatenate to original df
        d = dict()
        for i in range(temp.shape[1]):
            d[f"NIGEL_protocol_{i}"] = temp[:, i]

        return pd.concat([df, pd.DataFrame(d)], axis=1)
    
    return df
