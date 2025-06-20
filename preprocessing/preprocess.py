import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_duplicates(df):
    dup_count = df.duplicated().sum()
    print(f"Jumlah data duplikat yang ditemukan: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
        print("Data duplikat telah dihapus.")
    return df

def iqr_capping(df, columns, factor=2.0):
    print("Memulai penanganan outlier dengan IQR Capping...")
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers_before = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        print(f"  - Kolom '{col}': {outliers_before} outlier ditangani.")
    return df

def feature_engineering(df):
    print("Membuat fitur baru (feature engineering)...")
    df['ph_original'] = df['ph']
    
    df['mineral_saturation'] = df['Hardness'] * (10**(-df['ph_original']))
    df['TDS_ratio'] = df['Solids'] / df['Conductivity']
    
    df['ph_category'] = pd.cut(
        df['ph_original'], 
        bins=[0, 6.5, 8.5, 14],
        labels=['Asam', 'Netral', 'Basa'],
        include_lowest=True
    )
    
    df['hardness_level'] = pd.cut(
        df['Hardness'],
        bins=[0, 60, 120, 180, np.inf],
        labels=['Sangat Lunak', 'Lunak', 'Keras', 'Sangat Keras'],
        include_lowest=True
    )
    
    df = df.drop(columns=['ph_original'])
    
    print("  - Fitur baru berhasil dibuat.")
    return df

def main(args):
    df = load_data(args.input_path)
    df = handle_duplicates(df)
    print("Memulai imputasi nilai yang hilang dengan KNNImputer...")
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print(f"Imputasi selesai. Jumlah nilai hilang setelahnya: {df_imputed.isna().sum().sum()}")
    outlier_cols = ['Solids', 'Trihalomethanes', 'Conductivity']
    df_capped = iqr_capping(df_imputed, outlier_cols, factor=2.0)

    df_featured = feature_engineering(df_capped)

    print("Memulai One-Hot Encoding untuk fitur kategorikal...")
    categorical_features = ['ph_category', 'hardness_level']
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df_featured[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_data, 
        columns=encoder.get_feature_names_out(categorical_features),
        index=df_featured.index
    )
    
    df_processed = pd.concat([df_featured, encoded_df], axis=1)
    df_processed = df_processed.drop(columns=categorical_features)
    print("Encoding selesai.")

    print("Memulai normalisasi dengan RobustScaler...")
    numerical_features = [
        'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes',
        'Turbidity', 'mineral_saturation', 'TDS_ratio'
    ]
    scaler = RobustScaler()
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    print("Normalisasi selesai.")

    preprocessing_artifacts = {
        'imputer': imputer,
        'encoder': encoder,
        'scaler': scaler,
        'outlier_params': {
            'columns': outlier_cols,
            'factor': 2.0
        }
    }
    os.makedirs(os.path.dirname(args.artifact_path), exist_ok=True)
    joblib.dump(preprocessing_artifacts, args.artifact_path)
    print(f"Artefak preprocessing disimpan di: {args.artifact_path}")


    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_processed.to_csv(args.output_path, index=False)
    print(f"Preprocessing selesai. Data bersih disimpan di: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script untuk preprocessing data kualitas air.")
    parser.add_argument('--input-path', type=str, required=True, help='Path ke file CSV data mentah.')
    parser.add_argument('--output-path', type=str, required=True, help='Path untuk menyimpan file CSV hasil proses.')
    parser.add_argument('--artifact-path', type=str, required=True, help='Path untuk menyimpan artefak preprocessing.')
    
    args = parser.parse_args()
    main(args)