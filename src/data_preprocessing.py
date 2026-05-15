import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

def load_and_preprocess_data(data_dir='data', output_file='data/processed_f1_data.csv'):
    print("Loading datasets...")
    
    race_details = pd.read_csv(os.path.join(data_dir, 'race_details.csv'))
    starting_grids = pd.read_csv(os.path.join(data_dir, 'starting_grids.csv'))
    qualifyings = pd.read_csv(os.path.join(data_dir, 'qualifyings.csv'))
    
    race_details['finishing_position'] = pd.to_numeric(race_details['Pos'], errors='coerce').fillna(99).astype(int)
   
    race_details['Podium'] = (race_details['finishing_position'] <= 3).astype(int)
       
    starting_grids['grid_position'] = pd.to_numeric(starting_grids['Pos'], errors='coerce').fillna(20).astype(int)
    
    qualifyings['qualifying_position'] = pd.to_numeric(qualifyings['Pos'], errors='coerce').fillna(20).astype(int)


    print("Merging datasets...")
    
    df = pd.merge(
        race_details[['Year', 'Grand Prix', 'Driver', 'Car', 'Podium', 'finishing_position']],
        starting_grids[['Year', 'Grand Prix', 'Driver', 'grid_position']],
        on=['Year', 'Grand Prix', 'Driver'],
        how='left'
    )
    
    df = pd.merge(
        df,
        qualifyings[['Year', 'Grand Prix', 'Driver', 'qualifying_position']],
        on=['Year', 'Grand Prix', 'Driver'],
        how='left'
    )
    
    df['grid_position'] = df['grid_position'].fillna(20)
    df['qualifying_position'] = df['qualifying_position'].fillna(20)
    
    print("Encoding categorical features...")
    le_driver = LabelEncoder()
    le_grand_prix = LabelEncoder()
    le_car = LabelEncoder()
    
    df['driver_encoded'] = le_driver.fit_transform(df['Driver'].astype(str))
    df['gp_encoded'] = le_grand_prix.fit_transform(df['Grand Prix'].astype(str))
    df['car_encoded'] = le_car.fit_transform(df['Car'].astype(str))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_driver, 'models/le_driver.pkl')
    joblib.dump(le_grand_prix, 'models/le_grand_prix.pkl')
    joblib.dump(le_car, 'models/le_car.pkl')
    
    features = ['Year', 'gp_encoded', 'driver_encoded', 'car_encoded', 'grid_position', 'qualifying_position']
    target = 'Podium'
    
    X = df[features]
    y = df[target]
    
    print("Normalizing numerical features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    
    processed_df = pd.DataFrame(X_scaled, columns=features)
    processed_df['Podium'] = y.values
    processed_df.to_csv(output_file, index=False)
    
    print(f"Preprocessing complete. Processed data saved to {output_file}")
    return processed_df

if __name__ == "__main__":
    load_and_preprocess_data()
