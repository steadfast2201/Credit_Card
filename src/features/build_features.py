import pandas as pd
import pathlib
import sys
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(X, n_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return pd.DataFrame(X_pca)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["build_features"]

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    TARGET = 'Class'

    train_df = pd.read_csv(f"{input_path}/train.csv")
    test_df = pd.read_csv(f"{input_path}/test.csv")

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    X_train_pca = apply_pca(X_train, params["n_components"])
    X_test_pca = apply_pca(X_test, params["n_components"])

    # Add target back
    train_pca_df = X_train_pca.copy()
    train_pca_df[TARGET] = y_train.values
    test_pca_df = X_test_pca.copy()
    test_pca_df[TARGET] = y_test.values

    train_pca_df.to_csv(f"{output_path}/train.csv", index=False)
    test_pca_df.to_csv(f"{output_path}/test.csv", index=False)

if __name__ == "__main__":
    main()
