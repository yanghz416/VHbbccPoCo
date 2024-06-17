import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import argparse
import glob
print(lgb.__version__)
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, log_loss


# Function to load data
def load_data(dir_path, signal_name, background_names):
    signal_files = glob.glob(f"{dir_path}/{signal_name}/**/*.parquet", recursive=True)
    all_files = glob.glob(f"{dir_path}/**/*.parquet", recursive=True)

    background_files = []
    for background_name in background_names:
        background_files.extend([file for file in all_files if background_name in file])

    data_files = [file for file in all_files if "DATA" in file]

    if not signal_files:
        raise ValueError(f"No signal files found in directory {dir_path}/{signal_name}")
    if not background_files:
        raise ValueError(f"No background files found in directory {dir_path}")
    if not data_files:
        raise ValueError(f"No DATA files found in directory {dir_path}")

    signal_dfs = [pd.read_parquet(file) for file in signal_files]
    background_dfs = [pd.read_parquet(file) for file in background_files]
    data_dfs = [pd.read_parquet(file) for file in data_files]

    signal_df = pd.concat(signal_dfs, ignore_index=True)
    background_df = pd.concat(background_dfs, ignore_index=True)
    data_df = pd.concat(data_dfs, ignore_index=True)

    return signal_df, background_df, data_df

def train_model(X, y, X_data, signal_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    valid_sets = [valid_data]

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': 2,
    }

    bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=valid_sets, 
                    callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # Create directory for the model if it doesn't exist
    model_dir = f'Models/{signal_name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model with the signal name
    bst.save_model(f'{model_dir}/model_DY.txt')
    evaluate_model(bst, X_test, y_test, X_data, signal_name)

    return bst

# Function to evaluate model
def evaluate_model(model, X, y, X_data, signal_name):
    y_pred = model.predict(X)
    data_pred = model.predict(X_data)

    # Calculate metrics
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    loss = log_loss(y, y_pred)

    print(f'AUC: {auc}')
    print(f'Log Loss: {loss}')

    # Create directory for the plots if it doesn't exist
    plot_dir = f'Models/{signal_name}/Plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot histogram of BDT scores
    plt.figure()
    plt.hist(y_pred[y == 0], bins=50, alpha=0.5, label='Background', density=True)
    plt.hist(y_pred[y == 1], bins=50, alpha=0.5, label='Signal', density=True)
    plt.hist(data_pred, bins=50, alpha=0.75, label='Data', density=True)
    plt.xlabel('BDT score')
    plt.legend()
    plt.savefig(f'{plot_dir}/BDT_score.png')

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(f'{plot_dir}/roc_curve.png')


# Main function to load data and train/evaluate model
def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a LightGBM model.')
    parser.add_argument('mode', type=str, help='Mode: "train" or "eval".')
    parser.add_argument('dir_path', type=str, help='Path to the directory containing input files.')
    parser.add_argument('--signal', type=str, help='Name of the signal data file.')
    parser.add_argument('--background', type=str, nargs='+', help='Name of the bkg data files.')
    parser.add_argument('--model', type=str, help='Path to the model file.')
    args = parser.parse_args()

    signal_df, background_df, data_df = load_data(args.dir_path, args.signal, args.background)

    # Add target column to signal and background dataframes
    signal_df['target'] = 1
    background_df['target'] = 0

    # Concatenate signal and background dataframes
    df = pd.concat([signal_df, background_df], ignore_index=True)


    X = df.drop('target', axis=1)  # 'target' is the name of your target variable
    y = df['target']
    

    if args.mode == 'train':
        model = train_model(X, y, data_df, args.signal)
    elif args.mode == 'eval':
        model = lgb.Booster(model_file=args.model)
        evaluate_model(model, X, y, data_df, args.signal)

if __name__ == "__main__":
    main()