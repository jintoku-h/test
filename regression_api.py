#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneGroupOut, train_test_split
import shap

class regression_def:
    def __init__(self):
        pass
        
    def feature_creation(self, df, target, threshold=0.4):
        explanatory_variables = [col for col in df.columns if col != target]
        new_df = pd.DataFrame()
        
        for combo in combinations(explanatory_variables, 2):
            new_column = df[combo[0]] * df[combo[1]]
            correlation = new_column.corr(df[target])
            if abs(correlation) >= threshold:
                new_df[f'{combo[0]}_{combo[1]}'] = new_column
        
        new_df[target] = df[target]
        return new_df
    
    def evaluate(self, true, pred):
        mae = np.round(mean_absolute_error(true, pred), 3)
        rmse = np.round(np.sqrt(mean_squared_error(true, pred)), 3)
        r2 = np.round(r2_score(true, pred), 3)
        return mae, rmse, r2
    
    def k_fold(self, X, y, model_fn, n_splits=5, random_state=28):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []
        models = []
        preds_list = []
        fold_number = 0
        
        for train_idx, test_idx in kf.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            model = model_fn(X_train, y_train, X_test, y_test)
            y_pred = model.predict(X_test)
            evaluation_scores = self.evaluate(y_test, y_pred)
            scores.append(evaluation_scores)
            models.append(model)
            preds_list.append(pd.DataFrame({'y_test': y_test, 'y_pred': y_pred, 'fold': fold_number, 'MAE': evaluation_scores[0], 'RMSE': evaluation_scores[1], 'R2': evaluation_scores[2]}))
            fold_number += 1
        
        results_df = pd.concat(preds_list, ignore_index=True)
        scores_df = pd.DataFrame(scores, columns=['MAE', 'RMSE', 'R2'])
        print(scores_df)
        
        return models, results_df, scores_df

        
    def error_plot(self, x, y, y_err, output_dir, title):
        mae, rmse, r2 = self.evaluate(x, y)
        plt.figure(figsize=(6,6))
        ax = plt.subplot(111)
        ax.set_xlabel('True value', fontsize=15)
        ax.set_ylabel('Predict value', fontsize=15)
        plt.errorbar(x, y, yerr=y_err, capsize=3, fmt='o', markerfacecolor='w',
                     markersize=5, ecolor='b', markeredgecolor="b", color='b', linewidth=0.8, label="test")
        plt.title("y-y plot for " + title, size=15)
        ax.set_xlim(y.min()-y.max()*0.2, y.max()*1.3)
        ax.set_ylim(y.min()-y.max()*0.2, y.max()*1.3)
        plt.axline((0, 0), (1, 1), linewidth=0.5, color='k')
        plt.grid(True, linestyle='dotted')
        ax.text(0.65, 0.15, f'R2 = {r2}', transform=ax.transAxes, fontsize=15)
        ax.text(0.65, 0.1, f'MAE = {mae}', transform=ax.transAxes, fontsize=15)
        ax.text(0.65, 0.05, f'RMSE = {rmse}', transform=ax.transAxes, fontsize=15)
        plt.savefig(output_dir + 'yy_all_' + title + ".jpg")
        plt.show()
        return mae, rmse, r2
    
    #K-foldの平均を出力
    def fold_average(self, models, X):
        preds = [model.predict(X) for model in models]
        mean = pd.DataFrame(preds).describe().T["mean"]
        std = pd.DataFrame(preds).describe().T["std"]
        return mean, std
    
    #K-foldした平均値でSHAPを表示
    def average_shap(self, models, X, output_dir, title):
        plt.rcParams['font.family'] = 'Meiryo'
        all_shap_values = []
        
        for model in models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            all_shap_values.append(shap_values)
        
        avg_shap_values = np.mean(all_shap_values, axis=0)
        
        fig = plt.gcf()
        shap.summary_plot(avg_shap_values, X, max_display=10, show=True)
        fig.set_size_inches(15, 10, forward=True)
        fig.savefig(output_dir + "SHAP_" + title + ".jpg")
        plt.show()
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'shap_values': np.abs(avg_shap_values).mean(axis=0)
        })
        importance_df = importance_df.sort_values('shap_values', ascending=False)
        important_features = importance_df['feature'].tolist()
        return important_features

    #K-foldした平均値でSHAP scatter plotを表示
    def average_shap_scatter(self, models, X, top_num=5):
        plt.rcParams['font.family'] = 'Meiryo'
        all_shap_values = []
        
        for model in models:
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            all_shap_values.append(shap_values.values)
        
        avg_shap_values = np.mean(all_shap_values, axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'shap_values': np.abs(avg_shap_values).mean(axis=0)
        })
        importance_df = importance_df.sort_values('shap_values', ascending=False)
        important_features = importance_df['feature'].tolist()
        
        # 上位5つの特徴量のインデックスを取得
        top_features = important_features[:top_num]
        top_indices = [X.columns.get_loc(f) for f in top_features]
        
        # 上位5つの特徴量のみを選択したSHAP値オブジェクトを作成
        top_shap_values = shap.Explanation(values=avg_shap_values[:, top_indices],
                                           base_values=explainer.expected_value,
                                           data=X.values[:, top_indices],
                                           feature_names=[X.columns[i] for i in top_indices])
        
        shap.plots.scatter(top_shap_values)
        plt.show()
        return important_features

    
    #標準化
    def standarization(self, df):
        numerical_columns = df.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        df_std = df.copy()
        df_std[numerical_columns] = scaler.fit_transform(df_std[numerical_columns])
        return df_std

    #欠損値処理（平均で埋める）
    def fill_missing_values(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].fillna(0)
        return df

    def feature_selection(self, data, Target, var = 0.1, thresh=0):
        df = data.loc[:, data.var() > var]
        correlation_matrix = df.corr()
        correlation_with_target = correlation_matrix[Target]
        low_correlation_features = correlation_with_target[correlation_with_target.abs() <= thresh].index # 目的変数との相関が0.2以下の特徴量を抽出  
        df_highcorr = df.drop(low_correlation_features, axis=1)# 低相関の特徴量を削除
        return df_highcorr

    # LGBMと様々なk-foldを組み合わせた関数
    def various_cv_lgbm_model(self, df, target_columns, output_dir, drop_columns=None, group_column=None, cv_method='kfold', fold=5, seed=42, n_repeats=5,
                              test_size=0.2,lgbm_params=None, early_stopping_rounds=50, graph=1):
        if lgbm_params is None:
            lgbm_params = {
                'num_iterations': 1000,
                'max_depth': 20,
                'num_leaves': 20,
                'learning_rate': 0.03,
                'min_child_samples': 5,
                'reg_alpha': 0.001,
                'reg_lambda': 0.1,
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': ['l2', 'rmse'],
                'random_state': seed,
                'verbosity': -1,
            }
    
        eval_results = []
        models = []
    
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=0)
        ]
    
        # Define cross-validation method
        if cv_method == 'kfold':
            cv = KFold(n_splits=fold, shuffle=True, random_state=seed)
        elif cv_method == 'stratified':
            cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        elif cv_method == 'logo' and group_column is not None:
            cv = LeaveOneGroupOut()
        elif cv_method == 'repeated_holdout':
            cv = None  # Repeated holdout does not use a predefined CV method
        else:
            raise ValueError("group_column is wrong")
    
        for target in target_columns:
            temp_df = df.dropna(subset=[target])
            if drop_columns:
                X2 = temp_df.drop(drop_columns, axis=1)
            else:
                X2 = temp_df.copy()  # drop_columnsがNoneの場合、すべての列を使用
            X = pd.get_dummies(X2, dtype=int)
            y = temp_df[target]
            print(f"{target} shape is {X.shape}")
    
            if cv_method == 'stratified':
                y_bins = pd.qcut(y, q=10, duplicates='drop')
                y_bins = pd.factorize(y_bins)[0]  # y_binsをカテゴリの整数に変換
    
            if cv_method == 'repeated_holdout':
                for repeat in range(n_repeats):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed + repeat)
                    
                    model = lgb.LGBMRegressor(**lgbm_params)
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
                    y_pred = model.predict(X_test)
                    y_pred_train = model.predict(X_train)
    
                    # 評価指標の計算
                    r2_train = r2_score(y_train, y_pred_train)
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
                    r2_test = r2_score(y_test, y_pred)
                    mae_test = mean_absolute_error(y_test, y_pred)
                    rmse_test = mean_squared_error(y_test, y_pred, squared=False)
    
                    eval_results.append({
                        'Output': target,
                        'Repeat': repeat + 1,
                        'Train R2': r2_train,
                        'Train MAE': mae_train,
                        'Train RMSE': rmse_train,
                        'Test R2': r2_test,
                        'Test MAE': mae_test,
                        'Test RMSE': rmse_test
                    })
                    models.append(model)
    
                    if repeat == 0:  # 最初のリピートの結果のみをプロット
                        plt.figure(figsize=(6, 6))
                        plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle='--')
                        plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='Train')
                        plt.scatter(y_test, y_pred, color='red', alpha=0.5, label='Test')
                        plt.xlim(y.min() - y.max() * 0.2, y.max() * 1.3)
                        plt.ylim(y.min() - y.max() * 0.2, y.max() * 1.3)
                        plt.xlabel('Actual', size=15)
                        plt.ylabel('Predicted', size=15)
                        plt.title(f'yy-plot for {target}')
                        ax = plt.gca()
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        plt.text(xlim[1] * 0.95, ylim[0] * 0.9, f'Train R2: {r2_train:.2f}, Test R2: {r2_test:.2f}\nTrain MAE: {mae_train:.2f}, Test MAE: {mae_test:.2f}\nTrain RMSE: {rmse_train:.2f}, Test RMSE: {rmse_test:.2f}', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
                        plt.legend()
                        plt.show()
            else:
                if cv_method in ['logo'] and group_column is not None:
                    groups = temp_df[group_column].values
                else:
                    groups = None
    
                for i, (train_index, test_index) in enumerate(cv.split(X, y_bins if cv_method == 'stratified' else y, groups), start=1):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
                    model = lgb.LGBMRegressor(**lgbm_params)
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
                    y_pred = model.predict(X_test)
                    y_pred_train = model.predict(X_train)
    
                    # 評価指標の計算
                    r2_train = r2_score(y_train, y_pred_train)
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
                    r2_test = r2_score(y_test, y_pred)
                    mae_test = mean_absolute_error(y_test, y_pred)
                    rmse_test = mean_squared_error(y_test, y_pred, squared=False)
    
                    eval_results.append({
                        'Output': target,
                        'Fold': i,
                        'Train R2': r2_train,
                        'Train MAE': mae_train,
                        'Train RMSE': rmse_train,
                        'Test R2': r2_test,
                        'Test MAE': mae_test,
                        'Test RMSE': rmse_test
                    })
                    models.append(model)
    
                    if i == graph:  # 最初のフォールドの結果のみをプロット
                        plt.figure(figsize=(6, 6))
                        plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle='--')
                        plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='Train')
                        plt.scatter(y_test, y_pred, color='red', alpha=0.5, label='Test')
                        plt.xlim(y.min() - y.max() * 0.2, y.max() * 1.3)
                        plt.ylim(y.min() - y.max() * 0.2, y.max() * 1.3)
                        plt.xlabel('Actual', size=15)
                        plt.ylabel('Predicted', size=15)
                        plt.title(f'yy-plot for {target}')
                        ax = plt.gca()
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        plt.text(xlim[1] * 0.95, ylim[0] * 0.9, f'Train R2: {r2_train:.2f}, Test R2: {r2_test:.2f}\nTrain MAE: {mae_train:.2f}, Test MAE: {mae_test:.2f}\nTrain RMSE: {rmse_train:.2f}, Test RMSE: {rmse_test:.2f}', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
                        plt.legend()
                        plt.show()
    
        results_df = pd.DataFrame(eval_results)
        results_df.to_csv(output_dir + f"{cv_method}_{fold}-fold_results.csv", float_format='%.2f', index=False)
    
        # 各ターゲット毎に平均値を計算
        average_results = results_df.groupby(['Output']).agg({
            'Train R2': 'mean',
            'Train MAE': 'mean',
            'Train RMSE': 'mean',
            'Test R2': 'mean',
            'Test MAE': 'mean',
            'Test RMSE': 'mean'
        }).reset_index()
    
        # 平均値を行として追加
        for _, avg_row in average_results.iterrows():
            avg_row_dict = avg_row.to_dict()
            avg_row_dict.update({'Fold': 'Mean', 'Repeat': 'Mean'})
            avg_row_df = pd.DataFrame([avg_row_dict])
            results_df = pd.concat([results_df, avg_row_df], ignore_index=True)
    
        return results_df, models