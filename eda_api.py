# eda_api.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import plotly.express as px

class eda_def:
    def __init__(self, save_dir, save_flag=True):
        self.save_dir = save_dir
        self.save_flag = save_flag

    #ヒートマップ
    def hmap(self, df, corr_col):
        plt.rcParams['font.family'] = 'Meiryo'
        plt.figure(figsize=(12, 9))
        corr_matrix = df[corr_col].corr()
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr_matrix, annot=True, fmt='.1f', mask=mask,
                    vmin=-1, vmax=1, linewidths=0.5, cmap="coolwarm")
        sns.set(font_scale=1)
        if self.save_flag:
            plt.savefig(self.save_dir + "ヒートマップ.png")
        plt.show()
        plt.rcdefaults()

        
    #相関上位のヒートマップ
    def top_hmap(self, df, target, top=15, annot=True):
        plt.rcParams['font.family'] = 'Meiryo' 
        corr_matrix = df.corr()
        od_corr = corr_matrix[target].apply(abs).sort_values(ascending=False)
        top_features = od_corr.index[1:top+1]
        small_corr_matrix = df[top_features.tolist() + [target]].corr()
        
        plt.figure(figsize=(12, 9))
        mask = np.zeros_like(small_corr_matrix)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(small_corr_matrix, annot=annot, fmt='.2f', vmin=-1, vmax=1, linewidths=0.5, cmap="coolwarm", mask=mask)
        plt.title('Correlation of Top-related Features', size=15)
        if self.save_flag:
            plt.savefig(self.save_dir + target + "_相関上位ヒートマップ.png")
        plt.show()
        plt.rcdefaults()

        
    # 相関上位のペアプロット
    def top_pairplot(self, df, target, top=5, color_column=None):
        plt.rcParams['font.family'] = 'Meiryo'
        corr_matrix = df.corr()
        od_corr = corr_matrix[target].apply(abs).sort_values(ascending=False)
        top_features = od_corr.index[1:top+1].tolist() + [target]  # targetカラムを追加
        subset_df = df[top_features]
        
        if color_column and color_column in df.columns:
            sns.pairplot(subset_df, corner=True, hue=color_column, plot_kws={'alpha': 1, "s": 20, "edgecolor": "k"})
        else:
            sns.pairplot(subset_df, corner=True, plot_kws={'alpha': 1, "s": 20, "edgecolor": "k"})
        
        if self.save_flag:
            plt.savefig(self.save_dir + target + "_相関上位ペアプロット.png")
        plt.show()
        plt.rcdefaults()
        
        
    #指定の変数に対する全散布図
    def plot_scatter_matrix(self, df, y_column, hue_column=None, aspect_ratio=1.0, log_columns=[], y_log_scale=False):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.rcParams['font.family'] = 'Meiryo'
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
    
        if y_column in numeric_columns:
            numeric_columns.remove(y_column)
    
        num_plots = len(numeric_columns)
        ncols = 4
        nrows = -(-num_plots // ncols)
    
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.75 * nrows * aspect_ratio))
        
        if nrows == 1:
            axes = [axes]
        
        for idx, col in enumerate(numeric_columns):
            r, c = divmod(idx, ncols)
            if hue_column and hue_column in df.columns:
                sns.scatterplot(data=df, x=col, y=y_column, hue=hue_column, ax=axes[r][c])
            else:
                sns.scatterplot(data=df, x=col, y=y_column, ax=axes[r][c])
            
            if col in log_columns:
                axes[r][c].set_xscale('log')
            
            if y_log_scale:
                axes[r][c].set_yscale('log')
    
        for idx in range(num_plots, nrows * ncols):
            r, c = divmod(idx, ncols)
            fig.delaxes(axes[r][c])
        
        plt.tight_layout()
        if self.save_flag:
            plt.savefig(self.save_dir + y_column.replace('/', '_') + "_散布図" + ".png")
        plt.show()

        
    #指定の変数での散布図とヒートマップ
    def scatter_with_heatmap(self, df, X, Y, C=None, reverse_size=False):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.interpolate import griddata
    
        plt.rcParams['font.family'] = 'Meiryo'  # フォント設定
        df2 = df.dropna(subset=[X, Y] if C is None else [X, Y, C])
        x = df2[X]
        y = df2[Y]
        
        plt.figure(figsize=(8, 6))
    
        if C is not None:
            c = df2[C]
            min_size, max_size = 15, 100
            if reverse_size:
                size = max_size - (c - c.min()) / (c.max() - c.min()) * (max_size - min_size)
            else:
                size = min_size + (c - c.min()) / (c.max() - c.min()) * (max_size - min_size)
                
            grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
            grid_z = griddata((x, y), c, (grid_x, grid_y), method='linear')
            
            plt.contourf(grid_x, grid_y, grid_z, 20, cmap='plasma', alpha=0.7)
            plt.scatter(x, y, c=c, s=size, edgecolors='w', linewidths=0.5, cmap='plasma')
            cbar = plt.colorbar()
            cbar.set_label(C)
        else:
            plt.scatter(x, y, edgecolors='w', linewidths=0.5)
    
        #plt.title('Scatter Plot of Dispersion')
        plt.xlabel(X, size=15)
        plt.ylabel(Y, size=15)
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        if self.save_flag:
            save_name = f"{X.replace('/', '_')}_{Y.replace('/', '_')}"
            if C is not None:
                save_name += f"_{C.replace('/', '_')}"
            save_name += "_散布図カラーマップ.png"
            plt.savefig(self.save_dir + save_name)
        plt.show()



    # インタラクティブグラフ
    def interactive_scatter_plot(self, df, X, Y, C=None, save_title="interactive_scatter_plot"):
        import plotly.express as px
        
        hover_data = [col for col in df.columns if col not in [X, Y, C]]
        
        if C:
            fig = px.scatter(df, x=X, y=Y, color=C, hover_data=hover_data, size_max=10)
        else:
            fig = px.scatter(df, x=X, y=Y, hover_data=hover_data, size_max=10)
        
        fig.update_layout(width=700, height=600, plot_bgcolor='white')
        fig.update_layout(hoverlabel=dict(font=dict(family="Meiryo")))
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black', zeroline=True, zerolinewidth=1, zerolinecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black', zeroline=True, zerolinewidth=1, zerolinecolor='black', mirror=True)
        
        if self.save_flag:
            fig.write_html(f"{self.save_dir}{save_title}.html")
        
        fig.show()
    

        
    #ヒストグラム　色分け
    def target_hist(self, df, target, kind='hist', low = 0, high = 0.95, **kwargs):
        plt.rcParams['figure.figsize'] = (8, 4)  # グラフのサイズを調整
        plt.rcParams['font.family'] = 'Meiryo'  # フォント設定
        columns = [col for col in df.columns if col != target]  # ターゲット以外のカラムを選択
        n_cols = 3  # 一行に表示するグラフの数
        n_rows = (len(columns) + 2) // n_cols  # 必要な行数
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))  # subplotsでグラフのレイアウトを設定
        axes = axes.flatten()  # axesを1D配列に変換して簡単に扱えるようにする
        
        for i, column in enumerate(columns):
            ax = axes[i]
            
            lower_bound = df[column].quantile(low) # 外れ値を除外
            upper_bound = df[column].quantile(high) # 外れ値を除外
            filtered_data = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            pivot_data = filtered_data.pivot_table(index=filtered_data.index, columns=target, values=column)
            pivot_data.plot(kind=kind, title=column, ax=ax, **kwargs)
            
            # 凡例のラベルをチェックして適切に処理
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [label.split('=')[1] if '=' in label else label for label in labels]
            ax.legend(handles, new_labels)
        
        plt.tight_layout()
        if self.save_flag:
            plt.savefig(self.save_dir + target.replace('/', '_') + "_ヒストグラム" + ".png")
        plt.show()
