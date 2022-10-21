#Define functions used to clean data sets. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
import seaborn as sns
import scipy

class explore():
    '''
    Functions used to explore the data sets. 
    find_na_columns: Function to list columns that contain n/a's.
    regression_scatter: Function to create a scatter plot and calculate the r-squared term between x and y series
    transformation_comparison: Function to create plots to compare log, min_max_scaler, and standard_scaler transformation of a 
        numeric column.
    '''
    def find_na_columns(df: pd.DataFrame):
        """
        Function to list columns that contain n/a's
        Input: df: pd.DataFrame
        Output: List of column names
        """
        na_cols = []
        for col in df:
            if df[col].isna().any():
                na_cols.append(col)
        return(na_cols)
    
    def regression_scatter(x: pd.Series, y: pd.Series, xlabel=None, ylabel=None, title=None, save_loc=None):
        """
        Function to create a scatter plot and calculate the r-squared term between x and y series.
        
        Inputs: x: pd.Series of x values
                y: pd.Series of y values
                xlabel: str, xlabel for plot
                ylabel: str, ylabel for plot
                title: str, title for plot
                save_loc: str, directory to save plot
        Outputs:
                graph image
        """
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        ax = sns.regplot(x = x, y = y,
        line_kws = {
            'label': "R-squared: {}".format(round(r_value*r_value,3))
        })
        # plot legend
        _ = ax.legend()
        _ = plt.xlabel(xlabel)
        _ = plt.ylabel(ylabel)
        _ = plt.title(title)
        _ = plt.show()
        
        if save_loc:
            plt.savefig(save_loc+'.png')

    def transformation_comparison(df: pd.DataFrame, column: str):
        """
        Function to create plots to compare log, min_max_scaler, and standard_scaler transformation of a 
        numeric column.
        
        Inputs: df: pd.DataFrame
                column: str, the name of a numeric column in the dataframe
                
        Outputs:
                None, print a plot showing a histogram of the data with different transformations
                
        Source: https://towardsdatascience.com/how-to-differentiate-between-scaling-normalization-and-log-transformations-69873d365a94
        """
        try:
            #If the column is strictly positive, use the box-cox method of power transformation
            if df[column].min() > 0:
                pt = PowerTransformer('box-cox')
            else:
                pt = PowerTransformer('yeo-johnson')
            fig, axes = plt.subplots(1,4, figsize=(15,3), constrained_layout=True)
            title_text = 'Comparing Transformations of '+column
            _ = fig.suptitle(title_text)
            _ = df.hist(column, ax = axes[0])
            #df['log_col'] = df[col].apply(lambda x: np.log(x+1))
            df[["PowerTransformed"]] = pd.DataFrame(pt.fit_transform(df[[column]]), columns=[column])
            _ = df.hist('PowerTransformed', ax = axes[1])
            _ = plt.xlabel('Values')
            _ = plt.ylabel('Count')
            mms = MinMaxScaler()
            df[['MinMaxScaled']] = pd.DataFrame(mms.fit_transform(df[[column]]), columns=[column])
            _ = df.hist('MinMaxScaled', ax = axes[2])
            ss = StandardScaler()
            df[['StandardScaled']] = pd.DataFrame(ss.fit_transform(df[[column]]), columns=[column])
            _ = df.hist('StandardScaled', ax = axes[3])
            _ = plt.show()
        except:
            raise Exception(column, ' did not plot')

class clean():
    '''
    Functions used to clean the data sets. 
    impute_values: Function to impute missing values for numeric variables. Group column used to make imputations by mean more accurate.
    transform_columns: Function to transform column using box-cox or yeo-johnson method.
    remove_correlation: Function to remove correlated columns.
    '''
    #Imputation
    def impute_values(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str, group_column: str,  impute_value='mean'):
        """
        Function to impute missing values for numeric variables. Group column used to make imputations by mean more accurate.
        Input: train_df: pd.DataFrame, train data
                test_df: pd.DataFrame, test data
                column: str, column to impute
                group_column: str, column to group by for imputing
                impute_value: str {'mean', 'median'}
        Output: dataframe with column with all missing values imputed to impute value
        """
        if impute_value == 'mean':
            subset = train_df[[column,group_column]].groupby(group_column).mean()
        elif impute_value == 'median':
            subset = train_df[[column,group_column]].groupby(group_column).median()
        elif impute_value == 'zero':
            train_df[column] = np.where(train_df[column].isna(), 0, df[column])
            return(train_df)
        else:
            raise Exception('That impute value is invalid. Options include mean or median.')
        subset = subset.reset_index()
        subset = subset.rename(columns={column: 'impute_val'})
        train_df = train_df.merge(subset, how='left', on=group_column)
        train_df[column] = np.where(train_df[column].isna(), train_df['impute_val'], train_df[column])
        train_df = train_df.drop('impute_val', axis=1)
        
        test_df = test_df.merge(subset, how='left', on=group_column)
        test_df[column] = np.where(test_df[column].isna(), test_df['impute_val'], test_df[column])
        test_df = test_df.drop('impute_val', axis=1)
        return(train_df, test_df)

    def transform_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list(), transformer: str):
        """
            Function to transform column using box-cox or yeo-johnson method.

            Inputs:
                train_df: pd.DataFrame, training dataframe containing columns to be fit and transformed
                test_df: pd.DataFrame, testing dataframe containing columns to be transformed
                cols: list()
                transformer: str, name of the transformatoin method {box-cox, yeo-johnson}
            Outputs:
                train and test pd.Dataframes with columns transformed by provided method.
        """
        pt = PowerTransformer(transformer)
        pt.fit(train_df[cols])
        train_df[cols] = pt.transform(train_df[cols])
        test_df[cols] = pt.transform(test_df[cols])
        return(train_df, test_df)

    def remove_correlation(df: pd.DataFrame, target: str, threshold = .8):
        """
        Function to remove correlated columns. Find the columns that have a correlation above the absolute value of the
        threshold. Remove the column that is least correlated with the target variable until there are no more highly 
        correlated columns.
        Inputs:
            df: pd.DataFrame
            target: str, column name of target variable
            threshold: float, the correlation coefficient threshold to consider significant
        Outputs:
            pd.Dataframe with least correlated column removed
            list of dropped columns
        """
        corr_cols = ['placeholder']
        dropped_cols = []
        while len(corr_cols) > 0:
            correlation_matrix = df.corr()
            corr_df = pd.DataFrame(correlation_matrix.unstack(),columns = ['correlation_coef'])
            corr_df = corr_df.reset_index()
            corr_df_significant = corr_df[((corr_df['correlation_coef'] > .8)|(corr_df['correlation_coef'] < -.8))&(corr_df['level_0'] != corr_df['level_1'])]
            corr_cols = corr_df_significant['level_0'].unique()
            if target in corr_cols:
                corr_cols.remove(target)
            if len(corr_cols) == 0:
                return(df, dropped_cols)
            subset = corr_df[(corr_df['level_0'] ==target)&(corr_df['level_1'].isin(corr_cols))]
            subset['correlation_coef'] = abs(subset['correlation_coef'])
            drop_col = subset['level_1'].loc[subset['correlation_coef'].idxmin()]
            dropped_cols.append(drop_col)
            df = df.drop(drop_col, axis=1)