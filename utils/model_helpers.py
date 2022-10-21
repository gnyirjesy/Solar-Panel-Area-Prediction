#Define functions for model training and evaluation 
import pandas as pd
import numpy as np
import matplotlib.pyplplot as plt
import seaborn as sns
from tensorflow import keras


class evaluation():
    '''
    A set of functions to evaluate model performance and interpret results
    Functions:
    plot_feature_importance: Function to plot the most important features in a model.
    
    '''
    def plot_feature_importance(importance, names):
        '''
        Description: Function to plot the most important features in a model

        Inputs:
            importance: np.array with list of feature importances
            names: np.array with names of features corresponding to importance
        Outputs:
            pd.Dataframe with most important features and plot of feature importances

        this code is from here: 
        https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html
        '''
        #Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
        fi_df = fi_df[fi_df['feature_importance'] > .01]
        #Define size of bar plot
        plt.figure(figsize=(10,8))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        #Add chart labels
        plt.title('FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        return(fi_df)


# Display training progress by printing a single dot for each completed epoch - code from Pierre Gentine - ML for Environmental Engineering class
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

class training():
    '''
    A set of functions to show model training
    Functions:
    plot_feature_importance: Function to plot the most important features in a model.
    
    '''     
# Visualize the model's training progress using the stats stored in the history object. 
# We want to use this data to determine how long to train before the model stops making progress.
#code from Pierre Gentine - ML for Environmental Engineering class
def plot_history(history, title=None):
    '''
        Description: Function to plot how the model is doing during training.
        Visualize the model's training progress using the stats stored in the history object. 

        Inputs:
            history: model training history
            title: str, plot title
        Outputs:
            plot of the model train and validation loss

        Code Source: Pierre Gentine - ML for Environmental Engineering class
    '''
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Sq. Error')
    plt.title(title)
    plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
    plt.legend()
    #plt.ylim([0, 5])