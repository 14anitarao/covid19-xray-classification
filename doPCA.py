import random as rand  		  	   		     		  		  		    	 		 		   		 		    	   		     		  		  		    	 		 		   		 		  
'''
@author Anita Rao <arao338@gatech.edu>
'''

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
	     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    training1 = pd.read_csv('data/train_202012051842.csv')
    test1 = pd.read_csv('data/test_202012051842.csv')
    data1 = training1.append(test1, ignore_index=True)
    training2 = pd.read_csv('data/train_202012051917.csv')
    test2 = pd.read_csv('data/test_202012051917.csv')
    data2 = training2.append(test2, ignore_index=True)
    training3 = pd.read_csv('data/train_202012051940.csv')
    test3 = pd.read_csv('data/test_202012051940.csv')   
    data3 = training3.append(test3, ignore_index=True)
    training4 = pd.read_csv('data/train_202012052006.csv')
    test4 = pd.read_csv('data/test_202012052006.csv')   
    data4 = training4.append(test4, ignore_index=True)
    training5 = pd.read_csv('data/train_202012052024.csv')
    test5 = pd.read_csv('data/test_202012052024.csv')   
    data5 = training5.append(test5, ignore_index=True)        

    train_rows1, train_cols1 = training1.shape
    train_rows2, train_cols2 = training2.shape
    train_rows3, train_cols3 = training3.shape
    train_rows4, train_cols4 = training4.shape
    train_rows5, train_cols5 = training5.shape    

    train_rows = [train_rows1, train_rows2, train_rows3, train_rows4, train_rows5]
 
    datasets = [data1, data2, data3, data4, data5]

    for i in range(len(datasets)):

        d = datasets[i]
        rows = train_rows[i]
        X = d[d.columns[1:-2]]
        y = d[d.columns[-2]]

        data = pd.concat([X, y], axis=1, sort=False)
        data = data.rename(columns={"label": "results"})

        training = data[data.columns[:-1]]

        pca = PCA(n_components=40)
        X_pca = pca.fit_transform(training)

        X_pca_df = pd.DataFrame(X_pca)
        red_data = pd.concat([X_pca_df,data[data.columns[-1]]], axis=1, sort=False)
        red_data_train = red_data[:rows]
        red_data_test = red_data[rows:]
  
        red_data_train.to_csv(path_or_buf='code/data/train' + str(i) + '.csv',index=False)
        red_data_test.to_csv(path_or_buf='code/data/test' + str(i) + '.csv',index=False)
	   		     		  		  		    	 		 		   		 		  
