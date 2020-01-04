# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gHDLGwRq_VQCNErwPqaghii8QWrrh3i0
"""

def naive_data_preparation(dataset):

  '''
  
  Function used to fill in the gaps with mean for continous features and mode for the discrete ones and apply
  naive label encoding without concerns about the suitability of dummy variables.

  '''

  from sklearn.preprocessing import LabelEncoder


  for column in dataset.columns:
    if dataset[column].dtype != 'object':
        dataset[column].fillna(dataset[column].mean(), inplace= True)
    else:
        dataset[column] = dataset[column].astype('str')
        dataset[column].fillna(dataset[column].mode(), inplace= True)

  
  le = LabelEncoder()

  for column in dataset.columns:
    if dataset[column].dtype == 'object':
      dataset[column] = le.fit_transform(dataset[column])




def classifier_accuracy_graph(dataset, classifier, iterations= 5):  
  
  '''
  docstring
  '''


  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split
  import matplotlib.pyplot as plt
  import pandas as pd 


  colunas = dataset.columns

  accuracy_list_ = []
  x_axis_ = []
  y_axis_ac = []
  colunas_ = []

  for feat in colunas[:-1]:

    colunas_.append(feat)
    accuracy_list_ = []
    
    for it in range(iterations):
          
      x_train_, x_test_, y_train_, y_test_ = train_test_split(dataset[colunas_], dataset[colunas[-1]], test_size= 0.25, random_state= it)


      '''

      clf_ = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                            solver='svd', store_covariance=False, tol=0.0001)

      '''

      clf_ = classifier


      clf_.fit(x_train_, y_train_)
      y_pred_ = clf_.predict(x_test_)

      accuracy_list_.append(accuracy_score(y_pred_, y_test_))
            
    x_axis_.append(feat)
    y_axis_ac.append(pd.Series(accuracy_list_).mean())
    
  plt.title(str(classifier).split('(')[0])
  plt.plot(x_axis_, y_axis_ac)
  plt.xticks(rotation= 45)
  plt.grid()
  plt.show()




def classification_stakcing_model(dataset, model, df_resultados, iterations= 7, folds= 20):

  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score

  x_axis_ = []
  y_axis_ac = []
  
  for fold in range(folds):
    
    accuracy_list_ = []
    f1_list_ = []
    

    for it in range(iterations):

      x_train_, x_test_, y_train_, y_test_ = train_test_split(dataset[dataset.columns[0:-1]], dataset[dataset.columns[-1]], test_size= 0.25, random_state= fold*it)

      clf_ = model
     
      clf_.fit(x_train_, y_train_)
      y_pred_ = clf_.predict(x_test_)

      accuracy_list_.append(accuracy_score(y_pred_, y_test_))
          
      dataframe_ = pd.DataFrame({it: y_pred_}, index= y_test_.index)
      df_resultados = pd.concat([df_resultados, dataframe_], axis= 1)

  final = df_resultados.mode(axis= 1)[0]
  final_ = accuracy_score(final, dataset[dataset.columns[-1]])

  print('Accuracy score {} for {}'.format(final_, (str(model).split('(')[0])))