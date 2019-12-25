def forward_selection(df):
  
  '''

  user defined function to perform standard forward selection on random datafame

  '''

  array = df.values
  X = array[:,0: len(df.columns) - 1]
  Y = array[:, len(df.columns) - 1]

  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2

  # Feature extraction
  test = SelectKBest(score_func=chi2, k=len(df.columns) - 1)
  fit = test.fit(X, Y)

  # Summarize scores
  np.set_printoptions(precision=3)
  features = fit.transform(X)


  chi2_list = []
  for chi2 in fit.scores_:
    chi2_list.append(chi2)

  features_list = df.columns[0: len(df.columns) - 1]

  chi2_df = pd.DataFrame({'feature': features_list,
                          'chi2': chi2_list}).sort_values(by= ['chi2'], ascending= False).reset_index().drop(['index'], axis= 1)

  chi2_features = list(chi2_df['feature'])

  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import cross_val_score

  for i in range(len(df.columns) - 1):

    n = i + 1
    score = 0

    df_sorted = df[list(chi2_df['feature'])]
    array_sorted = df_sorted.values

    X_opt = array_sorted[:, 0: n]
    Y_opt = Y

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=100,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)
    
    score1 = cross_val_score(clf, X_opt, Y_opt, cv= 5, scoring= 'f1')
    

    if score1.mean() > score:
      score = score1
      
  print('Number of used features: ', n, '\n',
        'Used feadtures:', '\n',
        df_sorted.columns[0:n], '\n',
        'Score: ', score.mean())



  def forward_selection_regression(df):  

  used_features = []
  score = 1

  for i in range(len(df.columns) - 1):

    n = i + 1
    '''
    if i == 0:
      score = 1
    else:
      erro_opt_it
    '''
    df_sorted = df[list(chi2_df['feature'])]
    array_sorted = df_sorted.values

    used_features = []

    X_opt = array_sorted[:, 0: n]
    Y_opt = Y

    reg_opt_it = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0, warm_start=False)
    
    
    
    reg_opt_it.fit(x_opt_train, y_opt_train)
    y_opt_pred_it = reg_opt_it.predict(x_opt_test)

    erro_opt_it = np.sqrt(mean_squared_log_error(y_opt_test, y_opt_pred_it))
    erro_opt_it


    if erro_opt_it < score:
      score = erro_opt_it
      used_features.append(df_sorted.columns[i])

  return [len(used_features), score]