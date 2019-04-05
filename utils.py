import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, learning_curve, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_squared_error, auc, roc_curve, precision_recall_fscore_support

import composition

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_2d_grid_search(grid, midpoint=0.7, vmin=0, vmax=1):
    text_size = 20
    parameters = [x[6:] for x in list(grid.cv_results_.keys()) if 'param_' in x]

    param1 = list(set(grid.cv_results_['param_'+parameters[0]]))
    if parameters[1] == 'class_weight':
        param2 =list(set([d[1] for d in grid.cv_results_['param_'+parameters[1]]]))
    else:
        param2 =list(set(grid.cv_results_['param_'+parameters[1]]))
    scores = grid.cv_results_['mean_test_score'].reshape(len(param1),
                                                         len(param2))

    param1 = [round(param, 2) for param in param1]
    param2 = [round(param, 2) for param in param2]

    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint))
    plt.xlabel(parameters[1], size=text_size)
    plt.ylabel(parameters[0], size=text_size)    
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.colorbar()
    plt.xticks(np.arange(len(param2)), sorted(param2), rotation=90, size=text_size)
    plt.yticks(np.arange(len(param1)), sorted(param1), size=text_size)

    plt.title('grid search')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    adopted from:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve
                      .html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=(6, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

def rf_feature_importance(rf, X_train, N='all', std_deviation=False):
    '''Get feature importances for trained random forest object
    
    Parameters
    ----------
    rf : sklearn RandomForest object
    	This needs to be a sklearn.ensemble.RandomForestRegressor of RandomForestClassifier object that has been fit to data
    N : integer, optional (default=10)
    	The N most important features are displayed with their relative importance scores
    std_deviation : Boolean, optional (default=False)
    	Whether or not error bars are plotted with the feature importance. (error can be very large if maximum_features!='all' while training random forest
    Output
    --------
    graphic :
    	return plot showing relative feature importance and confidence intervals
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> rf = RandomForestRegressor(max_depth=20, random_state=0)
    >>> rf.fit(X_train, y_train)
    >>> rf_feature_importance(rf, N=15)
    ''' 
    if N=='all':
    	N=X_train.shape[1]
    importance_dic = {}
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
    			 axis=0)
    indices = np.argsort(importances)[::-1]
    indices = indices[0:N]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(0, N):
    	importance_dic[X_train.columns.values[indices[f]]]=importances[indices[f]]
    	print(("%d. feature %d (%.3f)" % (f + 1, indices[f], importances[indices[f]])),':', X_train.columns.values[indices[f]])
    
    
    # Plot the feature importances of the forest
    plt.figure(figsize=(6,6))
    plt.title("Feature importances")
    if std_deviation == True:
    	plt.bar(range(0, N), importances[indices], color="r", yerr=std[indices], align="center")
    else:
    	plt.bar(range(0, N), importances[indices], color="r", align="center")
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.xticks(range(0, N), indices, rotation=90)
    plt.xlim([-1, N])
    return X_train.columns.values[indices]

def plot_act_vs_pred(y_actual, y_predicted):
    text_size = 20
    plt.figure(figsize=(10, 10))
    plt.plot(y_actual, y_predicted, marker='o', markersize=14, mfc='#0077be', color='k', linestyle='none', alpha=0.6)
    plt.plot([min([min(y_actual), min(y_predicted)]), max([max(y_actual), max(y_predicted)])], [min([min(y_actual), min(y_predicted)]), max([max(y_actual), max(y_predicted)])], 'k--')
    # plt.title("actual vs. predicted values", size=text_size)
    plt.minorticks_on()
    plt.tick_params(direction='in', length=15, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in', length=7, bottom=True, top=True, left=True, right=True, which='minor')
    # limits = [min([min(y_actual), min(y_predicted)]), max([max(y_actual), max(y_predicted)])]
    limits = [0, max([max(y_actual), max(y_predicted)])]
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xticks(size=text_size)
    plt.yticks(size=text_size)
    plt.xlabel('Actual', size=text_size)
    plt.ylabel('Predicted', size=text_size)

def get_roc_auc(actual, probability, plot=False):
        fpr, tpr, tttt = roc_curve(actual, probability, pos_label=1)
        roc_auc = auc(fpr, tpr)
        if plot is True:
            plt.figure(2, figsize=(6, 6))
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([-0.02, 1.02])
            plt.ylim([-0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")

            plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
        return roc_auc

def get_classification_performance_metrics(actual, predicted, probability, plot=False):

    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    roc_auc = get_roc_auc(actual, probability, plot=plot) * 100

    recall = tp / (fn+tp) * 100
    precision = tp / (tp+fp) * 100

    # print('precision: {:0.2f}, recall: {:0.2f}'.format(precision, recall))
    fscore = 2  * (recall * precision) / (recall + precision)
    # print('f-score: {:0.2f}, ROC_auc: {:0.2f}'.format(fscore, roc_auc))
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return fscore, roc_auc

class MaterialsModel():
    def __init__(self, trained_model, scalar, normalizer):
        self.model = trained_model
        self.scalar = scalar
        self.normalizer = normalizer

    def predict(self, formula):
        '''
        Parameters
        ----------
        formula: str or list of strings
            input chemical formula or list of formulae you want predictions for
    
        Return
        ----------
        prediction: pd.DataFrame()
            predicted values generated from the given data
        '''
        # Store our formula in a dataframe. Give dummy 'taget value'.
        # (we will use composition.generate_features() to get the features)
        if type(formula) is str:
            df_formula = pd.DataFrame()
            df_formula['formula'] = [formula]
            df_formula['target'] = [0]
        if type(formula) is list:
            df_formula = pd.DataFrame()
            df_formula['formula'] = formula
            df_formula['target'] = np.zeros(len(formula))
        # here we get the features associated with the formula
        X, y, formula = composition.generate_features(df_formula)
        # here we scale the data (acording to the training set statistics)
        X_scaled = self.scalar.transform(X)
        X_scaled = self.normalizer.transform(X_scaled)
        y_predicted = self.model.predict(X_scaled)
        # save our predictions to a dataframe
        prediction = pd.DataFrame(formula)
        prediction['predicted value'] = y_predicted
        return prediction





