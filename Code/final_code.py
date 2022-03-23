#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 05:26:38 2021

@author: tarun
"""

#Data Manipulation
import numpy as np
import pandas as pd

#Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Random number generation
from random import randint

#Vizualization
import seaborn as sns
import matplotlib.pyplot as plt
import pydot
#from sklearn.tree import export_graphviz
#import graphviz




#Read in Abalone data into a Pandas dataframe and assign column names.
abalone_data_df = pd.read_table('abalone.data', delimiter=",", header=None)
abalone_data_df.columns = ['Sex', 'Length', 'Diametre', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

#Check if any null or NaN values in data.
print('Null or NaN values present in data:', abalone_data_df.isnull().values.any())

#Description of Data
abalone_desc = abalone_data_df.describe()
abalone_desc = abalone_desc.round(3)
abalone_desc.to_csv('abalone_description.csv', index = False)
zero_height = abalone_data_df[abalone_data_df['Height']==0]
zero_height.to_csv('zero_height.csv', index = False)




####### DATA PRE_PROCESSING #######

#Remove observations with height recorded as 0 mm.
abalone_data_df = abalone_data_df[abalone_data_df['Height']!=0]

#Encode the Sex attribute
abalone_data_df.iloc[:, 0] = abalone_data_df.iloc[:, 0].map({'M':0, 'F':1, 'I':-1})

#Encode the Rings attribute
def grouping_age(age):
    
    if 0 <= age <= 7:
        return 1
    elif 8 <= age <= 10:
        return 2
    elif 11 <= age <= 15:
        return 3
    else:
        return 4

abalone_data_df.iloc[:, 8] = abalone_data_df.iloc[:, 8].apply(grouping_age)
abalone_data_df.columns = ['Sex', 'Length', 'Diametre', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Age group']


######### VISUALIZATIONS #########

#https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
#https://seaborn.pydata.org/generated/seaborn.pairplot.html

def pairplot(abalone_data_df, include_hue = False):
    
    if include_hue:
        pplot = sns.pairplot(abalone_data_df.iloc[:,1:9], hue = 'Age group', palette="Set2",
                         height=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=0.5),
                         diag_kind="kde", diag_kws=dict(shade=True))
        fig = pplot.fig 
        fig.subplots_adjust(top=0.93, wspace=0.3)
        t = fig.suptitle('Abalone Continuous Features and Target Pairwise Plots', fontsize=15)
        plt.savefig('pair_plot_hue.png')
        plt.close()
    
    else:
        pplot = sns.pairplot(abalone_data_df.iloc[:,1:9], palette="Set2",
                         height=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=0.5),
                         diag_kind="kde", diag_kws=dict(shade=True))
        fig = pplot.fig 
        fig.subplots_adjust(top=0.93, wspace=0.3)
        t = fig.suptitle('Abalone Continuous Features and Target Pairwise Plots', fontsize=15)
        plt.savefig('pair_plot.png')
        plt.close()

pairplot(abalone_data_df)
pairplot(abalone_data_df, include_hue = True)


def corr_map(abalone_data_df):
    
    abalone_data_array = np.array(abalone_data_df)
    corr_matrix = np.corrcoef(abalone_data_array.T)
    ax = sns.heatmap(corr_matrix, annot=True, fmt='.2g', cmap="rocket", xticklabels=abalone_data_df.columns, yticklabels=abalone_data_df.columns)
    ax.set_title('Correlation Matrix')
    plt.savefig('correlation_matrix.png', bbox_inches = "tight")
    plt.close()

corr_map(abalone_data_df)


def make_boxplot(abalone_data_df, feature = ''):
    
    ax = sns.boxplot(data=abalone_data_df[feature], orient="h", palette="rocket", width = 0.4)
    ax.set_ylabel(feature)
    ax.set_title(f'Boxpot of {feature}')
    plt.savefig('height_boxplot.png')
    plt.close()

make_boxplot(abalone_data_df, feature = 'Height')



#Remove observations with height recorded as 0 mm.
abalone_data_df = abalone_data_df[abalone_data_df['Height']<0.50]
abalone_data_df.to_csv('clean_abalone_data.csv', index = False)

plt.style.use('ggplot')
def bar_plot(data_df = abalone_data_df, feature_col_name = []):
    
    for name in feature_col_name:
        
        bars = []
        height = []
        for k,v in sorted(dict(data_df[name].value_counts()).items()):
            bars.append(str(k))
            height.append(v)
        
        plt.bar(bars, height, color ='green', width = 0.3, alpha=0.7)
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {name}')
        plt.savefig(f'{name}_histogram.png')
        plt.close()


bar_plot(data_df = abalone_data_df, feature_col_name = ['Sex', 'Age group'])



#Class distribution
#abalone_data_df.iloc[:,0].value_counts()
#abalone_data_df.iloc[:,8].value_counts()






######################################################################
########################   MODELLING   ###############################
######################################################################




#Split data    
X,y = abalone_data_df.iloc[:, :8], abalone_data_df.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0, stratify = y)



#### CART DECISION TREE NO REG ####


def DT_no_reg(X_train, X_test, y_train, y_test, num_experiment=10):
    
    train_acc_list = []
    test_acc_list = []
    
    class_1_prec_list = []
    class_2_prec_list = []
    class_3_prec_list = []
    class_4_prec_list = []
    
    class_1_rec_list = []
    class_2_rec_list = []
    class_3_rec_list = []
    class_4_rec_list = []
    
    for experiment in range(num_experiment):
        
        state = randint(0,500)
        clf = DecisionTreeClassifier(random_state = state)
        clf.fit(X_train, y_train)
        
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        cm = confusion_matrix(y_test, y_pred_test)
        
        class_1_precision = round(cm[0,0]/cm[:,0].sum(),3)
        class_2_precision = round(cm[1,1]/cm[:,1].sum(),3)
        class_3_precision = round(cm[2,2]/cm[:,2].sum(),3)
        class_4_precision = round(cm[3,3]/cm[:,3].sum(),3)
        
        class_1_recall = round(cm[0,0]/np.bincount(y_test)[1],3)
        class_2_recall = round(cm[1,1]/np.bincount(y_test)[2],3)
        class_3_recall = round(cm[2,2]/np.bincount(y_test)[3],3)
        class_4_recall = round(cm[3,3]/np.bincount(y_test)[4],3)
        
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    
        class_1_prec_list.append(class_1_precision)
        class_2_prec_list.append(class_2_precision)
        class_3_prec_list.append(class_3_precision)
        class_4_prec_list.append(class_4_precision)
    
        class_1_rec_list.append(class_1_recall)
        class_2_rec_list.append(class_2_recall)
        class_3_rec_list.append(class_3_recall)
        class_4_rec_list.append(class_4_recall)
        
        if experiment == 9:
            
             info_dict = {'depth' : clf.tree_.max_depth,
                         'nodes' : clf.tree_.node_count,
                         'leaves' : clf.tree_.n_leaves}
             
             feat_imp = clf.feature_importances_
        
    results_df = pd.DataFrame({'Experiment' : [i for i in range(1,11)],
                               'Train Accuracy' : train_acc_list,
                               'Test Accuracy' : test_acc_list,
                               'Class 1 Precision' : class_1_prec_list,
                               'Class 1 Recall' : class_1_rec_list,
                               'Class 2 Precision' : class_2_prec_list,
                               'Class 2 Recall' : class_2_rec_list,
                               'Class 3 Precision' : class_3_prec_list,
                               'Class 3 Recall' : class_3_rec_list,
                               'Class 4 Precision' : class_4_prec_list,
                               'Class 4 Recall' : class_4_rec_list})
    
    return results_df, info_dict, feat_imp
    
results_dt_no_reg, info_dt_no_reg, feat_imp_no_reg = DT_no_reg(X_train, X_test, y_train, y_test, num_experiment=10)
results_dt_no_reg.to_csv('results_dt_no_reg.csv', index = False)  






#### CART DECISION TREE WITH REG ####




'''
def grid_search(X_train, X_test, k_fold = 5):
    
    dt = DecisionTreeClassifier()
    param_grid = {'max_depth' : np.arange(3,15), 'max_leaf_nodes' : np.arange(50,500,50), 'min_samples_leaf' : np.arange(5,60,10), 'min_samples_split' : np.hstack(([2], np.arange(5,66,10)))}
    clf_cv = GridSearchCV(dt, param_grid, refit = True, scoring = 'accuracy', cv = k_fold)
    clf_cv.fit(X_train, y_train)
    
    return (clf_cv.best_score_, clf_cv.best_params_)

best_score, best_params = grid_search(X_train, X_test, k_fold = 5)
'''
best_params = {'max_depth': 8, 'max_leaf_nodes': 50, 'min_samples_leaf': 35, 'min_samples_split': 5}

#{'max_depth': 6,
#           'max_leaf_nodes': 50,
#              'min_samples_leaf': 55,
#              'min_samples_split': 2}

#print('The best accuracy obtained from the grid search is:', round(best_score,3))
#print('The hyperparameters (best params) yielding this accuracy are:', best_params)


def DT_reg(X_train, X_test, y_train, y_test, params_dict = best_params, num_experiment=10):
    
    train_acc_list = []
    test_acc_list = []
    
    class_1_prec_list = []
    class_2_prec_list = []
    class_3_prec_list = []
    class_4_prec_list = []
    
    class_1_rec_list = []
    class_2_rec_list = []
    class_3_rec_list = []
    class_4_rec_list = []
    
    for experiment in range(num_experiment):
        
        state = randint(0,500)
        clf = DecisionTreeClassifier(max_depth = params_dict['max_depth'], max_leaf_nodes = params_dict['max_leaf_nodes'],
                                     min_samples_leaf = params_dict['min_samples_leaf'], min_samples_split = params_dict['min_samples_split'],
                                     random_state = state)
        clf.fit(X_train, y_train)
        
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        cm = confusion_matrix(y_test, y_pred_test)
        
        class_1_precision = round(cm[0,0]/cm[:,0].sum(),3)
        class_2_precision = round(cm[1,1]/cm[:,1].sum(),3)
        class_3_precision = round(cm[2,2]/cm[:,2].sum(),3)
        class_4_precision = round(cm[3,3]/cm[:,3].sum(),3)
        
        class_1_recall = round(cm[0,0]/np.bincount(y_test)[1],3)
        class_2_recall = round(cm[1,1]/np.bincount(y_test)[2],3)
        class_3_recall = round(cm[2,2]/np.bincount(y_test)[3],3)
        class_4_recall = round(cm[3,3]/np.bincount(y_test)[4],3)
        
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    
        class_1_prec_list.append(class_1_precision)
        class_2_prec_list.append(class_2_precision)
        class_3_prec_list.append(class_3_precision)
        class_4_prec_list.append(class_4_precision)
    
        class_1_rec_list.append(class_1_recall)
        class_2_rec_list.append(class_2_recall)
        class_3_rec_list.append(class_3_recall)
        class_4_rec_list.append(class_4_recall)
        
        if experiment == 9:
            
            
            
            dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['Sex', 'Length', 'Diametre',
                                                                               'Height', 'Whole weight',
                                                                               'Shucked weight', 'Viscera weight', 'Shell weight'],
                                            class_names = ['1','2','3','4'],
                                            filled=True, rounded=True, rotate = True, leaves_parallel = True)
            graph = pydot.graph_from_dot_data(dot_data)
            graph[0].write_png('tree_viz_reg.png')
            
            
            if clf.max_depth > 5:
                dot_data = tree.export_graphviz(clf, out_file=None, max_depth=5, feature_names=['Sex', 'Length', 'Diametre',
                                                                                                'Height', 'Whole weight',
                                                                               'Shucked weight', 'Viscera weight', 'Shell weight'],
                                            class_names = ['1','2','3','4'],
                                            filled=True, rounded=True, rotate = True, leaves_parallel = True)
                graph = pydot.graph_from_dot_data(dot_data)
                graph[0].write_png('tree_viz_reg_5.png')
            
                
            
            
            info_dict = {'depth' : clf.tree_.max_depth,
                         'nodes' : clf.tree_.node_count,
                         'leaves' : clf.tree_.n_leaves}
            
            feat_imp = clf.feature_importances_
            
        
    
    results_df = pd.DataFrame({'Experiment' : [i for i in range(1,11)],
                               'Train Accuracy' : train_acc_list,
                               'Test Accuracy' : test_acc_list,
                               'Class 1 Precision' : class_1_prec_list,
                               'Class 1 Recall' : class_1_rec_list,
                               'Class 2 Precision' : class_2_prec_list,
                               'Class 2 Recall' : class_2_rec_list,
                               'Class 3 Precision' : class_3_prec_list,
                               'Class 3 Recall' : class_3_rec_list,
                               'Class 4 Precision' : class_4_prec_list,
                               'Class 4 Recall' : class_4_rec_list})
    
    
    return results_df, info_dict, feat_imp
    
results_dt_reg, info_dt_reg, feat_imp_reg = DT_reg(X_train, X_test, y_train, y_test, params_dict = best_params, num_experiment=10)
results_dt_reg.to_csv('results_dt_reg.csv', index = False)  






### COST_COMPLEXITY PRUNING ###
#Code source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html






def DT_ccp(X_train, X_test, y_train, y_test, num_experiment=10):
    
    clf_CC = DecisionTreeClassifier(random_state=42)
    path = clf_CC.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("Effective Alpha")
    ax.set_ylabel("Total Impurity of Leaves")
    ax.set_title("Total Impurity vs Effective Alpha for Training Set")
    plt.savefig('impurity_vs_alpha.png')
    plt.close()
    
    
    #Next, train a decision tree using the effective alphas. 
    #The last value in ccp_alphas is the alpha value that prunes the whole tree,
    #leaving just the root node.
    
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clfs[-1].tree_.node_count, round(ccp_alphas[-1],5)))
    
    
    #Remove the last element in clfs and ccp_alphas, because it is the 
    #trivial tree with only one node.
    #Get viz for interaction between number of nodes 
    #and tree depth decreases as alpha increases.

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Number of Nodes")
    ax.set_title("Number of Nodes vs Alpha")    
    plt.savefig('nodes_vs_alpha.png')
    plt.close()
    
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Tree Depth")
    ax.set_title("Tree Depth vs Alpha")    
    plt.savefig('depth_vs_alpha.png')
    plt.close()
    
    #As alpha increases, more of the tree is pruned, creating 
    #a decision tree that generalizes better.
    
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    fig = plt.figure(figsize=(8,6))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel("Alpha", fontsize="medium")
    ax.set_ylabel("Accuracy", fontsize = "medium")
    ax.set_title("Accuracy vs Alpha for Training and Testing Sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
    ax.legend()
    plt.savefig('accuracy_train_test_alpha.png')
    plt.close()
    
    #Find max test_score and the ccp alpha
    
    chosen_alpha = ccp_alphas[np.argmax(test_scores)]    
    print('Maximum test accuracy achieved is:', round(np.max(test_scores),3), 'by alpha =', round(chosen_alpha,3))
    
    
    
    train_acc_list = []
    test_acc_list = []
    
    class_1_prec_list = []
    class_2_prec_list = []
    class_3_prec_list = []
    class_4_prec_list = []
    
    class_1_rec_list = []
    class_2_rec_list = []
    class_3_rec_list = []
    class_4_rec_list = []
    
    for experiment in range(num_experiment):
        
        state = randint(0,500)
        clf = DecisionTreeClassifier(random_state = state, ccp_alpha = chosen_alpha)
        clf.fit(X_train, y_train)
        
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        cm = confusion_matrix(y_test, y_pred_test)
        
        class_1_precision = round(cm[0,0]/cm[:,0].sum(),3)
        class_2_precision = round(cm[1,1]/cm[:,1].sum(),3)
        class_3_precision = round(cm[2,2]/cm[:,2].sum(),3)
        class_4_precision = round(cm[3,3]/cm[:,3].sum(),3)
        
        class_1_recall = round(cm[0,0]/np.bincount(y_test)[1],3)
        class_2_recall = round(cm[1,1]/np.bincount(y_test)[2],3)
        class_3_recall = round(cm[2,2]/np.bincount(y_test)[3],3)
        class_4_recall = round(cm[3,3]/np.bincount(y_test)[4],3)
        
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    
        class_1_prec_list.append(class_1_precision)
        class_2_prec_list.append(class_2_precision)
        class_3_prec_list.append(class_3_precision)
        class_4_prec_list.append(class_4_precision)
    
        class_1_rec_list.append(class_1_recall)
        class_2_rec_list.append(class_2_recall)
        class_3_rec_list.append(class_3_recall)
        class_4_rec_list.append(class_4_recall)
        
            
        if experiment == 9:
            
            
            dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['Sex', 'Length', 'Diametre',
                                                                               'Height', 'Whole weight',
                                                                               'Shucked weight', 'Viscera weight', 'Shell weight'],
                                            class_names = ['1','2','3','4'],
                                            filled=True, rounded=True, rotate = True, leaves_parallel = True)
            graph = pydot.graph_from_dot_data(dot_data)
            graph[0].write_png('tree_viz_ccp.png')
            
            if clf.tree_.max_depth > 5:
                dot_data = tree.export_graphviz(clf, out_file=None, max_depth=5, feature_names=['Sex', 'Length', 'Diametre',
                                                                                                'Height', 'Whole weight',
                                                                               'Shucked weight', 'Viscera weight', 'Shell weight'],
                                            class_names = ['1','2','3','4'],
                                            filled=True, rounded=True, rotate = True, leaves_parallel = True)
                graph = pydot.graph_from_dot_data(dot_data)
                graph[0].write_png('tree_viz_ccp_5.png')
                
                
            
            
            info_dict = {'depth' : clf.tree_.max_depth,
                         'nodes' : clf.tree_.node_count,
                         'leaves' : clf.tree_.n_leaves}
            
            feat_imp = clf.feature_importances_
        
    
    results_df = pd.DataFrame({'Experiment' : [i for i in range(1,11)],
                               'Train Accuracy' : train_acc_list,
                               'Test Accuracy' : test_acc_list,
                               'Class 1 Precision' : class_1_prec_list,
                               'Class 1 Recall' : class_1_rec_list,
                               'Class 2 Precision' : class_2_prec_list,
                               'Class 2 Recall' : class_2_rec_list,
                               'Class 3 Precision' : class_3_prec_list,
                               'Class 3 Recall' : class_3_rec_list,
                               'Class 4 Precision' : class_4_prec_list,
                               'Class 4 Recall' : class_4_rec_list})
    
    return results_df, info_dict, feat_imp


results_ccp, info_dt_ccp, feat_imp_ccp = DT_ccp(X_train, X_test, y_train, y_test, num_experiment=10)
results_ccp.to_csv('results_ccp.csv', index = False)  






###### RANDOM FORESTS ######





trees_list = [20,40,60,80,100,200,300,400,500]
def RF(X_train, X_test, y_train, y_test, num_trees = trees_list, num_experiment=10):
    
    all_res = []
    
    for trees in num_trees:
        
        train_acc_list = []
        test_acc_list = []
    
        for experiment in range(num_experiment):
            
            state = randint(0,500)
            clf = RandomForestClassifier(n_estimators = trees, max_depth=5, random_state = state)
            clf.fit(X_train, y_train)
        
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
        
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
        
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)    
            
            if trees == 500 and experiment == 9:
                feat_imp = clf.feature_importances_
        
        results = {'Number of Trees' : trees,
                                   'Mean Train Accuracy' : round(np.mean(train_acc_list),3),
                                   'Stdev Train Accuracy' : round(np.std(train_acc_list),3),
                                   'Mean Test Accuracy' : round(np.mean(test_acc_list),3),
                                   'Stdev Test Accuracy' : round(np.std(test_acc_list),3)}
                               
        #print(results)
        
        all_res.append(results)
        
    return pd.DataFrame(all_res), feat_imp
    
results_rf, feat_imp_rf = RF(X_train, X_test, y_train, y_test, num_trees = trees_list, num_experiment=10)
results_rf.to_csv('results_rf.csv', index = False)




#[info_dt_no_reg, info_dt_reg, info_dt_ccp]



def plot_feat_imp():
    
    feature_names = ['Sex', 'Length', 'Diametre', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
    
    
    
    dic = {feature_names[i] : feat_imp_no_reg[i] for i in range(len(feature_names))}
    dic = dict(sorted(dic.items(), key=lambda item: item[1]))  
    plt.figure(figsize=(8,6))
    #plt.tight_layout()
    plt.barh(list(dic.keys()), list(dic.values()), alpha=0.7, color = "blue")
    plt.xlabel('Feature Importance', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.title('Decision Tree Feature Importance (No Regularization)')
    plt.savefig('feat_imp_no_reg.png', bbox_inches = "tight")
    plt.close()
    
    
    dic = {feature_names[i] : feat_imp_reg[i] for i in range(len(feature_names))}
    dic = dict(sorted(dic.items(), key=lambda item: item[1]))
    plt.figure(figsize=(8,6))
    #plt.tight_layout()
    plt.barh(list(dic.keys()), list(dic.values()), alpha=0.7, color = "blue")
    plt.xlabel('Feature Importance', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.title('Decision Tree Feature Importance (With Regularization)')
    plt.savefig('feat_imp_with_reg.png', bbox_inches = "tight")
    plt.close()
    
    
    dic = {feature_names[i] : feat_imp_ccp[i] for i in range(len(feature_names))}
    dic = dict(sorted(dic.items(), key=lambda item: item[1]))
    plt.figure(figsize=(8,6))
    #plt.tight_layout()
    plt.barh(list(dic.keys()), list(dic.values()), alpha=0.7, color = "blue")
    plt.xlabel('Feature Importance', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.title('Decision Tree Feature Importance (CCP)')
    plt.savefig('feat_imp_ccp.png', bbox_inches = "tight")
    plt.close()
    
    
    dic = {feature_names[i] : feat_imp_rf[i] for i in range(len(feature_names))}
    dic = dict(sorted(dic.items(), key=lambda item: item[1]))
    plt.figure(figsize=(8,6))
    #plt.tight_layout()
    plt.barh(list(dic.keys()), list(dic.values()), alpha=0.7, color = "blue")
    plt.xlabel('Feature Importance', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.title('Random Forest Feature Importance (500 trees)')
    plt.savefig('feat_imp_rf.png', bbox_inches = "tight")
    plt.close()

plot_feat_imp()
###### END #######


