#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

import time
from datetime import datetime
from argparse import ArgumentParser
import os
import pickle

from lsn import *

if __name__ == "__main__":
    # parse user command-input arguments    
    parser = ArgumentParser(description="Wrapper to run the LSN code")
    parser.add_argument("-m", "--method", dest="method", type=str, default="LSN",
                        help="options: ANN, LR, RF, SVM, LSN, default=LSN")
    parser.add_argument("-f", "--features", dest="features", type=str, default="AAL",
                        help="options: AAL, PCA, RFE, RLR, default=AAL")
    parser.add_argument("-t", "--trajectory", dest="trajectory", type=str, default="MMSE",
                        help="options: MMSE, ADAS13, default=MMSE")
    parser.add_argument("-n", "--n_features", dest="n_features", type=int, default=78,
                        help="""number of features to be extracted 
                        (if feature type is not AAL) [default = %(default)s]""")
    parser.add_argument("-i", "--n_iterations", dest="n_iterations", type=int, default=10,
                        help="number of cross-validation iterations")
    parser.add_argument("--no_clinical", action="store_true", default=False)
    parser.add_argument("--no_followup", action="store_true", default=False)
    parser.add_argument("--only_clinical", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate of the LSN [default = %(default)s]")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs for training [default=%(default)s]")
    parser.add_argument("--validate_after", type=int, default=10,
                        help="validate after each n epochs [default=%(default)s]")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="size per gradient descent iteration [default=%(default)s]")
    parser.add_argument("--dropout", type=float, default=0.75,
                        help="keep probability during training [default=%(default)s]")
    parser.add_argument("--net_arch", type=int, nargs="+", default=[50, 50, 50, 50, 20, 10],
                        help="list of the sizes of hidden layers in the LSN default=%(default)s")
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--grid_search", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False)
    parser.add_argument("-o", "--out_path", dest="out_path", type=str, default="output/")    
    opt = parser.parse_args()
    
    # performance metrics to be measured
    perf = {'acc': [], 'auc': [], 'acc_BE': [], 'auc_BE': [], 
            'acc_FE': [], 'auc_FE': [], 'acc_CC': [], 'auc_CC': []}
    
    for i in range(opt.n_iterations):
        if opt.verbose:
            start_time = time.time()
            print("Cross-validation iteration {}".format(i))
            print("Loading data...")
    
        # load data and train-test splits specifications    
        sub_list = pd.read_csv("Exp_CL1_sub_list.csv")
        df = pd.read_csv("Exp_502_602_combined.csv")
        splits = pd.read_pickle("train_test_splits.pkl")
        train_split = splits["train"][i]
        test_split = splits["test"][i]
        ptid = sub_list["PTID"]
        train_ptid = ptid[train_split]
        test_ptid = ptid[test_split]
        df_train = df[df["PTID"].isin(train_ptid)]
        df_test = df[df["PTID"].isin(test_ptid)]
        print(sub_list.shape)
        print(df.shape)
        print(len(train_split))
        print(len(test_split))
        print(ptid.shape)
        print(train_ptid.shape)
        print(test_ptid.shape)
        print(df_train.shape)
        print(df_test.shape)
        
        # obtain the CT values grouped with ROI atlases
        if opt.features == "AAL":
            loc_bl = df_train.columns.get_loc("ACG.L_CT_bl")
            loc_var_tp = df_train.columns.get_loc("ACG.L_CT_var_tp")
            X_train_baseline = df_train.iloc[:, loc_bl:loc_bl+78].values
            X_train_followup = df_train.iloc[:, loc_var_tp:loc_var_tp+78].values
            X_test_baseline = df_test.iloc[:, loc_bl:loc_bl+78].values
            X_test_followup = df_test.iloc[:, loc_var_tp:loc_var_tp+78].values
            print(X_train_baseline.shape)
            print(X_train_followup.shape)
            print(X_test_baseline.shape)
            print(X_test_followup.shape)
        
        # create the reduced datasets with unsupervised feature selection
        elif opt.features == "PCA" or opt.features == "HCA":
            X_train_baseline = np.load("data/{}_bl_train_cv{}.npy".format(opt.features, i))
            X_train_followup = np.load("data/{}_vartp_train_cv{}.npy".format(opt.features, i))
            X_test_baseline = np.load("data/{}_bl_test_cv{}.npy".format(opt.features, i))
            X_test_followup = np.load("data/{}_vartp_test_cv{}.npy".format(opt.features, i))
            print(X_train_baseline.shape)
            print(X_train_followup.shape)
            print(X_test_baseline.shape)
            print(X_test_followup.shape)
        
        # create the reduced datasets with supervised feature selection
        elif opt.features == "RFE" or opt.features == "RLR":
            X_train_baseline = np.load("data/{}_bl_train_{}_cv{}.npy".format(opt.features, opt.trajectory, i))
            X_train_followup = np.load("data/{}_vartp_train_{}_cv{}.npy".format(opt.features, opt.trajectory, i))
            X_test_baseline = np.load("data/{}_bl_test_{}_cv{}.npy".format(opt.features, opt.trajectory, i))
            X_test_followup = np.load("data/{}_vartp_test_{}_cv{}.npy".format(opt.features, opt.trajectory, i))
            print(X_train_baseline.shape)
            print(X_train_followup.shape)
            print(X_test_baseline.shape)
            print(X_test_followup.shape)
    
        # load trajectory classes and auxiliary data
        if opt.trajectory == "MMSE":
            y_train_vector = df_train["MMSE_2c_traj"].values
            y_test_vector = df_test["MMSE_2c_traj"].values
            X_aux_train = df_train[["APOE4", "AGE", "MMSE_bl", "MMSE_var_tp"]].values
            X_aux_test = df_test[["APOE4", "AGE", "MMSE_bl", "MMSE_var_tp"]].values
        elif opt.trajectory == "ADAS13":
            y_train_vector = df_train["ADAS_3c_traj"].values
            y_test_vector = df_test["ADAS_3c_traj"].values
            X_aux_train = df_train[["APOE4", "AGE", "ADAS13_bl", "ADAS13_var_tp"]].values
            X_aux_test = df_test[["APOE4", "AGE", "ADAS13_bl", "ADAS13_var_tp"]].values
        print(y_train_vector.shape)
        print(y_test_vector.shape)
        print(X_aux_train.shape)
        print(X_aux_test.shape)
            
        if opt.verbose:
            delta_t = time.time() - start_time
            print("Time spent loading data: {}".format(delta_t))
            print("Ready to train the LSN model...")
        
        if opt.method == "LSN":    
            # get ready to train the LSN
            if opt.save_model:
                save_model_path = os.getcwd() + "/TF_trained_models/"
            MR_shape = X_train_baseline.shape[1]
            train_size = X_train_baseline.shape[0]
            test_size = X_test_baseline.shape[0]
            n_classes = int(np.amax(y_train_vector) + 1)
            X_MR_train = np.stack((X_train_baseline, X_train_followup), axis=1)
            X_MR_test = np.stack((X_test_baseline, X_test_followup), axis=1)
            y_train = np.zeros((train_size, n_classes))
            y_train[np.arange(train_size), y_train_vector] = 1
            y_test = np.zeros((test_size, n_classes))
            y_test[np.arange(test_size), y_test_vector] = 1
            print(X_MR_train.shape)
            print(X_MR_test.shape)
            print(y_train.shape)
            print(y_test.shape)
            print(X_aux_train.shape)
            print(X_aux_test.shape)
            
            # set the neural network architecture    
            net_arch = {'MR_shape':MR_shape,'n_layers':len(opt.net_arch)-1,'MR_output':opt.net_arch[-2],
                        'use_aux':True,'aux_shape':4,'aux_output':opt.net_arch[-2],'output':n_classes,'reg':0.01}
            for hidden_layer in range(len(opt.net_arch)-1):
                net_arch['l{}'.format(hidden_layer+1)] = opt.net_arch[hidden_layer]
            perf_df = pd.DataFrame(columns=['subject_id','label','pred_prob','pred_label'])
            print(net_arch)
            
            tf.reset_default_graph()
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            
                # ----------------------Train model    -------------------------------
                data = {'X_MR':X_MR_train,'X_aux':X_aux_train,'y':y_train}    
                if check_data_shapes(data,net_arch):      
                    print('train data <-> net_arch check passed')  
                    lsn = siamese_net(net_arch)        
                    optimizer = tf.train.AdamOptimizer(learning_rate = opt.lr).minimize(lsn.loss)
                    
                    tf.global_variables_initializer().run()
                    saver = tf.train.Saver()
                    
                    cur_time = datetime.time(datetime.now())
                    print('\nStart training time: {}'.format(cur_time))                
                    lsn, train_metrics = train_lsn(sess, lsn, data, optimizer, opt.n_epochs, opt.batch_size, 
                                                   opt.dropout, opt.validate_after, opt.verbose)
                    
                    # Save trained model 
                    if opt.save_model:
                        filename = "{}_{}_{}_{}".format(opt.features, opt.trajectory, MR_shape, i)
                        print('saving model at {}'.format("models/" + filename))
                        saver.save(sess, "models/" + filename)
                    
                    cur_time = datetime.time(datetime.now())
                    print('End training time: {}\n'.format(cur_time))  
                else:
                    print('train data <-> net_arch check failed')
        
                # Test model  (within same session)         
                data = {'X_MR':X_MR_test,'X_aux':X_aux_test,'y':y_test}
                if check_data_shapes(data,net_arch):
                    print('test data <-> net_arch check passed')   
                    _,test_metrics = test_lsn(sess,lsn,data)        
                    # populate perf dataframe
                    perf_df['subject_id']  = test_ptid
                    perf_df['label'] = np.argmax(y_test,1)
                    perf_df['pred_prob'] = list(test_metrics['test_preds'])
                    perf_df['pred_label'] = np.argmax(test_metrics['test_preds'],1)
                else:
                    print('test data <-> net_arch check failed')
                    
            # save model stats (accuracy, loss, performance)
            with open(opt.out_path + "{}_{}_{}_{}_train_metrics.pkl".format(
                        opt.features, opt.trajectory, MR_shape, i), "wb") as f:
                pickle.dump(train_metrics, f, pickle.HIGHEST_PROTOCOL)
            with open(opt.out_path + "{}_{}_{}_{}_test_metrics.pkl".format(
                        opt.features, opt.trajectory, MR_shape, i), "wb") as f:
                pickle.dump(test_metrics, f, pickle.HIGHEST_PROTOCOL)
            perf_df.to_csv(opt.out_path + "{}_{}_{}_{}_perf_df.csv".format(
                            opt.features, opt.trajectory, MR_shape, i))
            print(perf_df.shape)
            
            accuracy = test_metrics["test_acc"]
            
            # compute the micro-averaged ROC curve
            y_pred = test_metrics['test_preds']
            y_pred_vector = np.argmax(y_pred, axis=1)
            fpr,tpr,_ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_stats = {'fpr': fpr, 'tpr': tpr}
            roc_auc = auc(fpr, tpr)
        
        else:
            # train a baseline classifier
            MR_shape = X_train_baseline.shape[1]
            train_size = X_train_baseline.shape[0]
            test_size = X_test_baseline.shape[0]
            n_classes = int(np.amax(y_train_vector) + 1)
            print(X_train_baseline.shape)
            print(X_train_followup.shape)
            print(X_aux_train.shape)
            
            if opt.no_followup and opt.no_clinical:
                X_train = X_train_baseline
                X_test = X_test_baseline
            elif opt.no_followup and opt.only_clinical:
                X_train = X_aux_train[:, :3]
                X_test = X_aux_test[:, :3]
            elif opt.no_followup:
                X_train = np.concatenate((X_train_baseline, X_aux_train), axis=1)
                X_test = np.concatenate((X_test_baseline, X_aux_test), axis=1)
            elif opt.no_clinical:
                X_train = np.concatenate((X_train_baseline, X_train_followup), axis=1)
                X_test = np.concatenate((X_test_baseline, X_test_followup), axis=1)
            elif opt.only_clinical:
                X_train = X_aux_train
                X_test = X_aux_test
            else: # if use both followup and clinical attributes
                X_train = np.concatenate((X_train_baseline, X_train_followup, X_aux_train), axis=1)
                X_test = np.concatenate((X_test_baseline, X_test_followup, X_aux_test), axis=1)
            print(X_train.shape)
            print(X_test.shape)
            print(y_train_vector.shape)
            print(y_test_vector.shape)
            print(X_aux_train.shape)
            print(X_aux_test.shape)
            
            if opt.method == "LR":
                clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", verbose=opt.verbose)      
            if opt.method == "SVM":
                clf = SVC(gamma="scale", probability=True, verbose=opt.verbose) 
            if opt.method == "RF":
                clf = RandomForestClassifier(n_estimators=100, verbose=opt.verbose)
            if opt.method == "ANN": # same parameters as the LSN
                hidden_layer_sizes = opt.net_arch[:-2]
                clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, verbose=opt.verbose, 
                                               alpha=0.01, learning_rate_init=opt.lr, batch_size=opt.batch_size)
                                               
            # fit and score the classifier                                
            clf.fit(X_train, y_train_vector)
            y_pred_vector = clf.predict(X_test)
            accuracy = accuracy_score(y_test_vector, y_pred_vector)
            
            # compute ROC curve and AUC
            y_pred = clf.predict_proba(X_test)
            y_test = np.zeros((y_test_vector.shape[0], int(np.amax(y_test_vector)) + 1))
            y_test[np.arange(y_test_vector.shape[0]), y_test_vector] = 1
            fpr,tpr,_ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_stats = {'fpr': fpr, 'tpr': tpr}
            roc_auc = auc(fpr, tpr)
            
        # compute subgroup analysis (edge-cases vs cognitively consistent)
        for subgroup in ["BE", "FE", "CC"]:
            ind = np.where(df_test["{}_gr".format(opt.trajectory)] == subgroup)[0]
            acc_sg = accuracy_score(y_test_vector[ind], y_pred_vector[ind])
            fpr_sg,tpr_sg,_ = roc_curve(y_test[ind,:].ravel(), y_pred[ind,:].ravel())
            roc_auc_sg = auc(fpr_sg, tpr_sg)
            
            roc_stats["fpr_{}".format(subgroup)] = fpr_sg
            roc_stats["tpr_{}".format(subgroup)] = tpr_sg
            perf["acc_{}".format(subgroup)].append(acc_sg)
            perf["auc_{}".format(subgroup)].append(roc_auc_sg)
                
        perf['acc'].append(accuracy)
        perf['auc'].append(roc_auc)
        print("Fold {}: accuracy = {:.4f}, AUC = {:.4f}".format(i, accuracy, roc_auc))
        
    print(perf)
        
    # save ROC stats (true positive ratio, false positive ratio)
    with open(opt.out_path + "{}_{}_{}_{}_roc_stats.pkl".format(
              opt.method, opt.features, opt.trajectory, opt.n_features), "wb") as f:
        pickle.dump(roc_stats, f, pickle.HIGHEST_PROTOCOL)
    
    # save cross-validation model performance
    with open(opt.out_path + "model_performance.csv", "a") as f:
        if os.stat(opt.out_path + "model_performance.csv").st_size == 0:
            f.write("Method,FeatureType,Trajectory,NumberOfFeatures,NoClinical,NoFollowup," +
                    "AccuracyMean,AccuracySD,AUCMean,AUCSD,Accuracy0,Accuracy1,Accuracy2,Accuracy3," +
                    "Accuracy4,Accuracy5,Accuracy6,Accuracy7,Accuracy8,Accuracy9," +
                    "AUC0,AUC1,AUC2,AUC3,AUC4,AUC5,AUC6,AUC7,AUC8,AUC9," +
                    "AccBE0,AccBE1,AccBE2,AccBE3,AccBE4,AccBE5,AccBE6,AccBE7,AccBE8,AccBE9," +
                    "AucBE0,AucBE1,AucBE2,AucBE3,AucBE4,AucBE5,AucBE6,AucBE7,AucBE8,AucBE9," +
                    "AccFE0,AccFE1,AccFE2,AccFE3,AccFE4,AccFE5,AccFE6,AccFE7,AccFE8,AccFE9," +
                    "AucFE0,AucFE1,AucFE2,AucFE3,AucFE4,AucFE5,AucFE6,AucFE7,AucFE8,AucFE9," +
                    "AccCC0,AccCC1,AccCC2,AccCC3,AccCC4,AccCC5,AccCC6,AccCC7,AccCC8,AccCC9," +
                    "AucCC0,AucCC1,AucCC2,AucCC3,AucCC4,AucCC5,AucCC6,AucCC7,AucCC8,AucCC9\n")
                    
        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},".format(
                opt.method, opt.features, opt.trajectory, opt.n_features, opt.no_clinical, opt.no_followup, 
                np.mean(perf['acc']), np.std(perf['acc']), np.mean(perf['auc']), np.std(perf['auc']),
                perf['acc'][0], perf['acc'][1], perf['acc'][2], perf['acc'][3], perf['acc'][4], 
                perf['acc'][5], perf['acc'][6], perf['acc'][7], perf['acc'][8], perf['acc'][9], 
                perf['auc'][0], perf['auc'][1], perf['auc'][2], perf['auc'][3], perf['auc'][4], 
                perf['auc'][5], perf['auc'][6], perf['auc'][7], perf['auc'][8], perf['auc'][9]) + 
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},".format(
                perf['acc_BE'][0], perf['acc_BE'][1], perf['acc_BE'][2], perf['acc_BE'][3], perf['acc_BE'][4], 
                perf['acc_BE'][5], perf['acc_BE'][6], perf['acc_BE'][7], perf['acc_BE'][8], perf['acc_BE'][9], 
                perf['auc_BE'][0], perf['auc_BE'][1], perf['auc_BE'][2], perf['auc_BE'][3], perf['auc_BE'][4], 
                perf['auc_BE'][5], perf['auc_BE'][6], perf['auc_BE'][7], perf['auc_BE'][8], perf['auc_BE'][9]) +
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},".format(
                perf['acc_FE'][0], perf['acc_FE'][1], perf['acc_FE'][2], perf['acc_FE'][3], perf['acc_FE'][4], 
                perf['acc_FE'][5], perf['acc_FE'][6], perf['acc_FE'][7], perf['acc_FE'][8], perf['acc_FE'][9], 
                perf['auc_FE'][0], perf['auc_FE'][1], perf['auc_FE'][2], perf['auc_FE'][3], perf['auc_FE'][4], 
                perf['auc_FE'][5], perf['auc_FE'][6], perf['auc_FE'][7], perf['auc_FE'][8], perf['auc_FE'][9],) +
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                perf['acc_CC'][0], perf['acc_CC'][1], perf['acc_CC'][2], perf['acc_CC'][3], perf['acc_CC'][4], 
                perf['acc_CC'][5], perf['acc_CC'][6], perf['acc_CC'][7], perf['acc_CC'][8], perf['acc_CC'][9], 
                perf['auc_CC'][0], perf['auc_CC'][1], perf['auc_CC'][2], perf['auc_CC'][3], perf['auc_CC'][4], 
                perf['auc_CC'][5], perf['auc_CC'][6], perf['auc_CC'][7], perf['auc_CC'][8], perf['auc_CC'][9]))