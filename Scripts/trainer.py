# imports
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from data import *
import joblib

import warnings
warnings.filterwarnings('ignore')

gen_col_drop = ['Gen_RPM_Min', 'Gen_RPM_Min_av', 'Gen_RPM_Min_sd', 
                'Gen_RPM_Avg', 'Gen_RPM_Avg_av', 'Gen_RPM_Avg_sd',
                'Gen_Phase1_Temp_Avg', 'Gen_Phase1_Temp_Avg_av', 'Gen_Phase1_Temp_Avg_sd', 
                'Gen_Phase2_Temp_Avg', 'Gen_Phase2_Temp_Avg_av', 'Gen_Phase2_Temp_Avg_sd', 
                'Blds_PitchAngle_Min', 'Blds_PitchAngle_Min_av', 'Blds_PitchAngle_Min_sd', 
                'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Avg_sd', 
                'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Avg_sd', 
                'Amb_WindSpeed_Std', 'Amb_WindSpeed_Std_av', 'Amb_WindSpeed_Std_sd', 
                'Cont_Top_Temp_Avg', 'Cont_Top_Temp_Avg_av', 'Cont_Top_Temp_Avg_sd', 
                'Cont_VCP_Temp_Avg', 'Cont_VCP_Temp_Avg_av', 'Cont_VCP_Temp_Avg_sd',
                'Grd_InverterPhase1_Temp_Avg', 'Grd_InverterPhase1_Temp_Avg_av', 'Grd_InverterPhase1_Temp_Avg_sd']
bear_col_drop = ['Gen_RPM_Min', 'Gen_RPM_Min_av', 'Gen_RPM_Min_sd', 
                'Gen_RPM_Avg', 'Gen_RPM_Avg_av', 'Gen_RPM_Avg_sd',
                'Gen_Phase1_Temp_Avg', 'Gen_Phase1_Temp_Avg_av', 'Gen_Phase1_Temp_Avg_sd', 
                'Gen_Phase2_Temp_Avg', 'Gen_Phase2_Temp_Avg_av', 'Gen_Phase2_Temp_Avg_sd', 
                'Blds_PitchAngle_Min', 'Blds_PitchAngle_Min_av', 'Blds_PitchAngle_Min_sd', 
                'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Avg_sd', 
                'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Avg_sd', 
                'Amb_WindSpeed_Std', 'Amb_WindSpeed_Std_av', 'Amb_WindSpeed_Std_sd', 
                'Cont_Top_Temp_Avg', 'Cont_Top_Temp_Avg_av', 'Cont_Top_Temp_Avg_sd', 
                'Cont_VCP_Temp_Avg', 'Cont_VCP_Temp_Avg_av', 'Cont_VCP_Temp_Avg_sd',
                 'Grd_InverterPhase1_Temp_Avg', 'Grd_InverterPhase1_Temp_Avg_av', 'Grd_InverterPhase1_Temp_Avg_sd']
transf_col_drop = ['HVTrafo_Phase2_Temp_Avg', 'HVTrafo_Phase2_Temp_Avg_av', 'HVTrafo_Phase2_Temp_Avg_sd', 
                   'HVTrafo_Phase1_Temp_Avg', 'HVTrafo_Phase1_Temp_Avg_av', 'HVTrafo_Phase1_Temp_Avg_sd', 
                   'Blds_PitchAngle_Min', 'Blds_PitchAngle_Min_av', 'Blds_PitchAngle_Min_sd', 
                   'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Avg_sd', 
                   'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Avg_sd', 
                   'Amb_WindSpeed_Std', 'Amb_WindSpeed_Std_av', 'Amb_WindSpeed_Std_sd', 
                   'Cont_Top_Temp_Avg', 'Cont_Top_Temp_Avg_av', 'Cont_Top_Temp_Avg_sd', 
                   'Cont_VCP_Temp_Avg', 'Cont_VCP_Temp_Avg_av', 'Cont_VCP_Temp_Avg_sd', 
                   'Grd_Busbar_Temp_Avg', 'Grd_Busbar_Temp_Avg_av', 'Grd_Busbar_Temp_Avg_sd']
hyd_col_drop = ['Blds_PitchAngle_Min', 'Blds_PitchAngle_Min_av', 'Blds_PitchAngle_Min_sd', 
                'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Avg_sd', 
                'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Avg_sd', 
                'Amb_WindSpeed_Min', 'Amb_WindSpeed_Min_av', 'Amb_WindSpeed_Min_sd', 
                'Cont_Top_Temp_Avg', 'Cont_Top_Temp_Avg_av', 'Cont_Top_Temp_Avg_sd', 
                'Cont_VCP_Temp_Avg', 'Cont_VCP_Temp_Avg_av', 'Cont_VCP_Temp_Avg_sd', 
                'Grd_Busbar_Temp_Avg', 'Grd_Busbar_Temp_Avg_av', 'Grd_Busbar_Temp_Avg_sd']
gearbox_cols = ['Turbine_ID', 'Date','TTF', 'Failure', 'Gear_Oil_Temp_Avg', 'Gear_Bear_Temp_Avg',
                'Gear_Oil_Temp_Avg_av', 'Gear_Bear_Temp_Avg_av', 'Gear_Oil_Temp_Avg_sd', 'Gear_Bear_Temp_Avg_sd']

# Scale within each turbine
def scale (df_train, df_test, scaler='StandardScaler'):
    
    '''Scale within each given turbine
    
    Args:
            df_train      : Train datarame
            df_test       : Test dataframe
        
    Returns:
            array         : Scaled array of train and test'''
    
    
    # Scale for turbine T01 first
    X_train1 = df_train.loc[df_train['Turbine_ID']=='T01']
    X_test1 = df_test.loc[df_test['Turbine_ID']=='T01']

    X_train1 = X_train1.drop(columns=['Turbine_ID', 'Date', 'TTF', 'Failure'])
    X_test1 = X_test1.drop(columns=['Turbine_ID', 'Date', 'TTF', 'Failure'])
    
    if scaler == 'MinMaxScaler':
        sc = MinMaxScaler()
        X_train1 = sc.fit_transform(X_train1)
        X_test1 = sc.transform(X_test1) 
    else:
        sc = StandardScaler()
        X_train1 = sc.fit_transform(X_train1)
        X_test1 = sc.transform(X_test1) 
    
    # Scale on other turbines
    turbines = ['T06', 'T07', 'T09', 'T11']
    for turbine in turbines:
        X_train_ = df_train.loc[df_train['Turbine_ID']==turbine]
        X_test_ = df_test.loc[df_test['Turbine_ID']==turbine]

        X_train_ = X_train_.drop(columns=['Turbine_ID', 'Date', 'TTF', 'Failure'])
        X_test_ = X_test_.drop(columns=['Turbine_ID', 'Date', 'TTF', 'Failure'])

        if scaler == 'MinMaxScaler':
            sc = MinMaxScaler()
            X_train_ = sc.fit_transform(X_train_)
            X_test_ = sc.transform(X_test_)
        else:
            sc = StandardScaler()
            X_train_ = sc.fit_transform(X_train_)
            X_test_ = sc.transform(X_test_)
        
        # Concatenate
        X_train1 = np.concatenate((X_train1, X_train_))
        X_test1 = np.concatenate((X_test1, X_test_))
        
             
    return X_train1, X_test1


def data_aug(X_train, y_train):
    
    '''Function for data augmentation, using SMOTE
    
    Args:
            X_train                 : X_train
            y_train                 : y_train
        
    Returns:
            X_train, y_train         : X_train, y_train SMOTE training data '''
    
    
    all_classes = Counter(y_train)
    majority_class = all_classes.most_common(1)
    minority_class = all_classes.most_common()[1:]
    ratio = minority_class[0][1]/majority_class[0][1]
    
    # If ratio of training set < 10% 
    if ratio < 0.1:
        over = SMOTE(sampling_strategy=0.1, random_state=0) # 10% of positive examples
        X_train, y_train = over.fit_sample(X_train, y_train)    
    
    return X_train, y_train


class Trainer():
    def __init__(self, X_train, y_train, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        self.kwargs = kwargs
        self.X_train = X_train
        self.y_train = y_train
        self.component = self.kwargs.get("component")


    def get_estimator(self):
        """return estimator"""
        if self.component == "Generator":
            model =  GradientBoostingClassifier(learning_rate=0.8, random_state=42)
        elif self.component == "Generator Bearing":
            model = AdaBoostClassifier(learning_rate=0.6, n_estimators=10, random_state=42)
        elif self.component == "Transformer":
            model = GaussianNB()
        elif self.component == "Hydraulic":
            model = GradientBoostingClassifier(learning_rate=0.5, random_state=42)
        elif self.component == "Gearbox":
            model = AdaBoostClassifier(learning_rate=0.5, n_estimators=100, random_state=42)
        else:
            print('Invalid Component')

        return model

    def train(self):
        """set and train"""
        self.model = self.get_estimator()
        self.model.fit(self.X_train, self.y_train)

        return self.model

    def predict(self, X_test):
        """Predict"""
        y_score = self.model.predict_proba(X_test)[:,1]
        y_pred = self.model.predict(X_test)

        predictions = {'y_pred' : y_pred, 'y_score' : y_score}
        df_predictions = pd.DataFrame.from_dict(predictions)
        
        return df_predictions

def threshold(df, algo, trsh=0.5):
    '''Function to apply a threshold to try to minimize FP and improve costs savings
    
    Args:
            df                 : dataframe to apply
            algo               : model chosed
        
    Returns:
            df_costs           : a data frame with a new y_pred column for the chosen threshold'''
    
    # Concat with "True Failure" 
    df_costs = pd.concat([algo, df[['Turbine_ID', 'Date', 'TTF', 'Failure']]], axis=1)
    # Create a Column with a cutoff
    df_costs['y_trsh'] = np.where(((df_costs['y_score'] <= trsh)), 0, 1)
    
    return df_costs

def pos_neg(df, days=20):
    
    '''Function to apply financial engineering'''
    
    df['TP'] = 0 # True positives
    df['FN'] = 0 # False Negatives
    df['FP'] = 0 # False Positives
    df['GO'] = 0 # An employee go
    # True Positives
    df['TP'] = df.apply(lambda x: 1 if x['y_trsh'] == 1 and x['Failure'] == 1 else x['TP'], axis = 1)
    # False Positives
    df['FP'] = df.apply(lambda x: 1 if x['y_trsh'] == 1 and x['Failure'] == 0 else x['FP'], axis = 1)
    # False Negatives
    df['FN'] = df.apply(lambda x: 1 if x['y_trsh'] == 0 and x['Failure'] == 1 else x['FN'], axis = 1)
    # Aux column for the 20 days interval
    for i, (tp, fp) in enumerate(zip(df['TP'], df['FP'])):
        if tp == 1 or fp == 1: # Find the first FP or TP
            df['GO'][i] = 1
            break
    #From 20 to 20 days flag as 1
    i = df.index[df['GO'] == 1]
    df.iloc[i[0]::days, :]['GO'] = 1
    
    return df

def mat_cost(df, component):
    
    '''Function to calculate final costs savings for each component'''
    
    # Aux Variables
    count_fn = 0
    count_fp = 0
    tp_cost = 0
    aux_date = None
    aux_turbine = None
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Financial Engineering
    for i, (tp, fn, fp, go) in enumerate(zip(df['TP'], df['FN'], df['FP'], df['GO'])):
        if component == 'Gearbox':
            if tp == 1 and go == 1:
                cost = df_costs.loc[0, 'Replacement_Cost']-(df_costs.loc[0, 'Repair_Cost']+
                                    (df_costs.loc[0, 'Replacement_Cost']-df_costs.loc[0, 'Repair_Cost'])*(1-df['TTF'][i]/60))
                if aux_date is None: # If there is a TP, not count FN or other TP until failure
                    tp_cost += cost
                    aux_date = df['Date'][i]
                    aux_turbine = df['Turbine_ID'][i]
                else:
                    if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58: #label between 2-60 days
                        pass
                    else:
                        tp_cost += cost
                        aux_date = df['Date'][i]
                        aux_turbine = df['Turbine_ID'][i]
                
            elif fn == 1 and go == 1:
                if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58: #label between 2-60 days
                    pass
                else:
                    count_fn += 1

            elif fp == 1 and go == 1:
                count_fp += 1

            fn_cost = df_costs.loc[0, 'Replacement_Cost']*count_fn
            fp_cost = df_costs.loc[0, 'Inspection_cost']*count_fp
            
        
        if component == 'Generator':
            if tp == 1 and go == 1:
                cost = df_costs.loc[1, 'Replacement_Cost']-(df_costs.loc[1, 'Repair_Cost']+
                                    (df_costs.loc[1, 'Replacement_Cost']-df_costs.loc[1, 'Repair_Cost'])*(1-df['TTF'][i]/60))
                if aux_date is None: # If there is a TP, not count FN or other TP until failure
                    tp_cost += cost
                    aux_date = df['Date'][i]
                    aux_turbine = df['Turbine_ID'][i]
                else:
                    if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58: 
                        pass
                    else:
                        tp_cost += cost
                        aux_date = df['Date'][i]
                        aux_turbine = df['Turbine_ID'][i]
                
            elif fn == 1 and go == 1:
                if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58:
                    pass
                else:
                    count_fn += 1

            elif fp == 1 and go == 1:
                count_fp += 1

            fn_cost = df_costs.loc[1, 'Replacement_Cost']*count_fn
            fp_cost = df_costs.loc[1, 'Inspection_cost']*count_fp
            
            
        if component == 'Generator Bearing':
            if tp == 1 and go == 1:
                cost = df_costs.loc[2, 'Replacement_Cost']-(df_costs.loc[2, 'Repair_Cost']+
                                    (df_costs.loc[2, 'Replacement_Cost']-df_costs.loc[2, 'Repair_Cost'])*(1-df['TTF'][i]/60))
                if aux_date is None: # If there is a TP, not count FN or other TP until failure
                    tp_cost += cost
                    aux_date = df['Date'][i]
                    aux_turbine = df['Turbine_ID'][i]
                else:
                    if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58: 
                        pass
                    else:
                        tp_cost += cost
                        aux_date = df['Date'][i]
                        aux_turbine = df['Turbine_ID'][i]
                
                
            elif fn == 1 and go == 1:
                if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58:
                    pass
                else:
                    count_fn += 1

            elif fp == 1 and go == 1:
                count_fp += 1

            fn_cost = df_costs.loc[2, 'Replacement_Cost']*count_fn
            fp_cost = df_costs.loc[2, 'Inspection_cost']*count_fp
            
        if component == 'Transformer':
            if tp == 1 and go == 1:
                cost = df_costs.loc[3, 'Replacement_Cost']-(df_costs.loc[3, 'Repair_Cost']+
                                    (df_costs.loc[3, 'Replacement_Cost']-df_costs.loc[3, 'Repair_Cost'])*(1-df['TTF'][i]/60))
                if aux_date is None: # If there is a TP, not count FN or other TP until failure
                    tp_cost += cost
                    aux_date = df['Date'][i]
                    aux_turbine = df['Turbine_ID'][i]
                else:
                    if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58: 
                        pass
                    else:
                        tp_cost += cost
                        aux_date = df['Date'][i]
                        aux_turbine = df['Turbine_ID'][i]
                
            elif fn == 1 and go == 1:
                if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58:
                    pass
                else:
                    count_fn += 1

            elif fp == 1 and go == 1:
                count_fp += 1

            fn_cost = df_costs.loc[3, 'Replacement_Cost']*count_fn
            fp_cost = df_costs.loc[3, 'Inspection_cost']*count_fp
            

            
        if component == 'Hydraulic':
            if tp == 1 and go == 1:
                cost = df_costs.loc[4, 'Replacement_Cost']-(df_costs.loc[4, 'Repair_Cost']+
                                    (df_costs.loc[4, 'Replacement_Cost']-df_costs.loc[4, 'Repair_Cost'])*(1-df['TTF'][i]/60))
                if aux_date is None: # If there is a TP, not count FN or other TP until failure
                    tp_cost += cost
                    aux_date = df['Date'][i]
                    aux_turbine = df['Turbine_ID'][i]
                else:
                    if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58: 
                        pass
                    else:
                        tp_cost += cost
                        aux_date = df['Date'][i]
                        aux_turbine = df['Turbine_ID'][i]
                
            elif fn == 1 and go == 1:
                if df['Turbine_ID'][i] == aux_turbine and ((df['Date'][i] - aux_date)).days<58:
                    pass
                else:
                    count_fn += 1

            elif fp == 1 and go == 1:
                count_fp += 1

            fn_cost = df_costs.loc[4, 'Replacement_Cost']*count_fn
            fp_cost = df_costs.loc[4, 'Inspection_cost']*count_fp
            
           
        savings = tp_cost - fn_cost - fp_cost
    
    return savings


if __name__ == "__main__":
    # get data
    print("Getting Data")
    signals_df, failures_df, df_costs = get_data()
    # clean data
    print("Cleaning Data")
    signals_df, failures_df = clean_data(signals_df, failures_df)
    # Dataframe for each component
    print("Creating Dataframes components")
    df_generator, df_gen_bear, df_transformer, df_hydraulic, df_gearbox = choose_features(signals_df)
    #Dataframe for each component failure
    print("Creating Dataframes failures")
    df_fail_gen, df_fail_gen_bear, df_fail_transf, df_fail_hyd, df_fail_gearbox = choose_failure(failures_df)
    # Do the train test split before any other operation
    print("Do Train test Split")
    generator_train, generator_test = split_df(df_generator)
    gen_bear_train, gen_bear_test = split_df(df_gen_bear)
    transformer_train, transformer_test = split_df(df_transformer)
    hydraulic_train, hydraulic_test = split_df(df_hydraulic)
    gearbox_train, gearbox_test = split_df(df_gearbox, component='gearbox')
    df_fail_gen_train, df_fail_gen_test = split_df(df_fail_gen)
    df_fail_gen_bear_train, df_fail_gen_bear_test = split_df(df_fail_gen_bear)
    df_fail_transf_train, df_fail_transf_test = split_df(df_fail_transf)
    df_fail_hyd_train, df_fail_hyd_test = split_df(df_fail_hyd)
    df_fail_gearbox_train, df_fail_gearbox_test = split_df(df_fail_gearbox, component='gearbox')
    # Data Augmentation with a period of 60 days
    print("Rolling Averages")
    generator_train = add_features(generator_train, 60)
    generator_test = add_features(generator_test, 60)
    gen_bear_train = add_features(gen_bear_train, 60)
    gen_bear_test = add_features(gen_bear_test, 60)
    transformer_train = add_features(transformer_train, 60)
    transformer_test = add_features(transformer_test, 60)
    hydraulic_train = add_features(hydraulic_train, 60)
    hydraulic_test = add_features(hydraulic_test, 60)
    gearbox_train = add_features(gearbox_train, 60)
    gearbox_test = add_features(gearbox_test, 60)
    # Prepare Train and Test - Merge signals with failures and compute TTF
    print("Creating Final Dataframes")
    generator_train = prepare_df(generator_train, df_fail_gen_train)
    generator_test = prepare_df(generator_test, df_fail_gen_test)
    gen_bear_train = prepare_df(gen_bear_train, df_fail_gen_bear_train)
    gen_bear_test = prepare_df(gen_bear_test, df_fail_gen_bear_test)
    transformer_train = prepare_df(transformer_train, df_fail_transf_train)
    transformer_test = prepare_df(transformer_test, df_fail_transf_test)
    hydraulic_train = prepare_df(hydraulic_train, df_fail_hyd_train)
    hydraulic_test = prepare_df(hydraulic_test, df_fail_hyd_test)
    gearbox_train = prepare_df(gearbox_train, df_fail_gearbox_train)
    gearbox_test = prepare_df(gearbox_test, df_fail_gearbox_test)
    # Label as 1, all 60 days before the failure, for classification
    print("Labeling")
    generator_train['Failure'] = generator_train.apply(lambda x: label(x['TTF'], 60),axis=1)
    generator_test['Failure'] = generator_test.apply(lambda x: label(x['TTF'], 60),axis=1)
    gen_bear_train['Failure'] = gen_bear_train.apply(lambda x: label(x['TTF'], 60),axis=1)
    gen_bear_test['Failure'] = gen_bear_test.apply(lambda x: label(x['TTF'], 60),axis=1)
    transformer_train['Failure'] = transformer_train.apply(lambda x: label(x['TTF'], 60),axis=1)
    transformer_test['Failure'] = transformer_test.apply(lambda x: label(x['TTF'], 60),axis=1)
    hydraulic_train['Failure'] = hydraulic_train.apply(lambda x: label(x['TTF'], 60),axis=1)
    hydraulic_test['Failure'] = hydraulic_test.apply(lambda x: label(x['TTF'], 60),axis=1)
    gearbox_train['Failure'] = gearbox_train.apply(lambda x: label(x['TTF'], 60),axis=1)
    gearbox_test['Failure'] = gearbox_test.apply(lambda x: label(x['TTF'], 60),axis=1)
    print('Generator Model')
    #Drop Columns
    generator_train_ = generator_train.drop(columns = gen_col_drop)
    generator_test_ = generator_test.drop(columns = gen_col_drop)
    # Scale
    X_train_gen, X_test_gen = scale(generator_train_, generator_test_)
    # Define target and Data Augmentation on training data
    y_train_gen = generator_train['Failure']
    y_test_gen = generator_test['Failure']
    X_train_gen, y_train_gen = data_aug(X_train_gen, y_train_gen)
    # Train
    gen_model = Trainer(X_train_gen, y_train_gen, component='Generator')
    gen_model.train()
    joblib.dump(gen_model, 'generator_model.joblib')
    print('Generator Bearing Model')
    # Drop columns
    gen_bear_train_ = gen_bear_train.drop(columns = bear_col_drop)
    gen_bear_test_ = gen_bear_test.drop(columns = bear_col_drop)
    # Scale
    X_train_bear, X_test_bear = scale(gen_bear_train_, gen_bear_test_)
    # Define target and Data Augmentation on training data
    y_train_bear = gen_bear_train['Failure']
    y_test_bear = gen_bear_test['Failure']
    X_train_bear, y_train_bear = data_aug(X_train_bear, y_train_bear)
    # Train
    gen_bear_model = Trainer(X_train_bear, y_train_bear, component='Generator Bearing')
    gen_bear_model.train()
    joblib.dump(gen_model, 'generator_model.joblib')
    print('Transformer Model')
    # Drop columns
    transformer_train_ = transformer_train.drop(columns = transf_col_drop)
    transformer_test_ = transformer_test.drop(columns = transf_col_drop)
    # Scale
    X_train_transf, X_test_transf = scale(transformer_train_, transformer_test_)
    # Define target and Data Augmentation on training data
    y_train_transf = transformer_train['Failure']
    y_test_transf = transformer_test['Failure']
    X_train_transf, y_train_transf = data_aug(X_train_transf, y_train_transf)
    # Train
    transf_model = Trainer(X_train_transf, y_train_transf, component='Transformer')
    transf_model.train()
    joblib.dump(transf_model, 'transformer_model.joblib')
    print("Hydraulic Model")
    # Drop columns
    hydraulic_train_ = hydraulic_train.drop(columns = hyd_col_drop)
    hydraulic_test_ = hydraulic_test.drop(columns = hyd_col_drop)
    # Scale
    X_train_hyd, X_test_hyd = scale(hydraulic_train_, hydraulic_test_)
    # Define target and Data Augmentation on training data
    y_train_hyd = hydraulic_train['Failure']
    y_test_hyd = hydraulic_test['Failure']
    X_train_hyd, y_train_hyd = data_aug(X_train_hyd, y_train_hyd)
    # Train
    hyd_model = Trainer(X_train_hyd, y_train_hyd, component='Hydraulic')
    hyd_model.train()
    joblib.dump(hyd_model, 'hydraulic_model.joblib')
    print("Gearbox Model")
    gearbox_train_ = gearbox_train[gearbox_cols]
    gearbox_test_ = gearbox_test[gearbox_cols]
    # Scale
    X_train_gear, X_test_gear = scale(gearbox_train_, gearbox_test_)
    # Define target and Data Augmentation on training data
    y_train_gear = gearbox_train['Failure']
    y_test_gear = gearbox_test['Failure']
    X_train_gear, y_train_gear = data_aug(X_train_gear, y_train_gear)
    # Train
    gearbox_model = Trainer(X_train_gear, y_train_gear, component='Gearbox')
    gearbox_model.train()
    joblib.dump(gearbox_model, 'gearbox_model.joblib')
    print("Predicting")
    generator_pred = gen_model.predict(X_test_gen)
    gen_bear_pred = gen_bear_model.predict(X_test_bear)
    transf_pred = transf_model.predict(X_test_transf)
    hyd_pred = hyd_model.predict(X_test_hyd)
    gearbox_pred = gearbox_model.predict(X_test_gear)
    print("Savings:")
    generator_costs = threshold(generator_test.reset_index(), generator_pred, 0.6)
    generator_costs = pos_neg(generator_costs)
    generator_savings = mat_cost(generator_costs, 'Generator')
    print(f'Generator Savings: {generator_savings}')
    gen_bear_costs = threshold(gen_bear_test.reset_index(), gen_bear_pred, 0.6)
    gen_bear_costs = pos_neg(gen_bear_costs)
    gen_bear_savings = mat_cost(gen_bear_costs, 'Generator Bearing')
    print(f'Generator Bearing Savings: {gen_bear_savings}')
    transf_costs = threshold(transformer_test.reset_index(), transf_pred, 0.75)
    transf_costs = pos_neg(transf_costs)
    transf_savings = mat_cost(transf_costs, 'Transformer')
    print(f'Transformer Savings: {transf_savings}')
    hyd_costs = threshold(hydraulic_test.reset_index(), hyd_pred, 0.3)
    hyd_costs = pos_neg(hyd_costs)
    hyd_savings = mat_cost(hyd_costs, 'Hydraulic')
    print(f'Hydraulic Savings: {hyd_savings}')
    gear_costs = threshold(gearbox_test.reset_index(), gearbox_pred, 0.52)
    gear_costs = pos_neg(gear_costs)
    gear_savings = mat_cost(gear_costs, 'Gearbox')
    print(f'Gearbox Savings: {gear_savings}')

    
    

