import pandas as pd
import numpy as np

rooth_path = '../data/training/'


def get_data():
    failures_df = pd.read_csv(rooth_path + 'wind-farm-1-failures-training.csv')
    signals_df = pd.read_csv(rooth_path + 'wind-farm-1-signals-training.csv', delimiter=';')
    df_costs = pd.read_csv(rooth_path + 'HTW_Costs.csv')
    return signals_df, failures_df, df_costs


def clean_data(signals_df, failures_df):
    failures_df['Timestamp'] = pd.to_datetime(failures_df['Timestamp'])
    signals_df['Timestamp'] = pd.to_datetime(signals_df['Timestamp'])

    # Group by Turbine_ID and Timestamp
    signals_df = signals_df.groupby(['Turbine_ID', 'Timestamp']).mean().reset_index()

    # Fill null values by interpolation within each turbine
    turbines = ['T01', 'T06', 'T07', 'T09', 'T11']
    df_ = pd.DataFrame(columns=signals_df.columns)
    for turbine in turbines:
        temp = signals_df[signals_df['Turbine_ID']==turbine]
        temp = temp.interpolate(method='bfill', axis=0)
        df_ = pd.concat([df_, temp])
    signals_df = df_.copy()

    # Since we have very granular records, we will group by day the SCADA signals
    signals_df['Date'] = signals_df['Timestamp'].dt.date
    failures_df['Date'] = failures_df['Timestamp'].dt.date
    signals_df = signals_df.groupby(by=['Turbine_ID','Date']).mean().reset_index()

    # Remove the 'Remarks' and 'Timestamp' columns from failures_df
    failures_df = failures_df.drop(columns=['Remarks', 'Timestamp'])

    # Drop Electric Circuit Failure, since this failures are unpredictable
    failures_df = failures_df.drop(0) # index 0 for the short-circuit failures

    # Dummies for the Component Failures
    failures_df = pd.get_dummies(failures_df, columns=['Component'])

    return signals_df, failures_df


# Function to find str in columns of df
def component(component, col):
    
    '''Find a string of the component in features columns
    
    Args:
            component (str)     : The chosen component 
            col (list)           : list of columns to look in
        
    Returns:
            list: list with the columns of the component chosen'''
    
    pair_comp_col=[]
    for i in col:
        if component in i:
            pair_comp_col.append(i)
            
    return pair_comp_col


# Function to choose the right features of each component
def choose_features(df):
    
    '''Choose the correct input features of each component 
    
    Args:
            df (dataframe)     : input dataframe to choose the correct input features
        
    Returns:
            df (dataframe)     : a dataframe with the correct features for each component'''
    
    
    # Filter the important features based on the given PDF
    important_signals = ['Turbine_ID', 'Date', 'Gen_RPM_Max', 'Gen_RPM_Min', 'Gen_RPM_Avg', 'Gen_RPM_Std', 
                         'Gen_Bear_Temp_Avg', 'Gen_Phase1_Temp_Avg', 'Gen_Phase2_Temp_Avg', 'Gen_Phase3_Temp_Avg', 
                         'Hyd_Oil_Temp_Avg','Gear_Oil_Temp_Avg', 'Gear_Bear_Temp_Avg', 'Nac_Temp_Avg', 'Amb_WindSpeed_Max', 
                         'Amb_WindSpeed_Min', 'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Std', 'Amb_Temp_Avg',
                         'Prod_LatestAvg_TotActPwr', 'HVTrafo_Phase1_Temp_Avg', 'HVTrafo_Phase2_Temp_Avg',
                         'HVTrafo_Phase3_Temp_Avg', 'Grd_InverterPhase1_Temp_Avg', 'Cont_Top_Temp_Avg', 'Cont_Hub_Temp_Avg', 
                         'Cont_VCP_Temp_Avg', 'Gen_SlipRing_Temp_Avg', 'Spin_Temp_Avg', 'Blds_PitchAngle_Max', 
                         'Blds_PitchAngle_Min', 'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Std',
                         'Grd_Busbar_Temp_Avg', 'Gen_Bear2_Temp_Avg', 'Nac_Direction_Avg']
    
    df_ = df[important_signals]
    
    # Since we don't have a very deep knowledge on the domain, we will choose the all the "non related features" of each component
    time_id = ['Turbine_ID', 'Date']
    pair_gen = component('Gen', df_.columns) # Generator
    pair_hyd = component('Hyd', df_.columns) # Hydraulic
    pair_transf = component('Trafo', df_.columns) # Transformer
    pair_gear = component('Gear', df_.columns) # Gearbox
    pair_amb = component('Amb', df_.columns) # Ambient
    pair_blds = component('Blds', df_.columns) # Blades
    pair_cont = component('Cont', df_.columns) # Controler
    pair_nac = component('Nac', df_.columns) # Nacelle
    pair_spin = component('Spin', df_.columns) # Spin
    pair_prod = component('Prod', df_.columns) # Production
    pair_grid = component('Grd', df_.columns) # Grid
    
    #Create DF for each component
    df_generator = df[time_id + pair_gen + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_prod + pair_grid]
    df_gen_bear = df[time_id + pair_gen + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_prod + pair_grid]
    df_transformer = df[time_id + pair_transf + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_prod + pair_grid] 
    df_hydraulic = df[time_id + pair_hyd + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_prod + pair_grid] 
    df_gearbox = df[time_id + pair_gear + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_prod + pair_grid]
    
    return df_generator, df_gen_bear, df_transformer, df_hydraulic, df_gearbox



# Function to do the df of component failure
def choose_failure(df):
    
    '''Compute the Dataframe of failures for each component 
    
    Args:
            df (dataframe)     : input dataframe to choose the correct input failures
        
    Returns:
            df (dataframe)     : a dataframe with the correct failures for each component'''
    
    
    # Rename de column Generator Bearing to avoid mistakes with the column Generator
    df.rename(columns={"Component_GENERATOR_BEARING": "Component_BEARING"}, inplace=True)
    
    time_id = ['Turbine_ID', 'Date']
    gen = component('GENERATOR', df.columns)
    gen_bear = component('BEARING', df.columns)
    transf = component('TRANSFORMER', df.columns)
    hyd = component('HYDRAULIC', df.columns)
    gearbox = component('GEARBOX', df.columns)
    
    #Create DF for each component failure
    df_fail_gen = df[time_id + gen]
    df_fail_gen_bear = df[time_id + gen_bear]
    df_fail_transf = df[time_id + transf] 
    df_fail_hyd = df[time_id + hyd] 
    df_fail_gearbox = df[time_id + gearbox]
    
    return df_fail_gen, df_fail_gen_bear, df_fail_transf, df_fail_hyd, df_fail_gearbox


# Fucntion for split without shuffling
def split_df(df, component='other'):
    
    '''Train Test Split
    
    Args:
            df (dataframe)     : input dataframe 
        
    Returns:
            df (dataframe)     : a train dataframe and test dataframe'''
    
    
    # DateTime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # The gearbox failures occur only on 2016. To train and test a model for the gearbox we will do a split on 2019-09-01 for
    # this component. We will also use the longer period for the train
    if component == 'gearbox':
        # period to split
        split = '2016-09-01'
        
        df_train = df[df['Date'] >= split] #longer period for the train
        df_test = df[df['Date'] < split]
    
    else: # other component
        # period to split 
        split = '2017-01-01'

        # Train and Test split
        df_train = df[df['Date'] < split]
        df_test = df[df['Date'] >= split]
    
    return df_train, df_test


# Function for Rolling Aggregates
def add_features(df, rolling_win_size):
    
    ''' Do rolling aggregates
    
        Args:
            df (dataframe)          : input dataframe 
            rolling_win_size(int)   : window size, number of days to apply rolling aggregates
        
        Returns:
            df (dataframe)          : dataframe with the rolling means and std of each feature'''
    

    cols = df.columns[2:] #Features columns excluding turbine id and date
    
    rol_av_cols = [nm +'_av' for nm in cols] # Cols for Rolling averages
    rol_sd_cols = [nm +'_sd' for nm in cols] # Cols for Rolling std
#     rol_md_cols = [nm +'_md' for nm in cols] # Cols for Rolling median
#     rol_mn_cols = [nm +'_mn' for nm in cols] # Cols for Rolling minimum
#     rol_mx_cols = [nm +'_mx' for nm in cols] # Cols for Rolling maximum
    
    # New empty dataframe
    df_out = pd.DataFrame()
    
    ws = rolling_win_size
    
    #calculate rolling stats for each turbine. we don't want that the roll_av of one turbine have values from the previous turbine
    
    for n_id in pd.unique(df.Turbine_ID):
    
        # get a subset for each turbine sensors
        df_turbine = df[df['Turbine_ID'] == n_id]
        df_sub = df_turbine[cols]

        # get rolling mean for the subset
        rol_av = df_sub.rolling(ws, min_periods=1).mean()
        rol_av.columns = rol_av_cols

        # get the rolling standard deviation for the subset
        rol_sd = df_sub.rolling(ws, min_periods=1).std().fillna(method='bfill') # To not have 0 for the first Turbine occurance
        rol_sd.columns = rol_sd_cols
        
#         # get the rolling median for the subset
#         rol_md = df_sub.rolling(ws, min_periods=1).median() 
#         rol_md.columns = rol_md_cols
        
#         # get the rolling minimum for the subset
#         rol_mn = df_sub.rolling(ws, min_periods=1).min() 
#         rol_mn.columns = rol_mn_cols
        
#         # get the rolling standard deviation for the subset
#         rol_mx = df_sub.rolling(ws, min_periods=1).max()
#         rol_mx.columns = rol_mx_cols
        
    
        # combine the two new subset dataframes columns to the turbine subset
        new_ftrs = pd.concat([df_turbine, rol_av, rol_sd], axis=1) #...rol_md, rol_mn, rol_mx
    
        # add the new features rows to the output dataframe
        df_out = pd.concat([df_out, new_ftrs])
                
    return df_out

# Function to prepare train and test df
def prepare_df(df1, df2):
    
    '''Prepare Train and Test Dataframes for each component and compute TTF
    
       Args:
            df1 (dataframe)     : input dataframe to be merged with the failures
            df2 (dataframe)     : failures dataframe
        
        Returns:
            df (dataframe)      : final dataframe merged, with TTF computed'''
    
    
    # Merge df signals with df failures
    df = df1.merge(df2, how='left', on=['Turbine_ID', 'Date'])
    
    # Fill NaN with 0
    df.fillna(0, inplace=True)
    
    # New empty dataframe
    df_ = pd.DataFrame()
    
    turbines_list = list(df['Turbine_ID'].unique())
    
    # For each turbine
    for turbine in turbines_list:
        df1 = df[df['Turbine_ID']==turbine]
        
        # Retrieve the column with the component failure
        for col in df.columns:
            if 'Component' in col:
                name_col = col
                
                # 'aux_date' as an anchor to compute TTF
                if df1[name_col].nunique()>1:
                    index = df1[df1[name_col]==1]
                    index['aux_date'] = index['Date']
                    index = index[['aux_date','Date', 'Turbine_ID']]
                    df_merged = df1.merge(index, how='left', on=['Turbine_ID','Date'])
                    df_merged = df_merged.fillna(method='bfill')

                    #If there is not a failure after, hold present date
                    df_merged['aux_date'] = df_merged['aux_date'].fillna(df_merged['Date'])
                    
                    # Compute TTF (Time to Failure)
                    df_merged['TTF'] = round((df_merged['aux_date'] - df_merged['Date']) / np.timedelta64(1, 'D'),0)
                else:
                    df_merged = df1
                    df_merged['aux_date'] = df_merged['Date']
                    df_merged['TTF'] = 0 

                #Drop Column aux_date
                df_final = df_merged.drop(columns='aux_date')

                df_ = pd.concat([df_, df_final])
    
    # Drop column with str "Component" since we won't need it
    for col in df_.columns:
            if 'Component' in col:
                df_.drop(columns=[col], inplace=True)

    return df_


# Labeling
def label(days, period):
    
    '''Labeling for binary classification (target)'''
    
    if 2 <= days <= period:
        Flag=1
    else:
        Flag=0
    return Flag
