# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:54:54 2017

@author: herma
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def check_ProdID(df_mod_test):
    """
    Find replacement for unseen Product ID in test set
    Use existing PRoduct ID with similar Prod_cat123
    """
    
    # select new ProductID instances from test set
    critnew = df_mod_test.Product_ID == -1
    ProdID_Pcat123_new = df_mod_test.loc[critnew,'Prod_cat123'].drop_duplicates()
    
    # look for similar Product_ID and reasign this ID
    crit = ~critnew & df_mod_test.Prod_cat123.isin(ProdID_Pcat123_new)
    Prod_ID_repl = df_mod_test.loc[crit, :]
    df_map = Prod_ID_repl.groupby( ['Prod_cat123']).Product_ID.apply(lambda x: x.mode())
    df_map = df_map.xs(0, level=1)
    
    # add a new Product_ID column based on Prod_cat123
    df_mod_test.insert(df_mod_test.columns.get_loc('Product_ID')+1,\
                  'New_Product_ID', df_mod_test.Prod_cat123)
    # create dictionary for replacement operation (replace Prod_cat123 by new Product_ID)
    rpl_dict = { 'New_Product_ID' : df_map.to_dict()}
    df_mod_test.loc[ critnew , :] = df_mod_test.loc[ critnew , :].replace(rpl_dict)
    # update Product_ID columns from New_Product_ID
    df_mod_test.loc[ critnew , 'Product_ID'] = df_mod_test.loc[ critnew , 'New_Product_ID']
    # drop temporary column New_Product_ID
    df_mod_test.drop('New_Product_ID', axis = 1, inplace = True)
    
    return df_mod_test



def add_Combination(df_mod):
    """
        When using the 'Gender_Prod_cat123' encoder on the test set, it fails.
        There are unseen combination in the training where encoder is made.
        Thus I'm adding those missing combination based on existing similar instances.
        This is what this function does.
        
        Missing combinations:
            'M-10-11.0-nan'
            'F-9-nan-nan'
            'F-5-10.0-16.0'
    """
    # make all possible combination of Gender-Prodcat123 (Pandas, Numpy)
    Prodcat123_u = df_mod.Prod_cat123.unique()
    Gender_u = df_mod.Gender.unique()
    # repeat for one and tile for the other
    Prodcat123_list = np.tile(Prodcat123_u, Gender_u.shape[0])
    Gender_list = Gender_u.repeat(Prodcat123_u.shape[0])
    Combi_list = Gender_list + '-' + Prodcat123_list
    
    # find out missing combinations (use series/dataframe and values, keep Prodcat123 category)
    s_encoder_combi = pd.Series(df_mod.Gender_Prod_cat123.unique(), name = 'Incomplete')
    df_combi = pd.DataFrame(np.stack((Combi_list,Prodcat123_list), axis=1) \
                            , columns = ['Complete', 'Prodcat123_list'])
    df_merged = pd.concat([s_encoder_combi, df_combi], axis=1, join='outer')
    df_missing_combi = df_merged.loc[~df_merged.Complete.isin(df_merged.Incomplete),\
                                  ['Prodcat123_list', 'Complete']]
    
    # find replacement for missing instance based on Prod_cat123
    df_miss = df_mod[df_mod.Prod_cat123.isin(df_missing_combi.Prodcat123_list)]
    df_mode = df_miss.groupby(['Prod_cat123']).apply(lambda x: x.mode())
    
    # 3rd solution (second level is dropped by default)
    df_mode_u = df_mode.xs(0, level=1)
    # Replace Gender_Prod_cat123 column to missing values
    df_mode_u = pd.merge(df_mode_u, df_missing_combi, how='inner', \
             left_on=['Prod_cat123'], right_on=['Prodcat123_list'])
    df_mode_u.drop(['Prodcat123_list','Gender_Prod_cat123'], axis = 1, inplace = True)
    df_mode_u.rename(columns={'Complete': 'Gender_Prod_cat123'}, inplace = True)
    # keep dataframe in same order before concatenation
    df_mode_u = df_mode_u.reindex(columns=df_mod.columns)
    
    # add replacement for missing instances to our training set
    df_mod = pd.concat( [df_mod, df_mode_u], axis = 0 , ignore_index=True)
    
    return df_mod


def extract_DataFrame(df, addExtraRow = False):
    """
        return a dataframe modified per criteria found during data exploration
    """
    # start with a dataframe copy, ((optional)
    df_cpy = df.copy()
    
    # combine Prod_cat123 and add it to dataframe
    temp_prod123 = df.Product_Category_1.astype(str) + '-' \
                    + df.Product_Category_2.astype(str) + '-' \
                    + df.Product_Category_3.astype(str)
    # add Prod_cat123 column after Product_Category_3 (hence +1 below)
    df_cpy.insert(df_cpy.columns.get_loc('Product_Category_3')+1,\
                  'Prod_cat123', temp_prod123)
    # remove prod cat 1,2 and 3 from dataframe
    df_cpy.drop(\
    labels = ['Product_Category_1','Product_Category_2','Product_Category_3'],\
    inplace=True, axis = 1)
    
    # even out age first (see Introduction and Data Exploration notebook)
    df_cpy.Age.replace(['0-17', '18-25'], '0-25', inplace = True)
    df_cpy.Age.replace(['46-50', '51-55', '55+'], '46+', inplace = True)

    # combine age (string) and marital status (integer converted to string)
    temp_age_mar = df_cpy.Marital_Status.astype(str) + '-' + df_cpy.Age
    df_cpy.insert(df_cpy.columns.get_loc('Marital_Status')+1,\
                  'Marital_Status_Age', temp_age_mar)
    df_cpy.drop(labels = ['Age','Marital_Status'], inplace=True, axis = 1)

    # Combine Prod_cat123 and Gender. For Prod_cat123, it has to be converted
    # back to string via category
    temp_cpy_G_cat = df_cpy.Gender + '-' + df_cpy.Prod_cat123
    # add new column after Prod_cat123 column
    df_cpy.insert(df_cpy.columns.get_loc('Prod_cat123')+1,\
                  'Gender_Prod_cat123', temp_cpy_G_cat)

    # even out Stay_In_Current_City_Years
    df_cpy.Stay_In_Current_City_Years.replace(['0', '1'], '0-1', inplace = True)
    df_cpy.Stay_In_Current_City_Years.replace(['2', '3', '4+'], '2+', inplace = True)
    
    # add missing combination for Gender_Prod_cat123 (only for training set)
    if addExtraRow:
        df_cpy = add_Combination(df_cpy)
    
    # return modified Dataframe
    return df_cpy

def prepare_Data(df, encoders = None ):
    """
        returns One-hot encoded features
        also returns sklearn encoders if encoders are not specified
        
        encoders is a list or tuple where the first element are the categories
        and the second element are one-hot encoders
    """
    # feature extraction
    # Must be training set if encoders is NOT specified
    if encoders is None:
        df_cpy = extract_DataFrame(df, addExtraRow = True)
        # convert non-numeric columns to numbers
        category_dict = {}
        col2encode = df_cpy.select_dtypes(['object']).columns
        for col in col2encode:
            # convert to category
            temp_col = df_cpy[col].astype('category')
            category_dict[col] = temp_col.cat.categories
            df_cpy[col] = temp_col.cat.codes    
        
        # encode using sklearn
        X_features = []
        encoder_dict = {}
        feature_names = df_cpy.columns.tolist()
        if 'Purchase' in feature_names:
            feature_names.remove('Purchase')
        for f in feature_names:
            # OneHotEncoder() outputs sparse matrix by default
            this_encoder = OneHotEncoder()
            # populate X_features with sparse matrix and associated name as tuple
            X_features.append( \
            (this_encoder.fit_transform(df_cpy.loc[:,f].values.reshape(-1,1)), f) )
            # keep encoder
            encoder_dict[f] = this_encoder
            
        # return one-hot encoded features
        return X_features, category_dict, encoder_dict
        
    # feature extraction for test set
    else:
        df_cpy = extract_DataFrame(df)
        # convert non-numeric columns to numbers
        category_dict = encoders[0]
        col2encode = df_cpy.select_dtypes(['object']).columns
        for col in col2encode:
            # convert to category
            df_cpy[col] = df_cpy[col].astype('category',\
                  categories = category_dict[col]).cat.codes 
        # check for new value coded as -1
        df_cpy = check_ProdID(df_cpy)
        
        X_features = []
        encoder_dict = encoders[1]
        feature_names = df_cpy.columns.tolist()
        for f in feature_names:
            # populate X_features with sparse matrix and associated name as tuple
#            print('one hot on {}'.format(f))
            X_features.append( \
            (encoder_dict[f].transform(df_cpy.loc[:,f].values.reshape(-1,1)), f) )
            
        # return one-hot encoded features
        return X_features
    
#### The Main program, can be used as a script or as a module
if __name__ == "__main__":

    filename = './../train_oSwQCTC/train.csv'
    df = pd.read_csv(filename)
    # test feature preparation
    features,catcoders,encoders = prepare_Data(df)
    imp_feature = ['User_ID', 'Product_ID', 'Gender_Prod_cat123']
    X_features = tuple(f[0] for f in features if f[1] in imp_feature)
    from scipy.sparse import hstack
    X = hstack( X_features )
    # test dataframe modification
#    df_mod = extract_DataFrame(df, True)


    # test data preparation on test set
    filename = './../test_HujdGe7/test.csv'
    df_test = pd.read_csv(filename)
    # Test with encoders on files
#    import pickle
#    encoders = pickle.load( open( "../Onehotencoders.pkl", "rb" ) )
#    catcoders = pickle.load( open( "../Category_encoders.pkl", "rb" ) )
    features_test = prepare_Data(df_test, (catcoders, encoders))
    X_features_test = tuple(f[0] for f in features_test if f[1] in imp_feature)
    from scipy.sparse import hstack
    X_test = hstack( X_features_test )
    # test dataframe modification
#    df_mod_test = extract_DataFrame(df_test)
    
    

#    col2encode = df_mod_test.select_dtypes(['object']).columns
#    for col in col2encode:
#        # convert to category
#        df_mod_test[col] = df_mod_test[col].astype('category',\
#              categories = catcoders[col]).cat.codes 

#    # compare Product_ID from training ans test set
#    d = pd.concat([pd.Series(df_mod.Product_ID.unique()) \
#       , pd.Series(df_mod_test.Product_ID.unique())], axis=1)
#    # Test set Product_ID NOT in training set
#    d_test_new = d.loc[~d.iloc[:,1].isin(d.iloc[:,0]), 1 ].dropna()
#    # Training set Product_ID not in test set
##    d_train_new = d.loc[~d.iloc[:,0].isin(d.iloc[:,1]), 0 ]

    
    
    