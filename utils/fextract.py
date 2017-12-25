# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:54:54 2017

@author: herma
"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def extract_DataFrame(df):
    """
        return a dataframe modified per criteria found during data exploration
    """
    # start with a dataframe copy
    df_cpy = df.copy()
    
    # combine Prod cat123 and add it to dataframe
    temp_prod123 = df.Product_Category_1.astype(str) + '-' \
                    + df.Product_Category_2.astype(str) + '-' \
                    + df.Product_Category_3.astype(str)
    df_cpy.insert(len(df_cpy.columns)-1, 'Prod_cat123', temp_prod123)
    # remove prod cat 1,2 and 3 from dataframe
    df_cpy.drop(\
    labels = ['Product_Category_1','Product_Category_2','Product_Category_3'],\
    inplace=True, axis = 1)
    
    # even out age first (see Introduction and Data Exploration notebook)
    df_cpy.Age.replace(['0-17', '18-25'], '0-25', inplace = True)
    df_cpy.Age.replace(['46-50', '51-55', '55+'], '46+', inplace = True)

    # combine age (string) and marital status (integer converted to string)
    temp_age_mar = df_cpy.Marital_Status.astype(str) + '-' + df_cpy.Age
    df_cpy.insert(df_cpy.columns.get_loc('Age'), 'Marital_Status_Age', temp_age_mar)
    df_cpy.drop(labels = ['Age','Marital_Status'], inplace=True, axis = 1)

    # Combine Prod_cat123 and Gender. For Prod_cat123, it has to be converted
    # back to string via category
    temp_cpy_G_cat = df_cpy.Gender + '-' + df_cpy.Prod_cat123
    df_cpy.insert(len(df_cpy.columns)-1, 'Gender_Prod_cat123', temp_cpy_G_cat)

    # even out Stay_In_Current_City_Years
    df_cpy.Stay_In_Current_City_Years.replace(['0', '1'], '0-1', inplace = True)
    df_cpy.Stay_In_Current_City_Years.replace(['2', '3', '4+'], '2+', inplace = True)
    
    # return modified Dataframe
    return df_cpy

def prepare_Data(df, encoders = None):
    """
        returns One-hot encoded features
        also returns sklearn encoders if encoders are not specified
    """
    # wrangle input dataframe
    df_cpy = extract_DataFrame(df)
    
    col2encode = df_cpy.select_dtypes(['object']).columns
    # convert non-numeric columns to numbers
    for col in col2encode:
        # convert to category
        df_cpy[col] = df_cpy[col].astype('category').cat.codes
    
    # encode using sklearn
    X_features = []
    
    feature_names = df_cpy.columns.tolist()[:-1]
    if encoders is not None:
        encoder_list = encoders
        for en,f in enumerate(feature_names):
#            encoder_list[en].set_params(handle_unknown = 'ignore')
            # populate X_features with sparse matrix and associated name as tuple
            X_features.append( \
            (encoder_list[en].transform(df_cpy.loc[:,f].values.reshape(-1,1)), f) )
            
        # return one-hot encoded features
        return X_features
 
    else:
        encoder_list = []
        for f in feature_names:
            # OneHotEncoder() outputs sparse matrix by default
            this_encoder = OneHotEncoder()
            # populate X_features with sparse matrix and associated name as tuple
            X_features.append( \
            (this_encoder.fit_transform(df_cpy.loc[:,f].values.reshape(-1,1)), f) )
            # keep encoder
            encoder_list.append(this_encoder)
            
        # return one-hot encoded features
        return X_features, encoder_list

    
    
    
#### The Main program, can be used as a script or as a module
if __name__ == "__main__":

    filename = './../train_oSwQCTC/train.csv'
    df = pd.read_csv(filename)
    features, _ = prepare_Data(df)
    df_mod = extract_DataFrame(df)

    imp_feature = ['User_ID', 'Product_ID', 'Gender_Prod_cat123']
    X_features = tuple(f[0] for f in features if f[1] in imp_feature)
    from scipy.sparse import hstack
    X = hstack( X_features )



    # make all possible combination of Gender-Prodcat123 (Python)
    Prodcat123_list = df_mod.Prod_cat123.value_counts().index.tolist()
    Gender_list = df_mod.Gender.value_counts().index.tolist()
    Combi_list = [g+'-'+p for p in Prodcat123_list for g in Gender_list]
    # find out missing combinations
    encoder_combi = df_mod.Gender_Prod_cat123.value_counts().index.tolist()
    Missing_combi = [k for k in Combi_list if k not in encoder_combi]    
    
    # make all possible combination of Gender-Prodcat123 (Pandas, Numpy)
    import numpy as np
    Prodcat123_u = df_mod.Prod_cat123.unique()
    Gender_u = df_mod.Gender.unique()
    # repeat for one and tile for the other
    Prodcat123_list = np.tile(Prodcat123_u, Gender_u.shape[0])
    Gender_list = Gender_u.repeat(Prodcat123_u.shape[0])
    Combi_list = Gender_list + '-' + Prodcat123_list
    # find out missing combinations
    encoder_combi = df_mod.Gender_Prod_cat123.unique()
    df_combi = pd.DataFrame(Combi_list, columns = ['Complete'])
    df_combi['Incomplete'] = np.concatenate(\
            (encoder_combi, \
             np.array(np.nan).repeat(Combi_list.shape[0] - encoder_combi.shape[0]) ) )
    Missing_combi = df_combi.loc[~df_combi.Complete.isin(df_combi.Incomplete), 'Complete'].values
    
    
    encoder_combi = df_mod.Gender_Prod_cat123.value_counts()
    s_combi = pd.Series(Combi_list, index = Combi_list, columns = ['Complete'])
    
    # add the 3 missing instances


