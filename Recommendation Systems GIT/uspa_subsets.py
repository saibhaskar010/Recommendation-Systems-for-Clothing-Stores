#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import os
pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',200)
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
import datetime
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings("ignore")

def create_subset_dataframe(source_dataframe, column_list):
    ordered_columns = []
    for column in list(source_dataframe.columns):
        if column in column_list:
            ordered_columns.append(column)
    subset_dataframe = source_dataframe[ordered_columns]
    return subset_dataframe

def read_sql_table(table_name): 
    db = pymysql.connect(host='localhost', user="####",passwd="######" ) 
    cur = db.cursor()  
    sql="SELECT * FROM {} ".format(table_name)
    in_data = pd.read_sql(sql,db)
    return in_data

uspa_data = read_sql_table("recoms.uspa_transactions")
inventory_data=read_sql_table("recoms.uspa_inventory")

uspa_columns_considered = ['first_name','last_name','mobile','notes','bill_date','Bill_time','bill_date_time','quantity','reason','year','dim_membership_id','brand','branddescription','subbrand',
                     'subbranddescription','categorygroup','seasonyear','merchandiserange','merchandiserangedesciption',
                     'sleeve','sleevedesc','finish','fitdescription','fabricdescription','first_name','description',
                      'variantdescription','sizedfromtimestampescription','seasondescription',
                      'fit','mrp','gender','fabricpattern1','fabricpattern2','barcode','line_item_discount','total_quantity',
                      'bill_amount','bill_discount','initial_line_item_value','line_item_amount','line_item_discount','materialtypedescription','eandescription']

inventory_columns_considered = ['eandescription','mrp']
uspa_subset = create_subset_dataframe(uspa_data,uspa_columns_considered)
inventory_subset = create_subset_dataframe(inventory_data,inventory_columns_considered)
inventory_subset=inventory_subset.loc[inventory_subset['mrp']>=100]
#inventory_subset.drop(['mrp'],axis=1,inplace=True)

def df_to_sql_table(uspa_subset,inventory_subset):
    engine = create_engine('mysql+pymysql://###:###@localhost/recoms')
    Session = sessionmaker(bind=engine)
    session = Session()
    session.execute('''TRUNCATE TABLE uspa_subset''')
    session.execute('''TRUNCATE TABLE inventory_subset''')
    session.commit()
    session.close()
    uspa_subset.to_sql("uspa_subset",con=engine,if_exists='append', index=False)
    inventory_subset.to_sql("inventory_subset",con=engine,if_exists='append', index=False)
                            
df_to_sql_table(uspa_subset,inventory_subset)



