#!/usr/bin/env python3

#libraries
import pandas as pd
import numpy as np
import re
import os
pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',200)
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

def read_sql_table(table_name): 
    db = pymysql.connect(host='localhost', user="###",passwd="####" ) 
    cur = db.cursor() 
    sql="SELECT * FROM {} ".format(table_name)
    in_data = pd.read_sql(sql,db)
    return in_data

uspa_subset = read_sql_table("recoms.uspa_subset")
uspa_subset['mobile'] = uspa_subset['mobile'].astype(str)
billing_data = uspa_subset[['first_name','mobile','bill_date','Bill_time','bill_date_time','bill_amount','bill_discount','total_quantity']]
billing_data['key'] = billing_data['mobile'] + ' ' + billing_data['bill_date'].astype(str)
billing_data.drop_duplicates('key',inplace=True)
billing_avg = billing_data.groupby(['mobile']).agg({'bill_amount':'mean','bill_discount':'mean',
                                                    'total_quantity':'mean','key':'count',
                                                    'bill_date':'max'}).reset_index()
billing_avg_temp = billing_data.groupby(['mobile']).agg({'bill_date':'min'}).reset_index()
billing_avg.rename(columns = {'bill_amount':'average_bill_amount','bill_discount':'average_bill_discount',
                             'total_quantity':'quantities_per_visit','key':'visit_count','bill_date':'last_visit_date'},inplace=True)

def visit_freq_bins(visit_counts,frequency=45):
    visit_count_bin = visit_counts//frequency
    if (visit_counts > 359):
        formatted_bin = '> 1 year'
    else:
        formatted_bin = str(visit_count_bin * frequency) + "-"  + str((visit_count_bin +1) * frequency)
    return formatted_bin 

billing_avg_temp.rename(columns={'bill_date':'first_visit_date'},inplace=True)
billing_avg = pd.merge(billing_avg,billing_avg_temp,left_on = 'mobile',right_on = 'mobile',how='left')
billing_avg['last_visit_date'] = pd.to_datetime(billing_avg['last_visit_date'])
billing_avg['first_visit_date'] = pd.to_datetime(billing_avg['first_visit_date'])
billing_avg['total_days'] = (billing_avg['last_visit_date'] - billing_avg['first_visit_date']).dt.days
billing_avg['average_visit_days'] = billing_avg['total_days'] / billing_avg['visit_count']
billing_avg['average_visit_days_round'] = billing_avg['average_visit_days'].apply(np.ceil).astype(int)
temp = billing_avg['average_visit_days'].apply(lambda x:pd.Timedelta(x,unit='D'))
billing_avg['proj_next_visit'] = billing_avg['last_visit_date'] + temp
billing_avg['proj_next_visit'] = billing_avg['proj_next_visit'].dt.date
billing_avg['visit_count_bin'] = billing_avg['average_visit_days_round'].apply(visit_freq_bins)

uspa_color = uspa_subset[['mobile','variantdescription']]
uspa_color['key'] = uspa_color['mobile'] + ' ' + uspa_color['variantdescription']
uspa_color['key'] = uspa_color['key'].apply(lambda x:str(x).lower())


uspa_preferred_color = uspa_color['key'].value_counts().to_frame().reset_index().rename(columns={'index':'mob_col','key':'count'})
uspa_preferred_color = uspa_preferred_color[uspa_preferred_color['mob_col'] != 'nan']
uspa_preferred_color['mobile'] = uspa_preferred_color['mob_col'].apply(lambda x:x.split(' ')[0])
uspa_preferred_color['color'] = uspa_preferred_color['mob_col'].apply(lambda x:x.split(' ')[1])
uspa_preferred_color_temp = uspa_preferred_color.groupby(['mobile']).agg({'count':'max'}).reset_index()
uspa_preferred_color_temp['key_1'] = uspa_preferred_color_temp['mobile'].astype(str) + ' ' + uspa_preferred_color_temp['count'].astype(str)
uspa_preferred_color['key_1'] = uspa_preferred_color['mobile'].astype(str) + ' ' + uspa_preferred_color['count'].astype(str) 
uspa_color_final_1 = pd.merge(uspa_preferred_color,uspa_preferred_color_temp, left_on='key_1',right_on ='key_1',how='left')
uspa_color_final_1 = uspa_color_final_1.drop_duplicates(['mobile_x'])
uspa_color_dataframe = uspa_color_final_1[['mobile_x','color']]

billing_avg = pd.merge(billing_avg,uspa_color_dataframe, left_on = 'mobile',right_on='mobile_x',how = 'left')
billing_avg.drop(['mobile_x'],axis=1,inplace=True)

uspa_prod_type = uspa_subset[['mobile','materialtypedescription']]
uspa_prod_type['key'] = uspa_prod_type['mobile'] + ' ' + uspa_prod_type['materialtypedescription']
uspa_prod_type['key'] = uspa_prod_type['key'].apply(lambda x:str(x).lower())
uspa_preferred_prod = uspa_prod_type['key'].value_counts().to_frame().reset_index().rename(columns={'index':'mob_prod','key':'count'})
uspa_preferred_prod = uspa_preferred_prod[uspa_preferred_prod['mob_prod'] != 'nan']
uspa_preferred_prod['mobile'] = uspa_preferred_prod['mob_prod'].apply(lambda x:x.split(' ')[0])
uspa_preferred_prod['prod'] = uspa_preferred_prod['mob_prod'].apply(lambda x:x.split(' ')[1])
uspa_preferred_prod_temp = uspa_preferred_prod.groupby(['mobile']).agg({'count':'max'}).reset_index()
uspa_preferred_prod_temp['key_1'] = uspa_preferred_prod_temp['mobile'].astype(str) + ' ' + uspa_preferred_prod_temp['count'].astype(str)
uspa_preferred_prod['key_1'] = uspa_preferred_prod['mobile'].astype(str) + ' ' + uspa_preferred_color['count'].astype(str) 
uspa_prod_final_1 = pd.merge(uspa_preferred_prod,uspa_preferred_prod_temp, left_on='key_1',right_on ='key_1',how='left')
uspa_prod_final_1 = uspa_prod_final_1.drop_duplicates(['mobile_x'])
uspa_prod_dataframe = uspa_prod_final_1[['mobile_x','prod']]

billing_avg = pd.merge(billing_avg,uspa_prod_dataframe, left_on = 'mobile',right_on='mobile_x',how = 'left')
billing_avg.drop(['mobile_x'],axis=1,inplace=True)

uspa_prod_type = uspa_subset[['mobile','gender']]
uspa_prod_type['key'] = uspa_prod_type['mobile'] + ' ' + uspa_prod_type['gender']
uspa_prod_type['key'] = uspa_prod_type['key'].apply(lambda x:str(x).lower())
uspa_preferred_prod = uspa_prod_type['key'].value_counts().to_frame().reset_index().rename(columns={'index':'mob_prod','key':'count'})
uspa_preferred_prod = uspa_preferred_prod[uspa_preferred_prod['mob_prod'] != 'nan']
uspa_preferred_prod['mobile'] = uspa_preferred_prod['mob_prod'].apply(lambda x:x.split(' ')[0])
uspa_preferred_prod['gender'] = uspa_preferred_prod['mob_prod'].apply(lambda x:x.split(' ')[1])
uspa_preferred_prod_temp = uspa_preferred_prod.groupby(['mobile']).agg({'count':'max'}).reset_index()
uspa_preferred_prod_temp['key_1'] = uspa_preferred_prod_temp['mobile'].astype(str) + ' ' + uspa_preferred_prod_temp['count'].astype(str)
uspa_preferred_prod['key_1'] = uspa_preferred_prod['mobile'].astype(str) + ' ' + uspa_preferred_color['count'].astype(str) 
uspa_prod_final_1 = pd.merge(uspa_preferred_prod,uspa_preferred_prod_temp, left_on='key_1',right_on ='key_1',how='left')
uspa_prod_final_1 = uspa_prod_final_1.drop_duplicates(['mobile_x'])
uspa_prod_dataframe = uspa_prod_final_1[['mobile_x','gender']]
billing_avg = pd.merge(billing_avg,uspa_prod_dataframe, left_on = 'mobile',right_on='mobile_x',how = 'left')
billing_avg.drop(['mobile_x'],axis=1,inplace=True)

#find out the basket value in each bins
#add RFM 
#add item details
#figure out if the customer is a discount seeker
#identify how many times the customer has done high value purchases
#gather loyality points
#motivate customer to upgrade the loyality club
#confirm if the customer purchases for himself or other (NLP - name/gender prediction)


uspa_prod_type = uspa_subset[['mobile','sizedescription']]
uspa_prod_type['key'] = uspa_prod_type['mobile'] + ' ' + uspa_prod_type['sizedescription']
uspa_prod_type['key'] = uspa_prod_type['key'].apply(lambda x:str(x).lower())
uspa_preferred_prod = uspa_prod_type['key'].value_counts().to_frame().reset_index().rename(columns={'index':'mob_prod','key':'count'})
uspa_preferred_prod = uspa_preferred_prod[uspa_preferred_prod['mob_prod'] != 'nan']
uspa_preferred_prod['mobile'] = uspa_preferred_prod['mob_prod'].apply(lambda x:x.split(' ')[0])
uspa_preferred_prod['sizedescription'] = uspa_preferred_prod['mob_prod'].apply(lambda x:x.split(' ')[1])
uspa_preferred_prod_temp = uspa_preferred_prod.groupby(['mobile']).agg({'count':'max'}).reset_index()
uspa_preferred_prod_temp['key_1'] = uspa_preferred_prod_temp['mobile'].astype(str) + ' ' + uspa_preferred_prod_temp['count'].astype(str)
uspa_preferred_prod['key_1'] = uspa_preferred_prod['mobile'].astype(str) + ' ' + uspa_preferred_color['count'].astype(str) 
uspa_prod_final_1 = pd.merge(uspa_preferred_prod,uspa_preferred_prod_temp, left_on='key_1',right_on ='key_1',how='left')
uspa_prod_final_1 = uspa_prod_final_1.drop_duplicates(['mobile_x'])
uspa_prod_dataframe = uspa_prod_final_1[['mobile_x','sizedescription']]
billing_avg = pd.merge(billing_avg,uspa_prod_dataframe, left_on = 'mobile',right_on='mobile_x',how = 'left')
billing_avg.drop(['mobile_x'],axis=1,inplace=True)

customer_purchased_type = {}

for index, row in uspa_subset.iterrows():
    mobile_num = row['mobile']
    item_type = row['materialtypedescription']
    if mobile_num in customer_purchased_type.keys():
        customer_purchased_type[mobile_num].append(item_type)
    else:
        customer_purchased_type[mobile_num] = []
        customer_purchased_type[mobile_num].append(item_type)
    customer_purchased_type[mobile_num] = sorted(map(str,list(set(customer_purchased_type[mobile_num]))))
    customer_purchased_type[mobile_num] = [item for item in customer_purchased_type[mobile_num] if str(item) != 'nan']
    try:
        customer_purchased_type[mobile_num].remove('CARRY BAG')
        customer_purchased_type[mobile_num].remove('OTHERS')
    except ValueError:
        #print('Carry bag or other items not found in the purchase list',customer_purchased_type[mobile_num])
        pass
    
customer_purchased_type_series = pd.Series(customer_purchased_type)
customer_purchased_type_df = pd.DataFrame({'key_words':customer_purchased_type_series}).reset_index().rename(columns={'index':'mobile'})
customer_purchased_type_df['key_word_string'] = customer_purchased_type_df['key_words'].apply(lambda x:','.join(x))
customer_purchased_type_df['item_numbers'] = customer_purchased_type_df['key_words'].apply(lambda x:len(x))

#freq_matrix
freq_map  = {}
for index, row in customer_purchased_type_df.iterrows():
    for item in row['key_words']:
        if item not in freq_map.keys():
            freq_map[item] = {}
        for other_items in row['key_words']:
            if other_items not in freq_map[item].keys():
                freq_map[item][other_items] = 1
            else:
                freq_map[item][other_items] = freq_map[item][other_items] + 1
                
freq_matrix = pd.DataFrame(freq_map).T.fillna(0)


billing_avg = pd.merge(billing_avg,customer_purchased_type_df, left_on = 'mobile',right_on='mobile',how = 'left')
billing_avg.drop(['key_words'],axis=1,inplace=True)
freq_matrix.reset_index(inplace=True)

engine = create_engine('mysql+pymysql://###:###@localhost/recoms')
Session = sessionmaker(bind=engine)
session = Session()
session.execute('''TRUNCATE TABLE billing_avg''')
session.execute('''TRUNCATE TABLE billing_data''')
session.execute('''TRUNCATE TABLE freq_matrix''')
session.commit()
session.close()

billing_avg.to_sql("billing_avg",con=engine,if_exists='append', index=False)
billing_data.to_sql("billing_data",con=engine,if_exists='append', index=False)
freq_matrix.to_sql("freq_matrix",con=engine,if_exists='append',index=False)




