#!/usr/bin/env python
# coding: utf-8

import pandas as pd
pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',200)
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import datetime
import warnings
import pymysql
warnings.filterwarnings("ignore")

#taking the subset dataframe
#adding cosine similarity logic
#adding logic to remove last 2 words from the item description
#cleaning up the code
#removing the items that are already purchased
#see if previously purchased items can be used for recommendation
# organize the code

def read_sql_table(table_name): 
    db = pymysql.connect(host='localhost', user="###",passwd="#####" ) 
    cur = db.cursor()   
    sql="SELECT * FROM {} ".format(table_name)
    in_data = pd.read_sql(sql,db)   
    return in_data

inventory_subset = read_sql_table("recoms.inventory_subset") #uspa_create_subsets_dataframe
uspa_subset = read_sql_table("recoms.uspa_subset")           #uspa_create_subsets_dataframe
billing_data = read_sql_table("recoms.billing_data")         #uspa_data_analysis_v6
freq_matrix = read_sql_table("recoms.freq_matrix")           #uspa_data_analysis_v6
billing_avg = read_sql_table("recoms.billing_avg")      #uspa_data_analysis_v6

###########################################################################     RECOMMENDATIONS

def create_tfidf(inventory_dataframe,column):
    count_vect = CountVectorizer()
    text_counts  = count_vect.fit_transform(inventory_dataframe[column])
    tfidf_transformer = TfidfTransformer()
    text_tfidf = tfidf_transformer.fit_transform(text_counts)
    return text_tfidf,count_vect

def product_recommender(inventory_dataframe,text_tfidf,count_vect,customer_preference_details):
    similarity_list  = list(cosine_similarity(count_vect.transform([customer_preference_details]), text_tfidf))[0]
    similarity_frame = pd.DataFrame({"recommended_matches":similarity_list})
    recommended_dataframe = pd.concat([inventory_dataframe,similarity_frame],axis=1)
    #recommended_dataframe = recommended_dataframe.drop_duplicates(['eandescription_description','recommended_matches'])
    recommended_dataframe = recommended_dataframe.drop_duplicates(['eandescription_description'])
    recommended_dataframe.sort_values(['recommended_matches'],ascending=False,inplace=True)
    return recommended_dataframe

def add_description_column(inventory_subset):
    inventory_subset['eandescription_lower'] = inventory_subset['eandescription'].apply(lambda x:str(x).lower())
    inventory_subset['eandescription_description'] = inventory_subset['eandescription_lower'].apply(lambda x:' '.join(str(x).split()[:-2]))
    return inventory_subset

def get_relevant_columns(uspa_subset):
    uspa_subset_valid_mobile = uspa_subset[uspa_subset['mobile'] != 'not-interested'][['mobile','eandescription']]
    uspa_subset_valid_mobile['eandescription_lower'] = uspa_subset_valid_mobile['eandescription'].apply(lambda x:str(x).lower())
    uspa_subset_valid_mobile['eandescription_description'] = uspa_subset_valid_mobile['eandescription_lower'].apply(lambda x:' '.join(str(x).split()[:-2]))
    return uspa_subset_valid_mobile

def create_preference_keywords(uspa_subset):
    customer_purchase_description ={}
    customer_preference = {}
    for index, row in uspa_subset.iterrows():
        mobile_num = row['mobile']
        item_description = row['eandescription_lower'].split(' ')
        if mobile_num in customer_purchase_description.keys():
            customer_purchase_description[mobile_num].extend(item_description)
        else:
            customer_purchase_description[mobile_num] = []
            customer_purchase_description[mobile_num].extend(item_description)
        customer_purchase_description[mobile_num] = list(set(customer_purchase_description[mobile_num]))
        
    for key, words  in customer_purchase_description.items():
        customer_preference[key] = ' '.join(words)
        
    customer_preference_series = pd.Series(customer_preference)
    customer_preference_df = pd.DataFrame({'key_words':customer_preference_series}).reset_index().rename(columns={'index':'mobile'})
    
    return customer_preference_df

def recommend_items(customer_preference_df):
    customer_preference_df = customer_preference_df.reindex(columns = ['mobile','key_words','r1_text','r2_text','r3_text'])
    for index, row in customer_preference_df.iterrows():
        recommended_dataframe = product_recommender(inventory_subset,text_tfidf,count_vect,row['key_words'])[0:3]
        customer_preference_df.iloc[index,2] = ' '.join(recommended_dataframe.iloc[0,0].split()[:-2])
        customer_preference_df.iloc[index,3] = ' '.join(recommended_dataframe.iloc[1,0].split()[:-2])
        customer_preference_df.iloc[index,4] = ' '.join(recommended_dataframe.iloc[2,0].split()[:-2])
        #if (index > 2000):
            #break
    customer_preference_df.drop(['key_words'],axis=1,inplace=True)
    return customer_preference_df

#####################################################################################   PURCHASES

def last_purchased_items(uspa_subset_dataframe,number_of_items=3):
    purchased_items = {}
    for index, row in uspa_subset_dataframe.iterrows():
        mobile_num = row['mobile']
        item = row['description']
        if mobile_num in purchased_items.keys():
            if(len(purchased_items[mobile_num]) < number_of_items):
                purchased_items[mobile_num].append(item)
        else:
            purchased_items[mobile_num] = []
            purchased_items[mobile_num].append(item)
    purchased_item_dataframe = pd.DataFrame(list(purchased_items.items()),columns=['mobile','item_list'])
    return purchased_item_dataframe

def last_purchased_prices(uspa_subset_dataframe,number_of_items=3):
    purchased_prices = {}
    for index, row in uspa_subset_dataframe.iterrows():
        mobile_num = row['mobile']
        item = row['line_item_amount']
        if mobile_num in purchased_prices.keys():
            if(len(purchased_prices[mobile_num]) < number_of_items):
                purchased_prices[mobile_num].append(item)
        else:
            purchased_prices[mobile_num] = []
            purchased_prices[mobile_num].append(item)
    purchased_prices_dataframe = pd.DataFrame(list(purchased_prices.items()),columns=['mobile','item_prices'])
    return purchased_prices_dataframe

def remove_not_interested_records(uspa_subset_dataframe):
    uspa_subset_dataframe = uspa_subset_dataframe[~(uspa_subset_dataframe['mobile'] == 'not-interested')]
    return uspa_subset_dataframe

def create_item_columns(uspa_subset_dataframe):
    uspa_subset_dataframe['pp1_text'] = uspa_subset_dataframe['item_list'].apply(lambda x:x[0] if len(x) > 0 else '')
    uspa_subset_dataframe['pp2_text'] = uspa_subset_dataframe['item_list'].apply(lambda x:x[1] if len(x) > 1 else '')
    uspa_subset_dataframe['pp3_text'] = uspa_subset_dataframe['item_list'].apply(lambda x:x[2] if len(x) > 2 else '')
    uspa_subset_dataframe.drop(['item_list'],axis=1,inplace=True)
    return uspa_subset_dataframe

def create_price_columns(uspa_subset_dataframe):
    uspa_subset_dataframe['bill1'] = uspa_subset_dataframe['item_prices'].apply(lambda x:x[0] if len(x) > 0 else '')
    uspa_subset_dataframe['bill2'] = uspa_subset_dataframe['item_prices'].apply(lambda x:x[1] if len(x) > 1 else '')
    uspa_subset_dataframe['bill3'] = uspa_subset_dataframe['item_prices'].apply(lambda x:x[2] if len(x) > 2 else '')
    uspa_subset_dataframe.drop(['item_prices'],axis=1,inplace=True)
    return uspa_subset_dataframe

def combine_item_price(purchased_item_dataframe,last_purchased_prices):
    combine_dataframe = pd.merge(purchased_item_dataframe,last_purchased_prices,on='mobile',how='left').reset_index()
    combine_dataframe.drop(['index'],axis=1,inplace=True)
    return combine_dataframe

def add_name_column(combined_dataframe,uspa_subset_dataframe):
    combined_dataframe = pd.merge(combined_dataframe,uspa_subset_dataframe[['mobile','first_name','last_name']].drop_duplicates(['mobile']), on='mobile', how='left')
    combined_dataframe['first_name'] = combined_dataframe['first_name'].astype(str)
    combined_dataframe['last_name'] = combined_dataframe['last_name'].astype(str).apply(lambda x:x if x!= 'nan' else '')
    combined_dataframe['name'] = combined_dataframe['first_name'] + " " + combined_dataframe['last_name']
    combined_dataframe.drop(['first_name','last_name'],axis=1,inplace=True)
    combined_dataframe = combined_dataframe[['name','mobile', 'pp1_text', 'pp2_text', 'pp3_text', 'bill1', 'bill2', 'bill3']]
    return combined_dataframe
def add_recom_column(combined_dataframe,recom):
    combined_dataframe = pd.merge(combined_dataframe,recom[['mobile','r1_text','r2_text',"r3_text"]].drop_duplicates(['mobile']), on='mobile', how='left')
    return combined_dataframe

#########################################################################################  RFM Functions

def cust_disc_details_fun(billing_data,customer_disc_avg_dict):
    customer_disc_counter = 0
    customer_disc_details = {}
    for index, row in billing_data.iterrows():
        if row['mobile'] in customer_disc_avg_dict.keys():
            if row['bill_discount'] >= customer_disc_avg_dict[row['mobile']]:
                if row['mobile'] in customer_disc_details.keys():
                    customer_disc_details[row['mobile']] = customer_disc_details[row['mobile']] + 1
                else:
                    customer_disc_details[row['mobile']] = 1
    #        else:
    #            customer_disc_details[row['mobile']] = 0
        else:
            print('This mobile number is not found {0}'.format(row['mobile']))
    return customer_disc_details

def add_column_discount_above_zero(billing_data,billing_avg,customer_disc_avg_dict):
    customer_disc_counter = 0
    customer_disc_details = {}
    for index, row in billing_data.iterrows():
        if row['mobile'] in customer_disc_avg_dict.keys():
            if row['bill_discount'] > 0:
                if row['mobile'] in customer_disc_details.keys():
                    customer_disc_details[row['mobile']] = customer_disc_details[row['mobile']] + 1
                else:
                    customer_disc_details[row['mobile']] = 1
    #        else:
    #            customer_disc_details[row['mobile']] = 0
        else:
            print('This mobile number is not found {0}'.format(row['mobile']))

    customer_disc_series = pd.Series(customer_disc_details)
    customer_disc_df = pd.DataFrame({'discount_above_zero':customer_disc_series}).reset_index()
    billing_avg = pd.merge(billing_avg,customer_disc_df,left_on='mobile',right_on='index',how='left').drop(['index'],axis=1)
    billing_avg['discount_above_zero'].fillna(0,inplace=True)
    
    return billing_avg

def add_column_discount_above_avg(customer_disc_details,billing_avg):
    customer_disc_series = pd.Series(customer_disc_details)
    customer_disc_df = pd.DataFrame({'discount_above_avg':customer_disc_series}).reset_index()
    billing_avg = pd.merge(billing_avg,customer_disc_df,left_on='mobile',right_on='index',how='left').drop(['index'],axis=1)
    billing_avg['discount_above_avg'].fillna(0,inplace=True)
    return billing_avg

def add_column_high_value(billing_data,billing_avg,customer_disc_avg_dict):
    customer_disc_avg_dict = dict(zip(list(billing_avg['mobile']),list(billing_avg['average_bill_amount'])))
    customer_disc_counter = 0
    customer_disc_details = {}
    for index, row in billing_data.iterrows():
        if row['mobile'] in customer_disc_avg_dict.keys():
            if row['bill_amount'] >= billing_data['bill_amount'].quantile(.70):
                if row['mobile'] in customer_disc_details.keys():
                    customer_disc_details[row['mobile']] = customer_disc_details[row['mobile']] + 1
                else:
                    customer_disc_details[row['mobile']] = 1
    #        else:
    #            customer_disc_details[row['mobile']] = 0
        else:
            print('This mobile number is not found {0}'.format(row['mobile']))

    customer_disc_series = pd.Series(customer_disc_details)
    customer_disc_df = pd.DataFrame({'high_value_transactions':customer_disc_series}).reset_index()
    billing_avg = pd.merge(billing_avg,customer_disc_df,left_on='mobile',right_on='index',how='left').drop(['index'],axis=1)
    billing_avg['high_value_transactions'].fillna(0,inplace=True)
    billing_avg['percentage_discounted_trans'] = (billing_avg['discount_above_zero']/billing_avg['visit_count'])*100
    
    return billing_avg

def visit_zone(visits,row):
    length=row
    visits_num = float(visits)
    if visits_num >= 0 and visits_num < np.percentile(row,20):
        return "1"
    elif visits_num >= np.percentile(row,20) and visits_num < np.percentile(row,40):
        return "2"
    elif visits_num >= np.percentile(row,40) and visits_num < np.percentile(row,60):
        return "3"
    elif visits_num >= np.percentile(row,60) and visits_num < np.percentile(row,80):
        return "4"
    else :
        return "5"
    
def list_of_counts(billing_avg):
    rows_visits=billing_avg.visit_count.unique()
    row_length=len(billing_avg.visit_count.unique())
    rvc=[]
    for i in rows_visits:
        rvc.append(i)
    arr=sorted(rvc)
    return arr

def bill_unique_fun(billing_avg):
    billing_avg["total_bill"]=billing_avg.visit_count*billing_avg.average_bill_amount
    bill_unique=billing_avg.total_bill.unique()
    bill_unique=sorted(bill_unique)
    return bill_unique

def monitoring_zone(amount,row_bills):
    bill = float(amount)
    if bill >= 0 and bill < np.percentile(row_bills,20):
        return "1"
    elif bill >= np.percentile(row_bills,20) and bill < np.percentile(row_bills,40):
        return "2"
    elif bill >= np.percentile(row_bills,40) and bill < np.percentile(row_bills,60):
        return "3"
    elif bill >= np.percentile(row_bills,60) and bill < np.percentile(row_bills,80):
        return "4"
    else :
        return "5"
    
def add_column_current_date(billing_avg):
    to=datetime.datetime.today().strftime("%Y-%m-%d")
    to=datetime.datetime.strptime(to, "%Y-%m-%d")
    billing_avg["current_date"]= to
    billing_avg["last_bill_days"]= (billing_avg.current_date- billing_avg['last_visit_date'].values.astype('datetime64[ns]')).dt.days
    return billing_avg

def recency_zone(last_bill_day):
    bill = last_bill_day
    if bill >= 0 and bill <= 90:
        return "5"
    elif bill > 91 and bill <= 180:
        return "4"
    elif bill > 181 and bill <= 270:
        return "3"
    elif bill > 271 and bill <= 365:
        return "2"
    else :
        return "1"
    
def add_column_rfm(billing_avg):
    billing_avg=billing_avg[["mobile","R_score","F_score","M_score"]]
    billing_avg["RFM"]=""
    for i in range(len(billing_avg)):
        billing_avg["RFM"][i]=(int(billing_avg.R_score[i]),int(billing_avg.F_score[i]),int(billing_avg.M_score[i]))
    return billing_avg

def category_zone(score):
    New=[(4,1,4),(4,1,5),(4,2,4),(4,2,5),(5,1,4),(5,1,5),(5,2,4),(5,2,5),(5,2,2),(4,2,2),(4,2,3),(3,1,3),(3,2,3),(4,1,3),(5,1,3),(5,2,3),(5,1,1),(5,1,2),(5,2,1),(3,1,4),(3,1,5),(3,2,4),(3,2,5),(3,1,1),(3,1,2),(3,2,1),(3,2,2),(4,1,1),(4,1,2),(4,2,1)]

    Repeat= [(3,3,4) ,(3,3,5), (3,4,3) ,(4,3,4), (4,3,5), (4,4,3), (5,3,4), (5,3,5) ,(5,4,3),(3,3,1), (3,3,2) ,(3,3,3) ,(3,4,1) ,(3,4,2) ,(4,3,1) ,(4,3,2) ,(4,3,3), (4,4,1) ,(4,4,2) ,(5,3,1),(5,3,2), (5,3,3), (5,4,1), (5,4,2)]

    Loyal=[(3,4,4) ,(3,4,5) ,(3,5,3), (3,5,4), (3,5,5), (4,4,4), (4,4,5) ,(4,5,3), (4,5,4), (4,5,5), (5,4,4), (5,4,5), (5,5,3) , (5,5,4) ,(5,5,5),(3,5,1) ,(3,5,2) ,(4,5,1) ,(4,5,2) ,(5,5,1),(5,5,2)]
         
    HighPropensitytoChurn=[(2,4,4), (2,4,5), (2,5,4), (2,5,5),(2,1,4) ,(2,1,5), (2,2,4) ,(2,2,5), (2,3,4), (2,3,5), (2,4,1), (2,4,2), (2,4,3), (2,5,1), (2,5,2), (2,5,3),(2,1,1), (2,1,2), (2,1,3) ,(2,2,1), (2,2,2), (2,2,3) ,(2,3,1), (2,3,2), (2,3,3)]
    
    Churn=[ (1,4,4) ,(1,4,5), (1,5,4), (1,5,5),(1,3,4) ,(1,3,5), (1,4,1), (1,4,2), (1,4,3), (1,5,1), (1,5,2), (1,5,3),(1,1,1), (1,1,2), (1,1,3) ,(1,1,4), (1,1,5), (1,2,1), (1,2,2), (1,2,3), (1,2,4), (1,2,5), (1,3,1), (1,3,2), (1,3,3)]

    if score in New:
        return "New"
    elif score in Repeat :
        return "Repeat"
    elif score in Loyal :
        return "Loyal"
    elif score in HighPropensitytoChurn :
        return "High Propensity to Churn"
    else :
        return "Churn"

def convert_rfm_to_string(billing_avg):    
    for i in range(len(billing_avg)):
        billing_avg["RFM"][i]=str(billing_avg.R_score[i])+","+str(billing_avg.F_score[i])+","+str(billing_avg.M_score[i])
    billing_avg=billing_avg.drop(["R_score","F_score","M_score"],axis=1)
    return billing_avg

def df_to_sql_table(recommendations):
    engine = create_engine('mysql+pymysql://##:#######/recoms')
    recommendations["pp1_text"]=recommendations["pp1_text"].str.encode('ascii', 'ignore').str.decode("utf-8")
    recommendations["pp2_text"]=recommendations["pp2_text"].str.encode('ascii', 'ignore').str.decode("utf-8")
    recommendations["pp3_text"]=recommendations["pp3_text"].str.encode('ascii', 'ignore').str.decode("utf-8")
    recommendations["bill2"].replace(to_replace="",value=0.0,inplace= True)
    recommendations["bill3"].replace(to_replace="",value=0.0,inplace= True)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.execute('''TRUNCATE TABLE recommendations''')
    session.commit()
    session.close()
    recommendations.to_sql("recommendations",con=engine,if_exists='append', index=False)
    
def df_to_csv(recommendations):
    recommendations.to_csv("drive_data/recommedations_v1.csv")


# In[ ]:


##################### inventory analysis and recommendations
inventory_subset = add_description_column(inventory_subset)
uspa_subset_valid_mobile = get_relevant_columns(uspa_subset)
customer_preference_df = create_preference_keywords(uspa_subset_valid_mobile)
text_tfidf,count_vect = create_tfidf(inventory_subset,'eandescription_lower')
recom=recommend_items(customer_preference_df)
purchased_item_dataframe = last_purchased_items(uspa_subset)
last_purchased_prices = last_purchased_prices(uspa_subset)
purchased_item_dataframe = remove_not_interested_records(purchased_item_dataframe)
last_purchased_prices = remove_not_interested_records(last_purchased_prices)
purchased_item_dataframe = create_item_columns(purchased_item_dataframe)
last_purchased_prices = create_price_columns(last_purchased_prices)
combined_dataframe = combine_item_price(purchased_item_dataframe,last_purchased_prices)
combined_dataframe = add_name_column(combined_dataframe,uspa_subset)
combined_dataframe=add_recom_column(combined_dataframe,recom)


# In[ ]:


################## RFM - must be executed only once, running twice raises key error in add_column_discount functions
customer_disc_avg_dict = dict(zip(list(billing_avg['mobile']),list(billing_avg['average_bill_discount'])))
cust_disc_details=cust_disc_details_fun(billing_data,customer_disc_avg_dict)
billing_avg=add_column_discount_above_avg(cust_disc_details,billing_avg)
billing_avg=add_column_discount_above_zero(billing_data,billing_avg,customer_disc_avg_dict)
billing_avg=add_column_high_value(billing_data,billing_avg,customer_disc_avg_dict)
arr=list_of_counts(billing_avg)
billing_avg["F_score"] = billing_avg["visit_count"].apply(visit_zone,row=arr)
bill_unique=bill_unique_fun(billing_avg)
billing_avg["M_score"] = billing_avg["total_bill"].apply(monitoring_zone,row_bills=bill_unique)
billing_avg=add_column_current_date(billing_avg)
billing_avg["R_score"] = billing_avg["last_bill_days"].apply(recency_zone)
billing_avg=add_column_rfm(billing_avg)
billing_avg["Category"]=billing_avg["RFM"].apply(category_zone)
billing_avg=convert_rfm_to_string(billing_avg)
recommendations = pd.merge(combined_dataframe,billing_avg[['mobile','RFM']].drop_duplicates(['mobile']), on='mobile', how='left')


#df_to_csv(recommendations)
df_to_sql_table(recommendations)

