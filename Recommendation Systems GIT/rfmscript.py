#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
#Added librabies 
from sqlalchemy import create_engine
import datetime
import numpy as np
import pymysql
import warnings
from sqlalchemy.orm import sessionmaker
warnings.filterwarnings("ignore")


def read_sql_table(table_name):
    db = pymysql.connect(host='localhost', user="####",passwd="####" ) 
    cur = db.cursor()  
    sql="SELECT * FROM {} ".format(table_name)
    in_data = pd.read_sql(sql,db)
    return in_data


billing_data = read_sql_table("recoms.billing_data") 
billing_avg = read_sql_table("recoms.billing_avg") 

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

def bill_unique(billing_avg):
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
    billing_avg['last_visit_date']=billing_avg['last_visit_date'].values.astype('datetime64[D]')
    billing_avg["last_bill_days"]= (billing_avg.current_date- billing_avg.last_visit_date).dt.days
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
    New=[(4,1,4),(4,1,5),(4,2,4),(4,2,5),(5,1,4),(5,1,5),(5,2,4),(5,2,5),(5,2,2),(4,2,2),(4,2,3),
     (3,1,3),(3,2,3),(4,1,3),(5,1,3),(5,2,3),(5,1,1),(5,1,2),(5,2,1),(3,1,4),(3,1,5),(3,2,4),(3,2,5)
     ,(3,1,1),(3,1,2),(3,2,1),(3,2,2),(4,1,1),(4,1,2),(4,2,1)]

    Repeat= [(3,3,4) ,(3,3,5), (3,4,3) ,(4,3,4), (4,3,5), (4,4,3), (5,3,4), (5,3,5) ,(5,4,3),
        (3,3,1), (3,3,2) ,(3,3,3) ,(3,4,1) ,(3,4,2) ,(4,3,1) ,(4,3,2) ,(4,3,3), (4,4,1) ,(4,4,2) ,(5,3,1),(5,3,2), (5,3,3), (5,4,1), (5,4,2)]

    Loyal=[(3,4,4) ,(3,4,5) ,(3,5,3), (3,5,4), (3,5,5), (4,4,4), (4,4,5) ,(4,5,3), (4,5,4), (4,5,5), (5,4,4), (5,4,5), (5,5,3) , (5,5,4) ,(5,5,5),
      (3,5,1) ,(3,5,2) ,(4,5,1) ,(4,5,2) ,(5,5,1),(5,5,2)]
         
    HighPropensitytoChurn=[(2,4,4), (2,4,5), (2,5,4), (2,5,5),
                   (2,1,4) ,(2,1,5), (2,2,4) ,(2,2,5), (2,3,4), (2,3,5), (2,4,1), (2,4,2), (2,4,3), (2,5,1), (2,5,2), (2,5,3)
                   ,(2,1,1), (2,1,2), (2,1,3) ,(2,2,1), (2,2,2), (2,2,3) ,(2,3,1), (2,3,2), (2,3,3)]
    
    Churn=[ (1,4,4) ,(1,4,5), (1,5,4), (1,5,5),
       (1,3,4) ,(1,3,5), (1,4,1), (1,4,2), (1,4,3), (1,5,1), (1,5,2), (1,5,3),
       (1,1,1), (1,1,2), (1,1,3) ,(1,1,4), (1,1,5), (1,2,1), (1,2,2), (1,2,3), (1,2,4), (1,2,5), (1,3,1), (1,3,2), (1,3,3)]

    if score in New:
        return "New"
    elif score in Repeat :
        return "Repeat"
    elif score in Loyal :
        return "Loyal"
    elif score in Churn :
        return "Churn"
    else :
        return "HighPropensitytoChurn"

def convert_rfm_to_string(billing_avg):    
    for i in range(len(billing_avg)):
        billing_avg["RFM"][i]=str(billing_avg.R_score[i])+","+str(billing_avg.F_score[i])+","+str(billing_avg.M_score[i])
    billing_avg=billing_avg.drop(["R_score","F_score","M_score"],axis=1)
    return billing_avg

def df_to_sql_table(recommendations):
    engine = create_engine('mysql+pymysql://###:###@localhost/recoms')
    Session = sessionmaker(bind=engine)
    session = Session()
    session.execute('''TRUNCATE TABLE RFM''')
    session.commit()
    session.close()
    recommendations.to_sql("RFM",con=engine,if_exists='append', index=False)

customer_disc_avg_dict = dict(zip(list(billing_avg['mobile']),list(billing_avg['average_bill_discount'])))
cust_disc_details=cust_disc_details_fun(billing_data,customer_disc_avg_dict)
billing_avg=add_column_discount_above_avg(cust_disc_details,billing_avg)
billing_avg=add_column_discount_above_zero(billing_data,billing_avg,customer_disc_avg_dict)
billing_avg=add_column_high_value(billing_data,billing_avg,customer_disc_avg_dict)
arr=list_of_counts(billing_avg)
billing_avg["F_score"] = billing_avg["visit_count"].apply(visit_zone,row=arr)
bill_unique=bill_unique(billing_avg)
billing_avg["M_score"] = billing_avg["total_bill"].apply(monitoring_zone,row_bills=bill_unique)
billing_avg=add_column_current_date(billing_avg)
billing_avg["R_score"] = billing_avg["last_bill_days"].apply(recency_zone)
billing_avg=add_column_rfm(billing_avg)
billing_avg["Category"]=billing_avg["RFM"].apply(category_zone)
billing_avg=convert_rfm_to_string(billing_avg)
billing_avg=billing_avg.set_index("mobile").drop("not-interested",axis=0)
billing_avg.reset_index(inplace=True)
df_to_sql_table(billing_avg)




