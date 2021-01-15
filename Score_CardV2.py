#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:13:13 2020

@author: marconoronha
"""
#%%
# Importing some important librarys
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#%%
"Read data from dataset into a dataframe"
df = pd.read_csv('lending_club_loans_copy.csv')
#%%
"Drops undesired columns"
df.drop(["id","member_id","funded_amnt","grade","sub_grade",
         "verification_status","issue_d","pymnt_plan", 
         "url","desc","title","zip_code","emp_title","addr_state"],axis=1, inplace = True)
df.drop(df.columns[10:],axis = 1, inplace = True)
"Adds the split term column to the df in two new columns 'term' &'str'"
df[['term','str']] = df.term.str.split(expand=True) 
"Drops column "
df.drop(['str'],axis = 1, inplace = True)
df['emp_length'].astype(object)

"Prints any null values in term column"
print(df[df['term'].isnull()])
'removes any non numeric values and converts them to Nan'
df['term'] = pd.to_numeric(df['term'], errors='coerce')
'Drops any null values from column term'
df = df.dropna(subset=['term'])
'Drops any null values from column emp_length'
df = df.dropna(subset=['emp_length'])
'Drops any null values from column funded_amnt_inv'
df = df.dropna(subset=['funded_amnt_inv'])
"converts values in the column term to integers"
"df['term'] = df['term'].astype(object)"
df.dtypes
#%%
#MAKING BINS 
"Annual Income Bin"
annual_inc_labels = ["0-25k","25-50k","50-75k","75-100k","100-125k",
                     "125-150k", "150k+"]
annual_inc_cut_bins = [0,25000, 50000, 75000, 100000, 125000, 150000,
                       10000000]
df['AnnualIncomeBins'] = pd.cut(df['annual_inc'], bins=annual_inc_cut_bins, labels=annual_inc_labels).astype(object)

"Employment Length Bins"
emp_length_labels = ['0-2yrs', '2-4yrs', '4-6yrs','6-8yrs','8-10yrs']
emp_length_cut_bins = [0,2,4,6,8,10]
df["EmpLengthBins"] = pd.cut(df["emp_length"],bins = emp_length_cut_bins,labels = emp_length_labels).astype(object)

"Loan Ammount Bins"
loan_amnt_labels = ["0-7k", '7-14k', '14-21k','21-28k','28-35k']
loan_amnt_cut_bins = [0,7000,14000,21000,28000,35000]
df["LoanAmntBins"] = pd.cut(df["loan_amnt"],bins = loan_amnt_cut_bins,labels = loan_amnt_labels).astype(object)

"Funded Ammount Bins"
funded_amnt_labels = ["0-7k", '7-14k', '14-21k','21-28k','28-35k']
funded_amnt_cut_bins = [0,7000,14000,21000,28000,35000]
df["FundedAmntBins"] = pd.cut(df["funded_amnt_inv"],bins = funded_amnt_cut_bins,labels = funded_amnt_labels).astype(object)

"Interest Rate Bins"
interest_labels = ['0-5%', '5-10%', '10-15%','15-20%','20-25%']
interest_cut_bins = [0,5,10,15,20,25]
df["InterestBins"] = pd.cut(df["int_rate"],bins = interest_cut_bins,labels = interest_labels).astype(object)

"Installment Bins"
installment_labels = ['0-300', '300-600', '600-900','900-1200','1200-1500']
installment_cut_bins = [0,300,600,900,1200,1500]
df["InstallmentBins"] = pd.cut(df["installment"],bins = installment_cut_bins,labels = installment_labels).astype(object)

"Term Loan Bins"
term_labels = ['36 Months',"60 Months" ]
term_cut_bins = [0,36,60]
df["TermBins"] = pd.cut(df["term"],bins = term_cut_bins,labels = term_labels).astype(object)

"Drops old columns for new bin columns"
df.drop(["term","annual_inc","emp_length","loan_amnt", "funded_amnt_inv", "int_rate", "installment"], axis = 1, inplace = True)
df.dtypes
#%%
"import scorecard library as sc"
import scorecardpy as sc

"filter loan status target variable via fully paid status, iv, identical value rate"
dt_s = sc.var_filter(df, y="loan_status", iv_limit=0.00,
                     positive = "Unpaid")
print(dt_s.dtypes)
#%%
"breaking dt_s into train and test"
train, test = sc.split_df(dt_s, 'loan_status').values()
#%%
"WoE Binning"
bins = sc.woebin(dt_s, y="loan_status")
"Plots distribution and IV of bins"
sc.woebin_plot(bins)
#%%
#binning adjustment
"adjust breaks interactively"
breaks_adj = sc.woebin_adj(dt_s, "loan_status", bins) 
# # or specify breaks manually
breaks_adj = {
    'AnnualIncomeBins': ['0-25k', '25-50k','50-75k','75-100k','100-125k','125-150k','150k+' ],
    'EmpLenghtBins': ["0-2yrs", "2-4yrs", "4-6yrs","6-8yrs", "8-10yrs"],
    "FundedAmntBins":["0-7k","7-14k", "14-21k","21-28k","28-35k"],
    "LoanAmntBins":["0-7k","7-14k", "14-21k","21-28k","28-35k"],
    "InstallmentBins":["0-300","300-600","600-900","900-1200","1200-1500"], 
    "TermBins":["36 Months", "60 Months"], 
    "HomeOwnership":["Mortgage", "Own", "Rent", "Other"], 
    "Purpose":["wedding", "major_purchase","car","credit_card","home_improvement", "vacation", 
               "moving","medical","debt_consolidation", "house","renewable_energy", "small_business"], 
    "basepoint":["Base Points"]
     }
    
bins_adj = sc.woebin(dt_s, y="loan_status", breaks_list=breaks_adj)
#%%
"Converting train and test into woe values"
train_woe = sc.woebin_ply(train, bins_adj)
test_woe = sc.woebin_ply(test, bins_adj)
"Stores WoE data not including loan status column in training and test variables"
y_train = train_woe.loc[:,'loan_status']
X_train = train_woe.loc[:,train_woe.columns != 'loan_status']
y_test = test_woe.loc[:,'loan_status']
X_test = test_woe.loc[:,train_woe.columns != 'loan_status']
#%%
"Logistic Regression setup"
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
lr.coef_
lr.intercept_
#%%
#Predicting Proability
"Predicting Probability on training data"
train_pred = lr.predict_proba(X_train)[:,1]
"Predicting Probability on test data"
test_pred = lr.predict_proba(X_test)[:,1]
#%%
"Performance KS & ROC"
train_perf = sc.perf_eva(y_train, train_pred, title = "train")
test_perf = sc.perf_eva(y_test, test_pred, title = "test")
#%%
"Score Card"
"""Scorecard is created from the adjusted bins data, the model used is 
logistic regression, can change MAX points to anything, and can manipulate
base points per record and the points to double the odds (pdo)"""
card = sc.scorecard(bins_adj, lr, X_train.columns, points0= 650, odds0=1/19, 
                    pdo=50, basepoints_eq0=False)
"Credit Score"
train_score = sc.scorecard_ply(train, card, print_step=0)
test_score = sc.scorecard_ply(test, card, print_step=0)
#%%
#PSI
"Calculates population stability index which provides SC distribution based on the data"
"Measures how much the predicted variable has shifted across the test & Train data"
"PSI = 0.0003 indicates almost no change which is good"
sc.perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test}, 
  x_limits=None, x_tick_break=50,
  show_plot = True
)
#%%
# Scoring Logic
card

"Set base score for each customer"
score_count = 561
type(score_count)
#%%
#USER INCOME SCORE CALCULATOR 
user_inc = float(input('\nWhat is your annual income?:'))

if (user_inc < 25000):
    score_count = score_count - 29 
    print("You lose 29 points. New score:" , score_count)
    
elif (user_inc >=25000 and user_inc <= 50000):
        score_count = score_count - 16
        print("You lose 16 points. New score:" , score_count)
    
elif (user_inc >=50000 and user_inc < 75000) :
        score_count = score_count - 0
        print("You lose 0 points. New score:" ,score_count)
    
elif (user_inc >=75000 and user_inc < 100000):
        score_count = score_count + 20
        print("You gain 20 points. New score:" , score_count)
    
elif (user_inc >=100000 and user_inc < 125000):
        score_count = score_count + 32
        print("You gain 32 points. New score:" , score_count)

elif (user_inc >=125000 and user_inc < 150000):
        score_count = score_count + 33
        print("You gain 33 points. New score:" , score_count)
    
elif (user_inc >=150000):
        score_count = score_count + 36
        print("You gain 36 points. New score:", score_count)
    
else :
    print("\nPlease enter another figure!")
    
    
#%%
#USER INSTALLMENTS SCORE CALCULATOR 
user_inst = float(input('\nWhat is your monthly installment?:'))

if (user_inst < 300):
    score_count = score_count + 6 
    print("You gain 6 points. New score:" , score_count)
    
elif (user_inst >=300 and user_inst <= 600):
        score_count = score_count - 4
        print("You lose 4 points. New score:" , score_count)
    
elif (user_inst >=600 and user_inst < 900) :
        score_count = score_count - 13
        print("You lose 13 points. New score:" ,score_count)
    
elif (user_inst >=900 and user_inst < 1200):
        score_count = score_count - 27
        print("You lose 27 points. New score:" , score_count)
    
elif (user_inst >=1200 and user_inst <= 1500):
        score_count = score_count + 8
        print("You gain 8 points. New score:" , score_count)
        
else: 
    print("Installment size must be between 0 and 1500! Please enter another figure!")

#%%
#USER HOME OWNERSHIP SCORE CALCULATOR 

user_house = input("\n What is your residential Status?\n Rent, Own, Mortgage, Other\n").lower()

if (user_house == 'rent', "other"):
    score_count = score_count -2
    print("\n You lose 2 points. New score:", score_count)
    
elif (user_house == "mortgage"): 
    score_count = score_count + 2 
    print("\n You gain 2 points. New score:", score_count)
    
elif (user_house == "own"): 
    score_count = score_count +1
    print("\n You gain 1 points. New score:", score_count)
    
else:
    print("Please enter one of the options given!")
#%%
#FUNDED AMOUNT SCORE CALCULATOR 
"For use by commercial institiutions"
card

usr_fundamnt = float(input("\n How much of the loan are you funding?: \n"))

if (usr_fundamnt < 7000):
    score_count = score_count -2 
    print("You lose 2 points. New score:" , score_count)
    
elif (usr_fundamnt >=7000 and usr_fundamnt <= 14000):
        score_count = score_count - 2
        print("You lose 2 points. New score:" , score_count)
    
elif (usr_fundamnt >=14000 and usr_fundamnt < 21000) :
        score_count = score_count +3
        print("You gain 3 points. New score:" ,score_count)
    
elif (usr_fundamnt >=21000 and usr_fundamnt < 28000):
        score_count = score_count + 5 
        print("You gain 5 points. New score:" , score_count)
    
elif (usr_fundamnt >=28000 and usr_fundamnt <= 35000):
        score_count = score_count + 10
        print("You gain 10 points. New score:" , score_count)
        
else: 
    print("Funded amount must be between 0 & 35000! Please enter another figure!")
#%%
#LOAN AMOUNT CALCULATOR 
card

usr_loanamnt = float(input("\n How much of a loan are you looking for? \n"))

if (usr_loanamnt < 7000):
    score_count = score_count + 5 
    print("You gain 5 points. New score:" , score_count)
    
elif (usr_loanamnt >=7000 and usr_loanamnt <= 14000):
        score_count = score_count + 3
        print("You gain 3 points. New score:" , score_count)
    
elif (usr_loanamnt >=14000 and usr_loanamnt < 21000) :
        score_count = score_count- 5
        print("You lose 5 points. New score:" ,score_count)
    
elif (usr_loanamnt >=21000 and usr_fundamnt < 28000):
        score_count = score_count - 10
        print("You lose 10 points. New score:" , score_count)
    
elif (usr_loanamnt >=28000 and usr_loanamnt <= 35000):
        score_count = score_count - 19
        print("You lose 19 points. New score:" , score_count)
        
else: 
    print("Loan amount must be between 0 & 35000! Please enter another figure!")
#%%
#TERM BIN CALCULATOR
usr_term = float(input("\n What term loan lenght do you require? 36 months or 60 months? :"))

if (usr_term == 36): 
    score_count = score_count + 29
    print("\n You gain 29 points. New score:", score_count)
    
elif (usr_term == 60): 
    score_count = score_count - 54
    print("\n You lose 54 points. New score: ", score_count)
else: 
    print("Please choose either 36 or 60 months!\n")
    
    
#%%
#EMPLOYMENT LENGTH CALCULATOR
card 

usr_employ = float(input("\n How long have you been employed for? : \n"))

if (usr_employ < 2): 
    score_count = score_count + 3 
    print("\n You gain 3 points. New score:", score_count)
    
elif (usr_employ >= 2 and usr_employ < 6): 
    score_count = score_count + 2
    print("\n You gain 2 points. New score: ", score_count)
    
elif (usr_employ >= 6 and usr_employ < 10): 
    score_count = score_count - 4 
    print("\n You lose 4 points. New score: ", score_count)
    
else: 
    print("\n Please enter a figure between 1 - 10")
#%%
#LOAN PURPOSE CALCULATOR
card
"Need to create a drop down on the website only giving them these options"
user_purp = input("\n What is the purpose of your loan? \n"
                  "\n Select from the following :" 
                  "\n Major Purchase, Wedding, Car, Credit Card," 
                  "\n Home Improvement, Vacation, Moving, Medical"
                  "\n Debt Consolidation, House, Renewable Energy"
                  "\n Small Business : ").lower()


if (user_purp == 'major purchase', "wedding"):
    score_count = score_count + 30 
    print("\n You gain 30 points. New score:", score_count)
    
elif (user_purp == "car" , "credit card" ,"home improvement"): 
    score_count = score_count + 20
    print("\n You gain 20 points. New score:", score_count)
    
elif (user_purp == "vacation","moving", "medical", "debt consolidation"): 
    score_count = score_count - 5
    print("\n You lose 5 points. New score:", score_count)
    
elif (user_purp == "house", "renewable energy", "small business : "): 
    score_count = score_count - 43
    print("\n You lose 43 points. New score:", score_count)
    
else:
    print("Please choose from one of the options!")
#%%
final_score = score_count
if (final_score < 500):
    print("\n Your final score is :", final_score, "\n Due to it being less than 500 points, you are bing denied!")
    
elif (final_score > 500 ): 
    print("\n Your final score is :", final_score, "\n Congrats, your application has been successful!")
    
#%%
