#coding: utf-8
#author: Yidi Zhang, Xue Yang

"""
This the feature extraction for ['DefendantAppellant', 'DefendantRespondent',
'HarmlessError', 'ProsecutMisconduct', 'DocumentLength', 'Justice']
"""

import pandas as pd
import re
import os

df = pd.read_csv('./parsedfiles_2017/parsedfiles_20170.csv',sep='|')
path = './courtdoc/html/'

def clean(text):
    '''
    input unstructured string
    -----------------------
    output cleaned data
    '''
    cr = re.compile('<.*?>')
    text = re.sub(cr,'',text)
    text = re.sub('[()][0-9]+-[0-9]+[()]', '', text)
    text = text.lstrip(' ')
    text = text.rstrip('\n\t')
    return text

def drop_civilcase(path):
    '''
    Input: the path to get text data
    Output: the datsets in which the civil cases are dropped
    '''
    output = []

    filelist = os.listdir(path)
    for i in filelist:
        if i.endswith(".htm"):
            with open(path + i, 'r') as f:
                data = f.readlines()
                ss = data[2].replace('<TITLE>','')
                #print(ss[:9])
                if ss[:9] == "People v ":
                    output.append(i[:10])
    return output

#Is state of NY the 'appellant' or 'respondent'?
def respondent_or_appellant(output):
    '''
    Input the datsets in which the civil cases are dropped 
    ------------------------------------------------------
    Output the People of NY state is 'appellant' or 'respondent'
    '''
    p7 = re.compile(r'State of New York,\n(.*),', re.IGNORECASE)
    p8 = re.compile(r'State of New York,(.*),', re.IGNORECASE)
    p9 = re.compile(r'The People of the State of New York (.*)')
    #p8 = re.compile(r'The People of the State of New York, (.*)', re.IGNORECASE)
    output_da = {}
    #filelist = os.listdir(path)
    for i in output:
        #if i.endswith(".txt"):
        with open(path + i +'.htm', 'r') as f:
            data = f.read()
            data = clean(data)
            temp = []
            temp = p7.findall(data)
            temp.extend(p8.findall(data))
            temp.extend(p9.findall(data))
            if temp == []:
                print(i)
            output_da[i] = temp[0][:11]
    return output_da

def HarmlessError(output):
    '''
    Input the datsets in which the civil cases are dropped 
    ------------------------------------------------------
    Output the crime cases with harmless error is '1', without harmless error is '0'
    '''
    harmless = {}
    for i in output:
        #if i.endswith(".txt"):
        with open(path + i +'.htm', 'r') as f:
            data = f.readlines()
            #data = clean(data)
            for d in data:
                if 'harmless' in d and 'error' in d:
                    harmless[i]= 1
                else:
                    harmless[i] = 0
    return harmless

def ProsecutMisconduct(output):
    '''
    Input the datsets in which the civil cases are dropped 
    ------------------------------------------------------
    Output the crime cases with ProsecutMisconduct is '1', without harmless error is '0'
    '''
    misconduct = {}
    for i in output:
        #if i.endswith(".txt"):
        with open(path + i +'.htm', 'r') as f:
            data = f.readlines()
            #data = clean(data)
            for d in data:
                if 'prosecut' in d and 'misconduct' in d:
                    misconduct[i]= 1
                else:
                    misconduct[i]= 0
    return misconduct

def combine_data(misconduct, harmless, output_da):
    '''
    Input: the dictionary of ProsecutMisconduct, HarmlessError, respondent_or_appellant
    --------------------------------------------------------------------------------------
    Output: the dataframe concatenate the ProsecutMisconduct, HarmlessError, respondent_or_appellant
    '''
    miscond = pd.DataFrame.from_dict(misconduct, orient ='index')
    harml = pd.DataFrame.from_dict(harmless, orient ='index')
    DA = pd.DataFrame.from_dict(output_da,orient='index')
    DA.columns =['PSNY']
    DA = DA.join(harml)
    DA.columns =['PSNY','HarmlessError']
    DA = DA.join(miscond)
    DA.columns =['PSNY','HarmlessError','ProsecutMisconduct']
    return DA

if __name__ == '__main__':
    cri_case = drop_civilcase(path)
    dat_RA = respondent_or_appellant(cri_case)
    dat_harm = HarmlessError(cri_case)
    dat_prose = ProsecutMisconduct(cri_case)
    DA = combine_data(dat_RA, dat_harm, dat_prose)
    DA.to_csv('extract_data.csv')
