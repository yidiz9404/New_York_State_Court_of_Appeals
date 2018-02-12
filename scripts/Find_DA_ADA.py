import re
import pandas

'''
@ Wei Zhang wz1218

Change the path before you run the program.
This file include, how to find District Attorney, ADA.
if you have any question for these funtions, please contact me @ wz1218@nyu.edu
'''

def Find_DA(sentence):
    DA = re.compile('[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z]([a-z]+|\.)(?:,\s|)(?:Jr\.|Sr\.|IV|III|II|)(?:,\s|)District Attorney')
    dist1 = DA.search(sentence)
    if dist1:
        dist1 = dist1.group(0)
        dist1 = dist1.rsplit(',', 1)[0]
    return dist1


def Find_ADA(sentence):
    ADA_pattern2 = re.compile('\((.*?)\)')
    dist2 = ADA_pattern2.search(sentence)
    if dist2:
        dist2 = dist2.group(0)[1:-12]
    return dist2


if __name__ == '__main__':
    data = pd.read_csv("/Users/will/Desktop/hackathon/NY-Appellate-Scraping/2017-09-10/parsedfiles_2017/parsedfiles_20170.csv", sep='|')

    ADA = [None] * len(filelist)
    DA = [None] * len(filelist)

    path = "/Users/will/Desktop/hackathon/NY-Appellate-Scraping/2017-09-10/courtdoc/html/"
    filelist = os.listdir(path)

    list.sort(filelist)
    for index, i in enumerate(filelist):
        with open(path + i, 'r', encoding='utf-8') as f:
            for line in f:
                if "District Attorney" in line:
                    ADA[index] = Find_ADA(line)
                    DA[index] = Find_DA(line)

    DA = pd.Series(DA)
    ADA = pd.Series(ADA)
    new_data = pd.DataFrame()
    new_data["File"] = data["File"]
    new_data["DA"] = DA
    new_data["ADA"] = ADA

    new_data.to_csv("/Users/will/Desktop/hackathon/Find_DA_ADA.csv")
