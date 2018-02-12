# authors: Hezhi Wang, Han Zhao, Yuwei Tu

"""
This the feature extraction for ['Judges', 'Crime', 'Defense', 'ModeOfConviction', 'GroundOfAppeal']
"""
import os
import re
import string
import glob
import csv

path = 'NY-Appellate-Scraping/2017-09-10/courtdoc/txt/'
#judge = re.compile('\.\s.*JJ')
charsNOWHITE = '"#()\'*+,./<=>@[\\]^_`{|}~'
crimestopwords=['a','in','of','the','first','second','third','fourth','fifth','degree','one','two','three','four','five','six','counts','and','or']
mode = re.compile('plea\s*of\s*guilty|jury\s*verdict|nonjury\s*trial|guilty\s*plea', re.IGNORECASE)
convof = re.compile('conviction\s*of|convict', re.IGNORECASE)
Defense = re.compile('public\s*defender|conflict\s*defender|legal\s*aid\s*dureau|legal\s*aid\s*society', re.IGNORECASE)
sexoffender= re.compile('sex\s*offender\s*registration\s*act', re.IGNORECASE)

crimeset=set()

def read_file(filename):
	with open(filename, "r") as myfile:
		text = myfile.read()
		text = text.replace('\n', ' ').replace('\r', '')
		return text

def extract_judges(doc):
	#extract judges
	try:
		endIndex = doc.index('JJ')
		startIndex = doc[:endIndex].rindex('. ')
	except ValueError:
		return ''

	judges = doc[startIndex+1: endIndex+2]
	return judges

def extract_crimes(doc):
    # extract crimes
    conviction = mode.search(doc)
    idx=0
    if conviction is not None:
        end = doc[idx+conviction.end():].find('. ')
        end = idx+conviction.end()+end

        crimes=doc[idx+conviction.end():end].replace(' and ',',').split(',')
        for  i,y in enumerate(crimes):
            y=y.replace(charsNOWHITE, '').replace('\n',' ')
            crimes[i]=' '.join([x for x in y.split(' ') if x not in crimestopwords]).strip()

        crimeset.update(crimes)
        crime=';'.join(set(crimes)).strip(";")
        conviction=conviction.group()
    else:
        conviction=convof.search(doc[idx:])
        if conviction is not None:
            end = doc[idx+conviction.end():].find('. ')
            end = idx+conviction.end()+end
            crimes=doc[idx+conviction.end(): end].replace(' and ',',').split(',')
            for  i,y in enumerate(crimes):
                y=y.replace(charsNOWHITE, '').replace('\n',' ')
                crimes[i]=' '.join([x for x in y.split(' ') if x not in crimestopwords]).strip()
            crimeset.update(crimes)
            crime=';'.join(set(crimes)).strip(";")
            conviction=''
        else:
            conviction=''
            if sexoffender.search(doc[idx:]) is not None:
                crime='risk pursuant to Sex Offender Registration Act'
            else:
                crime=''
    return crime

def extract_defense(doc):
    
    #extract defense
    defenses = Defense.search(doc)
    if defenses is not None:
        defense = defenses.group()
    else:
        defense = 'Private'
    return defense

def extract_mode_of_conviction(doc):
    mode = re.compile('plea\s*of\s*guilty|jury\s*verdict|nonjury\s*trial', re.IGNORECASE)
    results = mode.search(doc)
    if results:
        return results.group(0)

def extract_ground_of_appeal(doc):
    ground = re.compile('insufficient|new\s*evidence|misconduct|improper\s*admission|ineffective\s*assistance', re.IGNORECASE)
    results = ground.search(doc)
    if results:
        return results.group(0)


if __name__ == '__main__':
    with open('CourtOutput.csv', 'w', newline='', encoding='utf-8') as myFile:
        myFields = ['File', 'Judges', 'Crime', 'Defense', 'ModeOfConviction', 'GroundOfAppeal']
        writer = csv.DictWriter(myFile, fieldnames=myFields)
        writer.writeheader()

        for filename in glob.glob(path + '*.txt'):

            doc = read_file(filename)
            judges = extract_judges(doc)
            crime = extract_crimes(doc)
            defense = extract_defense(doc)
            conviction = extract_mode_of_conviction(doc)

            ground = extract_ground_of_appeal(doc)

            filename = filename.replace(path, "")
            judges = judges.replace("—","-").replace("–","-")
            writer.writerow({'File' : filename, 'Judges': judges, 'Crime': crime, 'Defense': defense,
                'ModeOfConviction': conviction, 'GroundOfAppeal': ground})


