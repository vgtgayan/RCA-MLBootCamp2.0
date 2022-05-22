#!/depot/Python-3.6.2/bin/python
import os
import getpass
import sys
import re
from jira import JIRA
from jira.exceptions import JIRAError
import argparse


mn = 1000
delimiter = ","


#pass commandlne arguments
parser = argparse.ArgumentParser()
parser.add_argument('-jql', help='jql')
parser.add_argument('-o','--out', help='output name without file extension')
args = parser.parse_args()

#get username and password
usr = getpass.getuser()
filepath = os.path.expanduser("~/.jira_password")
try:
    with open(filepath, 'r') as file:
        pwd = file.read().rstrip("\n")
except:
    print("Cani't open password file in ~/.jira_password \nEnter your password:")
    pwd = getpass.getpass()

try:
  #create jira object
  jira_options = {'server': 'https://jira.internal.synopsys.com','verify':'/etc/ssl/certs/ca-bundle.crt'}
  jira = JIRA(options=jira_options, basic_auth=(usr,pwd),max_retries=0)
except JIRAError as e:
  if e.status_code == 401:
    print ("Login Failed check your password")
  sys.exit()

#read by jql
my_issues = jira.search_issues(args.jql, maxResults=mn,startAt=0)
total = my_issues.total


for i in range(mn,total,mn):
    my_issues2 = jira.search_issues(args.jql, maxResults=mn,startAt=i)
    my_issues += my_issues2


jkl_file = open(args.out+".jkl","w")
for issue in my_issues:
    jkl_file.write(issue.key+"\n")
jkl_file.close()


csv_file = open(args.out+".csv","w")

def writetocsv(data):
    csv_file.write(delimiter)

    if data != None:
        
        if type(data) == list :
            data = " ".join(data)
        if type(data) != str:
            data = str(data)
        data = re.sub(r",|\n|\r"," ", data)
        csv_file.write(data)



csv_file.write("key,l1,l2,l3,l4,summary,description,labels,logo,category,customer_severity,is_safty_related_issue\n")


with open(args.out+".jkl", "r") as a_file:
    for line in a_file:
        #get issue 
        key = line.strip()
        issue = jira.issue(key) 
        fields = issue.fields


        #key ------------------------------------------------
        csv_file.write(key) 

        #product l1 l2 l3 l4-----------------------------------------
        writetocsv(fields.customfield_11501)
        writetocsv(fields.customfield_11502)
        writetocsv(fields.customfield_11503)
        writetocsv(fields.customfield_11504)

        #summary --------------------------------------------
        writetocsv(fields.summary)
        
        #description ----------------------------------------
        writetocsv(fields.description)
        
        #labels
        writetocsv(fields.labels)
        
        #logo
        writetocsv(fields.customfield_11000)
        
        #category
        writetocsv(fields.customfield_10610)
        
        #customer severity
        writetocsv(fields.customfield_10302)
        
        #safty related issue
        writetocsv(fields.customfield_10625)
        
        #found version
        writetocsv(str(fields.customfield_10600))

        #linked issue

        """        #comments---------------------------------------------
        c = ""
        for comment in fields.comment.comments:
            c += comment.body
        writetocsv(c)
        """
        #end data write new line -----------------------------
        csv_file.write("\n") 


csv_file.close()



