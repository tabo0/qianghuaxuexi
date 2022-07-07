import json, time, datetime, math, csv, copy, sys

import numpy as np
from dateutil.parser import parse


# %read worker attribute: worker_quality
worker_quality = {}
csvfile = open("worker_quality.csv", "r")
csvreader = csv.reader(csvfile)
worker_answer=[]
for line in csvreader:
    if(line[1]=="worker_quality") : continue;
    if float(line[1]) > 0.0:
        worker_quality[int(line[0])] = float(line[1]) / 100.0
        # worker_answer[int(line[0])]={}
csvfile.close()


#% read project id
file = open("project_list.csv", "r")
project_list_lines = file.readlines()
file.close()
project_dir = "project/"
entry_dir = "entry/"

all_begin_time = parse("2018-01-01T0:0:0Z")

project_info = {}
entry_info = {}
limit = 24
industry_list = {}
for line in project_list_lines:
    line = line.strip('\n').split(',')

    project_id = int(line[0])
    entry_count = int(line[1])

    file = open(project_dir + "project_" + str(project_id) + ".txt", "r",encoding="utf_8")
    htmlcode = file.read()
    file.close()
    text = json.loads(htmlcode)

    project_info[project_id] = {}
    project_info[project_id]["sub_category"] = int(text["sub_category"]) #% project sub_category
    project_info[project_id]["category"] = int(text["category"]) #% project category
    project_info[project_id]["entry_count"] = int(text["entry_count"]) #% project answers in total
    project_info[project_id]["start_date"] = parse(text["start_date"]) #% project start date
    project_info[project_id]["deadline"] = parse(text["deadline"]) #% project end date

    if text["industry"] not in industry_list:
        industry_list[text["industry"]] = len(industry_list)
    project_info[project_id]["industry"] = industry_list[text["industry"]] #% project domain

    entry_info[project_id] = {}
    k = 0
    while (k < entry_count):
        file = open(entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r",encoding="utf_8")
        htmlcode = file.read()
        file.close()
        text = json.loads(htmlcode)

        for item in text["results"]:
            entry_number = int(item["entry_number"])
            entry_info[project_id][entry_number] = {}
            entry_info[project_id][entry_number]["entry_created_at"] = parse(item["entry_created_at"])# % worker answer_time
            #entry_info[project_id][entry_number]["worker"] = int(item["worker"]) #% work_id
            entry_info[project_id][entry_number]["author"] = int(item["author"]) # % work_id
            # if(int(item["author"]) not in worker_answer) :worker_answer[int(item["author"])]={}
            # worker_answer[int(item["author"])][project_id]= parse(item["entry_created_at"])
            temp={}
            temp['worker'] = int(item["author"])
            temp['project_id']= project_id
            temp['entry_created_at'] = parse(item["entry_created_at"])
            worker_answer.append(temp)
        k += limit
#np.save('entry_info.npy', entry_info)
np.save('worker_answer2.npy', worker_answer)
#np.save('project_info.npy', project_info)
print("finish read_data")