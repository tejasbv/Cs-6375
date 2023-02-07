import sys
import re
import math
attr = []
target = []

def read(path):
    file = open(path, "r")
    return file.read()


def analyze_data(path):
    content = read(path)
    rows = content.split("\n")
    # print(len(rows[0]))
    for r in rows:
        vals = r.split(" ")
        for i in range(0,len(vals)):
            if len(attr) > i and i != len(vals)-1:
                attr[i].append(vals[i])
            elif i is not len(vals)-1:
                attr.append([vals[i]])
            else:
                target.append(vals[i])
    
    print(len(attr))
    print(target)
    CalculateEntropyS0(target, [8,9])
    

def CalculateEntropyS0(att, set):
    counts = [0,0,0]
    for i in set:
        print(att[i])
        if att[i] == '0':
            counts[0]+=1
        elif att[i] == '1':
            counts[1]+=1
        else:
            counts[2]+=1
    
    total = counts[0]+counts[1]+counts[2]
    E = 0
    for p in counts:
        if p!=0:
            E += (p/total)*math.log(total/p,2)
    print(E)



if __name__ == "__main__":
    # print(sys.argv)
    analyze_data(sys.argv[1])
