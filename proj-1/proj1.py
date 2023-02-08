import sys
import re
import math
attr = []
target = []
sets = {}

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
    
    # print(len(attr))
    # print(target)
    
    

def analyze_partition(path):
    content = read(path)
    rows = content.split("\n")
    for r in rows:
        vals = r.split(" ")
        for i in range(0,len(vals)):
            if i == 0:
                sets[int(vals[i])] = []
            else:
                sets[int(vals[0])].append(int(vals[i]))

    # print(sets)


def Calculate_gain(set):
    Entropy_set = CalculateEntropyS0(target, set)
    Entropy_attr = []
    for a in attr:
        Entropy_attr.append(CalculateEntropy_attr(attr=a, target=target, set=set))
    
    Gain_attr = []
    for e in Entropy_attr:
        Gain_attr.append(Entropy_set-e)
    # print(Gain_attr)
    attr_index = Gain_attr.index(max(Gain_attr))
    return [attr_index, max(Gain_attr), set]
    

def split(att, set):
    newSet = [[],[],[]]
    for i in set:
        if att[i] == "0":
            newSet[0].append(i)
        elif att[i] == "1":
            newSet[1].append(i)
        elif att[i] == "2":
            newSet[2].append(i)
    # print(newSet)
    return newSet

def CalculateEntropyS0(att, set):
    counts = [0,0]
    for i in set:
        # print(att[i])
        if att[i] == '0':
            counts[0]+=1
        elif att[i] == '1':
            counts[1]+=1
        else:
            counts[2]+=1
    
    total = counts[0]+counts[1]
    E = 0
    for p in counts:
        if p!=0:
            E += (p/total)*math.log(total/p,2)
    # print(E)
    return E

def CalculateEntropy_attr(target, attr, set):
    counts = [[0,0],[0,0],[0,0]]
    for i in set:
        if attr[i] == '0':
            if target[i] == '0':
                counts[0][0]+=1
            elif target[i] == '1':
                counts[0][1]+=1
        elif attr[i] == '1':
            if target[i] == '0':
                counts[1][0]+=1
            elif target[i] == '1':
                counts[1][1]+=1
        elif attr[i] == '2':
            if target[i] == '0':
                counts[2][0]+=1
            elif target[i] == '1':
                counts[2][1]+=1
    E = 0
    for a in counts:
        total = a[0]+a[1]
        E2 = 0
        for p in a:
            if p!=0:
                E2 += (p/total)*math.log(total/p,2)
        E+=(total/len(set))*E2
    # print(E)
    return E
        


def updateSets(newSet, oldSet, attr_index):
    # print(sets)
    maxId = max(sets.keys())
    message = "Partition "
    for key, val in sets.items():
        if val == oldSet:
            sets.pop(key)
            message+=f"{key} was replaced with partitions "
            break
    
    for s in newSet:
        maxId+=1
        message+=f"{maxId},"
        sets[maxId] = s
    message = message[:-1]
    message+=f" using Attribute {attr_index}"
    print(message)

    
    

def writeSets(path):
    file = open(path, "w")
    for k, vals in sets.items():
        file.write(f"{k} ")
        for v in vals:
            file.write(f"{v} ")
        file.write("\n")

if __name__ == "__main__":
    # print(sys.argv)
    analyze_partition(sys.argv[2])
    analyze_data(sys.argv[1])

    setGains = []
    for k in sets.values():
        setGains.append(Calculate_gain(k))

    max_gain = -1
    gain = []
    for g in setGains:
        if  max_gain < g[1]:
            max_gain = g[1]
            gain = g
    # print(gain)


    newSet = split(attr[gain[0]], gain[2])
    updateSets(newSet, gain[2], gain[0])
    writeSets(sys.argv[3])
    
         
    




