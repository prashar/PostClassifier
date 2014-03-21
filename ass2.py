from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import label_binarize
from numpy import * 
from scipy import *
from sklearn import tree
from sklearn import svm 
from os import system
from sklearn import linear_model

X = array([[0,2,4],[0,1,0],[0,2,1],[1,0,4],[1,3,5],[1,1,5],[1,4,5],[2,3,4],[2,2,3],[3,5,2],[3,1,0]])
Y = array([1,0,0,0,1,0,1,1,0,1,0])
X1 = label_binarize(X[:,:1], classes=[0,1,2,3])
X2 = label_binarize(X[:,1:2], classes=[0,1,2,3,4,5])
X3 = label_binarize(X[:,2:3], classes=[0,1,2,3,4,5])
X1_app_X2 = append(X1,X2,1)
bin_data = append(X1_app_X2,X3,1)
bin_data_2 = append(bin_data,X3,1)
clf = BernoulliNB()
clf.fit(bin_data,Y)

print "Predict Input Data" 
for i in range(0,10):
  print(clf.predict(bin_data[i]))

## Predict First Question
X4 = label_binarize([2],classes=[0,1,2,3])
X5 = label_binarize([2],classes=[0,1,2,3,4,5])
X6 = label_binarize([4],classes=[0,1,2,3,4,5])
X4_app_X5 = append(X4,X5,1)
q1 = append(X4_app_X5,X6,1)
q1_2 = append(q1,X6,1)
print "predict q1"
print clf.predict(q1)
print clf.predict_proba(q1)

## Predict Second Question
X7 = label_binarize([0],classes=[0,1,2,3])
X8 = label_binarize([2],classes=[0,1,2,3,4,5])
X9 = label_binarize([5],classes=[0,1,2,3,4,5])
X7_app_X8 = append(X7,X8,1)
q2 = append(X7_app_X8,X9,1)
q2_2 = append(q2,X9,1)
print "predict q2"
print clf.predict(q2)
print clf.predict_proba(q2)

print "--------"

clf.fit(bin_data_2,Y)
print clf.predict(q1_2)
print clf.predict(q2_2)

print "--------------------"

data = array([[24,40000],[53,52000],[23,25000],[25,77000],[32,48000],[52,110000],[22,38000],[43,44000],[52,27000],[48,65000]])
output = array([1,-1,-1,1,1,1,1,-1,-1,1])
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(data,output)
dotfile = open("ans2.dot",'w')
dotfile = tree.export_graphviz(clf,out_file=dotfile)
dotfile.close()

regr = linear_model.LinearRegression()
regr.fit(data,output)
print regr.coef_

# -------------
# Q3.2 
# -------------
clf = svm.SVC(kernel='linear')
clf.fit(data,output)
w = clf.coef_[0]
print w[0],w[1],clf.intercept_[0]
income_coef = -1 * w[1]/clf.intercept_[0] 
age_coef = -1 * (w[1]/(clf.intercept_[0])) * (w[0]/w[1])
print income_coef,age_coef

dims = shape(data)
print "Check all points" 
for i in range(dims[0]):
    print age_coef*(data[i][0]) + income_coef*(data[i][1]) -1.0

print " Code to print all possible information gains " 
data_new = array([[24,40000,1],[53,52000,-1],[23,25000,-1],[25,77000,1],[32,48000,1],[52,110000,1],[22,38000,1],[43,44000,-1],[52,27000,-1],[48,65000,1]])

# -------------
# The main function that does all the splits
# -------------
def splitall(data_new,feature):
    ind = lexsort((data_new[:,1-feature],data_new[:,feature]))
    r_data = data_new[ind] 
    negs = size(where(data_new[:,2] == -1))
    pos = size(where(data_new[:,2] == 1))
    total = float(negs+pos) 
    if(total == 0):
        print "Nothing to classify"
        return -1 
    else:
        if(negs == 0):
            part_1_before = 0 
        else:
            part_1_before = (-negs/total)*log2(negs/total) 
        if(pos == 0):
            part_2_before = 0 
        else:
            part_2_before =  (-pos/total)*log2(pos/total)

    entropy_before = part_1_before + part_2_before 
    print "entropy_before, ", entropy_before
    max_ig = []

    for i in r_data[:,feature]:
        s1 = r_data[where(r_data[:,feature] <= i)]
        s2 = r_data[where(r_data[:,feature] > i)]
        s1_size = shape(s1)[0]
        s2_size = shape(s2)[0]
        s1_negs = size(where(s1[:,2] == -1))
        s1_pos = size(where(s1[:,2] == 1))    
        s2_negs = size(where(s2[:,2] == -1))
        s2_pos = size(where(s2[:,2] == 1))
        prop_1 = 0. 
        prop_2 = 0. 
        s1_prob_1 = 0 
        s1_prob_2 = 0 
        s2_prob_1 = 0 
        s2_prob_2 = 0 

        if(s1_size != 0):
            if(s1_negs == 0):
                s1_prob_1 = 0 
            elif(s1_pos == 0):
                s1_prob_2 = 0 
            else:
                s1_prob_1 = s1_negs/float(s1_size) 
                s1_prob_2 = (s1_pos)/float(s1_size)
            prop_1 = float(s1_size)/(s1_size + s2_size)

        if(s2_size != 0):
            if(s2_negs == 0):
                s2_prob_1 = 0 
            elif(s2_pos == s2_size):
                s2_prob_2 = 0 
            else:
                s2_prob_1 = s2_negs/float(s2_size) 
                s2_prob_2 = (s2_pos)/float(s2_size)
            prop_2 = float(s2_size)/(s1_size + s2_size)

        print "For i", i
        if(s1_prob_1 == 0 ):
            ones_part_1 = 0 
        else:
            ones_part_1 = (-s1_prob_1*log2(s1_prob_1))
        
        if(s1_prob_2 == 0 ):
            ones_part_2 = 0 
        else:
            ones_part_2 = (-s1_prob_2*log2(s1_prob_2))

        if(s2_prob_1 == 0 ):
            twos_part_1 = 0 
        else:
            twos_part_1 = (-s2_prob_1*log2(s2_prob_1))
        
        if(s2_prob_2 == 0 ):
            twos_part_2 = 0 
        else:
            twos_part_2 = (-s2_prob_2*log2(s2_prob_2))

        one_s =  (prop_1) * ((ones_part_1) + (ones_part_2))
        two_s =  (prop_2) * ((twos_part_1) + (twos_part_2))
        e_after = one_s + two_s 
        ig = entropy_before - e_after
        print "Entropy after", e_after
        print "IG:", ig
        max_ig.append((i,ig))
    total_arr = array(max_ig,dtype=float)
    print total_arr
    idx = argmax(total_arr[:,1])
    print "MAX IG ",total_arr[idx]
