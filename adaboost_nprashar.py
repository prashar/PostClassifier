from scipy import *
from numpy import *
from adaboost_test import *
import dstump as ds
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as la
#See http://scikit-learn.org/stable/modules/feature_extraction.html
import sklearn.feature_extraction as fe
import tok
from time import *

class HWQ32:
  def __init__(self,N=500,T=50):
    self.N = N
    self.T = T

  # ---------------------
  # Provided already .. 
  # ---------------------
  def TrainTwoLines(self):
    x_tl,y_tl = two_lines(self.N/2)
    return x_tl,y_tl

  # ---------------------
  # We need to split on the feature that gives us max classification
  # This function outputs the idx of the feature we need to split on. 
  # ---------------------
  def TrainFourClusters(self):
    x_tl,y_tl = four_clusters(self.N/4)
    return x_tl,y_tl

  # ---------------------
  # Used by adaboost to classify a certain feature column by a dv against y_out
  # ---------------------
  def classify(self, data, dim, dv, y_out,N=1):
    classn = where(data[:,dim]<dv,-1,1)
    ##print "in c", shape(classn)
    classn = classn.reshape(N)
    ##print "in c", shape(classn)

    ind = where(y_out!=classn,1,0)
    return classn, ind

  def plot_t_error(self,x,y,TrainOrValidate=0):
    plt.plot(x,y,'k-')
    if(TrainOrValidate == 0):
        plt.title("Training Error / AdaBoost Step")
        plt.ylabel("Training Error")
        plt.xlabel("Step")
    elif(TrainOrValidate == 1):
        plt.title("Validation Error / AdaBoost Step")
        plt.ylabel("Validation Error")
        plt.xlabel("Classifier")
    else:
        plt.title("Test Error / AdaBoost Step")
        plt.ylabel("Test Error")
        plt.xlabel("Classifier")

    #plt.axis([0,50,0,500])
    plt.grid()
    plt.show()

  def plot_alphas(self,x,y):
    plt.plot(x,y,'k-')
    plt.title("Abs(Alpha) / AdaBoost Step")
    plt.xlabel("Step")
    #plt.axis([0,50,0,500])
    plt.ylabel("alpha")
    plt.grid()
    plt.show()
  
  # ---------------------
  # We need to split on the feature that gives us max classification
  # This function outputs the idx of the feature we need to split on. 
  # ---------------------
  def splitonmaxarg(self,x_tl,y_tl,features,D_t,isSparse=0):
    ret = []
    pplus = sum(D_t * (y_tl > 0))
    for feature_i in range(features):
        (dv,err) = ds.stump_fit(x_tl[:,feature_i], y_tl, D_t, pplus)
        ret.append((feature_i,dv,err))
    a_ret = array(ret)
    arg = argmax(abs(0.5 - a_ret[:,2]))
    return a_ret[arg]

  # ---------------------
  # The main adaboost algorithm that I implemented.  
  # ---------------------
  def RunAdaBoost(self,x_tl,y_tl,features=1,N=500,T=50,d_i=None,isSparse=0):
    # Need to train the weighted data
    # Start with D_t set to 1/N
    if (d_i == None):
        D_t = ones((N,T+1),dtype=float) * 1/float(N)
    else:
        D_t = ones((N,T+1),dtype=float)
        D_t[:,0] = d_i
    cfs = zeros((2,T))
    err = zeros(T)
    errors = ones((N,T+1))
    alphas = zeros(T+1)
    poutput = zeros((T+1,N))
    po = zeros(T+1)

    # Deal with classification tasks
    if(isSparse):
        x_tl_dense = x_tl.todense()
    else:
        x_tl_dense = x_tl

    for step_t in range(0,T):

        ans = self.splitonmaxarg(x_tl,y_tl,features,D_t[:,step_t])
        cfs[0,step_t] = ans[0]
        cfs[1,step_t] = ans[1]


        # Need a decision variable for the Weighted_data
        # the stump_fit routine would give me that.
        print "dim,dv: ",  cfs[0,step_t],cfs[1,step_t]

        # The above returns the v for h_t, calculate the y_tilde for this variable.
        y_tilde = where(x_tl_dense[:,cfs[0,step_t]]>cfs[1,step_t],1,-1)
        #print shape(y_tilde), shape(y_tl)
        y_tilde = y_tilde.reshape(N)
        #print shape(y_tilde), shape(y_tl)
        errors[:,step_t] = where(y_tl!=y_tilde,1,0)

        # Get the e_t - Probably should normalize
        err[step_t] = sum(errors[:,step_t] * D_t[:,step_t])
        print "err[t]: ",err[step_t]

        # Stop if the err becoming too low or too large
        if(abs(err[step_t]) < 1e-30):
            self.plot_t_error(array(range(step_t)),po[0:step_t])
            return
        elif(abs(err[step_t]) >= 1.0):
            self.plot_t_error(array(range(step_t)),po[0:step_t])
            return

        # Get the alpha_t
        alphas[step_t] = 0.5 * log((1-err[step_t])/err[step_t])

        # Update D_t+1
        D_t[:,step_t+1] = D_t[:,step_t] * exp( alphas[step_t] * (2*(errors[:,step_t]) - 1))
        D_t[:,step_t+1] = D_t[:,step_t+1]/sum(D_t[:,step_t+1])
        print "Alpha_t,e_t",alphas[step_t],err[step_t]

        # Calculate the total number of errors till this point. This is an aggregate result
        outputs = zeros((N,step_t+1))
        if(step_t == 0):
            ans,errs = self.classify(x_tl_dense,cfs[0,0],cfs[1,0],y_tl,N)
            outputs[:,0] = ans
        else:
            for i in range(step_t):
                outputs[:,i],f_errors  = self.classify(x_tl_dense,cfs[0,i],cfs[1,i],y_tl,N)
        for n in range(N):
            poutput[step_t,n] = sum(alphas[:step_t+1]*outputs[n,:])

        poutput[step_t,:] = where(poutput[step_t,:]>0,1,-1)
        po[step_t] = shape(where(poutput[step_t,:]!=y_tl))[1]
        print "---# of Misclassifications for:  ", step_t, " : " , po[step_t]

    print po
    print alphas
    self.plot_t_error(array(range(T)),po[0:T]/float(N))
    self.plot_alphas(array(range(T)),abs(alphas[0:T]))

    outputs = zeros((N,shape(D_t)[1]))
    for t in range(T):
        outputs[:,t],errors  = self.classify(x_tl_dense,cfs[0,t],cfs[1,t],y_tl,N)

    output = zeros(N)
    for n in range(N):
        output[n] = sum(alphas*outputs[n,:])
    ans = where(output > 0, 1, -1)
    print "mistakes: ", sum(y_tl != ans)

    return alphas,cfs,D_t

  # ---------------------
  # Sort out the similarity of the post
  # ---------------------
  def classify_post_output(self,x_tl,alphas,cfs,y_tl,N,T):
    outputs = zeros((N,T+1))
    for t in range(T):
        outputs[:,t],errors  = self.classify(x_tl,cfs[0,t],cfs[1,t],y_tl,N)

    output = zeros(N)
    for n in range(N):
        output[n] = sum(alphas*outputs[n,:])
    ans = where(output > 0, 1, -1)
    p_ones = (ans == 1).sum()
    p_negones = (ans == -1).sum()
    print "+1 count: ", p_ones, "|  -1 count: ", p_negones

  # ---------------------
  # Predict validation error
  # ---------------------
  def predict_validation_errors(self,x_tl,y_tl,alphas,cfs,N=500,T=20,ValidateOrTest=1):
    if(sp.issparse(x_tl)):
        print "Changing to dense"
        x_tl = x_tl.todense()

    poutput = zeros((T+1,N))
    errors = ones((N,T+1))
    po = zeros(T+1)
    for step_t in range(T):
        print "Splitting on .. "
        print cfs[0,step_t], cfs[1,step_t]
        y_tilde = where(x_tl[:,cfs[0,step_t]]>cfs[1,step_t],1,-1)
        y_tilde = y_tilde.reshape(N)
        errors[:,step_t] = where(y_tl!=y_tilde,1,0)

        outputs = zeros((N,step_t+1))
        if(step_t == 0):
            ans,errs = self.classify(x_tl,cfs[0,0],cfs[1,0],y_tl,N)
            outputs[:,0] = ans
        else:
            for i in range(step_t+1):
                outputs[:,i],f_errors  = self.classify(x_tl,cfs[0,i],cfs[1,i],y_tl,N)

        for n in range(N):
            test1 = alphas[:step_t+1]
            test2 = outputs[n,:]
            test3 = test1*test2
            test4 = sum(test3)
            poutput[step_t,n] = sum(alphas[:step_t+1]*outputs[n,:])
        #print poutput[step_t]

        poutput[step_t,:] = where(poutput[step_t,:]>0,1,-1)
        #print sum(poutput[step_t,:])

        po[step_t] = shape(where(poutput[step_t,:]!=y_tl))[1]
        print "---After step:  ", step_t, " : " , po[step_t]

    if(ValidateOrTest):
        self.plot_t_error(array(range(T)),po[0:T]/float(N),TrainOrValidate=1)
    else:
        self.plot_t_error(array(range(T)),po[0:T]/float(N),TrainOrValidate=2)

    print po[0:T]
    idx = argmin(po[0:T])
    if(ValidateOrTest):
        mistakes = min(po[0:T])
        print "Validation T* @: ", idx, " : ", mistakes, " , v.error = ", mistakes/float(N)
    else:
        print "Final Test Error ", po[T-1]/float(N)
    return idx
  
  # -------------------
  # This function classifies all the posts in the corpus
  # It splits on .6/.2/.2 (train/validate/test) scheme
  # as well as prints all the words, the comparison between the posts. 
  # -------------------
  def classifyposts(self,N,T):

    #Read text, try removing comments, headers ... See tok.py for implementation.
    corpus = tok.fill_corpus(["alt.atheism", "comp.windows.x"])

    #Create training data
    ctr = reduce(list.__add__, map(lambda x: x[:600], corpus))
    ytr = zeros(len(ctr)); ytr[:600] = -1; ytr[600:] = 1

    #Train a bag-of-words feature extractor.
    #You're free to play with the parameters of fe.text.TfidfVectorizer, but your answers
    #*should be* answered for the parameters given here. You can find out more about these
    #on the scikits-learn documentation site.
    tfidf = fe.text.TfidfVectorizer(min_df=5, ngram_range=(1, 4), use_idf=True, encoding="ascii")

    #Train the tokenizer.
    ftr = tfidf.fit_transform(ctr)
    ftr = ftr.tocsc()
    alphas,cfs,dt = self.RunAdaBoost(ftr,ytr,features=11808,N=N,T=T,d_i=None,isSparse=1)
    print "Round 1: ", cfs[0,:]

    #This maps features back to their text.
    feature_names = tfidf.get_feature_names()
    for i in cfs[0,:]:
        print i,":",feature_names[int(i)]
    #This shouldn't take more than 20m.
    #<Adaboost goes here>

    #Create validation data
    cva = reduce(list.__add__, map(lambda x: x[600:800], corpus))
    yva = zeros(len(cva)); yva[:200] = -1; yva[200:] = 1

    #tfidf tokenizer is not trained here.
    fva = tfidf.transform(cva).tocsc()

    #<Validation code goes here>
    idx = self.predict_validation_errors(fva,yva,alphas,cfs,400,T)
    print "idx returned was ", idx

    #Create test data
    #Some lists have less than a thousand mails. You may have to change this.
    cte = reduce(list.__add__, map(lambda x: x[800:], corpus))
    yte = zeros(len(cte)); yte[:200] = -1; yte[200:] = 1

    fte = tfidf.transform(cte).tocsc()
    shape_t = shape(fte)[0]
    if(shape_t != 400):
        print shape_t
    self.predict_validation_errors(fte,yte,alphas,cfs,shape_t,idx,ValidateOrTest=0)

    paperlist = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
                  "misc.forsale","rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt",
                   "sci.electronics", "sci.med", "sci.space", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
                    "talk.religion.misc"]
    for i in paperlist:
        corpus = tok.fill_corpus([i])
        t_pred = reduce(list.__add__, map(lambda x: x[:1000], corpus))
        y_fake = zeros(len(t_pred)); y_fake[:500] = -1; y_fake[500:] = 1
        f_corpus = tfidf.transform(t_pred).tocsc()
        print "------ For posts in ", i
        self.classify_post_output(f_corpus.todense(),alphas,cfs,y_fake,1000,T)

# ---------------------
# HELPER FUNCTIONS TO KICK OFF DIFFERENT FUNCTIONALITIES
# ---------------------
def trainsets():
  trainer = HWQ32(500,50)
  x_tl,y_tl = trainer.TrainTwoLines()
  alphas,cfs,D_t = trainer.RunAdaBoost(x_tl,y_tl,2,500,50)
  ax_tl,ay_tl = trainer.TrainTwoLines()
  idx = trainer.predict_validation_errors(ax_tl, ay_tl, alphas, cfs, 500, 50)
  bx_tl,by_tl = trainer.TrainTwoLines()
  trainer.predict_validation_errors(bx_tl, by_tl, alphas, cfs, 500, idx, ValidateOrTest=0)

def trainclusters():
  trainer = HWQ32(500,50)
  x_tl,y_tl = trainer.TrainFourClusters()
  alphas,cfs,D_t = trainer.RunAdaBoost(x_tl,y_tl,2,500,50)
  ax_tl,ay_tl = trainer.TrainFourClusters()
  idx = trainer.predict_validation_errors(ax_tl, ay_tl, alphas, cfs, 500, 50)
  bx_tl,by_tl = trainer.TrainFourClusters()
  trainer.predict_validation_errors(bx_tl, by_tl, alphas, cfs, 500, idx, ValidateOrTest=0)

def trainposts():
  trainer = HWQ32(1200,30)
  trainer.classifyposts(1200,30)

# ---------------------
# MAIN()
# ---------------------
def main():
  #trainposts()
  #trainsets()
  trainclusters()

if __name__ == '__main__':
  main()

