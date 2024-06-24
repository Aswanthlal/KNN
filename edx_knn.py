import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

##K-Nearest Neighbors is a supervised learning algorithm. 
#Where the data is 'trained' with data points corresponding to their classification. 
#To predict the class of a given data point, it takes into account the classes of the 'K' nearest data points and 
#chooses the class in which the majority of the 'K' nearest data points belong to as the predicted class.

#loading dataset
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()

##telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. 
#If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. 
#It is a classification problem. That is, given the dataset,  with predefined labels, we need to build a model to be used to predict class of a new or unknown case.


#data visualization and analysis
df['custcat'].value_counts()
df.hist(column='income',bins=50)
plt.show()

df.columns
#To use scikit-learn library, convert the Pandas data frame to a Numpy array:
x=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed','employ', 'retire', 'gender', 'reside']].values.astype(float)
x[0:5]

y=df['custcat'].values
y[0:5]

#Normalize data
#Data Standardization gives the data zero mean and unit variance,
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x[0:5]

#Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print('Train set:', x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)


#Classification
#KNN
from sklearn.neighbors import KNeighborsClassifier

#training
k=4
neigh=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
neigh

#predicting
yhat=neigh.predict(x_test)
yhat[0:5]

#Accuracy evaluation
#(how closely the actual labels and predicted labels are matched in a test set)
from sklearn import metrics
print('Train set accuracy:',metrics.accuracy_score(y_train,neigh.predict(x_train)))
print('Test set accuracy:', metrics.accuracy_score(y_test,yhat))

#building a model with k=6
k=6
neigh6=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
yhat6=neigh6.predict(x_test)
print('Train set accuracy:',metrics.accuracy_score(y_train,neigh6.predict(x_train)))
print('Test set accuracy:', metrics.accuracy_score(y_test,yhat6))


#calculate the accuracy of KNN for different values of k.
ks=10
mean_acc=np.zeros((ks-1))
std_acc=np.zeros((ks-1))

for n in range(1,ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc

#Plotting the model accuracy for a different number of neighbours 
plt.plot(range(1,ks),mean_acc,'g')
plt.fill_between(range(1,ks),mean_acc-1*std_acc,mean_acc+1*std_acc,alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
