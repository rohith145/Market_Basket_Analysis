import pandas as pd #for using csv file in a simple way
import matplotlib.pyplot as plt #for plotting graphs
from sklearn.cluster import KMeans #to use the builtin package from sklearn


dataset=pd.read_csv("Mall_Customers.csv") #reading the csv file

x=dataset.iloc[:,[3,4]].values #as we are concerned only about the income and shopping score


inertia_list=[] #this has the summation of distances 
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=9)
    #n_clusters=i means we are choosing "i" as the number of clusters
    #init is the method of initialisation default is kmeans++
    #the maximum number of iterations that would be done if we didn't converge with choosen centroids
    #n_int this is the number of times we choose "i" clusters randomly
    kmeans.fit(x)
    #this will make i clusters
    inertia_list.append(kmeans.inertia_)
    #noting but the summation of distances of points in a cluster from cenntroid
    
plt.plot(range(1,11),inertia_list)
plt.title("Elbow Method")
plt.xlabel("number of clusters")
plt.ylabel("inerta_list")
plt.show()
#a graph for elbow method this is straight forward



kmeans=KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=9)
#the number of clusters is 5 from elbow method

y_means=kmeans.fit_predict(x)
#in fit method we calculate the kmeans clustering where as in fit_predict we assign each data point 
#the cluster it belongs to

#plotting the graph
color=["red","blue","green","cyan","magenta"]
for i in range(0,5):
    plt.scatter(x[y_means==i,0],x[y_means==i,1],s=100,c=color[i],label="cluster" +str(i))

#X[y_kmeans==i,0] selects all the rows where y_kmeans is equal to i and the first column [0] 
#X[y_kmeans==i,1] selects all the rows where y_kmeans is equal to i and the second column [1]
plt.title("cluster of clients")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()
#for the graph to show different clusters in different clusters as a scatter plot
