import pandas as pd
import csv
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import f_oneway
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances



movies = pd.read_csv('/Users/maya/Desktop/netflix_price.csv')
movies['movie_id'] = movies['movie_id'].astype(np.uint32,errors='ignore')
print("Movies_head netflix price csv \n")
print(movies.head())

movies.info()
pd.pivot_table(movies,values=['rating'],index =['movie_id'],aggfunc=['mean','count'])
pd.pivot_table(movies,values=['rating'],index =['movie_id','user_id'],aggfunc=['mean','count'])

# Movie Titles Dataset
with open('/Users/maya/Desktop/download/movie_titles.txt','r',encoding = 'ISO-8859-1') as w:
  #print(w.read())
  data = [line.strip().split(',') for line in w]

title = pd.DataFrame(data)
title = title.drop(columns = [3,4,5])
title = title.rename(columns={0:'movie_id',1: 'year',2: 'title'})
title['movie_id'] = title['movie_id'].apply(int)
print(title.head())
title.drop_duplicates(subset=['title', 'year'], inplace=True)

# Merging Nmovies dataset with movies
nmovies = movies.merge(title,on=['movie_id'])
nmovies = nmovies.sort_values(by=['user_id'])
nmovies.rename(columns={'year': 'release_date'}, inplace=True)
print("Printed nmovies \n")
print(nmovies.head())

# Creating cound and mean ratings
pivot_table = pd.pivot_table(nmovies, values='rating', index='movie_id', aggfunc=['mean', 'count'])
pivot_table.columns = pivot_table.columns.get_level_values(1)
merged_df = nmovies.merge(pivot_table, on='movie_id')
merged_df.columns = ['Unnamed: 0', 'user_id', 'rating', 'date', 'movie_id', 'release_date', 'title', 'mean_rating', 'count_rating']
print(merged_df.head())

movie_ratings_count = nmovies.groupby('movie_id')['rating'].count()



# New code here
# Create a ratings matrix

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.stats import f_oneway

movie_ratings_count = nmovies.groupby('movie_id')['rating'].count()

# Sort the movies based on the number of ratings in descending order
sorted_movies = movie_ratings_count.sort_values(ascending=False)
sorted_movies

top_n = 20  # Change this number to select a different number of movies

top_movies = sorted_movies.head(top_n)  # Select the top N movies by rating count
top_movies_ids = top_movies.index.tolist()  # Get the movie IDs of the top N movies

# Slice the dataset to include only the top N movies
filtered_data = nmovies[nmovies['movie_id'].isin(top_movies_ids)]
print("filtered data \n")
print(filtered_data.head())

movie_features_df=filtered_data.pivot_table(index='user_id',columns='movie_id',values='rating').fillna(0)
movie_features_df.head(100)

movie_feat=filtered_data[:20000]
print("filtered movie \n")
print(movie_feat.head)
print(movie_feat.shape)


# Select relevant features for clustering (user_id, movie_id, rating)
features = movie_feat[['user_id', 'movie_id', 'rating']]

# Split the data into training and test sets
train_data, test_data = train_test_split(features, test_size=0.2, random_state=42)

# Create a ratings matrix for both training and test sets
train_ratings_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Convert the ratings matrix to a NumPy array
train_ratings_matrix_array = train_ratings_matrix.values

# Apply k-means clustering on the training set
num_clusters = 100  # Adjust the number of clusters based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
train_data['cluster'] = kmeans.fit_predict(train_data[['user_id', 'movie_id', 'rating']])

# Perform ANOVA on the training set clusters
# Perform ANOVA on the training set clusters
most_significant_cluster = None
min_p_value = float('inf')  # Initialize with a large value

for cluster in range(num_clusters):
    cluster_ratings = train_data[train_data['cluster'] == cluster]['rating']
    
    # Check if there are at least two unique ratings within the cluster
    if len(cluster_ratings.unique()) > 1:
        print(f"ANOVA for Cluster {cluster} (Training Set):")
        f_statistic, p_value = f_oneway(cluster_ratings, train_data['rating'])
        print(f"F-statistic: {f_statistic}, p-value: {p_value}")
        
        # Update the most significant cluster if needed
        if p_value < min_p_value:
            min_p_value = p_value
            most_significant_cluster = cluster
        
        print()
    else:
        print(f"Insufficient data for ANOVA in Cluster {cluster} (Training Set)")

# Print the most significant cluster
print(f"The most significant cluster is Cluster {most_significant_cluster} with p-value {min_p_value}")

test_data['cluster'] = kmeans.predict(test_data[['user_id', 'movie_id', 'rating']])

# Merge test_data with the cluster means from the training set
test_data_with_means = test_data.merge(train_data.groupby('cluster')['rating'].mean().reset_index(), on='cluster', how='left')
test_data_with_means.rename(columns={'rating_x': 'actual_rating', 'rating_y': 'predicted_rating'}, inplace=True)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data_with_means['actual_rating'], test_data_with_means['predicted_rating']))
print(f"Root Mean Squared Error (RMSE): {rmse}")

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'train_data' contains the cluster information
plt.figure(figsize=(10, 6))

# Scatter plot of the data points with color representing clusters
sns.scatterplot(x='user_id', y='movie_id', hue='cluster', data=train_data, palette='viridis', s=50)
plt.title('Clustering Results')
plt.xlabel('User ID')
plt.ylabel('Movie ID')

# Add legend
plt.legend(title='Cluster', loc='upper right')
plt.show()
