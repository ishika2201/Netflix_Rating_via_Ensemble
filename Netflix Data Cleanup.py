import time
import pandas as pd
import csv

files = ['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']

# print(dfs)
count = 0
start = time.time()

# The netflix data is originally in 4 text files, and not organized in an accessible way. This script removes each movie ID line and appends each data point with it's respective movie ID.

with open("new_netflix_data.csv", 'w', newline='') as w:
        writer = csv.writer(w)
        writer.writerow(['CustId', 'Rating', 'Date', 'MovieId'])
    
        for file in files:            
            movie_id = None
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        movie_id = line.split(":")[0]
                    else:
                        line_components = line.split(",")
                        cust_id = line_components[0]
                        rating = line_components[1]
                        date = line_components[2]
                        row = [cust_id, rating, date, movie_id]
                        writer.writerow(row)
                        count+=1
                        current = time.time()
                        elapsed = current - start
                        average = elapsed / count
                        amount_left = 26000000 - count
                        seconds_left = amount_left * average
                        print(seconds_left/60)
                        

