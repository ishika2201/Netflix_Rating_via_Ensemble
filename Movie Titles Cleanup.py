import csv

# Some of the movie titles span over multiple columns. This script makes sure all movie titles are one column.
with open("new_movie_titles.csv", 'w', newline='') as w:
        writer = csv.writer(w)
        writer.writerow(['MovieID', 'Release Year', 'Movie Name'])            
        with open("movie_titles.csv", 'r') as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                for i in range(3, len(line)) :
                    line[2] = line[2] + line[i]
                row = [line[0], line[1], line[2]]
                writer.writerow(row)
                     
                        

