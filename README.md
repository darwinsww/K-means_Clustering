# K-means_Clustering

## Dependency
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/downloading.html)  
Downlard waka-x.x.x.zip in "Other platforms (Linux, etc.)". Unzip the zip file and you will find the necessary jar packages. I used the weka-3-8-2 here.


## Step-by-step instructions for setting up an IntelliJ project
1. Start IntelliJ  
2. Select "Create New Project"  
3. Select "Next"  
4. Select "Next"  
5. Enter "K-means_Clustering" as the project name
6. Select "Finish"
7. Select "Project Structure" from the "File" menu
8. Select "Libraries"
9. Click on "+" and choose "Java"
10. Navigate to the weka.jar file from the WEKA distribution you have downloaded and select it
11. Select "OK"
12. Click on "+" at the bottom of the panel on the right.
13. Navigate to the weka-src.jar file from the WEKA distribution and select it.
14. Untick "src/test/java" and select "OK"
15. Select "OK"
16. Right-click on "src" under "KMeansImageFilter" and select "New" -> "File"
17. Enter "weka/filters/unsupervised/attribute/KMeansImageFilter.java" and choose "OK"
18. Replace the blank KMeansImageFilter.java with the file in this repo
19. Select "Edit Configurations..." from the "Run" menu
20. Add a new configuration according to the following screenshot



21. Select "Run 'KMeansImageFilter'" from the "Run" menu
22. Check that output is given in the terminal


