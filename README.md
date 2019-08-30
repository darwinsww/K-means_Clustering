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
11. Use the same way to add the weka-src.jar, but untick "src/test/java"
12. Expand the weka.jar file, then you will find the following three jar files in the "weka-src/lib":  
    mtj.jar  
    arpack_combined_all.jar   
    core.jar  
13. Add them all in the same way
14. Final project structure should be like this:  
    ![image](https://github.com/darwinsww/K-means_Clustering/blob/master/img/Project_structure.png)  
15. Select "OK"
16. Right-click on "src" under "KMeansImageFilter" and select "New" -> "File"
17. Enter "weka/filters/unsupervised/attribute/KMeansImageFilter.java" and choose "OK"
18. Replace the blank KMeansImageFilter.java with the file in this repo
19. Select "Edit Configurations..." from the "Run" menu
20. Add a new configuration according to the following screenshot  
![image](https://github.com/darwinsww/K-means_Clustering/blob/master/img/Snapshot_of_the_project_configuration_in_IntelliJ.png)   
21. Select "Run 'KMeansImageFilter'" from the "Run" menu
22. Check that output is given in the terminal
