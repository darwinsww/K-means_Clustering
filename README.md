# Using K-means Clustering to Learn Representations for Image Classification

## References
Coates, A. & Ng, A. Y. (2012). Learning feature representations with k-means.
In Neural networks: Tricks of the trade (pp. 561{580). Springer.

Frank, E., Hall, M. A. & Witten, I. H. (2016). The WEKA Workbench. On-
line Appendix for "Data Mining: Practical Machine Learning Tools and
Techniques", Morgan Kaufmann, Fourth Edition, 2016.


@incollection{DBLP:series/lncs/CoatesN12,
  author    = {Adam Coates and
               Andrew Y. Ng},
  title     = {Learning Feature Representations with K-Means},
  booktitle = {Neural Networks: Tricks of the Trade - Second Edition},
  pages     = {561--580},
  year      = {2012},
  crossref  = {DBLP:series/lncs/7700},
  url       = {https://doi.org/10.1007/978-3-642-35289-8_30},
  doi       = {10.1007/978-3-642-35289-8_30},
  timestamp = {Tue, 16 May 2017 14:24:28 +0200},
  biburl    = {https://dblp.org/rec/bib/series/lncs/CoatesN12},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@book{DBLP:series/lncs/7700,
  editor    = {Gr{\'{e}}goire Montavon and
               Genevieve B. Orr and
               Klaus{-}Robert M{\"{u}}ller},
  title     = {Neural Networks: Tricks of the Trade - Second Edition},
  series    = {Lecture Notes in Computer Science},
  volume    = {7700},
  publisher = {Springer},
  year      = {2012},
  url       = {https://doi.org/10.1007/978-3-642-35289-8},
  doi       = {10.1007/978-3-642-35289-8},
  isbn      = {978-3-642-35288-1},
  timestamp = {Tue, 16 May 2017 14:24:27 +0200},
  biburl    = {https://dblp.org/rec/bib/series/lncs/7700},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@book{weka_book,
  author = {Ian H. Witten and Eibe Frank and Mark A. Hall},
  title = {Data Mining: Practical Machine Learning Tools and Techniques},
  publisher = {Morgan Kaufmann},
  year = 2011,
  address = {Burlington, MA},
  edition = 3,
  http = {http://www.cs.waikato.ac.nz/~ml/weka/book.html}
}


## Training Data
- [mnist](https://www.cs.waikato.ac.nz/ml/521/assignment1/mnist.tar.gz)
- [svhn](https://www.cs.waikato.ac.nz/ml/521/assignment1/svhn.tar.gz )
- [cifar-10](https://www.cs.waikato.ac.nz/ml/521/assignment1/cifar-10.tar.gz)
- [fashion-mnist](https://www.cs.waikato.ac.nz/ml/521/assignment1/fashion-mnist.tar.gz)


## Dependency
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/downloading.html)  
Downlard ```"waka-x.x.x.zip"``` in the section ```"Other platforms (Linux, etc.)"```.   
Unzip the zip file and you will find the necessary jar packages. I used the ```"weka-3-8-2"``` here.


## Step-by-step instructions for setting up an IntelliJ project
1. Start ```"IntelliJ"```
2. Select ```"Create New Project"```  
3. Select ```"Next"```  
4. Select ```"Next"```  
5. Enter ```"K-means_Clustering"``` as the project name
6. Select ```"Finish"```
7. Select ```"Project Structure"``` from the ```"File"``` menu
8. Select ```"Libraries"```
9. Click on ```"+"``` and choose ```"Java"```
10. Navigate to the ```"weka.jar"``` file from the WEKA distribution you have downloaded and select it  
11. Use the same way to add the ```"weka-src.jar"```, but untick ```"src/test/java"```
12. Expand the ```"weka.jar"``` file, then you will find the following three jar files in the ```"weka-src/lib"```:  
    ```
    mtj.jar  
    arpack_combined_all.jar   
    core.jar  
    ```
13. Add them all in the same way
14. Final ```"Project Structure"``` should be like this:  
    ![image](https://github.com/darwinsww/K-means_Clustering/blob/master/img/Project_structure.png)  
15. Select ```"OK"```
16. Right-click on ```"src"``` under ```"K-means_Clustering"``` and select ```"New"``` -> ```"File"```
17. Enter ```"weka/filters/unsupervised/attribute/KMeansImageFilter.java"``` and choose ```"OK"```
18. Replace the blank ```"KMeansImageFilter.java"``` with the file in this repo
19. Select ```"Edit Configurations"``` from the ```"Run"``` menu
20. Add a new ```"Run/Degbu Configuration"``` according to the following screenshot  
![image](https://github.com/darwinsww/K-means_Clustering/blob/master/img/Snapshot_of_the_project_configuration_in_IntelliJ.png)   
21. Select ```"Run 'KMeansImageFilter'"``` from the ```"Run"``` menu
22. Check that output is given in the terminal
23. Finally, you will find two outputs in the out folder:  
    ```
    KMeansImageFilter.class  
    KMeansImageFilter$1MyPanel.class  
    ```

    
