# Using K-means Clustering to Learn Representations for Image Classification

## Description
The k-means clustering algorithm, fast and relatively simple to implement, is one of the oldest algorithms in machine learning. It is still surprisingly useful and widely used in practice. One particularly interesting application is its use for feature extraction from images. The extracted features enable transformation of images into a representation that is suitable for standard classification
algorithms, yielding a fast and simple method for image classification.  

The basic method for extracting features from images using k-means is simple: Split the training images into equal-size "patches", represent each patch as a vector of numbers (consisting, for example, of RBG values), and cluster the vectors using k-means. The nal cluster centres, which can be viewed as representatives of the collection of image patches, are the extracted features.  

Once features (i.e., cluster centres) have been extracted, we can use each of them to process an image by using it in a filter that is run across the image: We calculate the cross-correlation between the feature and corresponding subregions of the image as we slide the filter across the image, and push the output of the cross-correlation operation, a single number for each subregion, through a nonlinear ("activation") function. The output of the filter - a "feature map" - will be a new "image" showing how well the feature aligns with each subregion of the input image. Note that there will be one feature map (corresponding to a "channel") for each cluster centre. The set of filters is called a "filter bank".  

Each feature map can be represented as a high-dimensional vector, just like the data from a colour channel in the original input image, and we can concatenate the vectors for all feature maps to make a very high-dimensional vector that we can use as a training instance (also called "feature vector") for a standard classification algorithm. However, in practice, we need to reduce dimensionality to make this process workable. There are two mechanism for doing this: 1) When a filter is run across an image, we can specify a step size|
also called "stride"|so that the size of the resulting feature map is reduced, and 2) after the feature maps have been created, we can reduce their size by splitting them into regions and pooling values from each region|for example, we can simply average all the numeric values in each region.  

It turns out that applying this process to raw image data does not work that well. To make the process work better, the input values from each raw image are normalised and decorrelated using a process called "whitening". The paper (Coates & Ng, 2012) describes all this in detail. The pseudo code for the preprocessing steps and the clustering process are given on page 6 of this paper. Note that the paper applies a variant of k-means called "spherical" k-means, which ensures that the vectors representing each cluster centre all have length 1. Section 4 of the paper describes how to process images based on the output "dictionary" of k-means|each cluster centre is viewed as a code word in a dictionary.  

The task in this project is to implement the method from (Coates & Ng, 2012) as a WEKA pre-processing tool (aka WEKA "filter") (Frank, Hall & Witten, 2016), but the deep learning method in Section 5 of the paper is not implemented. The filter should read a dataset with file names and one other attribute, the class attribute. The files contain images. The output of the filter needs to be the processed form of the images, suitable for other WEKA learning algorithms. The file names will be given as values of a string attribute. The class attribute can be nominal or numeric.


## References
Coates, A. & Ng, A. Y. (2012). Learning feature representations with k-means. In Neural networks: Tricks of the trade (pp. 561-580). Springer.

Frank, E., Hall, M. A. & Witten, I. H. (2016). The WEKA Workbench. On-line Appendix for "Data Mining: Practical Machine Learning Tools and Techniques", Morgan Kaufmann, Fourth Edition, 2016.

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

    
