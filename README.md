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


## Training Data
- [mnist](https://www.cs.waikato.ac.nz/ml/521/assignment1/mnist.tar.gz)
- [svhn](https://www.cs.waikato.ac.nz/ml/521/assignment1/svhn.tar.gz )
- [cifar-10](https://www.cs.waikato.ac.nz/ml/521/assignment1/cifar-10.tar.gz)
- [fashion-mnist](https://www.cs.waikato.ac.nz/ml/521/assignment1/fashion-mnist.tar.gz)


## Dependencies
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/downloading.html)  
Downlard ```"waka-x.x.x.zip"``` in the section ```"Other platforms (Linux, etc.)"```.   
Unzip the zip file and you will find the necessary jar packages. Here used the ```"weka-3-8-2.zip"```.   

- [netlibNativeLinux](netlibNativeLinux) Here used ```"netlibNativeLinux1.0.2"```    


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
    
## Compile & Package
If you don't want to use IntelliJ, following method could be an alternative:
- [Create Folders]: 
```
mkdir -p ~/ml

cd ml
mkdir netlibNativeLinux1.0.2
mkdir lib-stable-3-8-2
mkdir weka-3-8-2
mkdir K-means_Clustering

cd K-means_Clustering
mkdir -p ./data
mkdir -p ./src/weka/filters/unsupervised/attribute
mkdir -p ./out/production/K-means_Clustering
```

- [Copy Files]:   
Copy ```"netlibNativeLinux1.0.2.zip"``` to the folder ```"~/ml/netlibNativeLinux1.0.2"```   
Copy ```"weka-3-8-2.zip"``` to the folder ```"~/ml/weka-3-8-2"```   
Copy ```"mtj.jar"``` and ```"weka.jar"``` to the folder ```"~/ml/lib-stable-3-8-2"```      
Copy ```"mnist.tar.gz"``` to the folder ```"~/ml/K-means_Clustering/data"```     

- [Extraction]:
```
cd ~/ml/weka-3-8-2
unzip weka-3-8-2.zip

cd ~/ml/netlibNativeLinux1.0.2
unzip netlibNativeLinux1.0.2.zip

cd ~/ml/K-means_Clustering/data
tar -zxvf mnist.tar.gz
```

- [Compile]:  
```
cd ~/ml/K-means_Clustering
javac -classpath "~/ml/weka-3-8-2/weka.jar:~/ml/weka-3-8-2/mtj.jar" -d ./out/production/K-means_Clustering ./src/weka/filters/unsupervised/attribute/KMeansImageFilter.java
```

- [Package]:  
```
cd ~/ml/K-means_Clustering
jar -cvf KMeansImageFilter.jar ./out/production/K-means_Clustering/weka/filters/unsupervised/attribute/*.class
```


## Experiments
A lot of experiments had been done, following is an example command runs the test:
```
cd ~/ml/K-means_Clustering
java -Xmx12g -cp "~/ml/lib-stable-3-8-2/*:~/ml/netlibNativeLinux1.0.2/lib/*:out/production/means_Clustering/weka/filters/unsupervised/attribute/" \
weka.Run .FilteredClassifier -o -v -t ./data/mnist/training.arff -T ./data/mnist/testing.arff \
-F ".KMeansImageFilter -D ./data//mnist/ -Z 8 -N 1 -K 128 -T 4 -P 2 -S 0" \
-W .MultiClassClassifier -- -M 3 -W .SGD -- -N -M -F 0 -L 0.0001 -E 100 >> mnist-K128N1-r1.20 &
```

## Results
Several experiments have been conducted to test the performance of the k-means image filter. In these experiments, I used 4 difierent classification problems (mnist, cifar-10, fashion-mnist, and svhn), with difierent parameter settings for each of them. The explanations of the parameters are as follows:
-D: image directory, which specifies the directory of the input images.  
-S: seed of the random function, which specifies the positions of the patches and the coordinates of the initial centroids.  
-Z: patch size, which specifies the width/height of a patch truncated from an image.(Normally, the patch would be square.)  
-K: number of centroids/clusters.  
-N: number of patches per image, which specifies how many patches would be truncated from an image.  
-T: stride size, which specifies the size of the stride to use when creating features (both directions).  
-P: pool size, which specifies the size of the pool to use when creating features (both directions).  

For simplifying the experiments, the majority of the parameters above are fixed respectively in each classication problems. In the test of MNIST and FASHION-MNIST, which are with 28x28 pixel images, the settings of part parameters are: patch size = 8, stride = 4, pool size = 2; while in the test of CIFAR-10 and SVHN, which are with 32x32 pixel images, the corresponding settings are: patch size = 8, stride = 3, pool size = 3. Seed is set to 0 all the time.

So, the main parameters to vary are the number of clusters (-K) and the number of patches per image (-N). For comparison, I implemented 9 experiments with difierent combinations of K = 128, 500, 1000 and N = 1, 4, 8 for the 4 problems each.

By the experiments, the results are demonstrated in the following tables, which were generated using WEKA [2]. Unfortunately, I just could get the iteration times for convergence of SVHN when K = 1000 and N = 8. I failed to get its accuracy because the execution would spend such a long time, during which I encountered several times of connection lost.

### Accuracy
Regarding MNIST, FASHION-MNIST, and SVHN, the accuracies are nearly the same within their 9 experiments each. More precisely, yet, the accuracies gradually increase with the growth of K.   
As for CIFAR-10, the accuracy reached the bottom with the minimum of K. It reached the highest point when K = 500. It went a little bit down when K increased to 1000, however.
![image](https://github.com/darwinsww/K-means_Clustering/blob/master/img/Accuracy.png)

### Iteration times for convergence
For higher accuracy, I specified the error range of SSE < 0.01 in two iterations as the terminal condition of iteration.   
According to my experience, the iteration processes related to the same experiment are all the same if I used totally same parameters. In addition, the iteration times in the 4 problems each would generally increase with the growth of N, and decrease with the growth of K.
![image](https://github.com/darwinsww/K-means_Clustering/blob/master/img/Iteration_times_for_convergence.png)

### Time taken to test model on testing data (Unit: second)
The statistics may not be accurate since the computer is not exclusive when testing. However, it also could be regarded as an indicator of the efficiency of the algorithm. In each problem, the processing times with the same K are of the same order of magnitude. On the contrary, it would increase rapidly as the growth of K. This is because the bigger K is, the more features will be generated. Among all the situations, SVHN definitely took the longest time due to its larger size and higher complexity.
![image](https://github.com/darwinsww/K-means_Clustering/blob/master/img/Time_taken_to_test_model_on_testing_data.png)


## Conclusions
Images in the 4 problems have difierent characteristics. According to their overall accuracy, I ranked them as follows:
- [MNIST]: black-and-white images with numeric patterns.
- [FASHION-MNIST]: black-and-white images with difierent kinds of clothing.
- [SVHN]: colourful images with numeric patterns.
- [CIFAR-10]: colourful images with difierent types of objects.

To the same data scale, MNIST always has the highest accuracy, while CIFAR-10 has the lowest. FASHION-MNIST and SVHN, which the former
is a little bit higher than the latter, are both in the middle.

In conclusion, the k-means image filter algorithm I implemented is more suitable for extracting features from black-and-white images, especially the ones with numbers on it. On the contrary, the worst situation is a colourful image with non-numeric patterns. The reason is we transform RGB value into greyscale in the algorithm, which could result in distortion of the original image.


## References
- [1] Adam Coates and Andrew Y. Ng. Learning feature representations with kmeans. In Gregoire Montavon, Genevieve B. Orr, and Klaus-Robert Muller, editors, Neural Networks: Tricks of the Trade - Second Edition, volume 7700 of Lecture Notes in Computer Science, pages 561-580. Springer, 2012.

- [2] Ian H. Witten, Eibe Frank, and Mark A. Hall. Data Mining: Practical Machine Learning Tools and Techniques. Morgan Kaufmann, Burlington, MA, 3 edition, 2011.


