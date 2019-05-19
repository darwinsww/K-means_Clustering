package weka.filters.unsupervised.attribute;

import no.uib.cipr.matrix.*;
import no.uib.cipr.matrix.Matrix;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class KMeansImageFilter extends SimpleBatchFilter {
    /*
        Note: A (training) patch is a subregion of an image. We get "m_numPatchesPerImage" patches from one image, which
        every patch contains "m_cropSize x m_cropSize" pixels and its starting point is decided by a random number "m_Seed".
     */
    // The pixel width of a patch. Normally it's a square, so totally m_cropSize x m_cropSize pixels in this patch.
    protected int m_cropSize = 8;

    // A random starting point of a patch.
    protected int m_Seed = 0;

    // Specify the number of centroid/cluster
    protected int m_K = 10;

    // The variable is used to specify the number of images patches to randomly extract from a given image.
    // It is different from "m_FeatureNum". "m_numPatchesPerImage" is just used for calculating the final centroids.
    protected int m_numPatchesPerImage = 10;

    protected int m_Stride = 4;
    protected int m_Pooling = 2;

    // patches number in X/Y direction respectively when striding
    // X - Width, Y - Height
    protected int m_numX_stride = 0;
    protected int m_numY_stride = 0;

    // patches number in X/Y direction respectively when pooling
    // X - Width, Y - Height
    protected int m_numX_pooling = 0;
    protected int m_numY_pooling = 0;

    // Calculated based on the image size, the number of centroid "m_K", filter size "m_cropSize", stride "m_Stride", and pool size "m_Pooling".
    // It represents the number of features/attributes, also named as the length of the feacture vector.
    protected int m_FeatureNum = 0;

    // Specify the iteration times when we get the centroids (matrix D)
    protected int m_IterationTimes = 1000;

    protected String m_FeatureName = "X";
    protected static double m_Precision = 0.01;

    // saving the covariance matrix which had already made the eigenvalue decomposition
    protected Matrix m_eigendecomp = null;  // it is VxD_IxVt

    // saving all the final centroids
    protected Matrix m_D = null;

    protected double m_SSE = 0;

    // Full path of current project
    private String imageDirectory = System.getProperty("last.dir", System.getProperty("user.dir"));
    //protected String imageDirectory = "/home/ml/521/assignment1/mnist";

    public String globalInfo() {
        return "This filter performs feature extraction from images using the spherical k-means algorithm.";
    }

    protected String getFeatureNamePrefix() {
        return m_FeatureName;
    }

    // Get the number of features/attributes, also named as the length of the feature vector.
    protected int getNumFeatures(String fileName) {
        BufferedImage img = null;

        try {
            // Load the training image into a temporary variable "img"
            img = ImageIO.read(new File(fileName));
        } catch (IOException e) {
            System.err.println("File " + fileName + " could not be read");
        }

        // We assume all the input image are all square, however, I'd like to calculate its width and height
        int imgSizeX = img.getWidth();
        int imgSizeY = img.getHeight();

        if (imgSizeX < m_cropSize || imgSizeY < m_cropSize) {
            throw new IllegalArgumentException("Illegal Arguments: Stride size is larger than image size.");
        }

        // Calculate the number of features in the X & Y direction when striding.
        m_numX_stride = 1 + (int)Math.floor((imgSizeX - m_cropSize) / m_Stride);
        m_numY_stride = 1 + (int)Math.floor((imgSizeY - m_cropSize) / m_Stride);

        if (m_numX_stride % m_Pooling != 0 || m_numY_stride % m_Pooling != 0) {
            throw new IllegalArgumentException("Illegal Arguments: Pool size not compatible with raw features.");
        }

        // Calculate the number of features in the X & Y direction when pooling.
        m_numX_pooling = m_numX_stride / m_Pooling;
        m_numY_pooling = m_numY_stride / m_Pooling;

        // return the length of the feature vector (that is the number of features/attributes)
        m_FeatureNum = m_numX_pooling * m_numY_pooling * m_K;

        if (m_Debug) {
            System.out.println("m_cropSize = " + m_cropSize + "; imgSizeX = " + imgSizeX + "; imgSizeY = " + imgSizeY);
            System.out.println("m_Stride = " + m_Stride + "; m_numX_stride = " + m_numX_stride + "; m_numY_stride = " + m_numY_stride);
            System.out.println("m_Pooling = " + m_Pooling + "; m_numX_pooling = " + m_numX_pooling + "; m_numY_pooling = " + m_numY_pooling);
            System.out.println("m_FeatureNum = " + m_FeatureNum);
        }

        return m_FeatureNum;
    }

    // In this function:
    // Firstly, calculate the number of features (equals to the length of one feature vector)
    // Secondly, construct the header/name for the features, save them in the result instance to return.
    // refer to AbstractImageFilter.determineOutputFormat
    public Instances determineOutputFormat(Instances data) {
        Instances result = new Instances(data, 0);

        // Get the full name of one input image
        String file_full_path = imageDirectory + File.separator + data.instance(0).stringValue(0);;
        // get file name of current image
        //String fileName = data.instance(0).stringValue(0);

        try {
            // Get the number of the features.
            m_FeatureNum = getNumFeatures(file_full_path);

            // Check whether the images are all square and they all have the same width and height.
            int imgSize = -1;
            for (int i = 0; i < data.numInstances(); i++) {
                file_full_path = imageDirectory + File.separator + data.instance(i).stringValue(0);
                BufferedImage img = null;
                try {
                    img = ImageIO.read(new File(file_full_path));
                } catch (Exception ex) {
                    System.err.println("Could not load: " + file_full_path);
                }

                if (img.getWidth() != img.getHeight()) {
                    throw new IllegalArgumentException("Image " + file_full_path + " is not square.");
                }

                if (imgSize == -1) {
                    imgSize = img.getWidth();
                } else if (imgSize != img.getWidth()) {
                    throw new IllegalArgumentException("Image " + file_full_path + " has different size.");
                }
            }
        }
        catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            return new Instances(data, 0); // This just returns a copy of the input without any data instances
        }

        for (int index = m_FeatureNum - 1; index >= 0; index--) {
			result.insertAttributeAt(new Attribute(getFeatureNamePrefix() + index), 1);
		}

		// For using trainging data & testing data together. We need to remove attribute "file name".
		result.deleteAttributeAt(0);

		return result;
    }

    /**
     * Plots a vector as a grayscale image, assuming the image is square.
     *
     * @param v the vector to plot
     */
    protected void plotVector(Vector v) {

        class MyPanel extends JPanel {
            // for instance, vector v has 9 values (3x3), and current MyPanel has a dimension 9x6. Then we will divide
            // this panel into 9 pieces of dimension 3x2, and use v(i) to set colour of the ith piece.
            protected void paintComponent(Graphics g) {
                // invoke function paintComponent of base class.
                super.paintComponent(g);
                int dim = (int) Math.round(Math.sqrt(v.size()));

                // Returns the current width and height of this component by using getWidth() and getHeight().
                int xSize = getWidth() / dim;
                int ySize = getHeight() / dim;
                int x = 0;
                int y = -ySize;
                for (int i = 0; i < v.size(); i++) {
                    if (i % dim == 0) {
                        x = 0;
                        y += ySize;
                    }
                    int t = 127 + (int)(v.get(i) * 127); // Something more intelligent should be done here
                    if (t < 0) t = 0;
                    if (t > 255) t = 255;
                    g.setColor(new Color(t, t, t));
                    // The rectangle is filled using the graphics context's current color.
                    g.fillRect(x, y, xSize, ySize);
                    x += xSize;
                }
            }
        }

        MyPanel mainPanel = new MyPanel();
        // The main class for creating a dialog window.
        JDialog d = new JDialog();
        // specifies whether dialog blocks input to other windows when shown
        d.setModal(true);
        // Sets the operation that will happen by default when the user initiates a "close" on this dialog.
        d.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        // Appends the specified component to the end of this container. That is append mainPanel(MyPanel) to d(JDialog).
        d.add(mainPanel);
        // Set the size of this Dialog d by using the given width and height
        d.setSize(10 * (int) Math.round(Math.sqrt(v.size())), 2 + 10 * (int) Math.round(Math.sqrt(v.size())));
        // Sets the dialog d is not resizable by the user.
        d.setResizable(false);
        // Shows the Dialog d.
        d.setVisible(true);
    }

    /**
     * Returns a constant DenseVector with the given value in each slot.
     *
     * @param value the constant to use
     * @param length the length of the vector
     * @return DenseVector
     */
    protected DenseVector constantVector(double value, int length) {

        double[] v =  new double[length];
        // Assigns double value "value" to every element of the array "v"
        Arrays.fill(v, value);
        // Constructor for DenseVector
        // DenseVector, which represents a column vector, is a class in mtj.
        return new DenseVector(v);
    }

    // get the feature maps and create the final dataset
    protected Instances createNewInstances(Instances data) {
        // create the new dataset, get attributes names and amount, then save them in the Instances result.
        // This new instances "result" is the final dataset. The number of "feature vector"s in the result equals to
        // the amount of instances/original images.
        // Each "feature vector" corresponds to one instance(image), length of the "feature vector"(also called
        // elements amount in the "feature vector") is the attributes amount.
        Instances ret = new Instances(determineOutputFormat(data), 0);

        // "ret.attributes" equals to "data.numAttributes()" means that there is no other attributes added to the ret,
        // so there is nothing changed in function "determineOutputFormat".
        // Normally "data.numAttributes()" originally includes 2 attributes: "file name" and "class".
        // However we had already remove "file name" attribute in "determineOutputFormat", so in the following
        // if condition, we need to ues "data.numAttributes() - 1"
        if (ret.numAttributes() == data.numAttributes() - 1)
            return new Instances(data, 0); // This just returns a shallow copy of the input

        if (m_Debug) {
            System.out.println("Getting Features. Starts");
        }

        // Entrance of managing training data and test data.
        // iterate over the instances (first time is the training data and second is the test data), one instance(image) each time
        for (int i = 0; i < data.numInstances(); i++) {
            // Get the full name of one input image
            String file_full_path = imageDirectory + File.separator + data.instance(i).stringValue(0);

            BufferedImage img = null;
            try {
                img = ImageIO.read(new File(file_full_path));
            } catch (IOException e) {
                System.err.println("File " + file_full_path + " could not be read");
            }

            // extract features from the buffered image
            // here we stride, normalize, whitening, and pooling for every patch, in order to get the features of this image.
            double[] features = getFeatures(img);
            //double[] features = {i*10+i, i*20+i};

            // create "values" as the final "feature vector" for the instance/image
            double[] feature_vector = new double[ret.numAttributes()];

            // copy the features to the values array
            int values_index = 0;//1; // leave first value for the filename
            for (int index = 0; index < features.length; index++)
                feature_vector[values_index++] = features[index];

            // "data" is the input image/instance.
            // copy any additional attributes from the old instance to values
            for (int index = 1; index < data.numAttributes(); index++)
                feature_vector[values_index++] = data.instance(i).value(index);

            // create and an instance for the new dataset
            DenseInstance newInst = new DenseInstance(1, feature_vector);
            newInst.setDataset(ret);

            // add the filename
            //newInst.setValue(0, file_full_path);

            // done
            ret.add(newInst);
        }

        if (m_Debug) {
            System.out.println("Getting Features. Ends");
        }

        return ret;
    }

    // Get feature map(all the attributes/features) of the input image
    protected double[] getFeatures(BufferedImage img) {
        // used as a feature vector, which contains all the calculated features/attributes of the input image.
        double[] fVals = new double[m_FeatureNum]; // totally: m_numX_pooling * m_numY_pooling * m_K features
        int fIndex = 0;

        // patchLength: pixels per patch, that is the dimensions of a patch
        int patchLength = m_cropSize * m_cropSize;
        // numPatches: total number of training patches
        int numPatches_stride = m_numX_stride * m_numY_stride;

        Matrix normalizedX = new DenseMatrix(patchLength, numPatches_stride);

        // For each image, get its "numPatches_stride" subregions by striding, each subregion is regarded as a patch.
        for (int p = 0; p < numPatches_stride; p++) {
            // Picking a subregion(patch) from the orginal image by striding.
            int x = m_Stride * (p % m_numX_stride);
            int y = m_Stride * (p / m_numX_stride);
            if (m_Debug) {
                System.out.println("(x, y) = (" + x + ", " + y + ")");
            }

            BufferedImage patch = img.getSubimage(x, y, m_cropSize, m_cropSize);

            // Read from the patch above, and get the normalized vector corresponding.
            Vector normalizedVec = getNormalizdVector(patch);

            // Set the values of normalizedVec(as a column) into matrix X.
            // AbstractDenseMatrix.set(int row, int column, double value)
            for (int r = 0; r < normalizedVec.size(); r++) {
                normalizedX.set(r, p, normalizedVec.get(r));
            }
        }

        // Whitening Matrx X
        // For whitening the training data and testing data, just reuse the result of eigenvalue decomposition. T
        // The reason could be referred as the comment right after the calculation after the Matrix cov in function getCovMatrix.
        assert(m_eigendecomp != null);
        //Matrix whitenedX = getWhitenMatrix(normalizedX);

        // whitenedX: dimension of patchLength * numPatches_stride
        // m_D: dimension of patchLength * m_K
        // so whitenedXT * m_D(named as rawData in the following): dimension of numPatches_stride * m_K
        // Each column of the matrix rawData is regarded as a feature map(also called "channel" in the paper), on which
        // should do the operation the pooling.
        // Actually, one feature map(channel) represents the distance between all patches of an image and one
        // centroid of K-centroids. That's why we finally have k feature maps.

        // 	transAmult(Matrix B, Matrix C) means: C = AT * B, C is the return value.
        Matrix rawData = new DenseMatrix(numPatches_stride, m_K);
        //rawData = whitenedX.transAmult(m_D, rawData);  // Get feature map: rawData = whitenedXT * m_D;
        rawData = normalizedX.transAmult(m_D, rawData);

        // Iterate all m_K feature maps each by each: pooling them, and save them in the feature vector "fVals".
        for (int col = 0; col < rawData.numColumns(); col++) {
             // Operate one column in matrix rawData, which represents a feature map(also called channel in the paper, page 10).
            DenseVector col_vec = Matrices.getColumn(rawData, col);
            double[][] featuresRaw = new double[m_numY_stride][m_numX_stride];

            // For pooling, transform the above vector "col_vec", which actually is a feature map, into a 2-dimension array.
            for (int row = 0; row < rawData.numRows(); row++) {
                int i = row / m_numX_stride;
                int j = row % m_numX_stride;

                // Double.max(0, vals.get(j)) is a famous rectifier activation function used in neural networks,
                // which can substantially improve accuracy.
                featuresRaw[i][j] = Double.max(0, rawData.get(row, col));
                //featuresRaw[i][j] = rawData.get(row, col);
            }

            // Calculate pooled features
            if ((m_numX_stride % m_Pooling) != 0) {
                throw new IllegalArgumentException("Pool size not compatible with raw features.");
            }
            if ((m_numY_stride % m_Pooling) != 0) {
                throw new IllegalArgumentException("Pool size not compatible with raw features.");
            }

            // pooling
            for (int k = 0; k < m_numX_pooling; k++) {
                 int locX = k * m_Pooling;
                 for (int j = 0; j < m_numY_pooling; j++) {
                     int locY = j * m_Pooling;
                     double sum = 0;
                     for (int q = 0; q < m_Pooling; q++) {
                         for (int r = 0; r < m_Pooling; r++) {
                             sum += featuresRaw[locX + q][locY + r];
                         }
                     }
                     fVals[fIndex++] = sum / (m_Pooling * m_Pooling);
                 }
             }
        }

        return fVals;
    }

    // This is the function where mainly operations are executed.
    // Instances: public class Instances extends AbstractList...
    public Instances process(Instances data) {
        //System.out.println("Function process. Starts");

        if (m_Debug) {
            System.out.println("Image directory: " + imageDirectory);
            System.out.println("Patch size: " + m_cropSize);
            System.out.println("Patch number per image: " + m_numPatchesPerImage);
            System.out.println("K: " + m_K);
            System.out.println("Stride size: " + m_Stride);
            System.out.println("Pool size: " + m_Pooling);
            System.out.println("Seed: " + m_Seed);
        }

        // true if the first batch has been processed
        if (!isFirstBatchDone()) {
            System.out.println("Training data in 1st batch to get the m_D. Starts");
            // For getting centroids m_D.

            preprocess(data);

            if (m_Debug) {
                printDebugMsg("Print transpose of final matrix of centroids",
                        "Each line represents a centroid.", m_D, true);
            }
            System.out.println("Training data in 1st batch to get the m_D. Ends");
        }

        // totally done here, return new dataset
        Instances ret = createNewInstances(data);
        //System.out.println("Function process. Ends\n");

        return ret;
    }

    // reads images, extracts patches at random locations, fills up X with normalized data from the patches,
    // and calculates the covariance matrix from the centred version of X.
    // Finally get the m_K centroids(m_D).
    protected void preprocess(Instances data) {
        // We will need a random number generator
        Random rand = new Random(m_Seed);

        // Establish number of rows and columns for data matrix X
        // patchLength: pixels per patch, that is the dimensions of a patch
        int patchLength = m_cropSize * m_cropSize;
        // numPatches: total number of training patches
        int numPatches = m_numPatchesPerImage * data.numInstances();

        // Read image patches as greyscale, normalize patches, and turn them into columns in the matrix X.
        // Dense matrix: It is a good all-round matrix structure, with fast access and efficient algebraic operations.
        // The matrix is stored column major in a single array, as follows: a11	a21	a31	a41	a12	a22	a32	a42	a13	a23	a33	...
        Matrix normalizedX = new DenseMatrix(patchLength, numPatches);
        int colIndex = 0;

        System.out.println("Normalize patches of the input images. Starts");

        // Manipulate every image, by random collecting its "m_numPatchesPerImage" subregions.
        // In the end of this for loop, we will get the matrix X, which dimension is patchLength * numPatches.
        for (int i = 0; i < data.numInstances(); i++) {
            // Get the full name of one input image
            String file_full_path = imageDirectory + File.separator + data.instance(i).stringValue(0);

            BufferedImage img = null;
            try {
                // Load the training image into a temporary variable "img"
                img = ImageIO.read(new File(file_full_path));

                // Get the boundary when picking a subregion from the orginal image.
                int xmax = 1 + img.getWidth() - m_cropSize;
                int ymax = 1 + img.getHeight() - m_cropSize;
                // For each image, get its "m_numPatchesPerImage" subregions, each subregion represents a patch.
                for (int p = 0; p < m_numPatchesPerImage; p++) {
                    // Get a random subregion "patch" from the given image.
                    // Random.nextInt returns a pseudorandom between 0 (inclusive) and the specified value xmax/ymax (exclusive).
                    BufferedImage patch = img.getSubimage(rand.nextInt(xmax), rand.nextInt(ymax), m_cropSize, m_cropSize);

                    // Read from the patch above, and get the normalized vector corresponding.
                    Vector normalizedVec = getNormalizdVector(patch);

                    // Set the values of normalizedVec(as a column) into matrix X.
                    // AbstractDenseMatrix.set(int row, int column, double value)
                    for (int r = 0; r < normalizedVec.size(); r++) {
                        normalizedX.set(r, colIndex, normalizedVec.get(r));
                    }
                    colIndex++;
                }
            } catch (IOException e) {
                System.err.println("File " + file_full_path + " could not be read");
            }
        }

        if (m_Debug) {
            printDebugMsg("Print transpose of normalizedX",
                    "Each line is a normalized vector of an image patch.", normalizedX, true);
        }

        System.out.println("Normalize patches of the input images. Ends");

        // Whitening Matrx X
        m_eigendecomp = getCovMatrix(normalizedX, patchLength, numPatches);
        Matrix whitenedX = getWhitenMatrix(normalizedX);

        // Iterating to get the final centroid matrix D:
        m_D = getFinalCentroid(whitenedX);

        // Plot all the final centroids.
        for (int col = 0; col < m_D.numColumns(); col++) {
            DenseVector col_vec = Matrices.getColumn(m_D, col);
            //plotVector(col_vec);
        }
    }

    // Substep of step 1(in page 6): Get a normalizd vector from a corresponding patch.
    protected Vector getNormalizdVector(BufferedImage patch)
    {
        // patchLength: pixels per patch, that is the dimensions of a patch
        int patchLength = m_cropSize * m_cropSize;

        // Create constant vectors that we will reuse many times to center the values in each patch
        // The length of two following vectors are both "patchLength", that is the length of a patch,
        // and the values are 1/patchLength and 1 respectively.
        Vector oneOverPatchLength = constantVector(1.0 / patchLength, patchLength);
        Vector allOnesPatchLength = constantVector(1.0, patchLength);

        int index = 0;

        // Transform the "m_cropSize x m_cropSize" subregion into a vector which length is "patchLength",
        // get the rgb value of every pixel in the image, transform it into grayscale and save it into
        // the corresponding part of the vector.
        // DenseVector, which represents a column vector, is a class in mtj.
        Vector vec = new DenseVector(patchLength);
        for (int j = 0; j < patch.getWidth(); j++) {
            for (int k = 0; k < patch.getHeight(); k++) {
                int rgb = patch.getRGB(k, j);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = (rgb & 0xFF);
                vec.set(index++, (r + g + b) / 3.0);
            }
        }
/*
        if (m_Debug) {
            System.out.println("\nIn the function of \"getNormalizdVector\":");
            printDebugMsg("Print transpose of unnormalized vector getting from an image patch.", null,
                    new DenseMatrix(vec), true);
        }
*/
        //plotVector(vec);
        // 1. Normalize input:
        // DenseVector.dot(Vector y) means: xT*y
        // DenseVector.add(double alpha, Vector y) means: x = alpha*y + x
        // eg: vecT=(3,7,10,1,9) => oneOverPatchLengthT=(1/5,1/5,1/5,1/5,1/5)
        // vec.dot(oneOverPatchLength) = 6 (this is the mean(x(i)))
        // vec.add(-vec.dot(oneOverPatchLength), allOnesPatchLength) = -6 * (1,1,1,1,1)T + (3,7,10,1,9)T = (-3,1,4,-5,3)T
        // So: centeredVec is x(i)-mean(x(i))
        Vector centeredVec = vec.add(-vec.dot(oneOverPatchLength), allOnesPatchLength);
/*
        if (m_Debug) {
            printDebugMsg("Print transpose of x(i)-mean(x(i)) vector getting from an image patch.", null,
                    new DenseMatrix(centeredVec), true);
        }
*/
        // Vector.norm(Vector.Norm type) means: Computes the given norm of the vector
        // Here is to calculate L-2, that is the root of sum of squares of each element in vector centeredVec.
        double norm = centeredVec.norm(Vector.Norm.Two);
        //System.out.println("norm: " + norm);

        // DenseVector.scale(double alpha) means: x = alpha * x
        // Here, (norm * norm) / vec.size() means: var(x(i)) (var means variance)
        // So, normalizedVec is the final result in step 1. It is a vector which length is 1.
        Vector normalizedVec = centeredVec.scale(1.0 / Math.sqrt((norm * norm) / vec.size() + 10));
/*
        if (m_Debug) {
            printDebugMsg("Print transpose of normalized vector getting from an image patch.", null,
                    new DenseMatrix(normalizedVec), true);
        }
*/
        //plotVector(normalizedVec);

        return normalizedVec;
    }

    // Step 2.1(in page 6): Whitening Matrix X - Finish the eigenvalue decomposition.
    protected Matrix getCovMatrix(Matrix X, int patchLength, int numPatches) {
        System.out.println("Whitening Matrix X - Get the eigenvalue decomposition. Starts");

        // Calculate mean value for each pixel in X
         // DenseMatrix.mult(Vector x, Vector y) means: y = A*x
        // constantVector(1.0 / numPatches, numPatches) is a vector with length "numPatches" and value "1.0 / numPatches".
        // y is the return value, that is the "new DenseVector(patchLength)"
        // So, Vector mean actually equals to X * constantVector(1.0 / numPatches, numPatches).
        // So, Vector mean is the mean vector of all the patches in matrix X.
        Vector mean = X.mult(constantVector(1.0 / numPatches, numPatches), new DenseVector(patchLength));
        if (m_Debug) {
            printDebugMsg("Print mean value of each line of matrix normalizedX",
                    null, new DenseMatrix(mean), true);
        }

        // Calculate centered version of X and store it in S
        // Construct matrix S by using deep copy of matrix X.
        Matrix S = new DenseMatrix(X);
        if (m_Debug) {
            printDebugMsg("Print transpose of matrix S", "It should the same of normalizedX.", S, true);
        }

        // DenseMatrix.transBmultAdd(double alpha, Matrix B, Matrix C) means: C = alpha * A * BT + C
        // A is "new DenseMatrix(mean)": a matrix with just one column, which its values come from the mean vector calculated above.
        // B is "new DenseMatrix(constantVector(1.0, numPatches))": a matrix with just one column, which all the values is set to 1.
        // C is the return value, that is matrix S. The initial value of S is X, which contains all the normalized patches.
        // Finally, S(new) = - DenseMatrix(mean) * DenseMatrix(constantVector(1.0, numPatches)) + S(old)->X
        // DenseMatrix(mean) * DenseMatrix(constantVector(1.0, numPatches)): generates a matrix with dimension of patchLength * numPatches,
        // every column in the matrix are the same, which is the mean vector. So, the result of this multiplication is to
        // generate a matrix with numPatches of same mean vectors. We could named it as Z.
        // So, S(new) = -Z + S(old)->X = X - Z. That is, S is a matrix with columns which represent the distance from the
        // original patches to the mean vector of all these patches.
        S = (new DenseMatrix(mean)).transBmultAdd(-1.0, new DenseMatrix(constantVector(1.0, numPatches)), S);
        if (m_Debug) {
            printDebugMsg("Print transpose of matrix S processed",
                    "Every S[i][j] should subtract its mean value corresponded .", S, true);
        }

        // Calculate covariance matrix from centered version of the data
        // class UpperSPDDenseMatrix means: Upper symmetrical positive definite dense matrix.
        // "new UpperSPDDenseMatrix(patchLength))" means: create a matrix with dimension of patchLength * patchLength.
        // UpperSPDDenseMatrix.rank1(double alpha, Matrix C) means: A = alpha * C * CT + A.
        // A is the return value, that is matrix cov, also "new UpperSPDDenseMatrix(patchLength)". The initial values of cov are all 0.
        // C is S, a centered version of the data, which is calculated above.
        // Note: cov(x, y) = E[(x-E(x))(y-E(y))], so in the cov matrix, each item, such as a(ij) means covariance of patch(i) and patch(j).
        Matrix cov = (new UpperSPDDenseMatrix(patchLength)).rank1(1.0 / numPatches, S);
        // Important Note: Cov is defined as a metrix with deminsion of patchLength * patchLength in above statement.
        // That is because: each item (such as a(ij)) in cov represents covariance of attribute(i) and attribute(j) !
        // Do not mistake it as patch(i) and patch(j). So, it is correct that first parameter of rank1 should be "1.0/numPatches",
        // which is the number of original attributes(also called feature).
        // That's why we could reuse the cov matrix which calculated in step 2 page 6.
        if (m_Debug) {
            printDebugMsg("Print covarince matrix cov",
                    "It should the a symmetrical matrix and equals to S * ST / numPathces.", cov, false);
        }

        double[] eVals = null;
        Matrix V = null;
        try {
            // Computes all the eigenvalue decomposition of the covariance matrix "cov" and save them in the double array "EVals".
            SymmDenseEVD sdEVD = SymmDenseEVD.factorize(cov);

            // Get all the eigenvalues into a double array.
            eVals = sdEVD.getEigenvalues();
            if (m_Debug) {
                printDebugMsg("Print eigenvalues of covariance matrix cov(x).",
                        null, new DenseMatrix(new DenseVector(eVals)), true);
            }

            // Get the eigenvector matrix V
            V = sdEVD.getEigenvectors();
            if (m_Debug) {
                printDebugMsg("Print eigenvectors of covariance matrix cov(x).", null, V, false);
            }

        }
        catch (NotConvergedException e)
        {
            System.out.println(e.getReason());
            e.printStackTrace();
        }

        double e_zca = 0.1;
        // Calculate the matrix D which composed by the eigenvalues, just 0 values except leading diagonal.
        // Initial a matrix D_I, setting its each element in the leading diagonal plus a minor epsilon.
        Matrix D_I = new DenseMatrix(eVals.length, eVals.length);
        D_I.zero();
        for (int i = 0; i < eVals.length; i++) {
            D_I.set(i, i, 1.0/Math.sqrt(eVals[i] + e_zca));
        }

        if (m_Debug) {
            printDebugMsg("Print matrix D_I(-1/2).", null, D_I, false);
        }

        // VxD_I is the multiply of V and D_I
        Matrix VxD_I = new DenseMatrix(V.numRows(), D_I.numColumns());
        VxD_I = V.mult(D_I, VxD_I);
        if (m_Debug) {
            printDebugMsg("Print matrix VxD_I(-1/2).", null, VxD_I, false);
        }

        // VxD_IxVT is the multiply of V and D_I and transpose of V
        // public Matrix transBmult(Matrix B, Matrix C) means: C = A * BT, C is the return value.
        Matrix VxD_IxVt = new DenseMatrix(VxD_I.numRows(), V.numRows());
        VxD_IxVt = VxD_I.transBmult(V, VxD_IxVt);
        if (m_Debug) {
            printDebugMsg("Print matrix VxD_I(-1/2)xVt.", null, VxD_IxVt, false);
        }

        System.out.println("Whitening Matrix X - Get the eigenvalue decomposition. Ends");
        return VxD_IxVt;
    }

    // Step 2.2(in page 6): Whitening Matrix X - Get the whitened matrix X.
    protected Matrix getWhitenMatrix(Matrix X)
    {
        System.out.println("Whitening Matrix X - Get the whitened matrix X. Starts");

        // Get the matrix X after whitening process. Here, X is normalizedX.
        Matrix whitenedX = new DenseMatrix(X);
/*
        if (m_Debug) {
            printDebugMsg("Print transpose of normalizedX for whitening.", null, X, true);
            printDebugMsg("Print m_eigendecomp.", null, m_eigendecomp, false);
        }
*/

        // A.mult(Matrix B, Matrix C): C = A * B.
        // So it means: whitenedX = m_eigendecomp * X(normalizedX)

        whitenedX = m_eigendecomp.mult(X, whitenedX);
/*
        if (m_Debug) {
            printDebugMsg("Print transpose of finally whitened matrix whitenedX.", null, whitenedX, true);
        }
*/
        System.out.println("Whitening Matrix X - Get the whitened matrix X. Ends");
        return whitenedX;
    }

    // Step 3(in page 6): Iterating to get the final centroid matrix D:
    protected Matrix getFinalCentroid(Matrix whitenedX) {
        System.out.println("Iterating to get the final centroid matrix D. Starts");
        Matrix D = getInitialCentroids();
        if (m_Debug) {
            printDebugMsg("Print transpose of finally whitened matrix whitenedX.", null, whitenedX, true);
            printDebugMsg("Print transpose of initial centroids.", "Each line is a centroid", D, true);
        }

        double oldSSE = 0;
        for (int i = 0; i < m_IterationTimes; i++) {
            m_SSE = 0;  // Must set m_SSE to 0, otherwise it will be accumulated.

            Matrix D_new = getCentroids(D, whitenedX);
            System.out.println("Iteration time: " + i + "; oldSSE: " + oldSSE + "; SSE: " + m_SSE + "; (oldSSE - SSE): " + (oldSSE - m_SSE));

            // Terminate the process of find the final centroids by using SSE.
            if (Math.abs(oldSSE - m_SSE) < m_Precision) {
                System.out.println("converge count is : " + i);
                D = D_new;
                break;
            }

            D = D_new;
            oldSSE = m_SSE;
        }

        System.out.println("Iterating to get the final centroid matrix D. Ends");
        return D;
    }


    protected Matrix getCentroids(Matrix D_old, Matrix whitenedX) {
        int patchLength = whitenedX.numRows();
        int numPatches = whitenedX.numColumns();
/*
        if (m_Debug) {
            printDebugMsg("Print transpose of current centroids.", null, D_old, true);
        }
*/
        // 1.1 Calculate S = DT * X
        Matrix S = new DenseMatrix(m_K, numPatches);

        // 	A.transAmult(Matrix B, Matrix C) means: C = AT * B, C is the return value.
        S = D_old.transAmult(whitenedX, S);  // S = DT * X
/*
        if (m_Debug) {
            printDebugMsg("Print matrix S(numPatches x K).",
                    "Before calculating the maximum value in each column of S", S, false);
        }
*/
        // In each column of S, We must find the maximum absolute value, which represents the distance between x(i)
        // and its corresponding centroid. Set 0 to other elements in each column except the previous distance.
        for (int col = 0; col < numPatches; col++) {
            // current maximum distance
            double maxVal = S.get(0, col);
            // row number of current maximum distance
            int row_num = 0;

            for (int row = 1; row < m_K; row++) {
                double val = S.get(row, col);
                // Determine whether the new distance is more closer.
                //if (val > maxVal) {
                if (Math.abs(val) > Math.abs(maxVal)) {
                    maxVal = val;
                    // Set the previous max value to 0.
                    S.set(row_num, col, 0);
                    row_num = row;
                } else {
                    S.set(row, col, 0);
                }
            }
        }
/*
        if (m_Debug) {
            printDebugMsg("Print matrix S(numPatches x K).",
                    "Column i represents the centroid to which its corresponding x(i) belongs", S, false);
        }
*/
        // 1.2 Calculate the SSE(sum of squared errors). This is the termination condition of the process of finding centroids.
        for (int i = 0; i < whitenedX.numColumns(); i++) {
            // Calculate the distance between one patch to its corresponding centroid. Then save it in SSE.
            Vector projected = D_old.mult(Matrices.getColumn(S, i), new DenseVector(whitenedX.numRows()));
            double n = Matrices.getColumn(whitenedX, i).add(-1.0, projected).norm(Vector.Norm.Two);
            m_SSE += n * n;
        }

        // 2. D_new = X * ST + D_old, D_new represents new centorids.
        Matrix D_new = new DenseMatrix(D_old, true);
        // A.transBmultAdd(Matrix B, Matrix C) means: C = A * BT + C, C is return value.
        D_new = whitenedX.transBmultAdd(S, D_new);
/*
        if (m_Debug) {
            printDebugMsg("Print transpose of matrix contains (K x patchlength) new centroids.",
                    null, D_new, true);
        }
*/
        // 3. Normalize all the vectors in matrix D_new. Each of the vectors will be regard as a new centroid.
        for (int col = 0; col < D_new.numColumns(); col++) {
            DenseVector col_vec = Matrices.getColumn(D_new, col);
            // the root of sum of squares of the elements in the col_vec
            double norm = col_vec.norm(Vector.Norm.Two);

            for (int row = 0; row < D_new.numRows(); row++) {
                D_new.set(row, col, D_new.get(row, col) / norm);
            }
        }
/*
        if (m_Debug) {
            printDebugMsg("Print transpose of normalized matrix of new centroids(K x patchlength).",
                    null, D_new, true);
        }
*/
        return D_new;
    }

/*
        // set vector "vec" to the "col"th column of the matrix x
        protected void setVecToMatrix (Matrix X, Vector vec, int col) {
            assert(X.numRows() == vec.size());

            for (int i = 0; i < vec.size(); i++) {
                X.set(i, col, vec.get(i));
            }
        }

        // Select K centroid randomly from original patches which had already normalized and whitened.
        protected Matrix getInitialCentroids(Matrix whitenedX) {
            int patchLength = whitenedX.numRows();
            int numPatches = whitenedX.numColumns();

            Random rand = new Random(m_Seed);
            // selectedLabel[i] = 1 means the ith column in X is selected as one of the initial centroid.
            int selectedLabel[] = new int[numPatches];
            // Save all the indices of selected column as a centroid.
            int centroid_index[] = new int[m_K];

            // Randomly find k centroids.
            for (int i = 0; i < m_K; ) {
                int random_index = rand.nextInt(numPatches);

                if (selectedLabel[random_index] == 0) {
                    selectedLabel[random_index] = 1;
                    centroid_index[i] = random_index;
                    i++;
                }
            }

            // Set the randomly selected k patches as centroids.
            DenseMatrix initCentroidsMatrix = new DenseMatrix(patchLength, m_K);
            for (int i = 0; i < m_K; i++) {
                int col = centroid_index[i];
                for (int j = 0; j < patchLength; j++) {
                    initCentroidsMatrix.set(j, i, whitenedX.get(j, col));
                }
            }

            return initCentroidsMatrix;
        }
    */

    // Generate random K centroids by using function Random.nextGaussian.
    protected DenseMatrix getInitialCentroids() {
        Random rand = new Random(m_Seed);

        int patchLength = m_cropSize * m_cropSize;
        DenseMatrix centroids = new DenseMatrix(patchLength, m_K);

        for (int i = 0; i < m_K; i++) {
            DenseVector v = new DenseVector(patchLength);
            for (int j = 0; j < patchLength; j++) {
                v.set(j, rand.nextGaussian() * 100);
            }
            double norm = v.norm(Vector.Norm.Two);
            for (int j = 0; j < patchLength; j++) {
                centroids.set(j,i, v.get(j) / norm);
            }
        }

        return  centroids;
    }

    protected void printMatrix(Matrix matrix, boolean needTranspose) {
        int row = needTranspose ? matrix.numColumns() : matrix.numRows();
        int col = needTranspose ? matrix.numRows() : matrix.numColumns();

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                double val = needTranspose ? matrix.get(j, i) : matrix.get(i, j);
                System.out.print(String.format("%.3f  ", val));
            }
            System.out.println();
        }
    }

    protected void printDebugMsg(String msg, String submsg, Matrix matrix, boolean needTranspose) {
        System.out.println();
        System.out.println(msg +  " Starts:");
        if (submsg != null) {
            System.out.println(submsg);
        }
        printMatrix(matrix, needTranspose);
        System.out.println(msg + " Ends.");
        System.out.println();
    }

    /**
     * Capabilities, indicating that any type of attribute with or without a class is OK
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enableAllAttributes();

        // class
        result.enableAllClasses();
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @OptionMetadata(
            displayName = "Image directory",
            description = "The directory of the images.",
            displayOrder = 1,
            commandLineParamName = "D",
            commandLineParamSynopsis = "-D")
    public String getImageDirectory() {
        return imageDirectory;
    }
    public void setImageDirectory(String imageDirectory) {
        this.imageDirectory = imageDirectory;
        System.setProperty("last.dir", imageDirectory);
    }

    @OptionMetadata(
            displayName = "Patch size",
            description = "The size of the patches.",
            displayOrder = 2,
            commandLineParamName = "Z",
            commandLineParamSynopsis = "-Z")
    public int getCropSize() {
        return m_cropSize;
    }
    public void setCropSize(int cropSize) {
        this.m_cropSize = cropSize;
    }

    @OptionMetadata(
            displayName = "Patch number per image",
            description = "The number of the patches cutting from an image.",
            displayOrder = 3,
            commandLineParamName = "N",
            commandLineParamSynopsis = "-N")
    public int getNumPatchPerImage() {
        return m_numPatchesPerImage;
    }
    public void setNumPatchPerImage(int numPatchesPerImage) {
        this.m_numPatchesPerImage = numPatchesPerImage;
    }

    @OptionMetadata(
            displayName = "K",
            description = "The number of centroids",
            displayOrder = 4,
            commandLineParamName = "K",
            commandLineParamSynopsis = "-K")
    public int getK() {
        return m_K;
    }
    public void setK(int K) {
        this.m_K = K;
    }

    @OptionMetadata(
            displayName = "Stride size",
            description = "The size of the stride to use when creating features (both directions).",
            displayOrder = 5,
            commandLineParamName = "T",
            commandLineParamSynopsis = "-T")
    public int getStrideSize() {
        return m_Stride;
    }
    public void setStrideSize(int Stride) {
        this.m_Stride = Stride;
    }

    @OptionMetadata(
            displayName = "Pool size",
            description = "The size of the pool to use when creating features (both directions).",
            displayOrder = 6,
            commandLineParamName = "P",
            commandLineParamSynopsis = "-P")
    public int getPoolSize() {
        return m_Pooling;
    }
    public void setPoolSize(int Pooling) {
        this.m_Pooling = Pooling;
    }

    @OptionMetadata(
            displayName = "Seed",
            description = "The seed is used to generate a stream of pseudorandom numbers.",
            displayOrder = 7,
            commandLineParamName = "S",
            commandLineParamSynopsis = "-S")
    public int getSeed() {
        return m_Seed;
    }
    public void setSeed(int Seed) {
        this.m_Seed = Seed;
    }

    /**
     * We need to have access to the full input format so that we can read the images.
     *
     * @return true
     */
    public boolean allowAccessToFullInputFormat() {
        return true;
    }

    public static void main(String[] options) {
        String[] file_arff = {""};
        weka.gui.explorer.Explorer exp = new weka.gui.explorer.Explorer();
        exp.main(file_arff);
        runFilter(new KMeansImageFilter(), options);
    }
}