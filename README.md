# Mask-Detection-using-Uncertainty-of-Facial-Landmark

## Introduction
Facial landmarks are key features of human facial image, which generally represent 68 points of facial region 
such as eyes, nose, mouth, and contour of chin. In this project, I propose mask detection using facial landmark assessment. I use LUVLi (the joint estimation of landmark location, uncertainty, and visibility likelihood )
network to estimate uncertainty of predicted facial landmark on masked face dataset. 
I implement a machine learning model to classify whether mask is present or absent

## Dataset


## Implementation Detail
I train the LUVLi model on 300W dataset. The LUVLi network use transfer learning, which allows to leverage the knowledge learned from pre-training process. The dataset has 61,825 images with ground truth that has 68 landmark locations per image. For the details, please take a look at the LUVLi research paper. The trained model outputs four NumPy arrays: mean array, Cholesky array, images array, and ground truth array. The mean represents estimated location of predicted landmark. The Cholesky array stores the elements of lower-triangular matrix of each landmark. The images and ground truth array are image and ground truth of dataset that are transformed into arrays. The shape of the Cholesky array is as follows: *(num_img, num_landmark, 3)*. The last dimension of the array represents a lower-triangular matrix. The lower-triangular matrix is 2x2 matrix whose elements are three positive elements and zero. The Cholesky array stores the estimated three positive elements except zero. To obtain a covariance matrix, I multiply a lower triangular matrix by the conjugate of it. Then, I perform eigen decomposition to the covariance to get the two eigenvalues. Since the eigenvalues are directly related to the area of covariance ellipse, I preprocess the Cholesky array into eigenvalues array. To classify the data, I choose logistic regression, support vector machine, random forest classifier, and gaussian naïve classifier. The inputs of classifiers are numpy stacked images, means array, and cholesky array. The evaluation results are provided in table 3. It shows that All of the models demonstrate a performance higher than 90%. Especially, logistic regression model with 3000 max iterations slightly outperforms other three models. The SVM show second-best performance when using polynomial kernel with 0.01 coefficient. In conclusion, logistic regression model is the most optimal model as classification algorithm for our data. 

## References
Kumar, A., Marks, T.K., Mou, W., Wang, Y., Cherian, A., Jones, M.J., Liu, X., Koike-Akino, T., Feng, C., "LUVLi Face Alignment: Estimating Landmarks’ Location, Uncertainty, and Visibility Likelihood", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), DOI: 10.1109/​CVPR42600.2020.00826, June 2020.

## Resources
- [LUVLi Face Alignment: Estimating Landmarks’ Location, Uncertainty, and Visibility Likelihood Research Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kumar_LUVLi_Face_Alignment_Estimating_Landmarks_Location_Uncertainty_and_Visibility_Likelihood_CVPR_2020_paper.pdf)
- [LUVLi Source Code Request](https://www.merl.com/research/license/LUVLi)
- [300W dataset]

