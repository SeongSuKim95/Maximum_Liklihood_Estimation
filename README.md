# Pattern Recognition Computer Exercise
## Maximum Liklihood Estimation of multi dimension Gaussian model data


![Pattern_recognition_Computer_Exercise](https://user-images.githubusercontent.com/62092317/106402503-2337bb80-646d-11eb-83a9-a46f00483d49.PNG)

* 1.Make Gaussian Density model 
```python
def Gaussian_density_probability(mean, variance, data):

    sigma = sqrt(variance)
    temp = data-mean
    temp = temp/sigma
    Gaussian_constant = 1 / sqrt(2 * pi)
    probability = (Gaussian_constant/sigma)*exp(-(1/2)*(np.power(temp,2)))

    return probability

def multi_dimension_Gaussian_density_probability(mean,cov_matrix,data,dimension):

    det = np.linalg.det(cov_matrix)
    if det <0 :
        probability = np.zeros((data.shape[1],data.shape[1]))
    elif det >0:
        cov_inverse = np.linalg.inv(cov_matrix)
        temp = data- mean
        constant = 1/((sqrt(2*pi)**dimension)*sqrt(det))
        probability =constant*exp((-1/2)*np.dot(np.dot(temp.T,cov_inverse),temp))

    return probability
```

* 2.Log liklihood function
```python
def log_likelihood_function(probability):
    log_probability = log(probability)
    log_probability_sum = np.sum(log_probability,axis=0)
    return log_probability_sum
```

* 3.Make grid for searching mean and variance
  * I used np.linspace to make grid.
```python
  mean_resolution_1 = 50
  mean_resolution_2 = 50
  mean_resolution_3 = 50
  var_resolution_1 = 10
  var_resolution_2 = 10
  var_resolution_3 = 10

  mean_value1 = np.linspace(-1,1,num=mean_resolution_1,endpoint=False)
  mean_value2 = np.linspace(-1,1,num=mean_resolution_2,endpoint=False)
  mean_value3 = np.linspace(-1,1,num=mean_resolution_3,endpoint=False)

  var_value1 = np.linspace(4, 4.5, num=var_resolution_1, endpoint=False)
  var_value2 = np.linspace(4, 4.5, num=var_resolution_2, endpoint=False)
  var_value3 = np.linspace(0.3, 1, num=var_resolution_3, endpoint=False)
```
* 4.Computing Optimum Mean and Variance by grid search
  * 2 dimension
  ```python
  def maximum_likelihood_mean_2d(mean_value_1,mean_value_2,cov_mat,data):

      mean_value = Mean_matrix_2d(mean_value_1,mean_value_2)
      result_buffer = np.zeros((mean_value.shape[0]))
      for index,mean in enumerate(mean_value):
          probability = multi_dimension_Gaussian_density_probability(mean.reshape((2,1)),cov_mat,data,2)
          prob_diag = np.diag(probability)
          result_buffer[index] = log_likelihood_function(prob_diag)

      # x= mean_value[:,0]
      # y= mean_value[:,1]
      # fig =plt.figure(figsize=(10,5))
      # ax= fig.add_subplot(111,projection='3d')
      # ax.plot(x,y,result_buffer)
      # ax.set_xlabel('mean space 1')
      # ax.set_ylabel('mean space 2')
      # ax.set_zlabel('log likelihood')
      # plt.show()
      MLE_mean = mean_value[np.argmax(result_buffer)]

      return MLE_mean

  def maximum_likelihood_var_2d(mean_value,data):

      global var_value1,var_value2,var_value3

      Cov_mat = Var_matrix_2d(var_value1,var_value2,var_value3)
      result_buffer = np.zeros((var_value1.shape[0]*var_value2.shape[0]*var_value3.shape[0]))
      for index,cov in enumerate(Cov_mat):
          probability = multi_dimension_Gaussian_density_probability(mean_value.reshape((2,1)),cov,data,2)
          prob_diag = np.diag(probability)
          #print(prob_diag)
          result_buffer[index] = log_likelihood_function(prob_diag)

      MLE_var = Cov_mat[np.argmax(result_buffer)]

      return MLE_var
  ```
  * 3 dimension
  ```python
  def maximum_likelihood_mean_3d(mean_value1,mean_value2,mean_value3,data):

      mean_value = Mean_matrix_2d(mean_value1,mean_value2)
      result_buffer = np.zeros((mean_value3.shape[0],4))
      #Cov_mat = Covariance_matrix_3d(data_test1, data_test2, data_test3)
      Cov_mat = np.array([[1,0,0],[0,1,0],[0,0,1]])
      for index, mean3 in enumerate(mean_value3):
          temp = np.ones(mean_value1.shape[0]*mean_value2.shape[0])*(mean3)
          temp = temp.reshape((mean_value1.shape[0]*mean_value2.shape[0],1))
          mean_value_stack = np.hstack((mean_value,temp))
          MLE_mean = MLE_mean_stacker(mean_value_stack,Cov_mat,data)
          result_buffer[index] =MLE_mean

      temp = np.argmax(result_buffer,axis=0)
      #print(temp,'d')
      #print(temp)
      MLE_mean= result_buffer[temp[3]]
      #print(MLE_mean)
      MLE_mean= MLE_mean[0:3]
      return MLE_mean

  def maximum_likelihood_var_3d(MLE_mean,data):

      global var_value_31,var_value_32,var_value_33,var_value_34,var_value_35,var_value_36

      mean_value = MLE_mean
      result_buffer = np.zeros((var_value_31.shape[0]*var_value_32.shape[0]*var_value_33.shape[0]*var_value_34.shape[0]*var_value_35.shape[0]*var_value_36.shape[0]))
      Cov_mat = Var_matrix_3d(var_value_31,var_value_32,var_value_33,var_value_34,var_value_35,var_value_36)
      for index, cov in enumerate(Cov_mat):
          probability = multi_dimension_Gaussian_density_probability(mean_value.reshape((3,1)),cov,data,3)
          prob_diag = np.diag(probability)
          #print(prob_diag)
          result_buffer[index] = log_likelihood_function(prob_diag)

      MLE_var = Cov_mat[np.argmax(result_buffer)]

      return MLE_var
  ```
* 5.Comparing with MLE mean, Var

------------------------------------------------------------------------------------------------------------------------

## Prob 1_(a) Visualizing value of log likelihood in 1d mean space
![Problem_1](https://user-images.githubusercontent.com/62092317/106403896-14a0d280-6474-11eb-9f7a-454ba6ead024.PNG)
![x1](https://user-images.githubusercontent.com/62092317/106403931-38fcaf00-6474-11eb-82f8-708f222ecb5c.PNG)
![x2](https://user-images.githubusercontent.com/62092317/106403937-3b5f0900-6474-11eb-95a4-90c918b86451.PNG)
![x3](https://user-images.githubusercontent.com/62092317/106403941-3d28cc80-6474-11eb-88a6-db9c3bf2ad5f.PNG)

## Prob 1_(b) Visualizing value of log liklihood in 2d mean space
![x1x3](https://user-images.githubusercontent.com/62092317/106404271-a78e3c80-6475-11eb-91c3-82b847a8d40e.PNG)
![x2x3](https://user-images.githubusercontent.com/62092317/106404273-a826d300-6475-11eb-90a2-f6cd50a7b3ef.PNG)
![x1x2](https://user-images.githubusercontent.com/62092317/106404275-a8bf6980-6475-11eb-9ab8-2b1bea2b4637.PNG)

## Prob 1_(c)~(f) Read detail answers in [HERE](https://github.com/SeongSuKim95/Maximum_Liklihood_Estimation/blob/master/Maximum_Liklihood_Estimation.pdf)
