import tensorflow as tf
import numpy as np
import math
import sys
sys.path.insert(1, '/home/workspace/solution')
from utils import check_softmax, check_acc, check_model, check_ce

def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits tensor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    # IMPLEMENT THIS FUNCTION
    # Softmax function signma(x)_i = exp(x_i)/sum(e_j)
    # Numerator
    num = tf.exp(logits)
    # denominator (num, 1, keepdims=True) - num tensor to reduce (sum), dimension to reduce, keepdims = If true, retains reduced dimensions with length 1.)
    den = tf.math.reduce_sum(num, 1, keepdims=True)
    # calculate the soft_logits
    return num / den
    '''
    Old code not valid - Only kept for documentation
    # Convert the logits into an array to go through it
    logits_arr = logits.numpy()
    # init soft_logits
    soft_logits = np.empty(0)
    # initialize the sum of the exp logits
    expologits = 0
    for logi in logits_arr:
        # Calcaulate the expontential
        explogi = math.exp(logi)
        # append to the end the exponential of logits
        #print(explogi)
        soft_logits = np.append(soft_logits,explogi)
        # add the value to the total sum
        expologits += explogi
    # divide the whole list by the total exponential
    soft_logits /= expologits
    # Create a tensorflow tensor from the array
    soft_logits = tf.convert_to_tensor(arr)
    
    return soft_logits
    '''

def cross_entropy(scaled_logits, one_hot):
    """
    Old code not valid - Only kept for documentation
    Cross entropy loss implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy 
    """
    # IMPLEMENT THIS FUNCTION
    # dot product scaled_logits, one_hot
    masked_logits = tf.boolean_mask(scaled_logits, one_hot) 
    # log of the results
    return -tf.math.log(masked_logits)
    '''
    Old code not valid - Only kept for documentation
    # Convert the logits into an array to go through it
    scaled_logits_arr = scaled_logits.numpy()
    one_hot_arr = one_hot.numpy()
    # multiply arrays
    loss_arr = one_hot_arr.dot(np.log(scaled_logits_arr))
    # init loss
    loss = - loss_arr.sum()
    return loss
    '''

def model(X, W, b):
    """
    logistic regression model
    args:
    - X [tensor]: input HxWx3
    - W [tensor]: weights
    - b [tensor]: bias
    returns:
    - output [tensor]
    """
    # IMPLEMENT THIS FUNCTION
    #Y = W*X+b
   
    # Reshape the vector X to 1 dimension
    flatten_X = tf.reshape(X, (-1, W.shape[0]))
    # Return softmax(X*W) +b - The model and normalise output
    # Becuase of softmax sum of the output tensor is 1
    return softmax(tf.matmul(flatten_X, W) + b)
   


def accuracy(y_hat, Y):
    """
    calculate accuracy as
    sum(max(Y_hat) = Y) / size(Y)
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    
    """
    # IMPLEMENT THIS FUNCTION
    # calculate argmax and save it with same Y data type
    argmax = tf.cast(tf.argmax(y_hat, axis=1), Y.dtype)
   
    # calculate acc. When the max value of y_hat is equal to Y
    acc = tf.math.reduce_sum(tf.cast(argmax == Y, tf.int32)) / Y.shape[0]
   
    return acc
    

if __name__ == "__main__":
    
    check_softmax(softmax)
    check_ce(cross_entropy)
    check_model(model)
    check_acc(accuracy)
 
   