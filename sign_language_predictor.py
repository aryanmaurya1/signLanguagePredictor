from library import model, load_image
from tf_utils import load_dataset, convert_to_one_hot, predict



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))



parameters = model(X_train, Y_train, X_test, Y_test)
image_one = load_image("test.jpeg")
image_two = load_image("test2.jpeg")
image_three = load_image("test3.jpeg")

print(predict(image_one.T, parameters))
print(predict(image_two.T, parameters))
print(predict(image_three.T, parameters))
