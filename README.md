# Deep_Learning_Challenge

# Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. 

# Step 1: Preprocessing the Data

**StandardScaler()** was use for preprocessing the data 
    Reading in the charity_data.csv file to a Pandas DataFrame
        https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv
    Dropping EIN and NAME columns
    Determining the number of unique values for each columns
    Using [pd.get_dumies()] to encode categorical variables. 
    Split the preprocessed data into a "features" array, x, and a "target" array,y. Use these arrays and the [train_test_split] function to split the data into training and testing datasets. 
    Scale the training and testing features datasets by creating a [StandarScaler()] instance, fittting it to the training data, and then using the [transform] function. 
    
# Step 2: Compile, Train and Evaluate the Model

Using my knowledge of [TensorFlow], design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. 
    Create a neural network model by assigning the number of input features and nodes for each layer using [TensorFlow] and [Keras]
    Create the first hidden layer and choose an appropriate activation function. 
    If necessary, add a second hidden layer.
    Create an output layer
    Check the structure of the model
    Compile and train the model
    Evaluate the model using th test data to determine the loss of accuracy. 
    save and export the results to an HDF5 file. 

# Main
    For the main attempt, the columns EIN and NAME were dropped. Cutoff values for [Application_type] was set at 500 and for [Classification] was set at 1800. [get_dumies] was used to convert the categorical data. 

    **Defining the model**
        number of input features was set to columns. 
        Two hidden layers and and output layers where used with 80 and 30 nodes respectively. the activation functions were 'relu' and 'sigmoid'. 100 epochs
            
            # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
            number_input_feature = len(X_train.columns)
            hidden_nodes_layer1= 80
            hidden_nodes_layer2= 30

            nn = tf.keras.models.Sequential()

            # First hidden layer
            nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim= number_input_feature, activation='relu'))

            # Second hidden layer
            nn.add(tf.keras.layers.Dense(units= hidden_nodes_layer2, activation='relu'))

            # Output layer
            nn.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))

            # Check the structure of the model
            nn.summary()
    
    These were the results:
        268/268 - 0s - loss: 0.5635 - accuracy: 0.7301 - 218ms/epoch - 813us/step
        Loss: 0.563511312007904, Accuracy: 0.7301457524299622

# Optimization Attempt 1
    For this attempt a third hidden layer was used. The number of nodes were 100, 60 and 30. The activation functions were 'relu', 'tanh', 'swish' and 'sigmoid'. It ran for 250 epochs. 

        # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
        number_input_feature= len(X_train.columns)
        hidden_nodes_layer1= 100
        hidden_nodes_layer2= 60
        hidden_nodes_layer3= 30

        nn = tf.keras.models.Sequential()

        # First hidden layer
        nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim= number_input_feature, activation='relu'))

        # Second hidden layer
        nn.add(tf.keras.layers.Dense(units= hidden_nodes_layer2, activation='tanh'))

        # Third hidden layer
        nn.add(tf.keras.layers.Dense(units= hidden_nodes_layer3, activation='swish'))

        # Output layer
        nn.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))

        # Check the structure of the model
        .summary()

        These were the results
        268/268 - 0s - loss: 0.5828 - accuracy: 0.7277 - 232ms/epoch - 866us/step
        Loss: 0.5827525854110718, Accuracy: 0.7276967763900757