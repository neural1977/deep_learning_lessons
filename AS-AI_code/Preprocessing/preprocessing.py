# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:09:52 2020

@author: francesco
"""

class Preprocessing: 
    @staticmethod
    def load_garbage(datapath, validation_split, test_split, shuffle, limit, input_size, rescale):        
        # load data
        load_start_time = timeit.default_timer()
        # grab the image paths and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(datapath)))
        # loop over the input images
        data = []
        labels = []
        count = 0
        for imagePath in imagePaths:
            
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath)
                        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if rescale == True: 
                image = cv2.resize(image, (input_size, input_size))
            image = img_to_array(image)

            data.append(image)
            # extract the class label from the image path and update the labels list
            label = imagePath.split(os.path.sep)[-2]
                        
            if label == "glass":
                label = 0
            elif label == "metal":
                label = 1
            elif label == "paper":
                label = 2
            elif label == "plastic":
                label = 3
            elif label == "trash":
                label = 4
            labels.append(label)

            if limit is not None:
                count += 1
                
                if count>limit:
                    break

        data_set = np.array(data, dtype="float")
        labels = np.array(labels)
        
        # split the data into a training set and a validation set
        indices = np.arange(data_set.shape[0])

        if shuffle == True: 
            np.random.shuffle(indices)

        data_set = data_set[indices]
        labels = labels[indices]

        num_validation_samples = int(round(validation_split * data_set.shape[0]))
        num_test_samples = int(round(test_split * data_set.shape[0]))
        train_set_x = data_set[:-(num_test_samples+num_validation_samples)]
        train_set_y = labels[:-(num_test_samples+num_validation_samples)]
        val_set_x = data_set[-(num_test_samples+num_validation_samples):-num_test_samples]
        val_set_y = labels[-(num_test_samples+num_validation_samples):-num_test_samples]
        test_set_x = data_set[-num_test_samples:]
        test_set_y = labels[-num_test_samples:]

        print ('\n\nLoading time: %.2f minutes\n' % ((timeit.default_timer() - load_start_time) / 60.))

        return [train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y]

        
        
        
    @staticmethod    
    def load_satellite_imagery():
        return None