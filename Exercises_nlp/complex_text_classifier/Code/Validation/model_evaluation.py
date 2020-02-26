'''
Created on 31/01/2019

@author: Francesco Pugliese
'''

# Classes prediction from probabilities, return classes and probabilities
def predictClasses(model, test_set_x, batch_size, verbose = 0):
    prob = model.predict(test_set_x, batch_size=batch_size, verbose=verbose)
    if prob.shape[-1] > 1:
        return [prob.argmax(axis=-1), prob]
    else:
        return [(prob > 0.5).astype('int32'), prob]