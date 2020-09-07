from tensorflow import keras

__all__ = ['get_optimizer']

def get_optimizer(args):
    algo = args.optimizer.algorithm
    optimizer = None

    if algo == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=args.optimizer.config.learning_rate, 
                                         momentum=args.optimizer.config.momentum, 
                                         nesterov=args.optimizer.config.nesterov)
    elif algo == 'adam':
        
        # TF 1.x.x
        optimizer = keras.optimizers.Adam(lr=args.optimizer.config.learning_rate, 
                                          beta_1=0.9, 
                                          beta_2=0.999, 
                                          epsilon=None, 
                                          decay=1e-6)
        '''
        # TF 2.x.x
        optimizer = keras.optimizers.Adam(learning_rate=args.optimizer.config.learning_rate, 
                                          beta_1=0.9, 
                                          beta_2=0.999, 
                                          epsilon=None, 
                                          decay=1e-6, 
                                          amsgrad=False)
        '''
    else:
        raise ValueError(algo)

    return optimizer