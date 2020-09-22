import tensorflow as tf


__all__ = ['get_model']


def get_model(args, config=None):
    model = None
    name = args.model.name

    if name == 'lstm_enc_dec':
        if args.model.config.env_only:
            from .lstm_enc_dec import Encoder_Decoder_Env
            model = Encoder_Decoder_Env(input_shapes=args.model.config.input_shapes, 
                                        output_shape=args.model.config.output_shape)
        else:
            from .lstm_enc_dec import Encoder_Decoder
            model = Encoder_Decoder(input_shapes=args.model.config.input_shapes, 
                                    output_shape=args.model.config.output_shape)
    
    elif name == 'decision_tree':
        from .decision_tree import get_decision_tree_model
        model = get_decision_tree_model(args)

    elif name == 'multiout_decision_tree':
        from .decision_tree import get_decision_tree_model
        model = get_multiout_decision_tree_model(args)

    elif name == 'basic_lstm':
        from .basic_lstm import get_basic_lstm_model
        model = get_basic_lstm_model(config)
        '''
        from .basic_lstm import Basic_LSTM
        model = Basic_LSTM(config)
        '''
        
    else:
        raise ValueError(args.model.name)
    
    return model