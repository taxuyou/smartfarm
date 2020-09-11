import tensorflow as tf

__all__ = ['get_dataloader']

def get_dataloader(args):

    name = args.model.name

    if name == 'lstm_enc_dec':
        if args.model.config.env_only:
            from .lstm_enc_dec_dataloader import dataloader4lstm_enc_dec_env
            dataloader = dataloader4lstm_enc_dec_env(args)
        else:
            from .lstm_enc_dec_dataloader import dataloader4lstm_enc_dec
            dataloader = dataloader4lstm_enc_dec(args)

    elif name == 'decision_tree':
        from .rda_dataloader import get_dt_dataset
        dataloader = get_dt_dataset(args)

    elif name == 'basic_lstm':
        from .rda_dataloader import get_bl_dataset
        dataloader = get_bl_dataset(args)
        

    elif name == 'mlp':
        from .mlp_dataloader import mlp_dataloader
        dataloader = mlp_dataloader(args)

    else:
        raise ValueError(args.model.name)

    return dataloader
    