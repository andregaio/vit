from thop import profile


def model_info(model, x):
    flops, params = profile(model, inputs=(x,), verbose=False)
    return params * 1e-6, flops * 1e-9


def print_model_info(params, flops):
    print(f'Params: {params}M')
    print(f'FLOPS: {flops}G')