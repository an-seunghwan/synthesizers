# %%
import pandas as pd
import argparse
import torch
from torch.autograd.variable import Variable


# %%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# %%
def calculate_gradient_penalty(discriminator, penalty, real_data, fake_data, device):
    real_data = real_data.data
    fake_data = fake_data.data

    alpha = torch.rand(len(real_data), 1)
    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)
    discriminator_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=discriminator_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(discriminator_interpolates).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty


# %%
def postprocess(
    syndata, OutputInfo_list, colnames, discrete_dicts, discrete_dicts_reverse
):
    samples = []
    st = 0
    for j, info in enumerate(OutputInfo_list):
        ed = st + info.dim
        logit = syndata[:, st:ed]

        """argmax"""
        _, logit = logit.max(dim=1)

        samples.append(logit.unsqueeze(1))
        st = ed

    samples = torch.cat(samples, dim=1)
    syndata = pd.DataFrame(samples.cpu().numpy(), columns=colnames)

    """reverse to original column names"""
    for dis, disdict in zip(colnames, discrete_dicts_reverse):
        syndata[dis] = syndata[dis].apply(lambda x: disdict.get(x))
    for dis, disdict in zip(colnames, discrete_dicts):
        syndata[dis] = syndata[dis].apply(lambda x: disdict.get(x))
    return syndata


# %%
