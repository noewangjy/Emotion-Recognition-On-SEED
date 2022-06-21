from graphviz import Digraph
from torch.autograd import Variable
import torch
from src.models.MMD_AAE import MMD_AAE_model
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch.nn.functional as F


def make_dot(var, params=None):
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style="filled", shape="box", align="left", fontsize="12", ranksep="0.1", height="0.2")
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
                dot.edge(str(id(var.grad_fn)), str(id(var)))
                var = var.grad_fn
            if hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var)
    return dot


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    num_dim: int = 448
    inputs = torch.randn(num_dim, 310)
    class_labels = torch.ones(num_dim).long()
    real_labels = torch.ones(num_dim)
    fake_labels = torch.zeros(num_dim)
    fake_encodings = torch.randn(num_dim, 128)

    model = MMD_AAE_model(cfg)
    output = model.forward(inputs)

    all_encodings = torch.cat([output.encodings, fake_encodings], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)
    gan_labels = torch.cat([fake_labels, real_labels], dim=0)

    discriminator_pred = model.discriminator(all_encodings)
    encodings = output.encodings.view(-1, 2, 128)
    task_loss = F.nll_loss(output.class_pred, class_labels)
    mmd_loss = model.mmd_loss_function.calculate(encodings)
    generator_loss = F.mse_loss(discriminator_pred.squeeze(-1).square(), gan_labels.float())
    decoder_loss = F.mse_loss(output.decoded, output.norm_input)

    result = task_loss + mmd_loss + generator_loss + decoder_loss

    make_dot(result).view()


if __name__ == "__main__":
    run()
