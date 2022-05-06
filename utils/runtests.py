import abc
import tensorflow as tf
from pathlib import Path
import json

from models import *
from utils.common import empty_directory
from utils.save import convert_numpy_types_to_natives


class ModelGetter:
    def __init__(self, input_dim, latent_factor, lr_d, lr_g):
        self.input_dim = input_dim
        self.latent_factor = latent_factor
        self.lr_d, self.lr_g = lr_d, lr_g

    @abc.abstractmethod
    def get(self, structure, *args, **kwargs) -> BaseGAN:
        pass


class Info:
    def __init__(self, **kwargs):
        # model info
        self.input_dim = 0
        self.latent_factor = 0
        self.lr_d = self.lr_g = 0
        self.opt = ''
        self.d_opt = ''
        self.g_opt = ''
        self.struct = None
        self.gan_type = 'Invalid'

        # training info
        self.batch_sz = 0
        self.dg_r = 0
        self.done = False
        self.trained_epochs = 0

        self.__dict__.update(**kwargs)

    @classmethod
    def load(cls, path):
        i = Info()
        with open(path, 'r') as f:
            i.__dict__ = json.load(f)
        return i

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        pass

    @classmethod
    def from_model(cls, model: BaseGAN, structure, batch_sz, dg_r=None):
        # BaseGAN info
        info = Info()
        info.__dict__['model'] = convert_numpy_types_to_natives(model.get_config())
        info.struct = structure
        info.batch_sz, info.dg_r = batch_sz, dg_r
        return info


def folder_name_getter(i: Info):
    """
    GAN_lrd={lr_d}_lrg={lr_g}_bs={batch_size}_dgr={dg_r}_opt={optimizer_type}_lf={latent_factor}_strct={json hash:X}
    Use hashlib for stable hashing.
    :return:
    """

    name = f'{i.gan_type}_lrd={i.lr_d}_lrg={i.lr_g}_bs={i.batch_sz}_dgr={i.dg_r}_opt={i.opt}' \
           f'_lf={i.latent_factor}_strct={i.struct}'
    return name


def get_cases_to_run(path: Path, repeat_times):
    cases_to_run = []
    if path.exists():
        try:
            existed_info = Info.load(path / 'config.json')
        except FileNotFoundError:
            print(f'Config not found , retrain all cases in path {path}')
            empty_directory(path)

            cases_to_run = list(range(repeat_times))
        else:
            trained_cases = set()
            for folder in sorted(path.glob('case*')):
                s = str(folder)
                case_num = int(s[s.find('-') + 1:])
                try:
                    case_info = Info.load(folder / 'config.json')
                except FileNotFoundError:
                    cases_to_run.append(case_num)
                else:
                    if not case_info.done:
                        cases_to_run.append(case_num)
                    else:
                        trained_cases.add(case_num)
                pass

            max_case = max(trained_cases) if trained_cases else -1
            assert len(trained_cases) == max_case + 1
            if existed_info.done and max_case + 1 == repeat_times:
                print(f'Training all done, skipped in path {path}')
            else:
                cases_to_run.extend([
                    i for i in range(repeat_times) if i not in trained_cases and i not in cases_to_run
                ])
    else:
        path.mkdir(parents=True)
        cases_to_run = list(range(repeat_times))  # run all cases
        pass
    return cases_to_run


def run_single_case(path, case, param_builder, model_builder, clear_before_run=True):
    kwargs = param_builder()

    case_path = path / f'case-{case}'
    case_path.mkdir(exist_ok=True)
    if clear_before_run:
        empty_directory(case_path)

    # write undone case-config
    info = kwargs['info']
    info.done = False
    info.save(case_path / 'config.json')

    # build model
    model: BaseGAN = model_builder(**kwargs)

    sampler = kwargs['sampler']
    sampler.set_path(case_path)

    training_params = kwargs['training_params']
    training_params['sampler'] = sampler
    losses, metrics = model.train(**training_params)

    # save the model
    model.save(case_path / 'model')
