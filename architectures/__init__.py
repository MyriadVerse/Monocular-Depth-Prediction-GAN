import importlib
from architectures.base_architecture import BaseArchitecture


def find_architecture_using_name(arch_name):
    # 给定选项 --architecture [architecture]
    # 将导入文件 "architectures/{}_architecture.py"
    arch_filename = "architectures." + arch_name + "_architecture"
    arch_lib = importlib.import_module(arch_filename)

    # 在文件中，名为 [ArchiterctureName]Architecture() 的类将被实例化。
    # 它必须是 BaseArchitecture 的子类，并且大小写不敏感
    architecture = None
    target_model_name = arch_name.replace("_", "") + "architecture"
    for name, cls in arch_lib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(
            cls, BaseArchitecture
        ):
            architecture = cls

    if architecture is None:
        print(
            "No architecture class with name {} was found in {}.py,".format(
                target_model_name, arch_filename
            )
        )
        exit(0)

    return architecture


def create_architecture(args):
    model = find_architecture_using_name(args.architecture)
    instance = model(args)
    print("architecture [{}] was created".format(instance.architecture))
    return instance
