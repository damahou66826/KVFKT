import subprocess

#datasets = ['assist2017','assist2012', 'Junyi', 'fsai','NeurIPS','EdNet']
datasets = ['NeurIPS']
memory_sizes = [10, 20, 50, 100]
state_dims = [10, 50, 100, 200]
forget_cycles = [60000, 600000, 6000000]
n_epochs = [10, 15, 20]

model_profiles = []
for dataset in datasets:
    for memory_size in memory_sizes:
        for state_dim in state_dims:
            for forget_cycle in forget_cycles:
                for n_epoch in n_epochs:
                    model_profiles.append(
                        {
                            'dataset': dataset,
                            'memory_size': memory_size,
                            'value_memory_state_dim': state_dim,
                            'key_memory_state_dim': state_dim,
                            'forget_cycle': forget_cycle,
                            'n_epoch': n_epoch
                        }
                    )

for model_profile in model_profiles:
    # base
    command = ["python", "main02.py"]

    # add dataset
    command.append("--dataset")
    command.append("{}".format(model_profile['dataset']))

    # add memory_size
    command.append("--memory_size")
    command.append("{}".format(model_profile['memory_size']))

    # add value_memory_state_dim
    command.append("--value_memory_state_dim")
    command.append("{}".format(model_profile['value_memory_state_dim']))

    # add memokey_memory_state_dimry_size
    command.append("--key_memory_state_dim")
    command.append("{}".format(model_profile['key_memory_state_dim']))

    # add forget_cycle
    command.append("--forget_cycle")
    command.append("{}".format(model_profile['forget_cycle']))

    # add n_epoch
    command.append("--n_epochs")
    command.append("{}".format(model_profile['n_epoch']))

    # run command
    print("run:", command)
    subprocess.run(command)