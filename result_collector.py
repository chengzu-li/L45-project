import os
import argparse
import json
import pandas as pd
import numpy as np


def collect_results(args, model_names, experiment_dir, task_names, aggrs):
    for model_name in model_names:
        model_result = {}
        for task_name in task_names:
            model_result[task_name] = {}
            for aggr in aggrs:
                searching_dir = os.path.join(experiment_dir, f"output-{model_name}/{task_name}/{aggr}")
                if not args.multiple_seeds:
                    print("Only use one seed: 42")
                    random_seeds = ["42"]
                else:
                    random_seeds = []
                    for i in os.listdir(searching_dir):
                        try:
                            _ = int(i)
                            random_seeds.append(i)
                        except:
                            pass

                seed_acc = []
                seed_comp = []
                for seed in random_seeds:
                    result_dir = os.path.join(searching_dir, seed)
                    acc_file = os.path.join(result_dir, "accuracy.txt")
                    completion_file = os.path.join(result_dir, "results.txt")

                    with open(acc_file, "rb") as f:
                        acc = f.readline()
                        acc = str(acc).split("[")[-1]
                        acc = acc.split(']')[0].split(",")
                        acc = [float(i.strip("\' ").split(": ")[-1]) for i in acc]

                    with open(completion_file, 'r') as f:
                        comp = f.readline()
                        comp = float(comp.split(",")[-1].split(']')[0].strip("' "))
                    seed_acc.append(acc[-1])
                    seed_comp.append(comp)

                final_results = {
                    "test_acc": "{:.4f}+-{:.4f}".format(float(np.mean(seed_acc)), float(np.std(seed_acc))),
                    "completion_score": "{:.4f}+-{:.4f}".format(float(np.mean(seed_comp)), float(np.std(seed_comp)))
                }

                model_result[task_name][aggr] = final_results

        # convert to dataframe
        result_dict = {}
        result_dict['Aggr'] = aggrs
        for task_name in task_names:
            result_dict[f"{task_name}-acc"] = [model_result[task_name][aggr]['test_acc'] for aggr in aggrs]
            result_dict[f"{task_name}-completion"] = [model_result[task_name][aggr]['completion_score'] for aggr in aggrs]

        df = pd.DataFrame.from_dict(result_dict)
        df.set_index('Aggr')

        output_file_path = os.path.join(experiment_dir, f"output-{model_name}/{model_name}-result-seed{random_seeds}.csv")
        df.to_csv(output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs="*",
                        default=['customize'],
                        choices=['customize', 'novel_node', 'novel_edge',
                                 'novel_sim', 'gat', 'novel_gat', 'gat_n_sim'])
    parser.add_argument("--task_names", nargs="*",
                        default=['BA_Shapes', 'BA_Community', 'Tree_Cycle', 'Cora'],
                        choices=['BA_Shapes', 'BA_Community', 'Tree_Cycle', 'Cora'])
    parser.add_argument("--aggrs", nargs="*",
                        default=['add', 'mean', 'max', 'min', 'multi'],
                        choices=['add', 'mean', 'max', 'min', 'multi'])
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--multiple_seeds", action="store_true")

    args = parser.parse_args()

    collect_results(
        args=args,
        model_names=args.model_names,
        experiment_dir=args.experiment_dir,
        task_names=args.task_names,
        aggrs=args.aggrs
    )
