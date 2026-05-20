import os
import json


def save_args(args, output_dir):
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(
            {k: v for k, v in vars(args).items()},
            f,
            indent=4
        )


