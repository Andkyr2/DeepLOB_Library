from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn as nn

def log_experiment(results:dict,
                    model:nn.Module = None,
                    config:dict = None,):

    RESULTS_DIR = Path("results")
    
    t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #create results directory
    results_dir = RESULTS_DIR/f"{t}"
    results_dir.mkdir(parents=True,exist_ok=True)
    #save results as json
    with open(results_dir/"results.json","w") as f:
        json.dump(results,f)
    #save model
    if model is not None:
        torch.save(model.state_dict(),results_dir/"model.pth")
    #save config
    if config is not None:
        with open(results_dir/"config.json","w") as f:
            json.dump(config,f)