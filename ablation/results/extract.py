import json

def extract_top_models(file_path, top_n=3):
    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    data = data[0]['trials']
    # Sort the trials by the best_model's best_iou in descending order
    sorted_trials = sorted(data, key=lambda trial: trial['best_model']['best_iou'], reverse=True)
    
    # Select the top_n models
    top_models = sorted_trials[:top_n]
    
    return top_models

def print_model_parameters(models):
    for idx, model in enumerate(models, 1):
        trial_num = model.get("trial_number", "N/A")
        parameters = model.get("parameters", {})
        best_model = model['best_model']
        print(f"Model {idx} (Trial {trial_num}):")
        print(f"  Best epoch: {best_model['epoch']}")
        print(f"  Best IoU: {best_model['best_iou']}")
        print(f"  Accuracy: {best_model['test']['accuracy']}")
        print("  Parameters:")
        for key, value in parameters.items():
            if key == "scheduler":
                continue
            print(f"    {key}: {value}")
        # print("  Test metrics:")
        # for key, value in best_model['test'].items():
        #     print(f"    {key}: {value}")
        print("-" * 40)

if __name__ == '__main__':
    file_path = 'parsed_segnext.txt'
    top_models = extract_top_models(file_path, top_n=3)
    print("Top 3 models by best IoU:")
    print_model_parameters(top_models)
