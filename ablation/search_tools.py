import csv

def save_results_to_csv(results, filename="grid_search_results.csv"):
    """
    Save grid search results

    :param results: grid search results
    :param filename: file to save to
    """
    all_fields = set()
    for res in results:
        params = res["parameters"]
        all_fields.update([
            "threshold_low", 
            "threshold_high",
            *params.keys()
        ])
        all_fields.discard("thresholds")
        all_fields.update(res["metrics"].keys())
        all_fields.add("iou")

    ordered_fields = [
        "threshold_low",
        "threshold_high",
        *sorted([f for f in all_fields if f in [
            "batch_size", 
            "loss_fn", 
            "optimizer", 
            "scheduler"
        ]]),
        "iou",
        *sorted([f for f in all_fields if f not in [
            "threshold_low",
            "threshold_high",
            "batch_size",
            "loss_fn",
            "optimizer",
            "scheduler",
            "iou"
        ]])
    ]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ordered_fields)
        writer.writeheader()
        for res in results:
            row = {}
            params = res["parameters"]
            row["threshold_low"] = params["thresholds"][0]
            row["threshold_high"] = params["thresholds"][1]
            for k, v in params.items():
                if k != "thresholds":
                    row[k] = v
            row["iou"] = res["iou"]
            for k, v in res["metrics"].items():
                row[k] = v
            for field in ordered_fields:
                if field not in row:
                    row[field] = None

            writer.writerow(row)
