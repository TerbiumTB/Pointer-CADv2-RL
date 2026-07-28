import os
import re
import json
import argparse
import statistics
import pandas as pd
from rich import print
from loguru import logger
from collections import Counter





def main():
    parser=argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("-i", "--input_path", default="./log/comparison", help="Predicted result")
    # parser.add_argument("--verbose",action='store_true')

    args = parser.parse_args()
    
    logger.info("Evaluation for Design History")

    result_dir = [os.path.join(args.input_path, dir_name) for dir_name in os.listdir(args.input_path)]
    json_paths = [os.path.join(d, "results.json") for d in result_dir if os.path.isfile(os.path.join(d, "results.json"))]

    results = []
    for json_path in json_paths:
        logger.info(f"Loading results from {json_path}")
        with open(json_path, "r") as fp:
            results.append(json.load(fp))

    model_ids = []
    for result in results:
        model_ids.append([mid for mid in list(result.keys()) if result[mid] is not None and result[mid]["status"] and result[mid]["chamfer distance"] <= 1000])
    model_ids = set.intersection(*[set(mids) for mids in model_ids])

    eval_results = []
    for result_path, result in zip(json_paths, results):
        error_message = []
        chamfer_distance = []
        line_f1 = []
        arc_f1 = []
        circle_f1 = []
        extrude_f1 = []
        chamfer_f1 = []
        fillet_f1 = []
        watertightness = []

        for model_id, data in result.items():
            if model_id in model_ids:
                assert data is not None and data["status"]
                if data["chamfer distance"] is not None:
                    chamfer_distance.append(data["chamfer distance"])
                if data["f1"] is not None:
                    if "line" in data["f1"]:
                        line_f1.append(data["f1"]["line"])
                    if "arc" in data["f1"]:
                        arc_f1.append(data["f1"]["arc"])
                    if "circle" in data["f1"]:
                        circle_f1.append(data["f1"]["circle"])
                    if "extrude" in data["f1"]:
                        extrude_f1.append(data["f1"]["extrude"])
                    if "chamfer" in data["f1"]:
                        chamfer_f1.append(data["f1"]["chamfer"])
                    if "fillet" in data["f1"]:
                        fillet_f1.append(data["f1"]["fillet"])
                if data["is watertight"] is not None:
                    watertightness.append(data["is watertight"])
    
        eval_dict = {}

        eval_dict["failure"] = {}
        eval_dict["failure"]["rate"] = (len(error_message) / len(result) if len(result) > 0 else 0) * 100
        error_types = error_message.copy()
        error_counter = Counter(error_types)
        eval_dict["failure"]["detail"] = {}
        total_failures = len(error_types)
        for error, count in error_counter.items():
            eval_dict["failure"]["detail"][error] = (count / total_failures if total_failures > 0 else 0) * 100
            
        eval_dict['chamfer distance'] = {}
        eval_dict['chamfer distance']['median'] = statistics.median(chamfer_distance)
        eval_dict['chamfer distance']['mean'] = statistics.mean(chamfer_distance)
        
        eval_dict['f1'] = {
            "line": statistics.mean(line_f1),
            "arc": statistics.mean(arc_f1),
            "circle": statistics.mean(circle_f1),
            "extrude": statistics.mean(extrude_f1),
            "chamfer": statistics.mean(chamfer_f1) if len(chamfer_f1) > 0 else "N/A",
            "fillet": statistics.mean(fillet_f1) if len(fillet_f1) > 0 else "N/A",
        }

        eval_dict['watertightness'] = statistics.mean(watertightness) * 100

        json_formatted_str = json.dumps(eval_dict, indent=4)
        print("\n\n")
        print("=" * 10, "Evaluation Results", "=" * 10)
        print(json_formatted_str)
        print("=" * 40)
        print("\n\n")

        report_path = os.path.join(args.input_path, f"{os.path.basename(os.path.dirname(result_path))}.json")
        with open(report_path, "w") as f:
            json.dump(eval_dict, f, indent=4)

        logger.success(f"Evaluation completed: evaluated {len(result)} items. Results saved to {report_path}")

if __name__=="__main__":
    main()
