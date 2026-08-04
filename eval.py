import os
import re
import json
import argparse
import statistics
from rich import print
from loguru import logger
from collections import Counter


def mean_or_na(values):
    return statistics.mean(values) if values else "N/A"


def median_or_na(values):
    return statistics.median(values) if values else "N/A"


def main():
    parser=argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("-i", "--input_path", default="./log", help="Predicted result")
    parser.add_argument("-d", "--date_format", action='store_true', default=False)
    # parser.add_argument("-I", "--intersection", action='store_true', default=False)
    # parser.add_argument("--verbose",action='store_true')

    args = parser.parse_args()
    
    logger.info("Evaluation for Design History")

    json_path = None
    if os.path.isdir(args.input_path):
        json_pathes = []
        pattern = r'\b\d{4}-\d{2}-\d{2}/\d{2}:\d{2}\b'
        for root, dirs, files in os.walk(args.input_path):
            if args.date_format:
                match = re.search(pattern, root)
                if not match:
                    continue

            for file in files:
                if file.endswith(".json"):
                    json_pathes.append(os.path.join(root, file))
        
        json_path = sorted(json_pathes)[-1]
    else:
        json_path = args.input_path

    logger.info(f"Loading results from {json_path}")
    with open(json_path, "r") as fp:
        results: dict = json.load(fp)

    error_message = []
    chamfer_distance = []
    line_f1 = []
    arc_f1 = []
    circle_f1 = []
    extrude_f1 = []
    chamfer_f1 = []
    fillet_f1 = []
    watertightness = []

    for model_id, data in results.items():
        if data is not None and data["status"]:
            if data["chamfer distance"] is not None:
                cd = data["chamfer distance"]
                if cd >= 1000:
                    logger.warning(f"Abnormal chamfer distance {cd} for model {model_id}")
                else:
                    chamfer_distance.append(cd)
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
        else:
            if data is None:
                error_message.append("No response")
            else:
                error_message.append(data["error_message"])
    
    eval_dict = {}

    eval_dict["failure"] = {}
    eval_dict["failure"]["rate"] = (len(error_message) / len(results) if len(results) > 0 else 0) * 100
    error_types = error_message.copy()
    error_counter = Counter(error_types)
    eval_dict["failure"]["detail"] = {}
    total_failures = len(error_types)
    for error, count in error_counter.items():
        eval_dict["failure"]["detail"][error] = (count / total_failures if total_failures > 0 else 0) * 100
        
    eval_dict['chamfer distance'] = {}
    eval_dict['chamfer distance']['median'] = median_or_na(chamfer_distance)
    eval_dict['chamfer distance']['mean'] = mean_or_na(chamfer_distance)
    
    eval_dict['f1'] = {
        "line": mean_or_na(line_f1),
        "arc": mean_or_na(arc_f1),
        "circle": mean_or_na(circle_f1),
        "extrude": mean_or_na(extrude_f1),
        "chamfer": mean_or_na(chamfer_f1),
        "fillet": mean_or_na(fillet_f1),
    }

    eval_dict['watertightness'] = (
        statistics.mean(watertightness) * 100 if watertightness else "N/A"
    )

    json_formatted_str = json.dumps(eval_dict, indent=4)
    print("\n\n")
    print("=" * 10, "Evaluation Results", "=" * 10)
    print(json_formatted_str)
    print("=" * 40)
    print("\n\n")

    report_path = os.path.join(os.path.dirname(json_path), "report.json")
    with open(report_path, "w") as f:
        json.dump(eval_dict, f, indent=4)

    logger.success(f"Evaluation completed: evaluated {len(results)} items. Results saved to {report_path}")

if __name__=="__main__":
    main()
