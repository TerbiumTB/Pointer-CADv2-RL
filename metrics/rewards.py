import os
import sys
import copy
import math
import signal
import base64
import pickle
import itertools
import subprocess

from cadmodel.model import CADModel
from cadmodel.extrude import Extrude
from measurements.accuracy import accuracy
from measurements.chamfer_distance import chamfer_distance
from misc import TOKEN, STANDARD_PLANES



class Text2CADRewardWorker:
    def __init__(self, valid=1, chamfer_distance=1, length=1, accuracy=1, accuracy_v=1, 
                 accuracy_e=1, accuracy_f=1, chamfer_distance_num=5, chamfer_distance_target=0.2, 
                 chamfer_distance_transition=1000.0, chamfer_distance_cutoff=10000.0):
        self.weight_valid = valid
        self.weight_chamfer_distance = chamfer_distance
        self.weight_length = length
        self.weight_accuracy = accuracy
        self.weight_accuracy_v = accuracy_v
        self.weight_accuracy_e = accuracy_e
        self.weight_accuracy_f = accuracy_f
        self.cd_num = chamfer_distance_num
        self.cd_target = chamfer_distance_target
        self.cd_transition = chamfer_distance_transition
        self.cd_cutoff = chamfer_distance_cutoff


    def __map_cd_reward(self, cd_metric: float):
        # 1. 超烂，直接0
        if cd_metric >= self.cd_cutoff:
            return 0.0

        # 2. 中等 ~ 烂 (cd_transition~cd_cutoff)：线性衰减到 0.05
        if cd_metric >= self.cd_transition:
            return 0.05 * (self.cd_cutoff - cd_metric) / (self.cd_cutoff - self.cd_transition)

        # 3. 关键段 (<cd_transition)：指数高分辨率，但和上面在1000处连续
        alpha_fine = -math.log(0.05) / (self.cd_transition - self.cd_target)
        return math.exp(-alpha_fine * (cd_metric - self.cd_target))


    def calc(self, pred_ids, pred_parameter_map, pred_label, pred_parameter, pred_pointer, gt_response_length, gt_json):
        valid_reward = cd_reward = acc_reward = length_reward = 0
        valid_metric = cd_metric = length_metric = -1
        acc_metric = {"vertex": -1, "edge": -1, "face": -1}

        gt_model = CADModel.from_dict(gt_json)
        pred_model = copy.deepcopy(gt_model).submodel(-2)

        try:
            pred_vector = list(zip(pred_label, [max(x, y) for x, y in zip(pred_parameter, pred_pointer)]))
            pred_model_temp = copy.deepcopy(pred_model)
            pred_model_temp.from_vector(pred_vector, pred_parameter_map)
            pred_model = pred_model_temp
            valid_reward = 1
        except:
            try:
                pred_model_temp = copy.deepcopy(pred_model)
                pred_model_temp.from_vector(pred_vector, pred_parameter_map, strict=False)
                pred_model = pred_model_temp
                valid_reward = 0.5
            except:
                pass
        
        if valid_reward == 0:
            return 0, valid_reward, cd_reward, acc_reward, length_reward, valid_metric, cd_metric, acc_metric, length_metric

        assert isinstance(pred_model.seq[-1], Extrude) and isinstance(gt_model.seq[-1], Extrude), "The last operation must be Extrude."

        if pred_model.seq[-1].operation != gt_model.seq[-1].operation:
            valid_reward -= 0.25
        valid_metric = valid_reward == 1

        gt_model_last = CADModel([copy.deepcopy(gt_model.seq[-1])])
        pred_model_last = CADModel([copy.deepcopy(pred_model.seq[-1])])

        operation_valid = gt_model_last.seq[-1].operation == pred_model_last.seq[-1].operation
        gt_model_last.seq[-1].operation = "NewBodyFeatureOperation"
        pred_model_last.seq[-1].operation = "NewBodyFeatureOperation"

        try:
            cd_metric = sum([chamfer_distance(pred_model_last, gt_model_last, 8192) * 1000 for _ in range(self.cd_num)]) / self.cd_num
            cd_reward = self.__map_cd_reward(cd_metric * (1 if operation_valid else 1.2))
        except:
            pass

        try:
            v_acc, e_acc, f_acc = accuracy(pred_model_last, gt_model_last)
            if not isinstance(v_acc, float): v_acc = -1
            if not isinstance(e_acc, float): e_acc = -1
            if not isinstance(f_acc, float): f_acc = -1

            acc_metric = {"vertex": v_acc, "edge": e_acc, "face": f_acc}
            acc_reward = ((self.weight_accuracy_v * v_acc) if v_acc >= 0 else 0 + 
                          (self.weight_accuracy_e * e_acc if e_acc >= 0 else 0) + 
                          (self.weight_accuracy_f * f_acc if f_acc >= 0 else 0))
            acc_reward /= (self.weight_accuracy_v + self.weight_accuracy_e + self.weight_accuracy_f)
        except:
            pass

        pred_token_num = len(pred_ids) - pred_ids.count(151643)  # contain both plan and vector tokens
        length_metric = (pred_token_num + 1) / (gt_response_length + 1)
        length_reward = min((gt_response_length + 1) / (pred_token_num + 1), 1.2)

        return (
            self.weight_valid * valid_reward + self.weight_chamfer_distance * cd_reward + self.weight_accuracy * acc_reward + self.weight_length * length_reward,
            valid_reward, cd_reward, acc_reward, length_reward, valid_metric, cd_metric, acc_metric, length_metric
        )


    def __call__(self, pred_ids, pred_parameter_map, pred_label, pred_parameter, pred_pointer, gt_response_length, gt_json):
        rewards_list = []
        valid_list = []
        cd_list = []
        acc_list = []
        length_list = []
        valid_metric_list = []
        cd_metric_list = []
        acc_metric = {"vertex": [], "edge": [], "face": []}
        length_metric_list = []

        for sample in zip(pred_ids, pred_parameter_map, pred_label, pred_parameter, pred_pointer, itertools.repeat(gt_response_length), itertools.repeat(gt_json)):
            rewards, valid_r, cd_r, acc_r, length_r, valid_m, cd_m, acc_m, length_m = self.calc(*sample)
            rewards_list.append(rewards)
            valid_list.append(valid_r)
            cd_list.append(cd_r)
            acc_list.append(acc_r)
            length_list.append(length_r)
            valid_metric_list.append(valid_m)
            cd_metric_list.append(cd_m)
            acc_metric["vertex"].append(acc_m["vertex"])
            acc_metric["edge"].append(acc_m["edge"])
            acc_metric["face"].append(acc_m["face"])
            length_metric_list.append(length_m)

        return {"rewards": rewards_list,
                "valid": valid_list,
                "chamfer_distance": cd_list,
                "accuracy": acc_list,
                "length": length_list,
                "valid_metric": valid_metric_list,
                "chamfer_distance_metric": cd_metric_list,
                "accuracy_metric": acc_metric,
                "length_metric": length_metric_list}



class Text2CADReward:
    def __init__(self, valid=1, chamfer_distance=1, length=1, accuracy=1, accuracy_v=1, 
                 accuracy_e=1, accuracy_f=1, chamfer_distance_num=5, timeout: float = 60.0):
        self.weight_valid = valid
        self.weight_chamfer_distance = chamfer_distance
        self.weight_length = length
        self.weight_accuracy = accuracy
        self.weight_accuracy_v = accuracy_v
        self.weight_accuracy_e = accuracy_e
        self.weight_accuracy_f = accuracy_f
        self.cd_num = chamfer_distance_num
        self.timeout = timeout


    def __zero_result(self, n):
        return {
            "rewards": [0] * n,
            "valid": [0] * n,
            "chamfer_distance": [0] * n,
            "accuracy": [0] * n,
            "length": [0] * n,
            "valid_metric": [-1] * n,
            "chamfer_distance_metric": [-1] * n,
            "accuracy_metric": {
                "vertex": [-1] * n,
                "edge": [-1] * n,
                "face": [-1] * n,
            },
            "length_metric": [-1] * n,
        }


    def __call__(self, pred_ids, pred_parameter_map, pred_label, pred_parameter, pred_pointer, gt_response_length, gt_json):
        sample_num = len(pred_ids)

        payload = {
            "configs": {
                "valid": self.weight_valid,
                "chamfer_distance": self.weight_chamfer_distance,
                "length": self.weight_length,
                "accuracy": self.weight_accuracy,
                "accuracy_v": self.weight_accuracy_v,
                "accuracy_e": self.weight_accuracy_e,
                "accuracy_f": self.weight_accuracy_f,
                "chamfer_distance_num": self.cd_num,
            },
            "parameters":{
                "pred_ids": [t.tolist() for t in pred_ids],
                "pred_parameter_map": [{
                    "length": m["length"].tolist() if "length" in m else [],
                    "angle": m["angle"].tolist() if "angle" in m else []
                } for m in pred_parameter_map],
                "pred_label": [t.tolist() for t in pred_label],
                "pred_parameter": [t.tolist() for t in pred_parameter],
                "pred_pointer": [t.tolist() for t in pred_pointer],
                "gt_response_length": gt_response_length,
                "gt_json": gt_json,
            }
        }

        r_fd, w_fd = os.pipe()
        proc = subprocess.Popen(
            [sys.executable, "-m", __name__, "--worker", "--fd", str(w_fd)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=[w_fd],
            preexec_fn=os.setsid,
        )

        os.close(w_fd)

        encoded_data = base64.b64encode(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        )

        try:
            stdout, stderr = proc.communicate(input=encoded_data, timeout=self.timeout)
            # print(stdout.decode())
            # print(stderr.decode())
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.kill()
            os.close(r_fd)
            print(f"Process PID {proc.pid} killed due to timeout during reward calculation.")
            return self.__zero_result(sample_num)
        
        if proc.returncode == 0:
            try:
                result_bytes = os.read(r_fd, 1 << 24)
                os.close(r_fd)
                result = pickle.loads(base64.b64decode(result_bytes))
                return result
            except Exception as e:
                pass

        return self.__zero_result(sample_num)



def _run_worker():
    def timeout_handler(signum, frame):
        sys.stdout.write("[Worker] Timeout: Force killing process after 10 minutes\n")
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)

    try:
        fd = int(sys.argv[sys.argv.index("--fd") + 1])

        raw = sys.stdin.buffer.read()
        payload = pickle.loads(base64.b64decode(raw))

        worker = Text2CADRewardWorker(**payload["configs"])
        result = worker(**payload["parameters"])

        out_bytes = base64.b64encode(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
        os.write(fd, out_bytes)
        os.close(fd)
        sys.exit(0)
    except Exception as e:
        sys.stdout.write(f"[Worker] Failed: {e}\n")
        sys.exit(1)



if __name__ == "__main__":
    if "--worker" in sys.argv:
        _run_worker()

    # from tqdm import tqdm
    # from loguru import logger
    # from dataset.dataset import Text2CAD_Dataset, get_dataloaders

    # @logger.catch
    # def test():
    #     # dataset = Text2CAD_Dataset(
    #     #     dataset_dir="/mnt/afs/wangchenyu/dataset/cadv3/Base",
    #     #     split_filepath="/mnt/afs/wangchenyu/dataset/cadv3/train_val_test.json",
    #     #     subset="train"
    #     # )

    #     # for data in tqdm(dataset):
    #     #     print(Text2CADReward().calc([1] * 100, data[-2], data[3], data[5], data[7], 100, data[-1]))

    #     dataloader = get_dataloaders(
    #         dataset_dir="/mnt/afs/wangchenyu/dataset/cadv3/Base",
    #         split_filepath="/mnt/afs/wangchenyu/dataset/cadv3/train_val_test.json",
    #         subsets=["train"],
    #         batch_sizes=8,
    #         shuffle=True,
    #         pin_memory=True,
    #         num_workers=4,
    #         prefetch_factor=16
    #     )[0]
    #     for data in tqdm(dataloader):
    #         reward_dict = Text2CADReward()([torch.tensor([1] * 100)] * 8, data["parameter_map"], data["label"], data["parameter"], data["pointer"], [100] * 8, data["json"])
    #         print(reward_dict)

    # test()
