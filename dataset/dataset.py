import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
import os
import dgl
import pickle
from loguru import logger
from itertools import product
import torch.distributed as dist
from dgl.data.utils import load_graphs
from .augmentation import format_cad_data
from misc import STANDARD_PLANES, TOKEN
from cadmodel.model import convert_json_from_deepcad



class Text2CAD_Dataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split_filepath: str,
        subset: str,
        prompt_choices = ["abs", "exp"],
        max_graph_edge_num = 1000,
        max_graph_node_num = 200,
    ):
        """
        Args:
            cad_seq_dir (string): Directory with all the .pth files.
            prompt_path (string): Directory with all the .npz files.
            split_filepath (string): Train_Test_Val json file path.
            subset (string): "train", "test" or "val"
        """
        super(Text2CAD_Dataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.prompt_choices = prompt_choices
        self.max_graph_edge_num = max_graph_edge_num
        self.max_graph_node_num = max_graph_node_num

        # open spilt json
        with open(os.path.join(split_filepath), "r") as f:
            self.split = json.load(f)

        self.data_id = self.split[subset]

        logger.info(f"Found {len(self)} samples for {subset} split.")


    def __len__(self):
        return len(self.data_id) * len(self.prompt_choices)


    def __flatten_vector(self, data):
        if data[0][0] == TOKEN.index("<|sketch_start|>"):
            values, pointers = [[[]]], []
            for pair in data:
                if len(pair) == 2 and not isinstance(pair[0], list) and not isinstance(pair[1], list):
                    assert pair[0] != TOKEN.index("<|pointer_enable|>")
                    values[-1][0].append((pair[0], pair[1]))
                    pointers.append(-len(STANDARD_PLANES) - 1)
                elif len(pair) == 2 and not isinstance(pair[0], list) and isinstance(pair[1], list):
                    assert pair[0] == TOKEN.index("<|pointer_enable|>")
                    values[-1][0].append((pair[0], -len(STANDARD_PLANES) - 1))
                    pointers.append(pair[1])
                else:
                    value = []
                    for vector_list in pair:
                        value.append([(a, b) for a, b in vector_list])
                    values.append(value)
                    values.append([[]])
                    pointers.extend([-len(STANDARD_PLANES) - 1] * len(pair[0]))

            values_unpacked = [[item for sublist in v for item in sublist] for v in product(*values)]

            label_source = [[i[0] for i in item] for item in values_unpacked]
            parameter_source = [[i[1] for i in item] for item in values_unpacked]

            value_idx = random.randint(0, len(values_unpacked) - 1)
            label = label_source[value_idx].copy()
            parameter = parameter_source[value_idx].copy()

            pointer = [random.choice(item) if isinstance(item, list) else item for item in pointers]
            pointer_source = pointers.copy()
        elif data[0][0] in [TOKEN.index("<|chamfer_start|>"), TOKEN.index("<|fillet_start|>")]:
            assert len(data) == 4, "Chamfer and fillet vectors should have 4 items."
            pointers = list(set(data[2][1]))
            extra_item_num = len(pointers)

            label = [data[0][0], data[1][0]]
            label.extend([data[2][0]] * extra_item_num)
            label.append(data[3][0])

            parameter = [data[0][1], data[1][1]]
            parameter.extend([-len(STANDARD_PLANES) - 1] * extra_item_num)
            parameter.append(data[3][1])

            pointer = [data[0][1], -len(STANDARD_PLANES) - 1]
            shuffled_items = pointers[:]
            random.shuffle(shuffled_items)
            pointer.extend(shuffled_items)
            pointer.append(data[3][1])

            label_source = [label[:]]
            parameter_source = [parameter[:]]
            pointer_source = pointer.copy()
            pointer_source[2] = pointers[:]
            for idx in range(3, len(pointer_source) - 1):
                pointer_source[idx] = []

        return label, label_source, parameter, parameter_source, pointer, pointer_source


    def __clip_graph(self, g: dgl.DGLGraph, keep_nodes, keep_edges):
        """
        g: dgl.Graph
        keep_nodes: list[int] 必须保留的节点 id
        keep_edges: list[int] 必须保留的边 id
        """
        remove_eids = list(range(g.num_edges() // 2, g.num_edges()))
        g_pruned = dgl.remove_edges(g, remove_eids)

        all_edges = set(range(g_pruned.num_edges()))
        selected_edges = set(keep_edges)

        unused_nodes = set(keep_nodes)
        for eid in range(g_pruned.num_edges()):
            src, dst = g_pruned.find_edges(eid)
            if src.item() in unused_nodes and dst.item() in unused_nodes:
                selected_edges.add(eid)
                unused_nodes.remove(src.item())
                unused_nodes.remove(dst.item())
            if len(unused_nodes) == 0: break

        for nid in unused_nodes:
            in_eids = g_pruned.in_edges(nid, form='eid')   # 进入该节点的边
            out_eids = g_pruned.out_edges(nid, form='eid') # 从该节点出的边
            if len(in_eids) > 0: selected_edges.add(in_eids.tolist()[0])
            else: selected_edges.add(out_eids.tolist()[0])

        if len(selected_edges) < self.max_graph_edge_num // 2:
            remaining = list(all_edges - selected_edges)
            extra = remaining[:self.max_graph_edge_num // 2 - len(selected_edges)]
            selected_edges.update(extra)
        # elif len(selected_edges) > self.max_graph_edge_num // 2:
        #     selected_edges = set(list(selected_edges)[:self.max_graph_edge_num // 2])

        sg: dgl.DGLGraph = g_pruned.edge_subgraph(list(selected_edges), relabel_nodes=True, store_ids=True)

        new_nid: list = sg.ndata[dgl.NID].tolist()
        new_eid: list = sg.edata[dgl.EID].tolist()

        node_map = {keep_node: new_nid.index(keep_node) for keep_node in keep_nodes if keep_node in new_nid}
        edge_map = {keep_edge: new_eid.index(keep_edge) for keep_edge in keep_edges if keep_edge in new_eid}

        if sg.num_nodes() <= self.max_graph_node_num:
            graph = dgl.add_reverse_edges(sg, copy_ndata=True, copy_edata=True)
            return graph, node_map, edge_map
        
        selected_nodes = set(node_map.values())
        for eid in list(edge_map.values()) if len(edge_map) > 0 else [0]:
            src, dst = sg.find_edges(eid)
            selected_nodes.add(src.item())
            selected_nodes.add(dst.item())

        if len(selected_nodes) < self.max_graph_node_num:
            remaining = list(set(range(sg.num_nodes())) - selected_nodes)
            extra = remaining[:self.max_graph_node_num - len(selected_nodes)]
            selected_nodes.update(extra)
        # elif len(selected_nodes) > self.max_graph_node_num:
        #     selected_nodes = set(list(selected_nodes)[:self.max_graph_node_num])

        ssg = sg.subgraph(list(selected_nodes), relabel_nodes=True, store_ids=True)
        ssg_new_nid: list = ssg.ndata[dgl.NID].tolist()
        ssg_new_eid: list = ssg.edata[dgl.EID].tolist()

        ssg_node_map = {keep_node: ssg_new_nid.index(keep_node_sg) for keep_node, keep_node_sg in node_map.items() if keep_node_sg in ssg_new_nid}
        ssg_edge_map = {keep_edge: ssg_new_eid.index(keep_edge_sg) for keep_edge, keep_edge_sg in edge_map.items() if keep_edge_sg in ssg_new_eid}

        graph = dgl.add_reverse_edges(ssg, copy_ndata=True, copy_edata=True)
        return graph, ssg_node_map, ssg_edge_map


    def __prepare_data(self, chunk, model_id, part_id, prompt_choice):
        ################ Load Prompt ################
        prompt_path = os.path.join(self.dataset_dir, chunk, model_id, f"prompt_{prompt_choice}.txt")
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt = file.read()

        ################ Load Plan ################
        if self.subset != "test":
            plan_path = os.path.join(self.dataset_dir, chunk, model_id, "plan", f"{model_id}_{part_id}.txt")
            with open(plan_path, "r", encoding="utf-8") as file:
                plan = file.read()
        else:
            plan = ""

        ################ Load Parameter ################
        if self.subset != "test":
            parameter_path = os.path.join(self.dataset_dir, chunk, model_id, "parameter", f"{model_id}_{part_id}.pkl")
            with open(parameter_path, "rb") as f:
                parameter_source = pickle.load(f)
                parameter = {
                    "length": torch.tensor([parameter_source["length"][idx]["value_m"] for idx in range(1, len(parameter_source["length"]) + 1)], dtype=torch.float32),
                    "angle": torch.tensor([parameter_source["angle"][idx]["value"] for idx in range(1, len(parameter_source["angle"]) + 1)], dtype=torch.float32)
                }
        else:
            parameter_source = {
                "length": [],
                "angle": []
            }
            parameter = {
                "length": torch.tensor([]),
                "angle": torch.tensor([])
            }

        ################ Load Vector ################
        if self.subset != "test":
            vec_path = os.path.join(self.dataset_dir, chunk, model_id, "vector", f"{model_id}_{part_id}.pkl")
            with open(vec_path, "rb") as f:
                vector = self.__flatten_vector(pickle.load(f))

                vector_label = torch.tensor(vector[0])
                vector_label_source = torch.tensor(vector[1])
                vector_parameter = torch.tensor(vector[2])
                vector_parameter_source = torch.tensor(vector[3])
                vector_pointer = torch.tensor(vector[4])
                vector_pointer_source = vector[5]

                assert vector_label.shape == vector_parameter.shape == vector_pointer.shape
                assert vector_label_source.shape == vector_parameter_source.shape
        else:
            vector_label = torch.tensor([])
            vector_label_source = torch.tensor([])
            vector_parameter = torch.tensor([])
            vector_parameter_source = torch.tensor([])
            vector_pointer = torch.tensor([])
            vector_pointer_source = []

        ################ Load Graph ################
        graph_path = os.path.join(self.dataset_dir, chunk, model_id, "graph", f"{model_id}_{part_id}.bin")
        if os.path.exists(graph_path):
            graph = load_graphs(graph_path)[0][0]
            graph.ndata["x"] = graph.ndata["x"].type(torch.float32)
            graph.edata["x"] = graph.edata["x"].type(torch.float32)
            if len(graph.ndata["x"].shape) == 4: graph.ndata["x"][:, :, :, -2] = torch.clamp(graph.ndata["x"][:, :, :, -2], min=-10, max=10)
            if len(graph.edata["x"].shape) == 3: graph.edata["x"][:, :, -3:] = torch.clamp(graph.edata["x"][:, :, -3:], min=-100, max=100)

            if graph.num_edges() > self.max_graph_edge_num or graph.num_nodes() > self.max_graph_node_num:
                used_face, used_edge = [], []
                for v, p in zip(vector_label, vector_pointer_source[1:]):
                    if isinstance(p, list):
                        if v == TOKEN.index("<|sketch_start|>"):
                            used_face.extend([x for x in p if x >= 0])
                        else:
                            used_edge.extend(p)

                graph, graph_face_map, graph_edge_map = self.__clip_graph(graph, used_face, used_edge)

                for idx in range(1, len(vector_pointer_source)):
                    if isinstance(vector_pointer_source[idx], list):
                        if vector_label[idx - 1] == TOKEN.index("<|sketch_start|>"):
                            pointer_map = graph_face_map
                        else:
                            pointer_map = graph_edge_map

                        if vector_pointer[idx].item() >= 0: vector_pointer[idx] = pointer_map[vector_pointer[idx].item()]
                        vector_pointer_source[idx] = [pointer_map[x] if x >= 0 else x for x in vector_pointer_source[idx]]

            if '_ID' in graph.ndata: del graph.ndata['_ID']
            if '_ID' in graph.edata: del graph.edata['_ID']
        else:
            graph = dgl.graph(([], []))
            graph.ndata['x'] = torch.zeros((0,), dtype=torch.float32)
            graph.edata["x"] = torch.zeros((0,), dtype=torch.float32)

        ################ Load Json ################
        json_path = os.path.join(self.dataset_dir, chunk, model_id, "json", f"{model_id}_{part_id}.json")
        if os.path.exists(json_path):
            # deepcad format json file
            with open(json_path, "r") as fp:
                json_dict = convert_json_from_deepcad(json.load(fp))
        else:
            json_path = os.path.join(self.dataset_dir, chunk, model_id, f"{model_id}.json")
            with open(json_path, "r") as fp:
                json_dict = json.load(fp)
                if 'properties' in json_dict:
                    json_dict = convert_json_from_deepcad(json_dict)
            sequence = json_dict["sequence"]
            json_dict["sequence"] = []
            for item in sequence:
                if "index" in item and item["index"] <= int(part_id):
                    json_dict["sequence"].append(item)

        return chunk, model_id, part_id, vector_label, vector_label_source, vector_parameter, vector_parameter_source, vector_pointer, vector_pointer_source, prompt, plan, graph, parameter, json_dict

    def __getitem__(self, idx):
        prompt_choice = self.prompt_choices[int(idx / len(self.data_id))]
        chunk, model_id, part_id = self.data_id[idx % len(self.data_id)].split("_")

        data = self.__prepare_data(chunk, model_id, part_id, prompt_choice)
        chunk, model_id, part_id, vector_label, vector_label_source, vector_parameter, vector_parameter_source, vector_pointer, vector_pointer_source, prompt, plan, graph, parameter, json_dict = data
        return chunk, model_id, part_id, vector_label, vector_label_source, vector_parameter, vector_parameter_source, vector_pointer, vector_pointer_source, format_cad_data(prompt), plan, graph, parameter, json_dict


def collate(batch):
    chunks = []
    model_ids = []
    part_ids = []
    labels = []
    label_sources = []
    parameters = []
    parameter_sources = []
    pointers = []
    pointer_sources = []
    prompts = []
    plans = []
    graphs = []
    parameter_maps = []
    jsons = []

    for sample in batch:
        chunk, model_id, part_id, vector_label, vector_label_source, vector_parameter, vector_parameter_source, vector_pointer, vector_pointer_source, prompt, plan, graph, parameter, json_dict = sample

        chunks.append(chunk)
        model_ids.append(model_id)
        part_ids.append(part_id)
        labels.append(vector_label)
        label_sources.append(vector_label_source)
        parameters.append(vector_parameter)
        parameter_sources.append(vector_parameter_source)
        pointers.append(vector_pointer)
        pointer_sources.append(vector_pointer_source)
        prompts.append(prompt)
        plans.append(plan)
        graphs.append(graph)
        parameter_maps.append(parameter)
        jsons.append(json_dict)

    return {
        'chunk': chunks,  # list of strings
        'model_id': model_ids,  # list of strings
        'part_id': part_ids,  # list of strings
        'label': labels,  # list of tensor
        'label_source': label_sources,  # list of tensor
        'parameter': parameters,  # list of tensor
        'parameter_source': parameter_sources,  # list of tensor
        'pointer': pointers,  # list of tensor
        'pointer_source': pointer_sources,  # list of list
        'prompt': prompts,  # list of strings
        'plan': plans,  # list of strings
        'graph': dgl.batch(graphs),
        'parameter_map': parameter_maps,  # list of dict
        'json': jsons,
    }


def get_dataloaders(
    dataset_dir: str,
    split_filepath: str,
    subsets: list[str],
    batch_sizes,
    shuffle: bool = True,
    pin_memory: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 16,
    prompt_choices = ["abs", "exp"],
):
    """
    Generate a DataLoader for the Text2CADDataset.

    Args:
    - cad_seq_dir (str): The directory containing the CAD sequence files.
    - prompt_path (str): The path to the CSV file containing the prompts.
    - split_filepath (str): The path to the JSON file containing the train/test/validation split.
    - subsets (list[str]): The subset to use ("train", "test", or "val").
    - batch_size (int): The batch size.
    - shuffle (bool): Whether to shuffle the data.
    - pin_memory (bool): Whether to pin memory.
    - num_workers (int): The number of workers.
    - prefetch_factor (int): The prefetch factor.

    Returns:
    - dataloader (torch.utils.data.DataLoader): The DataLoader object.
    """

    all_dataloaders = []

    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes] * len(subsets)

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = None  # os.sched_getaffinity not available on this OS
    if cpu_count is not None and num_workers is not None and num_workers > cpu_count:
        logger.warning(f"num_workers ({num_workers}) is greater than the number of CPU cores ({cpu_count}). This may cause performance issues.")

    for subset, batch_size in zip(subsets, batch_sizes):
        # Create an instance of the Text2CADDataset
        dataset = Text2CAD_Dataset(
            dataset_dir=dataset_dir,
            split_filepath=split_filepath,
            subset=subset,
            prompt_choices=prompt_choices
        )

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=True
        ) if dist.is_initialized() else None

        # Create a DataLoader with the specified parameters
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if dist.is_initialized() else shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,  # Set to True if using CUDA
            prefetch_factor=prefetch_factor,
            sampler=sampler,
            collate_fn=collate
        )
        all_dataloaders.append(dataloader)

    return all_dataloaders


# wrong data 00968240_00006 
if __name__ == "__main__":
    from tqdm import tqdm
    
    # dataset = Text2CAD_Dataset(
    #     dataset_dir="/mnt/afs_01e/mayi-folder/ParamCAD/dataset/fusion360/format/pointercadv2/dataset",
    #     split_filepath="/mnt/afs_01e/mayi-folder/ParamCAD/dataset/fusion360/format/pointercadv2/train_val_test.json",
    #     subset="test"
    # )

    # print(dataset[1])

    # for idx in tqdm(range(len(dataset))):
    #     print(f"index: {idx}")
    #     chunk, model_id, part_id, vector_label, vector_label_source, vector_parameter, vector_parameter_source, vector_pointer, vector_pointer_source, prompt, plan, graph, parameter, json_dict = dataset[idx]
    #     print(parameter)

    #     feature_list = ["None"] * 100
    #     for feature in json_dict["sequence"]:
    #         if "index" in feature:
    #             feature_list[feature["index"]] = feature["type"]
    #     feature_map = {"Sketch": 3, "ExtrudeFeature": 3, "ChamferFeature": 5, "FilletFeature": 6, "None": 0}
    #     feature_id = [feature_map[f] for f in feature_list]
    #     if vector_value[0] != feature_id[int(part_id)]:
    #         print("feature error", chunk, model_id, part_id, vector_value, feature_id)
    #     if torch.max(vector_value) >= len(TOKEN) + 2 ** 8 or torch.min(vector_value) < 0:
    #         print("token error", chunk, model_id, part_id, vector_value)
        # src, dst = graph.edges()
        # print(model_id, vector_pointer_source, "\n", list(zip(src.tolist(), dst.tolist())))
        # print()
        # if (graph.num_edges() > 1000 or graph.num_nodes() > 200):
        #     print(graph.num_edges(), graph.num_nodes(), model_id, part_id)

        # if model_id in ["00057549", "00053795"]:
        #     print(graph.num_edges(), graph.num_nodes(), model_id, part_id)
        # if [] in vector_pointer_source:
        #     pointer_list = vector_pointer_source[2]
        #     if len(pointer_list) != len(set(pointer_list)):
        #         print("pointer error", chunk, model_id, part_id, vector_pointer_source)
        # print(vector_value, vector_value_source, vector_pointer, vector_pointer_source)


    dataloader = get_dataloaders(
        dataset_dir="/mnt/afs_01e/mayi-folder/ParamCAD/dataset/fusion360/format/pointercadv2/dataset",
        split_filepath="/mnt/afs_01e/mayi-folder/ParamCAD/dataset/fusion360/format/pointercadv2/train_val_test.json",
        subsets=["test"],
        batch_sizes=8,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=16
    )[0]

    for data in tqdm(dataloader):
        pass
        # print(data["model_id"])
        # print(data["value_source"])
        # print(data["pointer"])
        # print(data["pointer_source"])
