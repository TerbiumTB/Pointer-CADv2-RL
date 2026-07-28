from misc import create_mesh
from cadmodel.model import CADModel



def is_watertight(pred: CADModel):
    pred_mesh = create_mesh(pred)
    
    return pred_mesh.is_watertight



if __name__ == "__main__":
    from loguru import logger
    from cadmodel.model import CADModel

    @logger.catch()
    def test():
        import json
        from cadmodel.model import convert_json_from_deepcad

        with open("/public/home/qidacheng/workspace/cad/Cad_Parser/examples/test.json", "r") as fp:
            data = json.load(fp)
            if 'properties' in data:
                data = convert_json_from_deepcad(data)

        model = CADModel.from_dict(data)
        print(is_watertight(model))
    test()