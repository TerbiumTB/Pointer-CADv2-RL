import re
import random



# 单位配置：含缩写、全称、换算比例、采样概率
UNIT_CONFIG = {
    'm':  {'full': 'meter',      'scale': 1,     'prob': 0.1},
    'cm': {'full': 'centimeter', 'scale': 100,   'prob': 0.3},
    'mm': {'full': 'millimeter', 'scale': 1000,  'prob': 0.6}
}


def enhance_text_with_units_en(
    text,
    main_unit='mm',
):
    def format_float(val):
        return f"{val:.5f}".rstrip('0').rstrip('.') if '.' in f"{val:.5f}" else str(val)

    def replace_fn(match):
        original_val = float(match.group(1))

        select_unit = main_unit
        if abs(original_val) > 1: select_unit = "m"
        elif 1e-6 < abs(original_val) < 0.1: select_unit = "mm"

        scale_factor = UNIT_CONFIG[select_unit]['scale']
        new_val = original_val * scale_factor
        formatted_val = format_float(new_val)

        space = random.choice([' ', '', '', ''])

        return f"{formatted_val}{space}{select_unit}"
    
    return re.sub(r"<v>(-?[\d\.]+)</v>", replace_fn, text)


def format_cad_data(prompt):
    """
    确定性格式化 CAD 文本与参数 (只用单位缩写 m 或 mm)
    """

    # 自动选择单位：只考虑 m 和 mm
    all_parameters = re.findall(r"<v>(-?\d+(?:\.\d+)?)</v>", prompt)
    all_parameters_float = [abs(float(v)) for v in all_parameters if abs(float(v)) > 1e-6]

    m_rate = sum(1 for v in all_parameters_float if v >= 0.1) / len(all_parameters_float) if all_parameters_float else 0
    mm_rate = sum(1 for v in all_parameters_float if v < 1) / len(all_parameters_float) if all_parameters_float else 0

    # 文本增强
    enhanced_prompt = enhance_text_with_units_en(
        text=prompt,
        main_unit='m' if m_rate >= mm_rate + 0.1 else 'mm',
    )

    return enhanced_prompt
