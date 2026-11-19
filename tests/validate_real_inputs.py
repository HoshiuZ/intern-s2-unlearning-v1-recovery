"""只用 CPU 检查真实恢复包、真实目标张量及一个真实专家打包矩阵。"""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import platform
import time
import torch
from safetensors import safe_open
import safetensors
import numpy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b26_restore_real_check', ROOT / 'scripts/restore_model.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def main():
    parser = argparse.ArgumentParser(description='真实恢复包和目标张量的 CPU 验证；不生成完整模型。')
    parser.add_argument('--base', type=Path, required=True, help='完整的未编辑基模目录')
    parser.add_argument('--package', type=Path, required=True, help='单独分发的 B@26 ZIP')
    parser.add_argument('--report', type=Path, required=True, help='写出本次中文验证报告的位置')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('报告路径已经存在，请指定新的文件名。')
    torch.set_num_threads(4)
    started = time.monotonic()
    bm = m.read_json(ROOT / 'checksums/base_manifest.json')
    cm = m.read_json(ROOT / 'checksums/checkpoint_manifest.json')
    if bm['original_inventory_sha256'] != cm['base_inventory_sha256']:
        raise ValueError('两份清单记录的基模身份不一致。')
    payload = m.load_checkpoint(args.package, cm, zipped=True)
    index_path = args.base / 'model.safetensors.index.json'
    index_spec = next(z for z in bm['files'] if z['path'] == index_path.name)
    if m.file_sha256(index_path) != index_spec['sha256']:
        raise ValueError('真实基模索引哈希不符。')
    index = m.read_json(index_path)['weight_map']
    weights = payload['parameter_states']
    groups = m.routed_groups(payload['ablated_experts'])
    by_shard = {}
    for name in set(weights) | set(groups):
        by_shard.setdefault(index[name], []).append(name)
    checked = 0
    shared = 0
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(args.base / shard), framework='pt', device='cpu') as reader:
            for name in names:
                value = reader.get_tensor(name)
                expected_shape = cm['shared_tensors'][name]['shape'] if name in weights else [256, 2048, 512]
                if list(value.shape) != expected_shape or value.dtype != torch.bfloat16:
                    raise ValueError(f'真实基模目标张量形状或类型不符：{name}')
                checked += 1
                if name in weights:
                    changed = m.patch_tensor(name, value, weights, groups)
                    if not torch.equal(changed, weights[name]) or m.tensor_sha256(changed) != cm['shared_tensors'][name]['sha256']:
                        raise ValueError(f'真实共享矩阵替换校验失败：{name}')
                    shared += 1
                    del changed
                del value
    name = 'model.language_model.layers.20.mlp.experts.down_proj'
    selected = groups[name]
    with safe_open(str(args.base / index[name]), framework='pt', device='cpu') as reader:
        original = reader.get_tensor(name)
        before = m.tensor_sha256(original)
        changed = m.patch_tensor(name, original, weights, groups)
        for expert in range(256):
            if expert in selected:
                if torch.count_nonzero(changed[expert]).item() != 0:
                    raise ValueError('真实专家切片没有被置零。')
            elif not torch.equal(changed[expert], original[expert]):
                raise ValueError('真实非目标专家被修改。')
        if m.tensor_sha256(original) != before:
            raise ValueError('源基模张量被原地修改。')
    report = {'状态': '通过', '验证时间': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              '环境': {'Python': platform.python_version(), 'PyTorch': str(torch.__version__), 'safetensors': safetensors.__version__, 'NumPy': numpy.__version__, '设备': 'CPU'},
              '恢复包SHA256': m.file_sha256(args.package), '检查点SHA256': cm['checkpoint']['sha256'],
              '恢复程序SHA256': m.file_sha256(ROOT / 'scripts/restore_model.py'),
              '真实检查点矩阵身份校验数': 40, '真实目标张量形状类型检查数': checked, '真实共享矩阵替换检查数': shared,
              '真实路由矩阵验证层': 20, '该层置零专家数': len(selected), '该层非目标专家保持不变': True, '源基模张量未被原地修改': True,
              '生成完整模型': False, 'GPU使用': False,
              '范围说明': '此步骤校验真实恢复包和目标张量，不是完整约73GB模型的落盘或推理实测。完整基模逐文件哈希校验请使用恢复程序的 --verify-only。',
              '耗时秒': time.monotonic() - started}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    m.write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main()
