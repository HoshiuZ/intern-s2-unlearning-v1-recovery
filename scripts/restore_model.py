#!/usr/bin/env python3
"""将固定基模与单独分发的 B@26 恢复包合成为完整模型，整个过程仅使用 CPU。"""
from __future__ import annotations
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import zipfile
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def checked_path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts or not part.parts:
        raise ValueError(f'清单包含非法相对路径：{relative}')
    path = root.joinpath(*part.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'清单文件指向基模目录外部：{relative}')
    return path

def load_checkpoint(path: Path, manifest: dict, zipped: bool) -> dict:
    """只接受固定哈希的独立文件或 ZIP；ZIP 在内存解包，不额外写入磁盘。"""
    meta = manifest['checkpoint']
    actual = file_sha256(path)
    expected = manifest['archive']['sha256'] if zipped else meta['sha256']
    if actual != expected:
        raise ValueError(f'恢复包哈希不符：实际 {actual}，预期 {expected}')
    if zipped:
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != [meta['filename']]:
                raise ValueError('ZIP 内文件清单不符；应当只有固定的 B@26 检查点。')
            info = archive.getinfo(meta['filename'])
            if info.file_size != meta['bytes']:
                raise ValueError('ZIP 内检查点大小不符。')
            body = archive.read(meta['filename'])
            if hashlib.sha256(body).hexdigest() != meta['sha256']:
                raise ValueError('ZIP 内检查点哈希不符。')
        payload = torch.load(io.BytesIO(body), map_location='cpu', weights_only=True)
    else:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    validate_checkpoint(payload, manifest)
    return payload

def validate_checkpoint(payload: dict, manifest: dict) -> None:
    """检查固定步数、完整专家名单及每个 BF16 张量的内容身份。"""
    if payload.get('locked_step') != 26:
        raise ValueError('检查点不是锁定的第 26 步。')
    experts = payload.get('ablated_experts')
    if experts != manifest['ablated_experts'] or len(experts) != 384:
        raise ValueError('384 个消融专家名单不符。')
    pairs = [tuple(pair) for pair in experts]
    if len(set(pairs)) != 384 or any(not (0 <= l < 40 and 0 <= e < 256) for l, e in pairs):
        raise ValueError('消融专家名单存在重复或越界。')
    expected = manifest['shared_tensors']
    names = {f'model.language_model.layers.{i}.mlp.shared_expert.down_proj.weight' for i in range(40)}
    weights = payload.get('parameter_states', {})
    if set(weights) != names or set(expected) != names:
        raise ValueError('共享权重必须恰好覆盖第 0–39 层的 40 个 down_proj。')
    for name, tensor in weights.items():
        spec = expected[name]
        if tensor.dtype != torch.bfloat16 or list(tensor.shape) != spec['shape']:
            raise ValueError(f'张量类型或形状不符：{name}')
        if tensor_sha256(tensor) != spec['sha256']:
            raise ValueError(f'张量哈希不符：{name}')
    if payload.get('base_model_inventory_sha256') != manifest['base_inventory_sha256']:
        raise ValueError('检查点记录的基模身份不符。')

def verify_base(base: Path, manifest: dict) -> dict:
    """先校验全部基模文件和分片；校验通过前不创建输出目录。"""
    if not base.is_dir():
        raise ValueError('基模目录不存在。')
    entries = manifest['files']
    if len({entry['path'] for entry in entries}) != len(entries):
        raise ValueError('基模清单有重复路径。')
    for i, entry in enumerate(entries, 1):
        path = checked_path(base, entry['path'])
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'基模文件缺失或大小不符：{entry["path"]}')
        if file_sha256(path) != entry['sha256']:
            raise ValueError(f'基模文件哈希不符：{entry["path"]}')
        print(f'基模校验 {i}/{len(entries)}：{entry["path"]}', flush=True)
    index = read_json(base / 'model.safetensors.index.json')
    mapped = set(index['weight_map'].values())
    expected = {entry['path'] for entry in entries if entry['path'].endswith('.safetensors')}
    if mapped != expected:
        raise ValueError('基模分片索引与文件清单不一致。')
    return index

def routed_groups(experts: list) -> dict:
    groups = {}
    for layer, expert in experts:
        key = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        groups.setdefault(key, []).append(expert)
    return groups

def patch_tensor(name: str, tensor: torch.Tensor, weights: dict, groups: dict) -> torch.Tensor:
    """共享矩阵完整替换；路由专家只将指定行切片置零，不改变 router。"""
    if name in weights:
        replacement = weights[name]
        if tensor.shape != replacement.shape or tensor.dtype != replacement.dtype:
            raise ValueError(f'基模共享张量形状或类型不符：{name}')
        return replacement.clone().contiguous()
    if name in groups:
        if tensor.ndim != 3 or tensor.shape[0] != 256 or tensor.dtype != torch.bfloat16:
            raise ValueError(f'路由专家打包矩阵形状或类型不符：{name}')
        changed = tensor.clone()
        changed[groups[name]] = 0
        return changed.contiguous()
    return tensor

def write_patched_shard(source: Path, dest: Path, weights: dict, groups: dict) -> dict:
    """一次只处理一个分片；写完后逐张量重新读取并校验。"""
    expected = {}
    tmp = dest.with_suffix(dest.suffix + '.part')
    if dest.exists() or tmp.exists():
        raise ValueError(f'输出分片已经存在：{dest.name}')
    with safe_open(str(source), framework='pt', device='cpu') as reader:
        metadata = reader.metadata()
        tensors = {}
        for key in reader.keys():
            value = patch_tensor(key, reader.get_tensor(key), weights, groups)
            tensors[key] = value
            expected[key] = {'sha256': tensor_sha256(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}
        save_file(tensors, str(tmp), metadata=metadata)
        del tensors, value
    gc.collect()
    with safe_open(str(tmp), framework='pt', device='cpu') as reader:
        if set(reader.keys()) != set(expected) or reader.metadata() != metadata:
            raise ValueError(f'写出后的张量集合或分片元数据不符：{dest.name}')
        for key, spec in expected.items():
            value = reader.get_tensor(key)
            if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype'] or tensor_sha256(value) != spec['sha256']:
                raise ValueError(f'写出后的张量校验失败：{key}')
        del value
    tmp.replace(dest)
    return {'文件': dest.name, '状态': '已修改并逐张量校验', '张量数': len(expected), 'sha256': file_sha256(dest)}

def validate_output_path(base: Path, output: Path) -> tuple[Path, Path]:
    """在读取大体积基模前拒绝已有路径、符号链接或基模内部输出。"""
    if output.exists() or output.is_symlink():
        raise ValueError('输出目录已存在或是符号链接；请指定全新的目录。')
    base, resolved = base.resolve(), output.resolve()
    if resolved == base or resolved.is_relative_to(base):
        raise ValueError('输出目录不能等于基模目录或位于基模目录内部。')
    return base, resolved


def restore(base: Path, output: Path, payload: dict, base_manifest: dict, checkpoint_manifest: dict, index: dict) -> dict:
    base, output = validate_output_path(base, output)
    weights = payload['parameter_states']
    groups = routed_groups(payload['ablated_experts'])
    targets = set(weights) | set(groups)
    mapping = index['weight_map']
    if not targets <= set(mapping):
        raise ValueError('分片索引缺少需要恢复的目标张量。')
    target_shards = {mapping[key] for key in targets}
    entries = base_manifest['files']
    total = sum(entry['bytes'] for entry in entries)
    largest = max(entry['bytes'] for entry in entries)
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    required = total + largest + 1024**3
    if shutil.disk_usage(parent).free < required:
        raise ValueError(f'输出磁盘空间不足；保守至少需要 {required / 10**9:.2f} GB 可用空间。')
    output.mkdir(parents=True)
    marker = output / 'RECOVERY_INCOMPLETE.json'
    write_json(marker, {'状态': '恢复中，请勿用于推理', '检查点': checkpoint_manifest['checkpoint']['sha256']})
    report = {'状态': '恢复中', '锁定步数': 26, '共享矩阵数': 40, '消融专家数': 384,
              '检查点sha256': checkpoint_manifest['checkpoint']['sha256'], '基模清单摘要': checkpoint_manifest['base_inventory_sha256'],
              '分片': [], '非权重文件': [], '说明': '仅替换40个共享down_proj和384个路由专家down_proj切片；router、其它专家和全部多模态/MTP权重保持不变。'}
    try:
        for entry in entries:
            rel = entry['path']
            source, dest = checked_path(base, rel), checked_path(output, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f'处理：{rel}', flush=True)
            if rel in target_shards:
                report['分片'].append(write_patched_shard(source, dest, weights, groups))
            else:
                shutil.copyfile(source, dest)
                actual = file_sha256(dest)
                if actual != entry['sha256']:
                    raise ValueError(f'原样复制校验失败：{rel}')
                item = {'文件': rel, '状态': '原样复制且哈希一致', 'sha256': actual}
                report['分片' if rel.endswith('.safetensors') else '非权重文件'].append(item)
        report['状态'] = '恢复完成，全部输出已校验'
        write_json(output / 'recovery_report.json', report)
        marker.unlink()
        return report
    except BaseException as error:
        write_json(marker, {'状态': '恢复失败，保留现场，请勿用于推理', '错误': str(error), '说明': '工具不会覆盖已有输出或自动删除失败目录。'})
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description='用固定 Intern-S2-Preview 基模和 B@26 恢复包离线生成完整模型；仅使用 CPU。')
    parser.add_argument('--base', type=Path, required=True, help='包含23个原始权重分片及配置、分词器、自定义源码的完整基模目录')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--package', type=Path, help='恢复包压缩文件 B26_restore_weights_v1.zip')
    group.add_argument('--checkpoint', type=Path, help='已解压的 B26_locked_BF16.pt；与 --package 二选一')
    parser.add_argument('--output', type=Path, help='完整清除后模型的全新输出目录，不能指向基模目录')
    parser.add_argument('--verify-only', action='store_true', help='只核验基模和恢复包，不生成完整模型')
    parser.add_argument('--threads', type=int, default=4, help='CPU计算线程数，默认4')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads 必须是正整数')
    if not args.verify_only and args.output is None:
        parser.error('恢复模型时必须提供 --output；只做校验请加 --verify-only')
    if not args.verify_only:
        validate_output_path(args.base, args.output)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    base_manifest = read_json(ROOT / 'checksums/base_manifest.json')
    checkpoint_manifest = read_json(ROOT / 'checksums/checkpoint_manifest.json')
    if base_manifest['original_inventory_sha256'] != checkpoint_manifest['base_inventory_sha256']:
        raise ValueError('基模清单与恢复包清单的身份不一致。')
    payload = load_checkpoint(args.package or args.checkpoint, checkpoint_manifest, zipped=args.package is not None)
    print('恢复包身份、40个BF16矩阵和384专家名单校验通过。', flush=True)
    index = verify_base(args.base.resolve(), base_manifest)
    if args.verify_only:
        print('所有输入校验通过；未生成完整模型。', flush=True)
        return
    restore(args.base, args.output, payload, base_manifest, checkpoint_manifest, index)
    print(f'完整模型已恢复到：{args.output.resolve()}；耗时 {time.monotonic() - started:.1f} 秒。', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'恢复终止：{error}', file=sys.stderr)
        sys.exit(1)
