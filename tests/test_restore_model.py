"""CPU 合成分片测试：验证恢复、完整性保护和失败现场保留。"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b26_restore', ROOT / 'scripts/restore_model.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.set_num_threads(2)

class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='b26-test-')
        self.root = Path(self.temp.name)
        self.base = self.root / 'base'
        self.base.mkdir()
        self.experts = [[i % 40, i // 40] for i in range(384)]
        self.weights = {f'model.language_model.layers.{i}.mlp.shared_expert.down_proj.weight': (torch.arange(6).reshape(3, 2) + 700 + i).bfloat16() for i in range(40)}
        groups = [{}, {}, {}]
        for i in range(40):
            prefix = f'model.language_model.layers.{i}.mlp'
            groups[i % 3][prefix + '.shared_expert.down_proj.weight'] = torch.full((3, 2), -i - 1, dtype=torch.bfloat16)
            groups[i % 3][prefix + '.experts.down_proj'] = (torch.arange(256 * 6).reshape(256, 3, 2) + i * 4096).bfloat16()
            groups[i % 3][prefix + '.gate.weight'] = torch.full((256, 2), i + 1, dtype=torch.bfloat16)
        groups[0]['mtp.layers.0.mlp.shared_expert.down_proj.weight'] = torch.full((3, 2), 901, dtype=torch.bfloat16)
        groups[1]['model.visual.test.weight'] = torch.eye(2, dtype=torch.bfloat16)
        mapping = {}
        for i, tensors in enumerate(groups):
            name = f'model-{i + 1:05d}-of-00003.safetensors'
            save_file(tensors, str(self.base / name), metadata={'format': 'pt', '测试': '仅合成数据'})
            mapping.update({k: name for k in tensors})
        self.index = {'metadata': {'total_size': sum(t.numel() * t.element_size() for g in groups for t in g.values())}, 'weight_map': mapping}
        (self.base / 'model.safetensors.index.json').write_text(json.dumps(self.index))
        (self.base / 'config.json').write_text('{"说明":"这是合成测试，不是真实模型"}')
        self.base_manifest = {'files': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': m.file_sha256(p)} for p in sorted(self.base.iterdir())]}
        self.payload = {'locked_step': 26, 'parameter_states': self.weights, 'ablated_experts': self.experts, 'base_model_inventory_sha256': '合成基模身份'}
        self.checkpoint = self.root / 'B26_locked_BF16.pt'
        torch.save(self.payload, self.checkpoint)
        self.package = self.root / 'B26_restore_weights_v1.zip'
        with zipfile.ZipFile(self.package, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(self.checkpoint, self.checkpoint.name)
        self.manifest = {'base_inventory_sha256': '合成基模身份', 'checkpoint': {'filename': self.checkpoint.name, 'sha256': m.file_sha256(self.checkpoint), 'bytes': self.checkpoint.stat().st_size}, 'archive': {'sha256': m.file_sha256(self.package)}, 'ablated_experts': self.experts, 'shared_tensors': {name: {'shape': list(t.shape), 'sha256': m.tensor_sha256(t)} for name, t in self.weights.items()}}

    def tearDown(self):
        self.temp.cleanup()

    def test_完整恢复且保持其余张量(self):
        payload = m.load_checkpoint(self.package, self.manifest, True)
        with contextlib.redirect_stdout(io.StringIO()):
            index = m.verify_base(self.base, self.base_manifest)
            report = m.restore(self.base, self.root / 'restored', payload, self.base_manifest, self.manifest, index)
        self.assertEqual(report['状态'], '恢复完成，全部输出已校验')
        self.assertFalse((self.root / 'restored/RECOVERY_INCOMPLETE.json').exists())
        selected = m.routed_groups(self.experts)
        for shard in set(self.index['weight_map'].values()):
            with safe_open(str(self.base / shard), framework='pt') as before, safe_open(str(self.root / 'restored' / shard), framework='pt') as after:
                self.assertEqual(set(before.keys()), set(after.keys()))
                self.assertEqual(before.metadata(), after.metadata())
                for key in before.keys():
                    a, b = before.get_tensor(key), after.get_tensor(key)
                    if key in self.weights:
                        self.assertTrue(torch.equal(b, self.weights[key]))
                    elif key in selected:
                        for expert in range(256):
                            self.assertTrue(torch.count_nonzero(b[expert]) == 0 if expert in selected[key] else torch.equal(a[expert], b[expert]))
                    else:
                        self.assertTrue(torch.equal(a, b))
        for entry in self.base_manifest['files']:
            self.assertEqual(m.file_sha256(self.base / entry['path']), entry['sha256'])
        for name in ['config.json', 'model.safetensors.index.json']:
            self.assertEqual((self.base / name).read_bytes(), (self.root / 'restored' / name).read_bytes())

    def test_未选入的整层专家保持原样(self):
        self.experts = [[layer, expert] for expert in range(10) for layer in range(40) if layer != 2][:384]
        self.payload['ablated_experts'] = self.experts
        self.manifest['ablated_experts'] = self.experts
        torch.save(self.payload, self.checkpoint)
        self.manifest['checkpoint'].update(sha256=m.file_sha256(self.checkpoint), bytes=self.checkpoint.stat().st_size)
        payload = m.load_checkpoint(self.checkpoint, self.manifest, False)
        with contextlib.redirect_stdout(io.StringIO()):
            m.restore(self.base, self.root / 'without_layer2', payload, self.base_manifest, self.manifest, self.index)
        key = 'model.language_model.layers.2.mlp.experts.down_proj'
        shard = self.index['weight_map'][key]
        with safe_open(str(self.base / shard), framework='pt') as before, safe_open(str(self.root / 'without_layer2' / shard), framework='pt') as after:
            self.assertTrue(torch.equal(before.get_tensor(key), after.get_tensor(key)))

    def test_异地工作目录与空格路径可校验(self):
        resources = self.root / '工具目录 空格'
        (resources / 'checksums').mkdir(parents=True)
        bm = dict(self.base_manifest, original_inventory_sha256='合成基模身份')
        (resources / 'checksums/base_manifest.json').write_text(json.dumps(bm))
        (resources / 'checksums/checkpoint_manifest.json').write_text(json.dumps(self.manifest))
        work = self.root / '运行目录 空格'
        work.mkdir()
        renamed = self.root / '基模 有空格'
        self.base.rename(renamed)
        argv = ['restore_model.py', '--base', '../基模 有空格', '--checkpoint', '../B26_locked_BF16.pt', '--verify-only']
        with mock.patch.object(m, 'ROOT', resources), mock.patch.object(sys, 'argv', argv), contextlib.chdir(work), contextlib.redirect_stdout(io.StringIO()):
            m.main()
        self.assertFalse((work / 'recovery_report.json').exists())

    def test_拒绝断开的输出符号链接(self):
        target = self.root / '不应创建的目录'
        link = self.root / '断开的符号链接'
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            m.restore(self.base, link, self.payload, self.base_manifest, self.manifest, self.index)
        self.assertTrue(link.is_symlink())
        self.assertFalse(target.exists())

    def test_无效输出在读取大模型前拒绝(self):
        argv = ['restore_model.py', '--base', str(self.base), '--checkpoint', str(self.checkpoint), '--output', str(self.base / '内部输出')]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(m, 'verify_base') as verify, mock.patch.object(m, 'load_checkpoint') as load:
            with self.assertRaisesRegex(ValueError, '输出目录'):
                m.main()
            verify.assert_not_called()
            load.assert_not_called()

    def test_空间不足不创建输出(self):
        target = self.root / '空间不足时不创建'
        with mock.patch.object(m.shutil, 'disk_usage', return_value=type('容量', (), {'free': 0})()):
            with self.assertRaisesRegex(ValueError, '磁盘空间不足'):
                m.restore(self.base, target, self.payload, self.base_manifest, self.manifest, self.index)
        self.assertFalse(target.exists())

    def test_单独检查点入口(self):
        self.assertEqual(m.load_checkpoint(self.checkpoint, self.manifest, False)['locked_step'], 26)

    def test_拒绝损坏的包(self):
        self.package.write_bytes(self.package.read_bytes() + b'bad')
        with self.assertRaisesRegex(ValueError, '哈希不符'):
            m.load_checkpoint(self.package, self.manifest, True)

    def test_拒绝额外的压缩包成员(self):
        with zipfile.ZipFile(self.package, 'a') as archive:
            archive.writestr('多余文件.txt', '不能混入其它文件')
        self.manifest['archive']['sha256'] = m.file_sha256(self.package)
        with self.assertRaisesRegex(ValueError, '文件清单不符'):
            m.load_checkpoint(self.package, self.manifest, True)

    def test_拒绝错误步数(self):
        payload = dict(self.payload, locked_step=25)
        with self.assertRaisesRegex(ValueError, '第 26 步'):
            m.validate_checkpoint(payload, self.manifest)

    def test_拒绝重复或不同专家(self):
        payload = copy.deepcopy(self.payload)
        payload['ablated_experts'][0] = payload['ablated_experts'][1]
        with self.assertRaisesRegex(ValueError, '名单不符'):
            m.validate_checkpoint(payload, self.manifest)

    def test_拒绝错误权重内容(self):
        payload = copy.deepcopy(self.payload)
        next(iter(payload['parameter_states'].values()))[0, 0] += 10
        with self.assertRaisesRegex(ValueError, '张量哈希不符'):
            m.validate_checkpoint(payload, self.manifest)

    def test_拒绝改变过的基模(self):
        (self.base / 'config.json').write_text('内容已改变')
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            m.verify_base(self.base, self.base_manifest)

    def test_拒绝路径越界(self):
        with self.assertRaisesRegex(ValueError, '非法相对路径'):
            m.checked_path(self.base, '../outside')

    def test_拒绝覆盖及修改基模(self):
        with self.assertRaisesRegex(ValueError, '输出目录'), contextlib.redirect_stdout(io.StringIO()):
            m.restore(self.base, self.base / 'output', self.payload, self.base_manifest, self.manifest, self.index)
        existing = self.root / 'exists'
        existing.mkdir()
        (existing / '保留.txt').write_text('原文件')
        with self.assertRaisesRegex(ValueError, '输出目录已存在'), contextlib.redirect_stdout(io.StringIO()):
            m.restore(self.base, existing, self.payload, self.base_manifest, self.manifest, self.index)
        self.assertEqual((existing / '保留.txt').read_text(), '原文件')

    def test_失败时留下明确标记(self):
        output = self.root / 'failed'
        with mock.patch.object(m, 'write_patched_shard', side_effect=RuntimeError('模拟写入失败')):
            with self.assertRaisesRegex(RuntimeError, '模拟写入失败'), contextlib.redirect_stdout(io.StringIO()):
                m.restore(self.base, output, self.payload, self.base_manifest, self.manifest, self.index)
        marker = m.read_json(output / 'RECOVERY_INCOMPLETE.json')
        self.assertIn('恢复失败', marker['状态'])
        self.assertFalse((output / 'recovery_report.json').exists())

if __name__ == '__main__':
    unittest.main(verbosity=2)
