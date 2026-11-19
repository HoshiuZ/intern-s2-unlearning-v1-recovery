# Intern-S2 B@26 恢复工具

本仓库用于将**完整的 Intern-S2-Preview 基模**与单独分发的 B@26 恢复包，恢复为可加载的完整模型目录。恢复包通过微信另行提供，仓库中不包含恢复包或模型权重。

## 仓库目录

```text
intern-s2-unlearning-v1-recovery/
├── README.md                         # 仓库目录、各部分用途和恢复入口
├── scripts/                          # 实际执行恢复的程序
│   └── restore_model.py              # 校验输入，逐分片恢复权重，并校验写出的完整模型
├── checksums/                        # 固定基模、恢复包和验证结果的清单
│   ├── base_manifest.json            # 基模全部文件的大小与 SHA256，含 23 个权重分片
│   ├── checkpoint_manifest.json      # ZIP/检查点哈希、40 个共享矩阵身份及 384 个专家名单
│   └── validation_report.json        # 本仓库发布前实际完成的验证记录与验证范围
├── docs/                             # 中文使用与校验文档
│   ├── 使用说明.md                   # 环境准备、输入要求、恢复命令及恢复后加载方法
│   ├── 校验说明.md                   # 哈希与张量校验方法、输出报告和失败处理
│   └── 验证记录.md                   # 已做的测试，以及未做完整模型落盘验证的说明
├── tests/                            # 不依赖训练数据的 CPU 测试
│   ├── test_restore_model.py         # 合成分片端到端测试，覆盖专家置零和保护原文件等逻辑
│   └── validate_real_inputs.py       # 对真实恢复包和基模目标张量进行 CPU 检查
├── requirements.txt                  # 离线恢复所需依赖
├── requirements-inference.txt        # 可选：恢复完成后加载模型的主要推理依赖
└── .gitignore                        # 防止权重、ZIP、环境及本地恢复输出进入仓库
```

## 使用入口

准备好基模与微信收到的 `B26_restore_weights_v1.zip`，安装依赖后运行：

```bash
python scripts/restore_model.py \
  --base /path/to/Intern-S2-Preview \
  --package /path/to/B26_restore_weights_v1.zip \
  --output /path/to/Intern-S2-B26
```

恢复程序仅使用 CPU，按分片处理，不需要在 GPU 上加载完整模型。建议准备 **24 GiB 内存**及 **90 GB 可用输出空间**；基模和输出必须分开存放。完整恢复后的权重约 **73.23 GB**，输出保留基模原有配置、分词器、自定义代码与多模态文件。

具体安装步骤、仅校验模式、已解压检查点的用法，以及恢复后的加载方式见 [使用说明](docs/使用说明.md)。
