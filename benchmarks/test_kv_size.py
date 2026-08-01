"""kv_size 的测试。

断言的期望值来自 experiments/configs/ 中的真实 config.json 快照，
每个用例同时钉住一个已知陷阱（见 docs/kv-cache/03-sizing.md 第 3 节）。

运行：python -m unittest discover -s benchmarks -v
"""

import unittest

from kv_size import (
    ConfigFieldError,
    UnsupportedArchitecture,
    kv_spec,
    load_config,
    total_bytes,
)


class TestGQAStandard(unittest.TestCase):
    """标准 HF 命名的 GQA：num_hidden_layers / num_key_value_heads。"""

    def test_qwen(self):
        # 28 层 × 2(K,V) × 4 头 × 128 × 2 字节
        spec = kv_spec(load_config("qwen2.5-7b-instruct-awq"))
        self.assertEqual(spec.scheme, "GQA")
        self.assertEqual(spec.bytes_per_token, 57_344)
        self.assertEqual(spec.constant_bytes, 0)

    def test_llama(self):
        # 32 层 × 2 × 8 头 × 128 × 2 字节
        spec = kv_spec(load_config("llama-3.1-8b-instruct-awq"))
        self.assertEqual(spec.scheme, "GQA")
        self.assertEqual(spec.bytes_per_token, 131_072)

    def test_head_dim_derived_when_absent(self):
        """陷阱 1：Qwen/Llama 无 head_dim 字段，需由 hidden_size/num_attention_heads 推导。"""
        spec = kv_spec(load_config("qwen2.5-7b-instruct-awq"))
        self.assertEqual(spec.detail["head_dim"], 128)
        self.assertIn("推导", spec.detail["head_dim_source"])

    def test_qwen_sliding_window_is_disabled(self):
        """陷阱 4：Qwen 有 sliding_window 字段但 use_sliding_window 为 false，不应走滑窗分支。"""
        cfg = load_config("qwen2.5-7b-instruct-awq")
        self.assertIn("sliding_window", cfg)          # 字段确实存在
        self.assertFalse(cfg["use_sliding_window"])   # 但被关闭
        self.assertEqual(kv_spec(cfg).constant_bytes, 0)  # 按全注意力处理

    def test_awq_quantization_does_not_affect_kv(self):
        """权重量化与 KV 精度无关：AWQ 4-bit 模型的 KV 仍为 2 字节。"""
        cfg = load_config("qwen2.5-7b-instruct-awq")
        self.assertEqual(cfg["quantization_config"]["bits"], 4)
        self.assertEqual(kv_spec(cfg).detail["dtype_bytes"], 2)


class TestGQAChatGLM(unittest.TestCase):
    """陷阱 5：ChatGLM 使用另一套字段名，标准 key 不存在。"""

    def test_glm(self):
        # 40 层 × 2 × 2 组 × 128 × 2 字节
        spec = kv_spec(load_config("glm-4-9b-chat"))
        self.assertEqual(spec.scheme, "GQA")
        self.assertEqual(spec.bytes_per_token, 40_960)

    def test_standard_field_absent(self):
        cfg = load_config("glm-4-9b-chat")
        self.assertNotIn("num_key_value_heads", cfg)
        self.assertEqual(cfg["multi_query_group_num"], 2)


class TestMLA(unittest.TestCase):
    """陷阱 3：DeepSeek 含 GQA 风格的残留字段，误用会高估 8.9 倍。"""

    def test_deepseek(self):
        # 27 层 × (kv_lora_rank 512 + qk_rope_head_dim 64) × 2 字节
        # 注意：不乘 2（K/V 联合压缩），不乘头数（全头共享）
        spec = kv_spec(load_config("deepseek-v2-lite-chat"))
        self.assertEqual(spec.scheme, "MLA")
        self.assertEqual(spec.bytes_per_token, 31_104)

    def test_must_not_use_gqa_formula(self):
        """回归防护：若误按 GQA 公式计算会得到 276_480。"""
        cfg = load_config("deepseek-v2-lite-chat")
        self.assertEqual(cfg["num_key_value_heads"], 16)  # 残留字段确实存在
        naive = cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * (
            (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]) + cfg["v_head_dim"]
        ) * 2
        self.assertEqual(naive, 276_480)
        self.assertNotEqual(kv_spec(cfg).bytes_per_token, naive)

    def test_mla_is_most_economical(self):
        """MLA 应低于所有 GQA 主线模型（README 光谱排序的依据）。"""
        mla = kv_spec(load_config("deepseek-v2-lite-chat")).bytes_per_token
        for name in ("glm-4-9b-chat", "qwen2.5-7b-instruct-awq",
                     "llama-3.1-8b-instruct-awq"):
            self.assertLess(mla, kv_spec(load_config(name)).bytes_per_token, name)


class TestSlidingWindow(unittest.TestCase):
    """GPT-OSS：12 层全注意力线性增长 + 12 层滑窗封顶。"""

    def test_gpt_oss_two_regimes(self):
        spec = kv_spec(load_config("gpt-oss-20b"))
        self.assertEqual(spec.scheme, "SWA")
        self.assertEqual(spec.bytes_per_token, 24_576)      # 12 层 × 2 × 8 × 64 × 2
        self.assertEqual(spec.constant_bytes, 3_145_728)    # 12 层 × 2048 × 128

    def test_head_dim_explicit_wins_over_derived(self):
        """陷阱 2：显式 head_dim=64，而推导值为 2880/64=45，盲目推导会低估 42%。"""
        cfg = load_config("gpt-oss-20b")
        self.assertEqual(cfg["hidden_size"] // cfg["num_attention_heads"], 45)
        spec = kv_spec(cfg)
        self.assertEqual(spec.detail["head_dim"], 64)
        self.assertIn("显式", spec.detail["head_dim_source"])

    def test_missing_torch_dtype_defaults_to_2(self):
        """陷阱 6：GPT-OSS 无 torch_dtype 字段，应回落到默认 2 字节并标注。"""
        cfg = load_config("gpt-oss-20b")
        self.assertNotIn("torch_dtype", cfg)
        self.assertEqual(kv_spec(cfg).detail["dtype_bytes"], 2)
        self.assertIn("默认", kv_spec(cfg).detail["dtype_source"])


class TestTotalBytes(unittest.TestCase):
    def test_gqa_is_linear(self):
        spec = kv_spec(load_config("qwen2.5-7b-instruct-awq"))
        self.assertEqual(total_bytes(spec, 1), 57_344)
        self.assertEqual(total_bytes(spec, 32_768), 57_344 * 32_768)

    def test_sliding_window_caps_at_window_size(self):
        spec = kv_spec(load_config("gpt-oss-20b"))
        # 上下文超过窗口：滑窗部分封顶为常数
        self.assertEqual(total_bytes(spec, 32_768), 24_576 * 32_768 + 3_145_728)
        # 上下文短于窗口：滑窗部分仍按实际长度线性增长
        self.assertEqual(total_bytes(spec, 64), 24_576 * 64 + 12 * 2048 * 64)

    def test_dtype_override_halves_size(self):
        """--kv-dtype fp8 应使体积减半。"""
        cfg = load_config("qwen2.5-7b-instruct-awq")
        self.assertEqual(kv_spec(cfg, dtype_bytes=1).bytes_per_token, 28_672)


class TestErrors(unittest.TestCase):
    def test_unknown_model_type_raises(self):
        with self.assertRaises(UnsupportedArchitecture):
            kv_spec({"model_type": "some_future_arch", "num_hidden_layers": 1})

    def test_missing_required_field_raises(self):
        """缺字段应明确报错，而非静默使用默认值。"""
        with self.assertRaises(ConfigFieldError):
            kv_spec({"model_type": "llama", "num_hidden_layers": 32})


if __name__ == "__main__":
    unittest.main()
