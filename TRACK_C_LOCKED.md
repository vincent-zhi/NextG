# Track C 最终锁定配置（2026-11-24）

来源：计划原文参数匹配有误（MoE 预算 66.8M 漏扣边界 Attention 12.853M 与 Tape 1.051M，按原文实现超 117.0M 目标约 6M）。经 track_c_fix.py 脚本复核 + 独立重算，三方数字一致后锁定如下。两个尺度的总参均**略低于**基线目标（刻意，避免"靠多参数赢"的质疑），按计划 0.1M 取整口径仍报 25.7M / 117.0M。

## Tier 0（25.587M non-emb，目标 25.7M，-0.44%）
- d=512，L_pre/core/post = 2/4/2
- micro_ff = 448（micro/dense = 448/1408 = 31.8%，与 Tier 1 的 31.6% 同构；MoE 占非嵌入参数 54.0%，Tier 1 为 52.0%）
- MoE：8 experts，top-2，expert_ff = 280（8 的倍数，Tensor Core 最低对齐；精确匹配值原为 282）
- 明细：micro core 4 层 6.951M + 边界 Attention 4 层 4.199M + MoE 13.779M + Tape 0.657M + final norm 512
- FLOP 校核（L_seq=2048，含因果 Attention）：核心线性投影降 45.9%，计入 Attention 后单层总 FLOP 降 34.6%，守住 ≥30% recurrent FLOP 缩减门槛（对比：选项一 768/242 仅降 23.0%，不达标，且 micro 比例 54.5% 破坏跨尺度同构，已否决）

## Tier 1（116.709M non-emb，目标 117.0M，-0.25%）
- d=896，L_pre/core/post = 2/8/2
- micro_ff = 768，MoE：8 experts，top-2，expert_ff = 704（64 的倍数；精确匹配值原为 707，奇数严重不对齐）
- 明细：micro core 8 层 42.221M + 边界 Attention 4 层 12.853M + MoE 60.584M + Tape 1.051M + final norm 896

## Tape 实现约束（写代码时必须遵守）
- 规格：m=8 槽，d_m=256，每 loop 共享一套（不是每层一套）。参数口径（Tier 1）：写查询 8×256 + 写 KV 投影 d→2·d_m + 读 q/k/v/out 四投影 = 1.051M。若更换 Tape 实现，须重跑 track_c_fix.py 更新数字。
- **因果性（硬约束）**：写池化禁止对整条序列做全局 cross-attention pooling（会把未来 token 信息写入槽位，造成跨 loop 因果泄露，Phase 0 Bit-Exact Causality Test 必挂）。必须按固定块（chunk size C=128 或 256）分段写入：每个已完成块产出 m 个槽位；读取加严格块级因果 mask，token i 只能读结束位置 ≤ i 的历史块槽位，且只读前序 loop 已写入的内容，与单步自回归解码保持 bit-exact 一致。
- **门控**：读回残差带零初始化标量/向量门控（保证 step 0 与 Flat-Loop 严格等价，过 Identity-Init 检查）；另设擦除衰减门 γ_t。两项合计 <0.003M，不影响 0.1M 取整，但实现时不得遗漏。
- **分析注意**：两尺度同用 d_m=256 时，Tier 0 的 d_m/d = 50%，Tier 1 为 28.6%——Tier 0 的 Tape 相对带宽更大，Phase 1 的 V 变量扩容增益不得线性外推到 Tier 1。
