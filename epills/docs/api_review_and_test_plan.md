# PharmaScan API — Review, Changes & Verification Backlog

> Based on the team-provided **PharmaScan API OpenAPI v0.4.0** and user-shared screenshot of one-pill `/scan` producing two overlapping boxes. Endpoint existence is **documented**, not a claim that its behavior is verified. API service version 0.4.0 is not automatically the model version.

## Decisions

1. **Owner of patient data:** main application PostgreSQL + private object storage; hosted API stateless `/match` computations unless its owner documents otherwise.
2. **Owner of medication plan and dose rules:** main backend deterministic policy, not generic vision API.
3. **Vision service:** embeddings and visual candidate retrieval only; no '99% confident' from cosine similarity.
4. **Image modes:** printed package OCR; loose pill embeddings; unlabeled blister reference only experimental; fully opaque unlabeled packaging abstain.
5. **No retraining from patient gallery:** `/feedback` is separate, opt-in and vetted.

## Endpoint decisions

| Method | Endpoint | Decision | Priority | Change / concern | Acceptance test |
|---|---|---|---|---|---|
| GET | /health | 保留 | P0 | 返回模型/目录可用状态；明确冷启动、版本及响应 schema；检测网络与延迟 | 健康检查返回准确；失败、冷启动不会伪装正常 |
| POST | /v1/embed | 保留 + 明确契约 | P0 | 只支持已裁剪裸片；返回 256-d 及 embedding_space_id、model_version、preprocess_version、normalized | 两次同图向量一致性；维度、NaN、预处理版本验证 |
| POST | /v1/match | MVP 保留 | P0 | Gallery 为请求载荷，不是 HF 服务器持久图库；明确 threshold、margin、tie/unknown 结果；按版本拒绝跨空间对比 | 同药/异药/相似规格/未知药 FPR、FNR、ambiguous 测试 |
| POST | /v1/identify | 保留候选检索 | P0 | 返回可能的美国目录 Top-k；需要 imprint/shape/colour 约束与非分数概率声明 | 真实 C3PI 标签 ground truth；Top-k、刻印消融、非目录药拒识 |
| POST | /v1/scan | 仅候选/调试，不能直接安全核验 | P0 | 现有 stand-in finder 一颗药检出两个框；拆出 detection 和 dose rule；不要把目录 score 当确认 | 单颗变双框回归测试、NMS、数量与背景测试、结构化 JSON schema |
| GET | /v1/pills/search | 保留 | P1 | 按药名、规格查询；返回产品 ID/市场/品牌/强度/剂型，防止同成分不同产品合并 | 500mg vs 850mg、IR vs XR、组合药排序 |
| GET | /v1/pills/{pill_id} | 保留 | P1 | 目录产品详情；将产品/strength/form/market 的标识完整返回 | 目录缺字段/不可查询 ID 的处理 |
| GET | /v1/pills/{pill_id}/image/{n} | 保留为辅助视觉参考 | P2 | C3PI 研发使用限制、产品图像不一定对应本地患者持有实物 | 图片 URL、许可说明和 404 处理 |
| POST | /v1/feedback | MVP 不默认调用 | P2 | 和患者自用 Gallery 严格分离；仅显式 opt-in、可撤回，label_source 需审核队列 | 无同意拒绝、重复上传、撤销后删除、未核验标签隔离 |
| DELETE | /v1/feedback/{consent_id} | 保留撤回功能 | P2 | 服务端实际删除的持久化验证及审计机制；不要预先声称删除成功 | 上传→撤回→确认不可恢复、重试幂等 |
| POST | /v1/safety/medications | 保留兼容，安全服务逻辑拆分 | P1 | DDInter 2.0 NC 授权；只有可映射成分可筛查；输出 checked/unchecked coverage | 已知相互作用阳性/阴性/未知成分，空告警不能证明安全 |
| POST | /v1/safety/dose | 迁移核心规则至主 Backend | P0 | 用药计划与患者事件只有主 App 知道；支持 actual time、PRN、停药和用户未确认 | 多次确认、历史停药、错规格、重复 dose、无照片事件 |
| GET | /v1/safety/label | 迁至 Medication Knowledge 工具 | P1 | 来源于美国 openFDA；不要将普通标签说明当个人化医嘱 | 产品链接准确性、缺失标签、来源日期和引用 |
| POST | /v1/detect | 建议新增（团队协商） | P1 | 单独输出 boxes、pill_count、NMS、quality_flags，或暂由主 App 提供 boxes | 1 pill→1 box, overlap, 2 pills→2 boxes, min box size |
| GET | /v1/model-info | 建议新增 | P0 | 服务版本≠模型版本。暴露 embedding_space_id、preprocess、阈值/边际、训练覆盖范围 | 端到端向量版本校验；升级后旧向量提示 re-embed |
| POST | /v1/search-by-embedding | 未来优化 | P2 | 避免 /embed 后 /identify 重跑 encoder，前提是目录搜索兼容 embedding | 与原 /identify 候选排序一致性/性能 |
| App | /api/medications/enroll | 主应用新建 | P0 | OCR、查重、draft、患者确认、Regimen 历史保留 | 缺失用法不得激活确定性 dose validation |
| App | /api/dose-check | 主应用新建 | P0 | Agent context + modality routing + external vision + deterministic policy + event | match 和 intake 两个状态分离，错误检测框不计数 |
| App | /api/agent/session | 主应用新建 | P0 | 受约束 tool-calling、恢复 state、跨会话上下文、权限与 tool-trace | 未知输入澄清、工具失败重试、重复写保护 |
| App | /api/care/summary | 主应用新建 | P1 | 长期趋势、health tracking、既定预防任务、就诊准备 | 不能从空记录推断临床依从率 |

## Blocking issues & experiment plans

### P0: 1 photo → 2 detections regression
- Capture the exact TEVA original image and response JSON, note model/version/crop and current requested `boxes`. Output displayed two overlapping boxes for one pill. Add NMS, minimum separation and count consistency or require one-patient-selected crop.
- If `/scan` stand-in boxes remain unreliable, **do not infer observed pill count or dose check result from them**; use single-pill `/embed` or `/identify` workflows while Stage 1 improves.

### P0: False high similarity / open-set rejection
- The shown scores 0.97–0.99 are similarity scores for plausible-looking but semantically conflicting catalogue candidates. Confirm similarity math (cosine? ANN? normalized?) and provide false acceptance / rejection distribution with held-out samples, calibration provenance and ambiguous thresholds.
- Test with true labels, held-out phones/backgrounds, different strengths, off-catalogue drugs, same imprint manufacturer and cases where shape/colour are indistinguishable.

### P0: API request and response schemas
- OpenAPI declares many 200 responses as `{}`. Capture actual `/embed`, `/match`, `/identify`, `/scan`, `/safety/*` responses, create Pydantic response schemas and contract tests; use timeout/retry/backoff, no fake successful fallbacks.
- Provide `/v1/model-info`: unique embedding space identifier, model weights version, preprocessing/crop normalization version, threshold and margin with evaluation provenance.

### P1: Safety API & licensing
- `DDInter 2.0` is declared CC BY-NC-SA 4.0 by API; **do not assume commercial rights**. Track per-ingredient mapping coverage; empty alerts do not establish safety.
- `/v1/safety/dose` can remain as comparison/demo but authoritative local validation must read actual current regimen, PRN, timestamps, previously logged events and record source status.

### P1: Gallery storage / privacy
- Store own photos and vectors with patient isolation; send only relevant query/reference vectors to external model when possible; test `/match` inputs to avoid sending full personal longitudinal history.
- Distinguish `product_id`, `patient_medication_id`, `reference_set_id` and `regimen_id`; do not assume matching image determines strength/dosage form or market.

## Test cases (minimum)

| Case | Input | Expected |
|---|---|---|
| D01 | 1 pill photo with no boxes | At most one actual pill detection, no duplicate count |
| D02 | Two separated pills | Two non-overlapping unique detections |
| D03 | Cropped image one pill | `/embed` returns finite vector with declared dimension/version |
| M01 | Same product, new photo | Personal match or calibrated ambiguous, measured error rates |
| M02 | Wrong strength same appearance | No automatic identification of strength without label evidence |
| M03 | Out-of-gallery/unknown pill | Unknown or ambiguous; no forced match |
| I01 | C3PI ground truth with known imprint | Top-k accuracy measured and recorded |
| I02 | Unknown US/non-US pill | Visual candidates not claimed as verified |
| P01 | Printed box/label | Extracted text with field provenance and user review |
| P02 | Unprinted clear blister | No unvalidated Stage 2 pill match |
| P03 | Opaque unprinted foil | Ask for original source / professional, abstain |
| R01 | Discontinued or changed regimen | No old schedule automatically considered active |
| R02 | Duplicate dose report | Idempotent event, warning based on verified history |
| R03 | Not photographed / self-reported | No visual verification flag; intake self-report retained |
| R04 | Patient forgot whether they took it | `unsure`, no inferred missed dose or auto-redose |
| A01 | Vision API timeout | Agent acknowledges failure, retains task state |
| A02 | User uploads ambiguous image | Agent asks intent and selects appropriate tool after reply |
| A03 | Enrollment source missing frequency | Draft only; ask for original instruction |
| S01 | DDI with unknown ingredient | `unchecked` coverage; no 'no interaction' claim |
| C01 | Caregiver not authorised/verified | Block notification and log denial |

## Evidence for judging
- Save real WorkBuddy development chat, actual schema/adapter diffs, failing and passing test output, and truthful limitation notes. Keep identity and sensitive data out of screenshots.